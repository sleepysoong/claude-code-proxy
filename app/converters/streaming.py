"""Stream an LiteLLM async generator as Anthropic SSE events."""

from __future__ import annotations

import json
import logging
import traceback
import uuid
from typing import Any, AsyncIterator

from app.models.request import MessagesRequest

logger = logging.getLogger("app")


async def handle_streaming(
    response_generator: AsyncIterator[Any],
    original_request: MessagesRequest,
):
    """Yield Anthropic-compatible SSE events from a LiteLLM streaming response."""
    message_id = f"msg_{uuid.uuid4().hex[:24]}"

    # --- message_start ---
    yield _sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": original_request.model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 0,
                },
            },
        },
    )

    # Open text content block [0]
    yield _sse(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
    )
    yield _sse("ping", {"type": "ping"})

    # Mutable state
    tool_index: int | None = None
    accumulated_text = ""
    text_sent = False
    text_block_closed = False
    input_tokens = 0
    output_tokens = 0
    has_sent_stop = False
    last_tool_index = 0
    anthropic_tool_index = 0

    try:
        async for chunk in response_generator:
            try:
                # Usage
                if hasattr(chunk, "usage") and chunk.usage is not None:
                    input_tokens = getattr(chunk.usage, "prompt_tokens", input_tokens)
                    output_tokens = getattr(
                        chunk.usage, "completion_tokens", output_tokens
                    )

                if not (hasattr(chunk, "choices") and len(chunk.choices) > 0):
                    continue

                choice = chunk.choices[0]
                delta = getattr(choice, "delta", getattr(choice, "message", {}))
                finish_reason = getattr(choice, "finish_reason", None)

                # --- Text delta ---
                delta_content = (
                    getattr(delta, "content", None)
                    if not isinstance(delta, dict)
                    else delta.get("content")
                )
                if delta_content:
                    accumulated_text += delta_content
                    if tool_index is None and not text_block_closed:
                        text_sent = True
                        yield _sse(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": 0,
                                "delta": {"type": "text_delta", "text": delta_content},
                            },
                        )

                # --- Tool-call deltas ---
                delta_tool_calls = (
                    getattr(delta, "tool_calls", None)
                    if not isinstance(delta, dict)
                    else delta.get("tool_calls")
                )

                if delta_tool_calls:
                    if tool_index is None:
                        # Close text block before first tool
                        if text_sent and not text_block_closed:
                            text_block_closed = True
                            yield _sse(
                                "content_block_stop",
                                {"type": "content_block_stop", "index": 0},
                            )
                        elif (
                            accumulated_text and not text_sent and not text_block_closed
                        ):
                            text_sent = True
                            yield _sse(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": 0,
                                    "delta": {
                                        "type": "text_delta",
                                        "text": accumulated_text,
                                    },
                                },
                            )
                            text_block_closed = True
                            yield _sse(
                                "content_block_stop",
                                {"type": "content_block_stop", "index": 0},
                            )
                        elif not text_block_closed:
                            text_block_closed = True
                            yield _sse(
                                "content_block_stop",
                                {"type": "content_block_stop", "index": 0},
                            )

                    if not isinstance(delta_tool_calls, list):
                        delta_tool_calls = [delta_tool_calls]

                    for tc in delta_tool_calls:
                        cur_idx = _get_attr(tc, "index", 0)

                        if tool_index is None or cur_idx != tool_index:
                            tool_index = cur_idx
                            last_tool_index += 1
                            anthropic_tool_index = last_tool_index

                            fn = _get_attr(tc, "function", {})
                            name = _get_attr(fn, "name", "") if fn else ""
                            tid = _get_attr(tc, "id", f"toolu_{uuid.uuid4().hex[:24]}")

                            yield _sse(
                                "content_block_start",
                                {
                                    "type": "content_block_start",
                                    "index": anthropic_tool_index,
                                    "content_block": {
                                        "type": "tool_use",
                                        "id": tid,
                                        "name": name,
                                        "input": {},
                                    },
                                },
                            )

                        fn = _get_attr(tc, "function", {})
                        arguments = _get_attr(fn, "arguments", "") if fn else ""
                        if arguments:
                            yield _sse(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": anthropic_tool_index,
                                    "delta": {
                                        "type": "input_json_delta",
                                        "partial_json": arguments,
                                    },
                                },
                            )

                # --- Finish ---
                if finish_reason and not has_sent_stop:
                    has_sent_stop = True

                    if tool_index is not None:
                        for i in range(1, last_tool_index + 1):
                            yield _sse(
                                "content_block_stop",
                                {"type": "content_block_stop", "index": i},
                            )

                    if not text_block_closed:
                        if accumulated_text and not text_sent:
                            yield _sse(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": 0,
                                    "delta": {
                                        "type": "text_delta",
                                        "text": accumulated_text,
                                    },
                                },
                            )
                        yield _sse(
                            "content_block_stop",
                            {"type": "content_block_stop", "index": 0},
                        )

                    stop_map = {
                        "stop": "end_turn",
                        "length": "max_tokens",
                        "tool_calls": "tool_use",
                    }
                    stop_reason = stop_map.get(finish_reason, "end_turn")

                    yield _sse(
                        "message_delta",
                        {
                            "type": "message_delta",
                            "delta": {
                                "stop_reason": stop_reason,
                                "stop_sequence": None,
                            },
                            "usage": {"output_tokens": output_tokens},
                        },
                    )
                    yield _sse("message_stop", {"type": "message_stop"})
                    yield "data: [DONE]\n\n"
                    return

            except Exception:
                logger.error(f"Error processing chunk: {traceback.format_exc()}")
                continue

        # Generator exhausted without finish_reason
        if not has_sent_stop:
            if tool_index is not None:
                for i in range(1, last_tool_index + 1):
                    yield _sse(
                        "content_block_stop", {"type": "content_block_stop", "index": i}
                    )
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
            yield _sse(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": output_tokens},
                },
            )
            yield _sse("message_stop", {"type": "message_stop"})
            yield "data: [DONE]\n\n"

    except Exception:
        logger.error(f"Streaming error: {traceback.format_exc()}")
        yield _sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "error", "stop_sequence": None},
                "usage": {"output_tokens": 0},
            },
        )
        yield _sse("message_stop", {"type": "message_stop"})
        yield "data: [DONE]\n\n"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _get_attr(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
