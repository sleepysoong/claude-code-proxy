"""Convert Anthropic-format requests to LiteLLM (OpenAI-compatible) format."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from app.models.request import MessagesRequest
from app.providers.registry import ProviderRegistry

logger = logging.getLogger("app")


def _parse_tool_result_content(content: Any) -> str:
    """Normalize heterogeneous tool-result content into a plain string."""
    if content is None:
        return "No content provided"
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, dict):
                parts.append(item.get("text", json.dumps(item)))
            elif isinstance(item, str):
                parts.append(item)
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(content, dict):
        if content.get("type") == "text":
            return content.get("text", "")
        return json.dumps(content)
    return str(content)


def convert_anthropic_to_litellm(request: MessagesRequest) -> Dict[str, Any]:
    """Build a LiteLLM-compatible request dict from an Anthropic *MessagesRequest*."""

    messages: list[dict[str, Any]] = []

    # --- System message ---
    if request.system:
        if isinstance(request.system, str):
            messages.append({"role": "system", "content": request.system})
        elif isinstance(request.system, list):
            text = "\n\n".join(
                (b.text if hasattr(b, "text") else b.get("text", ""))
                for b in request.system
            )
            if text:
                messages.append({"role": "system", "content": text.strip()})

    # --- Conversation messages ---
    for msg in request.messages:
        content = msg.content

        if isinstance(content, str):
            messages.append({"role": msg.role, "content": content})
            continue

        # User message with tool_result blocks → flatten to text
        if msg.role == "user" and any(
            getattr(b, "type", None) == "tool_result" for b in content
        ):
            parts: list[str] = []
            for block in content:
                bt = getattr(block, "type", None)
                if bt == "text":
                    parts.append(block.text)
                elif bt == "tool_result":
                    tid = getattr(block, "tool_use_id", "")
                    rc = _parse_tool_result_content(getattr(block, "content", ""))
                    parts.append(f"Tool result for {tid}:\n{rc}")
            messages.append({"role": "user", "content": "\n".join(parts).strip()})
            continue

        # Generic block list
        processed: list[dict[str, Any]] = []
        for block in content:
            bt = getattr(block, "type", None)
            if bt == "text":
                processed.append({"type": "text", "text": block.text})
            elif bt == "image":
                processed.append({"type": "image", "source": block.source})
            elif bt == "tool_use":
                processed.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )
            elif bt == "tool_result":
                rc = getattr(block, "content", "")
                if isinstance(rc, str):
                    rc = [{"type": "text", "text": rc}]
                elif not isinstance(rc, list):
                    rc = [{"type": "text", "text": str(rc)}]
                processed.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": getattr(block, "tool_use_id", ""),
                        "content": rc,
                    }
                )
        messages.append({"role": msg.role, "content": processed})

    # --- Cap max_tokens via provider ---
    max_tokens = request.max_tokens
    try:
        provider = ProviderRegistry.get(request.model)
        cap = provider.get_max_output_tokens()
        if cap is not None:
            max_tokens = min(max_tokens, cap)
    except ValueError:
        pass

    litellm_request: Dict[str, Any] = {
        "model": request.model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "temperature": request.temperature,
        "stream": request.stream,
    }

    # Thinking (Anthropic-only)
    if request.thinking and request.model.startswith("anthropic/"):
        litellm_request["thinking"] = request.thinking

    if request.stop_sequences:
        litellm_request["stop"] = request.stop_sequences
    if request.top_p:
        litellm_request["top_p"] = request.top_p
    if request.top_k:
        litellm_request["top_k"] = request.top_k

    # --- Tools ---
    if request.tools:
        try:
            provider = ProviderRegistry.get(request.model)
        except ValueError:
            provider = None

        openai_tools = []
        for tool in request.tools:
            td = tool.model_dump() if hasattr(tool, "model_dump") else dict(tool)
            schema = td.get("input_schema", {})
            if provider:
                schema = provider.clean_schema(schema)
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": td["name"],
                        "description": td.get("description", ""),
                        "parameters": schema,
                    },
                }
            )
        litellm_request["tools"] = openai_tools

    # --- Tool choice ---
    if request.tool_choice:
        tc = request.tool_choice
        ctype = tc.get("type")
        if ctype == "auto":
            litellm_request["tool_choice"] = "auto"
        elif ctype == "any":
            litellm_request["tool_choice"] = "any"
        elif ctype == "tool" and "name" in tc:
            litellm_request["tool_choice"] = {
                "type": "function",
                "function": {"name": tc["name"]},
            }
        else:
            litellm_request["tool_choice"] = "auto"

    return litellm_request
