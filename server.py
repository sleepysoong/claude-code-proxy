from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from typing import Any, Dict, List, Literal, Optional, Union

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="GitHub Copilot SDK 전용 Anthropic 프록시")


COPILOT_SUPPORTED_MODELS = {
    "claude-sonnet-4.5",
    "claude-haiku-4.5",
    "claude-opus-4.5",
    "claude-sonnet-4",
    "gpt-5.1-codex-max",
    "gpt-5.1-codex",
    "gpt-5.2",
    "gpt-5.1",
    "gpt-5",
    "gpt-5.1-codex-mini",
    "gpt-5-mini",
    "gpt-4.1",
    "gemini-3-pro-preview",
}

COPILOT_DEFAULT_MODEL = os.getenv("COPILOT_MODEL", "gpt-4.1")
COPILOT_SMALL_MODEL = os.getenv("COPILOT_SMALL_MODEL", COPILOT_DEFAULT_MODEL)
COPILOT_BIG_MODEL = os.getenv("COPILOT_BIG_MODEL", "gpt-5")
COPILOT_CLI_URL = os.getenv("COPILOT_CLI_URL")


class ContentBlockText(BaseModel):
    type: Literal["text"]
    text: str


class ContentBlockImage(BaseModel):
    type: Literal["image"]
    source: Dict[str, Any]


class ContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]


class ContentBlockToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], Dict[str, Any], List[Any], Any]


ContentBlock = Union[
    ContentBlockText,
    ContentBlockImage,
    ContentBlockToolUse,
    ContentBlockToolResult,
]


class SystemContent(BaseModel):
    type: Literal["text"]
    text: str


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, List[ContentBlock]]


class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]


class ThinkingConfig(BaseModel):
    enabled: bool = True


class MessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    thinking: Optional[ThinkingConfig] = None


class TokenCountRequest(BaseModel):
    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    tools: Optional[List[Tool]] = None
    thinking: Optional[ThinkingConfig] = None
    tool_choice: Optional[Dict[str, Any]] = None


class TokenCountResponse(BaseModel):
    input_tokens: int


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class MessagesResponse(BaseModel):
    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    content: List[Union[ContentBlockText, ContentBlockToolUse]]
    type: Literal["message"] = "message"
    stop_reason: Optional[
        Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]
    ] = None
    stop_sequence: Optional[str] = None
    usage: Usage


def _read_github_token() -> Optional[str]:
    return (
        os.getenv("COPILOT_GITHUB_TOKEN")
        or os.getenv("GH_TOKEN")
        or os.getenv("GITHUB_TOKEN")
    )


def _copilot_client_options() -> Dict[str, Any]:
    options: Dict[str, Any] = {}
    token = _read_github_token()
    if token:
        options["github_token"] = token
        options["use_logged_in_user"] = False
    if COPILOT_CLI_URL:
        options["cli_url"] = COPILOT_CLI_URL
    return options


def _load_copilot_sdk() -> Any:
    try:
        from copilot import CopilotClient
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "github-copilot-sdk가 설치되어 있지 않습니다. "
                "다음 명령으로 설치하세요: pip install github-copilot-sdk"
            ),
        ) from exc
    return CopilotClient


def _approve_all_permissions(_request: Any, _invocation: Any) -> Dict[str, str]:
    return {"kind": "approved"}


def _strip_provider_prefix(model: str) -> str:
    for prefix in ("anthropic/", "openai/", "gemini/"):
        if model.startswith(prefix):
            return model[len(prefix) :]
    return model


def _resolve_copilot_model(requested_model: str) -> str:
    clean = _strip_provider_prefix(requested_model)
    lowered = clean.lower()

    if clean in COPILOT_SUPPORTED_MODELS:
        return clean

    if "haiku" in lowered:
        return COPILOT_SMALL_MODEL

    if "sonnet" in lowered or "opus" in lowered:
        return COPILOT_BIG_MODEL

    return COPILOT_DEFAULT_MODEL


def _safe_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _extract_content_text(content: Union[str, List[ContentBlock]]) -> str:
    if isinstance(content, str):
        return content

    chunks: List[str] = []
    for block in content:
        block_type = block.type

        if block_type == "text":
            chunks.append(getattr(block, "text", ""))
            continue

        if block_type == "tool_use":
            chunks.append(
                f"[도구_호출] 이름={getattr(block, 'name', '')} "
                f"id={getattr(block, 'id', '')} "
                f"input={_safe_json(getattr(block, 'input', {}))}"
            )
            continue

        if block_type == "tool_result":
            chunks.append(
                f"[도구_결과] tool_use_id={getattr(block, 'tool_use_id', '')} "
                f"content={_safe_json(getattr(block, 'content', ''))}"
            )
            continue

        if block_type == "image":
            chunks.append("[이미지 내용 생략]")
            continue

        chunks.append(str(block))

    return "\n".join(part for part in chunks if part)


def _flatten_system(system: Optional[Union[str, List[SystemContent]]]) -> str:
    if not system:
        return ""

    if isinstance(system, str):
        return system

    lines: List[str] = []
    for block in system:
        if block.type == "text":
            lines.append(block.text)
    return "\n\n".join(lines)


def _format_tools_for_prompt(tools: Optional[List[Tool]]) -> str:
    if not tools:
        return ""

    tool_lines: List[str] = []
    for tool in tools:
        tool_lines.append(
            f"- {tool.name}: {tool.description or ''} schema={_safe_json(tool.input_schema)}"
        )
    return "\n".join(tool_lines)


def _build_prompt(
    messages: List[Message],
    system: Optional[Union[str, List[SystemContent]]],
    tools: Optional[List[Tool]],
) -> str:
    sections: List[str] = []

    system_text = _flatten_system(system).strip()
    if system_text:
        sections.append(f"시스템 지침:\n{system_text}")

    if tools:
        sections.append(f"사용 가능한 도구:\n{_format_tools_for_prompt(tools)}")

    transcript_lines: List[str] = []
    for message in messages:
        role = "사용자" if message.role == "user" else "어시스턴트"
        content = _extract_content_text(message.content).strip() or "[비어 있음]"
        transcript_lines.append(f"{role}: {content}")

    sections.append("대화 기록:\n" + "\n\n".join(transcript_lines))
    sections.append("다음 어시스턴트 답변만 생성하세요.")

    return "\n\n".join(sections)


def _event_type(event: Any) -> str:
    event_type: Any = getattr(event, "type", "")
    if hasattr(event_type, "value"):
        return str(event_type.value)
    return str(event_type)


def _event_data_value(event: Any, key: str, default: Any = None) -> Any:
    data = getattr(event, "data", None)
    if data is None:
        return default
    return getattr(data, key, default)


def _to_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _extract_usage(event: Any) -> Usage:
    return Usage(
        input_tokens=_to_int(_event_data_value(event, "input_tokens", 0)),
        output_tokens=_to_int(_event_data_value(event, "output_tokens", 0)),
        cache_creation_input_tokens=_to_int(
            _event_data_value(event, "cache_write_tokens", 0)
        ),
        cache_read_input_tokens=_to_int(
            _event_data_value(event, "cache_read_tokens", 0)
        ),
    )


def _response_id_from_event(event: Any) -> str:
    message_id = _event_data_value(event, "message_id")
    if message_id:
        return str(message_id)
    return f"msg_{uuid.uuid4().hex[:24]}"


def _sse(event_name: str, payload: Dict[str, Any]) -> str:
    return f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _close_session_and_client(session: Any, client: Any) -> None:
    try:
        await session.destroy()
    except Exception:
        logger.debug("세션 종료 중 오류가 발생했습니다.", exc_info=True)

    try:
        await client.stop()
    except Exception:
        logger.debug("클라이언트 종료 중 오류가 발생했습니다.", exc_info=True)


async def _stream_anthropic_events(
    session: Any,
    client: Any,
    prompt: str,
    requested_model: str,
) -> Any:
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    queue: asyncio.Queue[Any] = asyncio.Queue()
    usage = Usage(input_tokens=0, output_tokens=0)
    saw_delta = False
    last_message_content = ""

    def _on_event(event: Any) -> None:
        queue.put_nowait(event)

    unsubscribe = session.on(_on_event)
    send_task = asyncio.create_task(session.send({"prompt": prompt}))

    try:
        yield _sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": requested_model,
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
        yield _sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        )

        while True:
            event = await queue.get()
            event_name = _event_type(event)

            if event_name == "assistant.message_delta":
                delta = _event_data_value(event, "delta_content", "") or ""
                if delta:
                    saw_delta = True
                    yield _sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type": "text_delta", "text": delta},
                        },
                    )
                continue

            if event_name == "assistant.message":
                content = _event_data_value(event, "content", "") or ""
                if content:
                    last_message_content = content
                usage = _extract_usage(event)
                continue

            if event_name == "session.idle":
                break

        await send_task

        if not saw_delta and last_message_content:
            yield _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": last_message_content},
                },
            )

        yield _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
        yield _sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": usage.output_tokens},
            },
        )
        yield _sse("message_stop", {"type": "message_stop"})
        yield "data: [DONE]\n\n"

    except Exception as exc:
        logger.exception("스트리밍 요청 처리 중 오류가 발생했습니다.")
        yield _sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "text_delta",
                    "text": f"Copilot SDK 오류가 발생했습니다: {exc}",
                },
            },
        )
        yield _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
        yield _sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 0},
            },
        )
        yield _sse("message_stop", {"type": "message_stop"})
        yield "data: [DONE]\n\n"
    finally:
        if not send_task.done():
            send_task.cancel()
        if callable(unsubscribe):
            unsubscribe()
        await _close_session_and_client(session, client)


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    word_count = len(re.findall(r"\S+", text))
    char_estimate = max(1, len(text) // 4)
    return max(word_count, char_estimate)


@app.post("/v1/messages")
async def create_message(request: MessagesRequest, raw_request: Request) -> Any:
    del raw_request

    copilot_model = _resolve_copilot_model(request.model)
    prompt = _build_prompt(request.messages, request.system, request.tools)

    logger.info(
        "요청 처리: 원본 모델=%s, 매핑 모델=%s, 스트림=%s",
        request.model,
        copilot_model,
        bool(request.stream),
    )

    CopilotClient = _load_copilot_sdk()
    client = CopilotClient(_copilot_client_options())
    await client.start()

    session = await client.create_session(
        {
            "model": copilot_model,
            "streaming": bool(request.stream),
            "on_permission_request": _approve_all_permissions,
        }
    )

    if request.stream:
        return StreamingResponse(
            _stream_anthropic_events(session, client, prompt, request.model),
            media_type="text/event-stream",
        )

    try:
        event = await session.send_and_wait({"prompt": prompt})
        content = _event_data_value(event, "content", "") or ""
        usage = _extract_usage(event)

        return MessagesResponse(
            id=_response_id_from_event(event),
            model=request.model,
            role="assistant",
            content=[ContentBlockText(type="text", text=content)],
            stop_reason="end_turn",
            stop_sequence=None,
            usage=usage,
        )
    except Exception as exc:
        logger.exception("일반 요청 처리 중 오류가 발생했습니다.")
        raise HTTPException(status_code=500, detail=f"Copilot SDK 오류: {exc}") from exc
    finally:
        await _close_session_and_client(session, client)


@app.post("/v1/messages/count_tokens")
async def count_tokens(
    request: TokenCountRequest, raw_request: Request
) -> TokenCountResponse:
    del raw_request
    prompt = _build_prompt(request.messages, request.system, request.tools)
    return TokenCountResponse(input_tokens=_estimate_tokens(prompt))


@app.get("/")
async def root() -> Dict[str, str]:
    return {
        "message": "GitHub Copilot SDK 기반 Anthropic 프록시가 실행 중입니다.",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8082, log_level="info")
