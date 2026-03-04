"""Anthropic 형식 요청을 LiteLLM (OpenAI 호환) 형식으로 변환한다."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from app.models.request import MessagesRequest
from app.providers.registry import ProviderRegistry

logger = logging.getLogger("app")


def _parse_tool_result_content(content: Any) -> str:
    """이기종 도구 결과 콘텐츠를 평문 문자열로 정규화한다."""
    if content is None:
        return "제공된 콘텐츠 없음"
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
    """Anthropic *MessagesRequest*로부터 LiteLLM 호환 요청 딕셔너리를 생성한다."""

    messages: list[dict[str, Any]] = []

    # --- 시스템 메시지 ---
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

    # --- 대화 메시지 ---
    for msg in request.messages:
        content = msg.content

        if isinstance(content, str):
            messages.append({"role": msg.role, "content": content})
            continue

        # tool_result 블록이 포함된 사용자 메시지 → 텍스트로 평탄화
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
                    parts.append(f"{tid}의 도구 결과:\n{rc}")
            messages.append({"role": "user", "content": "\n".join(parts).strip()})
            continue

        # 일반 블록 리스트
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

    # --- 프로바이더를 통해 max_tokens 제한 ---
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

    # Thinking (Anthropic 전용)
    if request.thinking and request.model.startswith("anthropic/"):
        litellm_request["thinking"] = request.thinking

    if request.stop_sequences:
        litellm_request["stop"] = request.stop_sequences
    if request.top_p:
        litellm_request["top_p"] = request.top_p
    if request.top_k:
        litellm_request["top_k"] = request.top_k

    # --- 도구 변환 ---
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

    # --- 도구 선택 ---
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
