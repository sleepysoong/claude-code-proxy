"""LiteLLM (OpenAI 형식) 응답을 Anthropic 형식으로 역변환한다."""

from __future__ import annotations

import json
import logging
import traceback
import uuid
from typing import Any, Dict, Union

from app.models.request import MessagesRequest
from app.models.response import MessagesResponse, Usage

logger = logging.getLogger("app")


def convert_litellm_to_anthropic(
    litellm_response: Union[Dict[str, Any], Any],
    original_request: MessagesRequest,
) -> MessagesResponse:
    """LiteLLM ``ModelResponse`` (또는 dict)를 Anthropic ``MessagesResponse``로 매핑한다."""
    try:
        clean_model = original_request.model
        for prefix in ("anthropic/", "openai/", "gemini/"):
            if clean_model.startswith(prefix):
                clean_model = clean_model[len(prefix) :]
                break

        is_claude = clean_model.startswith("claude-")

        # --- ModelResponse 객체 또는 dict에서 추출 ---
        if hasattr(litellm_response, "choices") and hasattr(litellm_response, "usage"):
            choices = litellm_response.choices
            message = choices[0].message if choices else None
            content_text = getattr(message, "content", "") if message else ""
            tool_calls = getattr(message, "tool_calls", None) if message else None
            finish_reason = choices[0].finish_reason if choices else "stop"
            usage_info = litellm_response.usage
            response_id = getattr(litellm_response, "id", f"msg_{uuid.uuid4()}")
        else:
            response_dict = _to_dict(litellm_response)
            choices = response_dict.get("choices", [{}])
            message = choices[0].get("message", {}) if choices else {}
            content_text = message.get("content", "")
            tool_calls = message.get("tool_calls")
            finish_reason = (
                choices[0].get("finish_reason", "stop") if choices else "stop"
            )
            usage_info = response_dict.get("usage", {})
            response_id = response_dict.get("id", f"msg_{uuid.uuid4()}")

        # --- 콘텐츠 블록 생성 ---
        content: list[dict[str, Any]] = []

        if content_text:
            content.append({"type": "text", "text": content_text})

        if tool_calls:
            tool_calls = tool_calls if isinstance(tool_calls, list) else [tool_calls]

            if is_claude:
                for tc in tool_calls:
                    fn, tid, name, arguments = _extract_tool_call(tc)
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tid,
                            "name": name,
                            "input": arguments,
                        }
                    )
            else:
                # Claude가 아닌 모델의 경우 도구 호출을 텍스트로 렌더링
                tool_text = "\n\n도구 사용:\n"
                for tc in tool_calls:
                    fn, tid, name, arguments = _extract_tool_call(tc)
                    tool_text += (
                        f"도구: {name}\n인자: {json.dumps(arguments, indent=2)}\n\n"
                    )
                if content and content[0]["type"] == "text":
                    content[0]["text"] += tool_text
                else:
                    content.append({"type": "text", "text": tool_text})

        if not content:
            content.append({"type": "text", "text": ""})

        # --- 사용량 ---
        if isinstance(usage_info, dict):
            prompt_tokens = usage_info.get("prompt_tokens", 0)
            completion_tokens = usage_info.get("completion_tokens", 0)
        else:
            prompt_tokens = getattr(usage_info, "prompt_tokens", 0)
            completion_tokens = getattr(usage_info, "completion_tokens", 0)

        # --- 종료 사유 ---
        stop_map = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
        }
        stop_reason = stop_map.get(finish_reason, "end_turn")

        return MessagesResponse(
            id=response_id,
            model=original_request.model,
            content=content,  # type: ignore[arg-type]
            stop_reason=stop_reason,  # type: ignore[arg-type]
            stop_sequence=None,
            usage=Usage(input_tokens=prompt_tokens, output_tokens=completion_tokens),
        )

    except Exception as exc:
        logger.error(f"응답 변환 오류: {exc}\n{traceback.format_exc()}")
        return MessagesResponse(
            id=f"msg_{uuid.uuid4()}",
            model=original_request.model,
            content=[{"type": "text", "text": f"응답 변환 오류: {exc}"}],  # type: ignore[arg-type]
            stop_reason="end_turn",
            usage=Usage(input_tokens=0, output_tokens=0),
        )


# ------------------------------------------------------------------
# 헬퍼 함수
# ------------------------------------------------------------------


def _to_dict(obj: Any) -> dict:
    """객체를 딕셔너리로 변환한다."""
    if isinstance(obj, dict):
        return obj
    for method in ("model_dump", "dict"):
        fn = getattr(obj, method, None)
        if callable(fn):
            return fn()
    return getattr(
        obj, "__dict__", {"id": f"msg_{uuid.uuid4()}", "choices": [{}], "usage": {}}
    )


def _extract_tool_call(tc: Any) -> tuple[Any, str, str, dict]:
    """(function_obj, tool_id, name, parsed_arguments)를 반환한다."""
    if isinstance(tc, dict):
        fn = tc.get("function", {})
        tid = tc.get("id", f"tool_{uuid.uuid4()}")
        name = fn.get("name", "") if isinstance(fn, dict) else ""
        raw_args = fn.get("arguments", "{}") if isinstance(fn, dict) else "{}"
    else:
        fn = getattr(tc, "function", None)
        tid = getattr(tc, "id", f"tool_{uuid.uuid4()}")
        name = getattr(fn, "name", "") if fn else ""
        raw_args = getattr(fn, "arguments", "{}") if fn else "{}"

    if isinstance(raw_args, str):
        try:
            arguments = json.loads(raw_args)
        except json.JSONDecodeError:
            arguments = {"raw": raw_args}
    else:
        arguments = raw_args

    return fn, tid, name, arguments
