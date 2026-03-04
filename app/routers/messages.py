"""Anthropic 호환 /v1/messages 엔드포인트를 위한 API 라우터."""

from __future__ import annotations

import json
import time
import traceback
from typing import Any

import litellm
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.converters import (
    convert_anthropic_to_litellm,
    convert_litellm_to_anthropic,
    handle_streaming,
)
from app.logging import log_request, logger
from app.models import (
    MessagesRequest,
    MessagesResponse,
    TokenCountRequest,
    TokenCountResponse,
)
from app.providers import ProviderRegistry

router = APIRouter(prefix="/v1")


# ---------------------------------------------------------------------------
# 헬퍼 함수
# ---------------------------------------------------------------------------


def _display_model(model: str) -> str:
    """로그용으로 프로바이더 접두사를 제거한 모델명을 반환한다."""
    return model.split("/")[-1] if "/" in model else model


def _sanitize_for_json(obj: Any) -> Any:
    """*obj*를 JSON 직렬화 가능하도록 재귀적으로 변환한다."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(i) for i in obj]
    if hasattr(obj, "__dict__"):
        return _sanitize_for_json(obj.__dict__)
    if hasattr(obj, "text"):
        return str(obj.text)
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


# ---------------------------------------------------------------------------
# POST /v1/messages
# ---------------------------------------------------------------------------


@router.post("/messages")
async def create_message(
    request: MessagesRequest,
    raw_request: Request,
) -> Any:
    try:
        # 원본 모델명 파싱 (Pydantic이 해석하기 전)
        body = await raw_request.body()
        body_json = json.loads(body.decode("utf-8"))
        original_model = body_json.get("model", "unknown")
        display_model = _display_model(original_model)

        logger.debug(f"요청 처리 중: 모델={request.model}, 스트리밍={request.stream}")

        # 1. Anthropic 요청 → LiteLLM 딕셔너리 변환
        litellm_request = convert_anthropic_to_litellm(request)

        # 2. 프로바이더별 설정 (API 키, 베이스 URL, Vertex 등)
        try:
            provider = ProviderRegistry.get(request.model)
            litellm_request = provider.configure_request(litellm_request)

            # 프로바이더별 메시지 전처리 (예: OpenAI 평탄화)
            if "messages" in litellm_request:
                litellm_request["messages"] = provider.preprocess_messages(
                    litellm_request["messages"]
                )
        except ValueError:
            logger.warning(f"등록된 프로바이더가 없습니다: {request.model}")

        logger.debug(
            f"모델 요청: {litellm_request.get('model')}, "
            f"스트리밍: {litellm_request.get('stream', False)}"
        )

        # 공통 로깅 인자
        num_tools = len(request.tools) if request.tools else 0
        log_kwargs = dict(
            method="POST",
            path=raw_request.url.path,
            claude_model=display_model,
            mapped_model=litellm_request.get("model", ""),
            num_messages=len(litellm_request["messages"]),
            num_tools=num_tools,
            status_code=200,
        )

        # 3a. 스트리밍
        if request.stream:
            log_request(**log_kwargs)
            response_generator = await litellm.acompletion(**litellm_request)
            return StreamingResponse(
                handle_streaming(response_generator, request),
                media_type="text/event-stream",
            )

        # 3b. 비스트리밍
        log_request(**log_kwargs)
        start_time = time.time()
        litellm_response = litellm.completion(**litellm_request)
        logger.debug(
            f"응답 수신: 모델={litellm_request.get('model')}, "
            f"소요시간={time.time() - start_time:.2f}초"
        )
        return convert_litellm_to_anthropic(litellm_response, request)

    except Exception as exc:
        error_traceback = traceback.format_exc()

        error_details: dict[str, Any] = {
            "error": str(exc),
            "type": type(exc).__name__,
            "traceback": error_traceback,
        }
        for attr in ("message", "status_code", "response", "llm_provider", "model"):
            if hasattr(exc, attr):
                error_details[attr] = getattr(exc, attr)
        if hasattr(exc, "__dict__"):
            for key, value in exc.__dict__.items():
                if key not in error_details and key not in ("args", "__traceback__"):
                    error_details[key] = str(value)

        logger.error(
            f"요청 처리 오류: {json.dumps(_sanitize_for_json(error_details), indent=2)}"
        )

        error_message = f"오류: {exc}"
        if error_details.get("message"):
            error_message += f"\n메시지: {error_details['message']}"
        if error_details.get("response"):
            error_message += f"\n응답: {error_details['response']}"

        status_code = error_details.get("status_code", 500)
        raise HTTPException(status_code=status_code, detail=error_message)


# ---------------------------------------------------------------------------
# POST /v1/messages/count_tokens
# ---------------------------------------------------------------------------


@router.post("/messages/count_tokens")
async def count_tokens(
    request: TokenCountRequest,
    raw_request: Request,
) -> TokenCountResponse:
    try:
        original_model = request.original_model or request.model
        display_model = _display_model(original_model)

        # 변환기를 재사용하기 위해 임시 MessagesRequest 생성
        converted_request = convert_anthropic_to_litellm(
            MessagesRequest(
                model=request.model,
                max_tokens=100,  # 임의 값; 카운팅에는 사용되지 않음
                messages=request.messages,
                system=request.system,
                tools=request.tools,
                tool_choice=request.tool_choice,
                thinking=request.thinking,
            )
        )

        # 프로바이더별 메시지 전처리
        try:
            provider = ProviderRegistry.get(request.model)
            converted_request = provider.configure_request(converted_request)
            if "messages" in converted_request:
                converted_request["messages"] = provider.preprocess_messages(
                    converted_request["messages"]
                )
        except ValueError:
            pass

        from litellm import token_counter

        num_tools = len(request.tools) if request.tools else 0
        log_request(
            method="POST",
            path=raw_request.url.path,
            claude_model=display_model,
            mapped_model=converted_request.get("model", ""),
            num_messages=len(converted_request["messages"]),
            num_tools=num_tools,
            status_code=200,
        )

        token_counter_args: dict[str, Any] = {
            "model": converted_request["model"],
            "messages": converted_request["messages"],
        }
        from app.config import OPENAI_BASE_URL

        if request.model.startswith("openai/") and OPENAI_BASE_URL:
            token_counter_args["api_base"] = OPENAI_BASE_URL

        token_count = token_counter(**token_counter_args)
        return TokenCountResponse(input_tokens=token_count)

    except Exception as exc:
        logger.error(f"토큰 카운트 오류: {exc}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"토큰 카운트 오류: {exc}",
        )
