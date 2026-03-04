"""Anthropic 형식 API 요청을 위한 Pydantic 모델."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

from app.config import BIG_MODEL, PREFERRED_PROVIDER, SMALL_MODEL

logger = logging.getLogger("app")

# 접두사 해석에 사용되는 알려진 모델 목록
OPENAI_MODELS = [
    "o3-mini",
    "o1",
    "o1-mini",
    "o1-pro",
    "gpt-4.5-preview",
    "gpt-4o",
    "gpt-4o-audio-preview",
    "chatgpt-4o-latest",
    "gpt-4o-mini",
    "gpt-4o-mini-audio-preview",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
]

GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]


def _strip_prefix(model: str) -> str:
    """모델명에서 프로바이더 접두사를 제거한다."""
    for prefix in ("anthropic/", "openai/", "gemini/"):
        if model.startswith(prefix):
            return model[len(prefix) :]
    return model


def _resolve_model(original: str) -> str:
    """프로바이더 인식 모델 매핑 규칙을 적용한다."""
    clean = _strip_prefix(original)

    if PREFERRED_PROVIDER == "anthropic":
        return f"anthropic/{clean}"

    # haiku → SMALL_MODEL (소형 모델)
    if "haiku" in clean.lower():
        if PREFERRED_PROVIDER == "google" and SMALL_MODEL in GEMINI_MODELS:
            return f"gemini/{SMALL_MODEL}"
        return f"openai/{SMALL_MODEL}"

    # sonnet → BIG_MODEL (대형 모델)
    if "sonnet" in clean.lower():
        if PREFERRED_PROVIDER == "google" and BIG_MODEL in GEMINI_MODELS:
            return f"gemini/{BIG_MODEL}"
        return f"openai/{BIG_MODEL}"

    # 알려진 모델에 접두사 추가
    if clean in GEMINI_MODELS and not original.startswith("gemini/"):
        return f"gemini/{clean}"
    if clean in OPENAI_MODELS and not original.startswith("openai/"):
        return f"openai/{clean}"

    # 이미 접두사가 있거나 알 수 없는 모델은 그대로 전달
    if not original.startswith(("openai/", "gemini/", "anthropic/")):
        logger.warning(
            f"모델 '{original}'에 대한 접두사 또는 매핑 규칙이 없습니다. 원본 그대로 사용합니다."
        )
    return original


# --- 콘텐츠 블록 타입 ---


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


class SystemContent(BaseModel):
    type: Literal["text"]
    text: str


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[
        str,
        List[
            Union[
                ContentBlockText,
                ContentBlockImage,
                ContentBlockToolUse,
                ContentBlockToolResult,
            ]
        ],
    ]


class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]


class ThinkingConfig(BaseModel):
    enabled: bool = True


# --- 요청 모델 ---


class MessagesRequest(BaseModel):
    """Anthropic Messages API 요청 본문."""

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
    original_model: Optional[str] = None

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str, info) -> str:  # noqa: N805
        resolved = _resolve_model(v)
        values = info.data
        if isinstance(values, dict):
            values["original_model"] = v
        logger.debug(f"모델 매핑: '{v}' -> '{resolved}'")
        return resolved


class TokenCountRequest(BaseModel):
    """토큰 카운트 요청 본문."""

    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    tools: Optional[List[Tool]] = None
    thinking: Optional[ThinkingConfig] = None
    tool_choice: Optional[Dict[str, Any]] = None
    original_model: Optional[str] = None

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str, info) -> str:  # noqa: N805
        resolved = _resolve_model(v)
        values = info.data
        if isinstance(values, dict):
            values["original_model"] = v
        logger.debug(f"토큰 카운트 모델 매핑: '{v}' -> '{resolved}'")
        return resolved
