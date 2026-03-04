"""Anthropic 형식 API 응답을 위한 Pydantic 모델."""

from __future__ import annotations

from typing import List, Literal, Optional, Union

from pydantic import BaseModel

from app.models.request import ContentBlockText, ContentBlockToolUse


class Usage(BaseModel):
    """토큰 사용량 정보."""

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class MessagesResponse(BaseModel):
    """Anthropic Messages API 응답 객체."""

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


class TokenCountResponse(BaseModel):
    """토큰 카운트 응답 객체."""

    input_tokens: int
