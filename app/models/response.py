"""Pydantic models for Anthropic-format API responses."""

from __future__ import annotations

from typing import List, Literal, Optional, Union

from pydantic import BaseModel

from app.models.request import ContentBlockText, ContentBlockToolUse


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


class TokenCountResponse(BaseModel):
    input_tokens: int
