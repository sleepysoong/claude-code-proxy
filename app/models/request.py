"""Pydantic models for Anthropic-format API requests."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

from app.config import BIG_MODEL, PREFERRED_PROVIDER, SMALL_MODEL

logger = logging.getLogger("app")

# Known model lists (used for prefix resolution)
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
    for prefix in ("anthropic/", "openai/", "gemini/"):
        if model.startswith(prefix):
            return model[len(prefix) :]
    return model


def _resolve_model(original: str) -> str:
    """Apply provider-aware model mapping rules."""
    clean = _strip_prefix(original)

    if PREFERRED_PROVIDER == "anthropic":
        return f"anthropic/{clean}"

    # haiku → SMALL_MODEL
    if "haiku" in clean.lower():
        if PREFERRED_PROVIDER == "google" and SMALL_MODEL in GEMINI_MODELS:
            return f"gemini/{SMALL_MODEL}"
        return f"openai/{SMALL_MODEL}"

    # sonnet → BIG_MODEL
    if "sonnet" in clean.lower():
        if PREFERRED_PROVIDER == "google" and BIG_MODEL in GEMINI_MODELS:
            return f"gemini/{BIG_MODEL}"
        return f"openai/{BIG_MODEL}"

    # Add prefix to known models
    if clean in GEMINI_MODELS and not original.startswith("gemini/"):
        return f"gemini/{clean}"
    if clean in OPENAI_MODELS and not original.startswith("openai/"):
        return f"openai/{clean}"

    # If already prefixed or unknown, pass through
    if not original.startswith(("openai/", "gemini/", "anthropic/")):
        logger.warning(
            f"No prefix or mapping rule for model: '{original}'. Using as-is."
        )
    return original


# --- Content block types ---


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


# --- Request models ---


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
    original_model: Optional[str] = None

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str, info) -> str:  # noqa: N805
        resolved = _resolve_model(v)
        values = info.data
        if isinstance(values, dict):
            values["original_model"] = v
        logger.debug(f"MODEL MAPPING: '{v}' -> '{resolved}'")
        return resolved


class TokenCountRequest(BaseModel):
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
        logger.debug(f"TOKEN COUNT MAPPING: '{v}' -> '{resolved}'")
        return resolved
