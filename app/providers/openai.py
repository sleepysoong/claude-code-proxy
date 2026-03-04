"""OpenAI provider implementation."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from app.config import OPENAI_API_KEY, OPENAI_BASE_URL
from app.providers.base import AbstractProvider

logger = logging.getLogger("app")


class OpenAIProvider(AbstractProvider):
    """Provider for OpenAI models (GPT-4o, GPT-4.1, o-series, etc.)."""

    MODELS = [
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

    def get_model_prefix(self) -> str:
        return "openai"

    def get_supported_models(self) -> List[str]:
        return self.MODELS

    def get_max_output_tokens(self) -> int | None:
        return 16384

    def configure_request(self, litellm_request: Dict[str, Any]) -> Dict[str, Any]:
        litellm_request["api_key"] = OPENAI_API_KEY
        if OPENAI_BASE_URL:
            litellm_request["api_base"] = OPENAI_BASE_URL
            logger.debug(f"Using OpenAI custom base URL: {OPENAI_BASE_URL}")
        return litellm_request

    def preprocess_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Flatten content blocks to plain strings for OpenAI compatibility."""
        for i, msg in enumerate(messages):
            content = msg.get("content")

            # Handle list content -> string
            if isinstance(content, list):
                text = self._flatten_content_blocks(content)
                messages[i]["content"] = text if text.strip() else "..."

            # Handle None content
            elif content is None:
                messages[i]["content"] = "..."

            # Remove unsupported fields
            allowed = {"role", "content", "name", "tool_call_id", "tool_calls"}
            for key in list(msg.keys()):
                if key not in allowed:
                    del msg[key]

        return messages

    # ------------------------------------------------------------------

    @staticmethod
    def _flatten_content_blocks(blocks: list) -> str:
        """Recursively extract text from heterogeneous content blocks."""
        parts: list[str] = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")

            if btype == "text":
                parts.append(block.get("text", ""))

            elif btype == "tool_result":
                tool_id = block.get("tool_use_id", "unknown")
                parts.append(f"[Tool Result ID: {tool_id}]")
                rc = block.get("content", [])
                if isinstance(rc, list):
                    for item in rc:
                        if isinstance(item, dict):
                            parts.append(item.get("text", json.dumps(item)))
                elif isinstance(rc, str):
                    parts.append(rc)
                else:
                    try:
                        parts.append(json.dumps(rc))
                    except (TypeError, ValueError):
                        parts.append(str(rc))

            elif btype == "tool_use":
                name = block.get("name", "unknown")
                tid = block.get("id", "unknown")
                inp = json.dumps(block.get("input", {}))
                parts.append(f"[Tool: {name} (ID: {tid})]\nInput: {inp}")

            elif btype == "image":
                parts.append("[Image content]")

        return "\n".join(parts)
