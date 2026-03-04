"""Anthropic passthrough provider (for direct Anthropic API usage)."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from app.config import ANTHROPIC_API_KEY
from app.providers.base import AbstractProvider

logger = logging.getLogger("app")


class AnthropicProvider(AbstractProvider):
    """Passthrough provider for native Anthropic / Claude models."""

    def get_model_prefix(self) -> str:
        return "anthropic"

    def get_supported_models(self) -> List[str]:
        # Claude models are accepted dynamically; we don't restrict the list.
        return []

    def configure_request(self, litellm_request: Dict[str, Any]) -> Dict[str, Any]:
        litellm_request["api_key"] = ANTHROPIC_API_KEY
        logger.debug("Using Anthropic API key")
        return litellm_request

    def preprocess_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Anthropic messages are passed through as-is via LiteLLM."""
        return messages
