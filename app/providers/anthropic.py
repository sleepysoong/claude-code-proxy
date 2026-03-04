"""Anthropic 패스스루 프로바이더 (직접 Anthropic API 사용 시)."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from app.config import ANTHROPIC_API_KEY
from app.providers.base import AbstractProvider

logger = logging.getLogger("app")


class AnthropicProvider(AbstractProvider):
    """네이티브 Anthropic / Claude 모델용 패스스루 프로바이더."""

    def get_model_prefix(self) -> str:
        return "anthropic"

    def get_supported_models(self) -> List[str]:
        # Claude 모델은 동적으로 허용되므로 목록을 제한하지 않는다.
        return []

    def configure_request(self, litellm_request: Dict[str, Any]) -> Dict[str, Any]:
        litellm_request["api_key"] = ANTHROPIC_API_KEY
        logger.debug("Anthropic API 키 사용")
        return litellm_request

    def preprocess_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Anthropic 메시지는 LiteLLM을 통해 그대로 전달된다."""
        return messages
