"""Gemini 프로바이더 구현."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from app.config import GEMINI_API_KEY, USE_VERTEX_AUTH, VERTEX_LOCATION, VERTEX_PROJECT
from app.providers.base import AbstractProvider

logger = logging.getLogger("app")


class GeminiProvider(AbstractProvider):
    """Google Gemini 모델용 프로바이더."""

    MODELS = [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ]

    def get_model_prefix(self) -> str:
        return "gemini"

    def get_supported_models(self) -> List[str]:
        return self.MODELS

    def get_max_output_tokens(self) -> int | None:
        return 16384

    def configure_request(self, litellm_request: Dict[str, Any]) -> Dict[str, Any]:
        if USE_VERTEX_AUTH:
            litellm_request["vertex_project"] = VERTEX_PROJECT
            litellm_request["vertex_location"] = VERTEX_LOCATION
            litellm_request["custom_llm_provider"] = "vertex_ai"
            logger.debug(
                f"Vertex AI ADC 사용: 프로젝트={VERTEX_PROJECT}, 리전={VERTEX_LOCATION}"
            )
        else:
            litellm_request["api_key"] = GEMINI_API_KEY
            logger.debug("Gemini API 키 사용")
        return litellm_request

    def preprocess_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Gemini는 LiteLLM을 통해 OpenAI와 동일한 형식을 수용한다."""
        return messages

    def clean_schema(self, schema: Any) -> Any:
        """Gemini 도구 파라미터에서 지원하지 않는 필드를 제거한다."""
        return _clean_gemini_schema(schema)


def _clean_gemini_schema(schema: Any) -> Any:
    """Gemini용 JSON 스키마에서 지원되지 않는 필드를 재귀적으로 제거한다."""
    if isinstance(schema, dict):
        schema.pop("additionalProperties", None)
        schema.pop("default", None)

        if schema.get("type") == "string" and "format" in schema:
            allowed_formats = {"enum", "date-time"}
            if schema["format"] not in allowed_formats:
                schema.pop("format")

        for key, value in list(schema.items()):
            schema[key] = _clean_gemini_schema(value)

    elif isinstance(schema, list):
        return [_clean_gemini_schema(item) for item in schema]

    return schema
