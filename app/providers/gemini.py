"""Gemini provider implementation."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from app.config import GEMINI_API_KEY, USE_VERTEX_AUTH, VERTEX_LOCATION, VERTEX_PROJECT
from app.providers.base import AbstractProvider

logger = logging.getLogger("app")


class GeminiProvider(AbstractProvider):
    """Provider for Google Gemini models."""

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
                f"Using Vertex AI ADC: project={VERTEX_PROJECT}, location={VERTEX_LOCATION}"
            )
        else:
            litellm_request["api_key"] = GEMINI_API_KEY
            logger.debug("Using Gemini API key")
        return litellm_request

    def preprocess_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Gemini generally accepts the same format as OpenAI via LiteLLM."""
        return messages

    def clean_schema(self, schema: Any) -> Any:
        """Remove fields that Gemini tool parameters don't support."""
        return _clean_gemini_schema(schema)


def _clean_gemini_schema(schema: Any) -> Any:
    """Recursively strip unsupported fields from a JSON schema for Gemini."""
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
