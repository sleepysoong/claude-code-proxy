"""Abstract base class that every LLM provider must implement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class AbstractProvider(ABC):
    """Extension point for adding new LLM providers.

    To add a new provider:
      1. Create a new file in ``app/providers/`` (e.g. ``mistral.py``).
      2. Subclass ``AbstractProvider`` and implement every abstract method.
      3. Register the instance in ``app/providers/registry.py``.
    """

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @abstractmethod
    def get_model_prefix(self) -> str:
        """Return the LiteLLM model prefix (e.g. ``"openai"``, ``"gemini"``)."""

    @abstractmethod
    def get_supported_models(self) -> List[str]:
        """Return the list of model names this provider supports."""

    # ------------------------------------------------------------------
    # Request lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def configure_request(self, litellm_request: Dict[str, Any]) -> Dict[str, Any]:
        """Inject provider-specific config (API keys, base URLs, etc.)."""

    @abstractmethod
    def preprocess_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Transform messages into the format the provider expects."""

    # ------------------------------------------------------------------
    # Optional hooks (override when needed)
    # ------------------------------------------------------------------

    def clean_schema(self, schema: Any) -> Any:
        """Provider-specific tool-schema sanitization. Default: identity."""
        return schema

    def get_max_output_tokens(self) -> int | None:
        """Return a hard cap on ``max_tokens``, or ``None`` for no cap."""
        return None
