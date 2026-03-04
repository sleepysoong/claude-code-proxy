"""Provider registry -- central look-up for all registered providers."""

from __future__ import annotations

from typing import Dict

from app.providers.base import AbstractProvider


class ProviderRegistry:
    """Factory that maps model prefixes to provider instances.

    Usage::

        ProviderRegistry.register(OpenAIProvider())
        provider = ProviderRegistry.get("openai/gpt-4.1")
    """

    _providers: Dict[str, AbstractProvider] = {}

    @classmethod
    def register(cls, provider: AbstractProvider) -> None:
        cls._providers[provider.get_model_prefix()] = provider

    @classmethod
    def get(cls, model: str) -> AbstractProvider:
        """Return the provider for *model* based on its prefix."""
        for prefix, provider in cls._providers.items():
            if model.startswith(f"{prefix}/"):
                return provider
        raise ValueError(f"No registered provider for model: {model}")

    @classmethod
    def all_providers(cls) -> Dict[str, AbstractProvider]:
        return dict(cls._providers)


def register_default_providers() -> None:
    """Register the built-in providers (called once at startup)."""
    from app.providers.anthropic import AnthropicProvider
    from app.providers.gemini import GeminiProvider
    from app.providers.openai import OpenAIProvider

    ProviderRegistry.register(OpenAIProvider())
    ProviderRegistry.register(GeminiProvider())
    ProviderRegistry.register(AnthropicProvider())
