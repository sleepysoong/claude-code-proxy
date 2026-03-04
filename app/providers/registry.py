"""프로바이더 레지스트리 — 등록된 모든 프로바이더의 중앙 조회소."""

from __future__ import annotations

from typing import Dict

from app.providers.base import AbstractProvider


class ProviderRegistry:
    """모델 접두사를 프로바이더 인스턴스에 매핑하는 팩토리.

    사용 예시::

        ProviderRegistry.register(OpenAIProvider())
        provider = ProviderRegistry.get("openai/gpt-4.1")
    """

    _providers: Dict[str, AbstractProvider] = {}

    @classmethod
    def register(cls, provider: AbstractProvider) -> None:
        """프로바이더를 레지스트리에 등록한다."""
        cls._providers[provider.get_model_prefix()] = provider

    @classmethod
    def get(cls, model: str) -> AbstractProvider:
        """접두사를 기반으로 *model*에 해당하는 프로바이더를 반환한다."""
        for prefix, provider in cls._providers.items():
            if model.startswith(f"{prefix}/"):
                return provider
        raise ValueError(f"등록된 프로바이더가 없습니다: {model}")

    @classmethod
    def all_providers(cls) -> Dict[str, AbstractProvider]:
        """등록된 모든 프로바이더를 반환한다."""
        return dict(cls._providers)


def register_default_providers() -> None:
    """내장 프로바이더를 등록한다 (시작 시 한 번 호출)."""
    from app.providers.anthropic import AnthropicProvider
    from app.providers.gemini import GeminiProvider
    from app.providers.openai import OpenAIProvider

    ProviderRegistry.register(OpenAIProvider())
    ProviderRegistry.register(GeminiProvider())
    ProviderRegistry.register(AnthropicProvider())
