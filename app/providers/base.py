"""모든 LLM 프로바이더가 구현해야 하는 추상 기본 클래스."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class AbstractProvider(ABC):
    """새 LLM 프로바이더를 추가하기 위한 확장 포인트.

    새 프로바이더 추가 방법:
      1. ``app/providers/`` 디렉터리에 새 파일 생성 (예: ``mistral.py``).
      2. ``AbstractProvider``를 상속하고 모든 추상 메서드를 구현.
      3. ``app/providers/registry.py``에서 인스턴스를 등록.
    """

    # ------------------------------------------------------------------
    # 프로바이더 식별
    # ------------------------------------------------------------------

    @abstractmethod
    def get_model_prefix(self) -> str:
        """LiteLLM 모델 접두사를 반환한다 (예: ``"openai"``, ``"gemini"``)."""

    @abstractmethod
    def get_supported_models(self) -> List[str]:
        """이 프로바이더가 지원하는 모델명 목록을 반환한다."""

    # ------------------------------------------------------------------
    # 요청 생명주기
    # ------------------------------------------------------------------

    @abstractmethod
    def configure_request(self, litellm_request: Dict[str, Any]) -> Dict[str, Any]:
        """프로바이더별 설정을 주입한다 (API 키, 베이스 URL, Vertex 등)."""

    @abstractmethod
    def preprocess_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """메시지를 프로바이더가 기대하는 형식으로 변환한다."""

    # ------------------------------------------------------------------
    # 선택적 훅 (필요 시 오버라이드)
    # ------------------------------------------------------------------

    def clean_schema(self, schema: Any) -> Any:
        """프로바이더별 도구 스키마 정리. 기본값: 그대로 반환."""
        return schema

    def get_max_output_tokens(self) -> int | None:
        """``max_tokens``의 하드 캡을 반환한다. 제한 없으면 ``None``."""
        return None
