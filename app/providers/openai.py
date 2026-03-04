"""OpenAI 프로바이더 구현."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from app.config import OPENAI_API_KEY, OPENAI_BASE_URL
from app.providers.base import AbstractProvider

logger = logging.getLogger("app")


class OpenAIProvider(AbstractProvider):
    """OpenAI 모델용 프로바이더 (GPT-4o, GPT-4.1, o-시리즈 등)."""

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
            logger.debug(f"OpenAI 커스텀 베이스 URL 사용: {OPENAI_BASE_URL}")
        return litellm_request

    def preprocess_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """OpenAI 호환성을 위해 콘텐츠 블록을 평문 문자열로 평탄화한다."""
        for i, msg in enumerate(messages):
            content = msg.get("content")

            # 리스트 콘텐츠 → 문자열 변환
            if isinstance(content, list):
                text = self._flatten_content_blocks(content)
                messages[i]["content"] = text if text.strip() else "..."

            # None 콘텐츠 처리
            elif content is None:
                messages[i]["content"] = "..."

            # 지원하지 않는 필드 제거
            allowed = {"role", "content", "name", "tool_call_id", "tool_calls"}
            for key in list(msg.keys()):
                if key not in allowed:
                    del msg[key]

        return messages

    # ------------------------------------------------------------------

    @staticmethod
    def _flatten_content_blocks(blocks: list) -> str:
        """이기종 콘텐츠 블록에서 텍스트를 재귀적으로 추출한다."""
        parts: list[str] = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")

            if btype == "text":
                parts.append(block.get("text", ""))

            elif btype == "tool_result":
                tool_id = block.get("tool_use_id", "unknown")
                parts.append(f"[도구 결과 ID: {tool_id}]")
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
                parts.append(f"[도구: {name} (ID: {tid})]\n입력: {inp}")

            elif btype == "image":
                parts.append("[이미지 콘텐츠]")

        return "\n".join(parts)
