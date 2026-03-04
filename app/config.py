"""환경 변수에서 로드되는 애플리케이션 설정."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


# --- API 키 ---
ANTHROPIC_API_KEY: str | None = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY: str | None = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY: str | None = os.environ.get("GEMINI_API_KEY")

# --- Vertex AI ---
VERTEX_PROJECT: str = os.environ.get("VERTEX_PROJECT", "unset")
VERTEX_LOCATION: str = os.environ.get("VERTEX_LOCATION", "unset")
USE_VERTEX_AUTH: bool = os.environ.get("USE_VERTEX_AUTH", "False").lower() == "true"

# --- OpenAI ---
OPENAI_BASE_URL: str | None = os.environ.get("OPENAI_BASE_URL")

# --- 프로바이더 / 모델 선택 ---
PREFERRED_PROVIDER: str = os.environ.get("PREFERRED_PROVIDER", "openai").lower()
BIG_MODEL: str = os.environ.get("BIG_MODEL", "gpt-4.1")
SMALL_MODEL: str = os.environ.get("SMALL_MODEL", "gpt-4.1-mini")

# --- 서버 ---
HOST: str = os.environ.get("HOST", "0.0.0.0")
PORT: int = int(os.environ.get("PORT", "8082"))
