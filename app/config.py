"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


# --- API Keys ---
ANTHROPIC_API_KEY: str | None = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY: str | None = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY: str | None = os.environ.get("GEMINI_API_KEY")

# --- Vertex AI ---
VERTEX_PROJECT: str = os.environ.get("VERTEX_PROJECT", "unset")
VERTEX_LOCATION: str = os.environ.get("VERTEX_LOCATION", "unset")
USE_VERTEX_AUTH: bool = os.environ.get("USE_VERTEX_AUTH", "False").lower() == "true"

# --- OpenAI ---
OPENAI_BASE_URL: str | None = os.environ.get("OPENAI_BASE_URL")

# --- Provider / Model Selection ---
PREFERRED_PROVIDER: str = os.environ.get("PREFERRED_PROVIDER", "openai").lower()
BIG_MODEL: str = os.environ.get("BIG_MODEL", "gpt-4.1")
SMALL_MODEL: str = os.environ.get("SMALL_MODEL", "gpt-4.1-mini")

# --- Server ---
HOST: str = os.environ.get("HOST", "0.0.0.0")
PORT: int = int(os.environ.get("PORT", "8082"))
