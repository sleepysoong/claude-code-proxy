"""FastAPI 애플리케이션 팩토리."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request

from app.logging import logger
from app.providers.registry import register_default_providers
from app.routers import messages


def create_app() -> FastAPI:
    """완전히 설정된 FastAPI 애플리케이션을 생성하고 반환한다."""
    app = FastAPI(
        title="Anthropic Proxy for LiteLLM",
        description="Anthropic API 요청을 LiteLLM을 통해 다른 LLM으로 변환하는 프록시.",
        version="1.0.0",
    )

    # --- 미들웨어 ---

    @app.middleware("http")
    async def _log_requests(request: Request, call_next):  # noqa: ANN001
        logger.debug(f"요청: {request.method} {request.url.path}")
        return await call_next(request)

    # --- 프로바이더 등록 ---
    register_default_providers()

    # --- 라우터 등록 ---
    app.include_router(messages.router)

    # --- 헬스체크 / 루트 ---
    @app.get("/")
    async def root():
        return {"message": "Anthropic Proxy for LiteLLM"}

    return app


app = create_app()
