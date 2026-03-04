"""FastAPI application factory."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request

from app.logging import logger
from app.providers.registry import register_default_providers
from app.routers import messages


def create_app() -> FastAPI:
    """Build and return the fully-configured FastAPI application."""
    app = FastAPI(
        title="Anthropic Proxy for LiteLLM",
        description="Translates Anthropic API requests to any LLM via LiteLLM.",
        version="1.0.0",
    )

    # --- Middleware ---

    @app.middleware("http")
    async def _log_requests(request: Request, call_next):  # noqa: ANN001
        logger.debug(f"Request: {request.method} {request.url.path}")
        return await call_next(request)

    # --- Providers ---
    register_default_providers()

    # --- Routers ---
    app.include_router(messages.router)

    # --- Health / root ---
    @app.get("/")
    async def root():
        return {"message": "Anthropic Proxy for LiteLLM"}

    return app


app = create_app()
