"""Entry point — run with ``uvicorn server:app`` or ``python server.py``."""

from app.main import app  # noqa: F401 — re-exported for uvicorn

if __name__ == "__main__":
    import uvicorn
    from app.config import HOST, PORT

    uvicorn.run("server:app", host=HOST, port=PORT, reload=True)
