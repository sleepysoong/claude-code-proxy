"""진입점 — ``uvicorn server:app`` 또는 ``python server.py``로 실행."""

from app.main import app  # noqa: F401 — uvicorn을 위해 re-export

if __name__ == "__main__":
    import uvicorn
    from app.config import HOST, PORT

    uvicorn.run("server:app", host=HOST, port=PORT, reload=True)
