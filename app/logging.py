"""중앙 집중식 로깅 설정."""

from __future__ import annotations

import logging
import sys


class _MessageFilter(logging.Filter):
    """LiteLLM 및 HTTP 내부 라이브러리의 불필요한 로그 메시지를 차단하는 필터."""

    _BLOCKED = (
        "LiteLLM completion()",
        "HTTP Request:",
        "selected model name for cost calculation",
        "utils.py",
        "cost_calculator",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        msg = getattr(record, "msg", "")
        if isinstance(msg, str):
            return not any(phrase in msg for phrase in self._BLOCKED)
        return True


def setup_logging() -> logging.Logger:
    """애플리케이션 로거를 설정하고 반환한다."""
    logging.basicConfig(
        level=logging.WARN,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # uvicorn 로거 무음 처리
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        logging.getLogger(name).setLevel(logging.WARNING)

    # 전역 필터 적용
    root = logging.getLogger()
    root.addFilter(_MessageFilter())

    return logging.getLogger("app")


# 애플리케이션 전체에서 사용되는 모듈 수준 로거
logger = setup_logging()


# --- 요청 로그 출력 (컬러) ---


class Colors:
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"


def log_request(
    method: str,
    path: str,
    claude_model: str,
    mapped_model: str,
    num_messages: int,
    num_tools: int,
    status_code: int,
) -> None:
    """요청 요약 정보를 컬러로 stdout에 출력한다."""
    C = Colors

    claude_display = f"{C.CYAN}{claude_model}{C.RESET}"

    mapped_display = mapped_model
    if "/" in mapped_display:
        mapped_display = mapped_display.split("/")[-1]
    mapped_display = f"{C.GREEN}{mapped_display}{C.RESET}"

    tools_str = f"{C.MAGENTA}{num_tools} tools{C.RESET}"
    messages_str = f"{C.BLUE}{num_messages} messages{C.RESET}"

    if status_code == 200:
        status_str = f"{C.GREEN}✓ {status_code} OK{C.RESET}"
    else:
        status_str = f"{C.RED}✗ {status_code}{C.RESET}"

    endpoint = path.split("?")[0] if "?" in path else path
    print(f"{C.BOLD}{method} {endpoint}{C.RESET} {status_str}")
    print(f"{claude_display} → {mapped_display} {tools_str} {messages_str}")
    sys.stdout.flush()
