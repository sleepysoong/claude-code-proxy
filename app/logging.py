"""Centralized logging configuration."""

from __future__ import annotations

import logging
import sys


class _MessageFilter(logging.Filter):
    """Block noisy log messages from LiteLLM and HTTP internals."""

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
    """Configure and return the application logger."""
    logging.basicConfig(
        level=logging.WARN,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Silence uvicorn loggers
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        logging.getLogger(name).setLevel(logging.WARNING)

    # Apply global filter
    root = logging.getLogger()
    root.addFilter(_MessageFilter())

    return logging.getLogger("app")


# Module-level logger used across the application
logger = setup_logging()


# --- Pretty request logging ---


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
    """Pretty-print a request summary to stdout."""
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
