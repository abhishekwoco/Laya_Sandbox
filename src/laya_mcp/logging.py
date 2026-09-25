"""structlog JSON logging, with the stdlib (uvicorn, fastmcp, laya) bridged into the same output.

One JSON object per line to stderr and to a rotating file under `<data_dir>/logs/laya-mcp.log`.
Raw content (states, items, text) is redacted from structured events unless
`Settings.log_content` is true; FastMCP's request logger only includes payloads in that case too.
"""
from __future__ import annotations

import logging
import logging.handlers
import sys
from typing import Any

import structlog

from .config import Settings

_MARK = "_laya_mcp_handler"
CONTENT_KEYS = frozenset({"state", "states", "items", "text", "content", "payload", "arguments", "prompt", "body"})
_BRIDGED = ("uvicorn", "uvicorn.error", "uvicorn.access", "fastmcp", "laya", "huey", "httpx", "sqlalchemy")


class _StderrHandler(logging.StreamHandler):
    """Always writes to the *current* sys.stderr (survives stream swaps by test runners/services)."""

    @property  # type: ignore[override]
    def stream(self):
        return sys.stderr

    @stream.setter
    def stream(self, _value) -> None:
        pass


def _redactor(log_content: bool):
    def redact(_logger: Any, _name: str, event: dict[str, Any]) -> dict[str, Any]:
        if not log_content:
            for key in CONTENT_KEYS & event.keys():
                event[key] = "[redacted]"
        return event

    return redact


def configure_logging(settings: Settings) -> None:
    """Idempotent: safe to call once per app start (tests create several apps)."""
    level = logging.getLevelName(str(settings.log_level).upper())
    if not isinstance(level, int):
        level = logging.INFO

    shared: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        _redactor(settings.log_content),
    ]
    structlog.configure(
        processors=[
            *shared,
            structlog.processors.StackInfoRenderer(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
    )

    handlers: list[logging.Handler] = [_StderrHandler()]
    try:
        log_dir = settings.data_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.handlers.RotatingFileHandler(
                log_dir / "laya-mcp.log", maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8", delay=True
            )
        )
    except OSError as e:  # a read-only data dir must not stop the server
        print(f"laya-mcp: file logging disabled: {e}", file=sys.stderr)

    root = logging.getLogger()
    for h in list(root.handlers):
        if getattr(h, _MARK, False):
            root.removeHandler(h)
            h.close()
    for h in handlers:
        setattr(h, _MARK, True)
        h.setFormatter(formatter)
        root.addHandler(h)
    root.setLevel(level)

    # Route library loggers (fastmcp installs its own rich console handler) through the root handlers.
    for name in _BRIDGED:
        lg = logging.getLogger(name)
        for h in list(lg.handlers):
            lg.removeHandler(h)
        lg.propagate = True
    logging.getLogger("uvicorn.access").setLevel(max(level, logging.WARNING))
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str = "laya_mcp") -> Any:
    return structlog.get_logger(name)
