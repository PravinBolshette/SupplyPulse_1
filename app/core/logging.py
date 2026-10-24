"""
Structured Logging Configuration with Correlation Context Tracking.

Provides unified JSON and text formatters that automatically bind the active
request correlation ID from contextvars, ensuring complete distributed tracing.
"""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Optional

from app.core.config import Settings, get_settings

# Context variable holding the correlation ID for the active async execution context
correlation_id_ctx: ContextVar[str] = ContextVar("correlation_id", default="-")


def get_correlation_id() -> str:
    """Retrieves the active correlation ID or default placeholder."""
    return correlation_id_ctx.get()


def set_correlation_id(correlation_id: str) -> None:
    """Binds a correlation ID to the current task execution context."""
    correlation_id_ctx.set(correlation_id)


class JSONLogFormatter(logging.Formatter):
    """
    Serializes Python logging records into structured JSON lines.
    Includes ISO-8601 timestamps, log level, logger name, correlation ID,
    and exception tracebacks if present.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "request_id": correlation_id_ctx.get(),
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Inject extra attributes passed in logging call
        if hasattr(record, "extra_data") and isinstance(record.extra_data, dict):
            log_payload["extra"] = record.extra_data

        if record.exc_info:
            log_payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_payload, default=str)


class TextLogFormatter(logging.Formatter):
    """
    Standard human-readable text formatter with correlation ID tagging.
    """

    def format(self, record: logging.LogRecord) -> str:
        req_id = correlation_id_ctx.get()
        req_tag = f"[request_id={req_id}] " if req_id != "-" else ""
        asctime = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        msg = record.getMessage()
        formatted = f"{asctime} [{record.levelname:<8}] {req_tag}{record.name}: {msg}"
        if record.exc_info:
            formatted += "\n" + self.formatException(record.exc_info)
        return formatted


def setup_logging(settings: Optional[Settings] = None) -> logging.Logger:
    """
    Initializes root and application loggers with handlers and formatters.
    """
    if settings is None:
        settings = get_settings()

    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers to prevent duplicate output
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    if settings.LOG_FORMAT == "json":
        console_handler.setFormatter(JSONLogFormatter())
    else:
        console_handler.setFormatter(TextLogFormatter())

    root_logger.addHandler(console_handler)

    # Configure uvicorn loggers to use the same formatter
    for uvicorn_name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        uv_logger = logging.getLogger(uvicorn_name)
        uv_logger.handlers = []
        uv_logger.propagate = True

    app_logger = logging.getLogger("supplypulse")
    app_logger.setLevel(log_level)
    return app_logger


def get_logger(name: str) -> logging.Logger:
    """
    Factory function returning a scoped child logger.
    """
    return logging.getLogger(f"supplypulse.{name}")
