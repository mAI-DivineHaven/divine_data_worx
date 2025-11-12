"""Structured logging utilities for the DivineHaven backend."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

try:
    from opentelemetry import trace
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    trace = None  # type: ignore[assignment]


class JsonFormatter(logging.Formatter):
    """Simple JSON formatter emitting structured log lines."""

    def __init__(self, service_name: str | None = None) -> None:
        super().__init__()
        self._service_name = service_name

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        """Format the log record as a JSON payload."""

        base: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC)
            .isoformat()
            .replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if self._service_name:
            base["service"] = self._service_name

        span_context = None
        if trace is not None:
            span = trace.get_current_span()
            span_context = span.get_span_context() if span else None
        if span_context and span_context.trace_id:
            base["trace_id"] = format(span_context.trace_id, "032x")
            base["span_id"] = format(span_context.span_id, "016x")
            base["trace_flags"] = int(span_context.trace_flags)

        if record.exc_info:
            base["exception"] = self.formatException(record.exc_info)

        # Include any extra structured attributes that were passed via logger.bind
        for key, value in record.__dict__.items():
            if key.startswith("_"):
                continue
            if key in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
            }:
                continue
            base.setdefault("extra", {})[key] = value

        return json.dumps(base, default=_json_default)


def _json_default(value: Any) -> Any:
    """Fallback JSON serializer for unsupported types."""

    if isinstance(value, (datetime,)):
        return value.isoformat()
    return str(value)


def configure_logging(level: str = "INFO", service_name: str | None = None) -> None:
    """Configure application-wide structured logging."""

    logging_level = getattr(logging, level.upper(), logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter(service_name=service_name))

    logging.basicConfig(level=logging_level, handlers=[handler], force=True)


def get_logger(name: str = "divinehaven") -> logging.Logger:
    """Return a structured logger instance."""

    return logging.getLogger(name)
