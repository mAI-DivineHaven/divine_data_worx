"""Request logging middleware capturing structured metadata."""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterable

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from ..utils.logging import get_logger
from ..utils.metrics import observe_request

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Emit structured logs for every request/response pair."""

    def __init__(
        self,
        app,
        exempt_paths: Iterable[str] | None = None,
        metrics_enabled: bool = True,
    ) -> None:
        super().__init__(app)
        self.exempt_paths = set(exempt_paths or [])
        self.metrics_enabled = metrics_enabled

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        start = time.perf_counter()
        path = request.url.path
        route = request.scope.get("route")
        metric_path = getattr(route, "path", path)
        method = request.method

        try:
            response = await call_next(request)
        except Exception:  # noqa: BLE001
            duration = time.perf_counter() - start
            logger.exception(
                "request_error",
                extra={
                    "request_id": request_id,
                    "path": path,
                    "method": method,
                    "duration_ms": round(duration * 1000, 3),
                    "client_ip": request.client.host if request.client else None,
                },
            )
            if self.metrics_enabled:
                observe_request(method, metric_path, 500, duration)
            raise

        duration = time.perf_counter() - start
        user = getattr(request.state, "user", None) or {}
        user_id = user.get("sub") if isinstance(user, dict) else None

        if path not in self.exempt_paths:
            logger.info(
                "request",
                extra={
                    "request_id": request_id,
                    "path": path,
                    "method": method,
                    "status_code": response.status_code,
                    "duration_ms": round(duration * 1000, 3),
                    "client_ip": request.client.host if request.client else None,
                    "user_id": user_id,
                },
            )

        if self.metrics_enabled:
            observe_request(method, metric_path, response.status_code, duration)

        response.headers.setdefault("X-Request-ID", request_id)
        return response


__all__ = ["RequestLoggingMiddleware"]
