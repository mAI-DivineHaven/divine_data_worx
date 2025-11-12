"""Redis-backed rate limiting middleware with in-memory fallback."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterable

from fastapi import Request
from fastapi.responses import JSONResponse
from redis.exceptions import RedisError
from starlette.middleware.base import BaseHTTPMiddleware

from ..utils.logging import get_logger
from ..utils.metrics import RATE_LIMIT_REJECTIONS
from ..utils.redis import get_redis_client

logger = get_logger(__name__)


class _LocalLimiter:
    def __init__(self) -> None:
        self._hits: dict[str, tuple[int, float]] = {}
        self._lock = asyncio.Lock()

    async def increment(self, key: str, limit: int, window_seconds: int) -> tuple[int, int, int]:
        now = time.time()
        window_start = int(now // window_seconds) * window_seconds
        bucket_key = f"{key}:{window_start}"
        reset = window_start + window_seconds
        async with self._lock:
            count, expiry = self._hits.get(bucket_key, (0, reset))
            if expiry <= now:
                count = 0
                expiry = reset
            count += 1
            self._hits[bucket_key] = (count, reset)
            # Cleanup old buckets
            expired_keys = [k for k, (_, exp) in self._hits.items() if exp <= now]
            for k in expired_keys:
                self._hits.pop(k, None)
        remaining = max(limit - count, 0)
        return count, remaining, int(reset)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Apply rate limits per user (auth) or per IP (anonymous)."""

    def __init__(
        self,
        app,
        redis_url: str,
        limit: int,
        window_seconds: int,
        exempt_paths: Iterable[str] | None = None,
        enabled: bool = True,
    ) -> None:
        super().__init__(app)
        self.redis_url = redis_url
        self.limit = limit
        self.window_seconds = window_seconds
        self.exempt_paths = set(exempt_paths or [])
        self.enabled = enabled and limit > 0
        self._local = _LocalLimiter()
        if not self.enabled:
            logger.info("RateLimitMiddleware disabled via configuration")

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        if self._should_skip(request):
            return await call_next(request)

        if not self.enabled:
            return await call_next(request)

        identity = self._identifier_for(request)
        if identity is None:
            return await call_next(request)

        identifier, identity_type = identity

        allowed, remaining, reset = await self._check(identifier)
        if allowed:
            response = await call_next(request)
        else:
            logger.warning(
                "rate_limit_exceeded",
                extra={
                    "identifier": identifier,
                    "identity_type": identity_type,
                    "path": request.url.path,
                    "limit": self.limit,
                    "window_seconds": self.window_seconds,
                },
            )
            RATE_LIMIT_REJECTIONS.labels(identity_type=identity_type).inc()
            response = JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"},
            )

        limit_headers = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(max(remaining, 0)),
            "X-RateLimit-Reset": str(reset),
        }
        for header, value in limit_headers.items():
            response.headers[header] = value
        if not allowed:
            response.headers["Retry-After"] = str(max(reset - int(time.time()), 0))

        return response

    def _should_skip(self, request: Request) -> bool:
        if request.method.upper() == "OPTIONS":
            return True
        return request.url.path in self.exempt_paths

    def _identifier_for(self, request: Request) -> tuple[str, str] | None:
        user = getattr(request.state, "user", None)
        if user and isinstance(user, dict):
            subject = user.get("sub") or user.get("id")
            if subject:
                return f"user:{subject}", "authenticated"
        client = request.client
        if client and client.host:
            return f"ip:{client.host}", "anonymous"
        return None

    async def _check(self, identifier: str) -> tuple[bool, int, int]:
        redis = await get_redis_client(self.redis_url)
        if redis is None:
            count, remaining, reset = await self._local.increment(
                identifier, self.limit, self.window_seconds
            )
            return count <= self.limit, remaining, reset

        key = f"ratelimit:{identifier}"
        try:
            count = await redis.incr(key)
            if count == 1:
                await redis.expire(key, self.window_seconds)
            ttl = await redis.ttl(key)
        except (RedisError, OSError) as exc:
            logger.warning("Redis rate limit failed, falling back", extra={"error": str(exc)})
            count, remaining, reset = await self._local.increment(
                identifier, self.limit, self.window_seconds
            )
            return count <= self.limit, remaining, reset

        remaining = max(self.limit - count, 0)
        reset = int(time.time() + (ttl if ttl and ttl > 0 else self.window_seconds))
        return count <= self.limit, remaining, reset


__all__ = ["RateLimitMiddleware"]
