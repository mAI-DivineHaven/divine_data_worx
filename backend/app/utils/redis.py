"""Async Redis utility helpers with graceful fallbacks."""

from __future__ import annotations

import asyncio

from redis.asyncio import Redis
from redis.exceptions import RedisError

from ..config import settings
from .logging import get_logger

logger = get_logger(__name__)

_redis_client: Redis | None = None
_redis_lock = asyncio.Lock()


async def get_redis_client(redis_url: str) -> Redis | None:
    """Return a cached Redis client instance or ``None`` when unavailable."""

    if not redis_url:
        return None

    global _redis_client
    if _redis_client is not None:
        return _redis_client

    async with _redis_lock:
        if _redis_client is not None:
            return _redis_client
        try:
            client = Redis.from_url(
                redis_url,
                encoding="utf-8",
                decode_responses=False,
                health_check_interval=settings.REDIS_HEALTH_CHECK_INTERVAL,
                max_connections=settings.REDIS_MAX_CONNECTIONS,
                socket_timeout=_optional_timeout(settings.REDIS_SOCKET_TIMEOUT),
                socket_connect_timeout=_optional_timeout(settings.REDIS_SOCKET_CONNECT_TIMEOUT),
                retry_on_timeout=settings.REDIS_RETRY_ON_TIMEOUT,
                socket_keepalive=settings.REDIS_SOCKET_KEEPALIVE,
                client_name=settings.REDIS_CLIENT_NAME or None,
            )
            await client.ping()
        except (RedisError, OSError) as exc:
            logger.warning("Redis connection unavailable", extra={"error": str(exc)})
            return None
        else:
            _redis_client = client
            logger.info("Redis connection established", extra={"redis_url": redis_url})
            return _redis_client


async def close_redis() -> None:
    """Close the cached Redis connection if it exists."""

    global _redis_client
    if _redis_client is not None:
        await _redis_client.close()
        _redis_client = None
        logger.info("Redis connection closed")


def _optional_timeout(value: float | None) -> float | None:
    """Convert zero-ish timeout values into ``None`` for redis-py."""

    if value is None or value <= 0:
        return None
    return value
