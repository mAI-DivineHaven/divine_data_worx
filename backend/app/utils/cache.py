"""Caching utilities providing L1/L2 caching behaviour."""

from __future__ import annotations

import asyncio
import pickle
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from redis.exceptions import RedisError

from .logging import get_logger
from .metrics import CACHE_HITS, CACHE_MISSES, CACHE_SIZE
from .redis import get_redis_client

logger = get_logger(__name__)


@dataclass
class CacheEntry:
    value: Any
    expires_at: float


class L1Cache:
    """A simple asyncio-safe LRU cache with TTL support."""

    def __init__(self, max_items: int):
        self.max_items = max_items
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Any | None:
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if entry.expires_at and entry.expires_at < time.time():
                self._store.pop(key, None)
                return None
            # Move to end to denote recent use
            self._store.move_to_end(key)
            return entry.value

    async def set(self, key: str, value: Any, ttl: int) -> None:
        expires_at = time.time() + ttl if ttl else 0
        async with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = CacheEntry(value=value, expires_at=expires_at)
            if len(self._store) > self.max_items:
                self._store.popitem(last=False)
            CACHE_SIZE.set(len(self._store))

    async def delete_pattern(self, pattern: str) -> None:
        async with self._lock:
            keys_to_delete = [key for key in self._store if _match(pattern, key)]
            for key in keys_to_delete:
                self._store.pop(key, None)
            CACHE_SIZE.set(len(self._store))

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()
            CACHE_SIZE.set(0)


def _match(pattern: str, key: str) -> bool:
    if pattern == "*":
        return True
    if pattern.endswith("*"):
        return key.startswith(pattern[:-1])
    return pattern == key


class CacheManager:
    """Coordinates L1 (memory) and L2 (Redis) caches."""

    def __init__(
        self,
        redis_url: str,
        default_ttl: int,
        max_items: int,
        namespace: str,
    ) -> None:
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self.namespace = namespace
        self.l1 = L1Cache(max_items=max_items)
        self._redis_checked = False
        self._redis_available = False

    def _namespaced(self, key: str) -> str:
        return f"{self.namespace}:{key}"

    async def _redis(self):
        if not self.redis_url:
            return None
        if self._redis_checked and not self._redis_available:
            return None
        client = await get_redis_client(self.redis_url)
        self._redis_checked = True
        self._redis_available = client is not None
        return client

    async def get(self, key: str) -> Any | None:
        value = await self.l1.get(key)
        if value is not None:
            CACHE_HITS.labels(layer="l1").inc()
            return value

        CACHE_MISSES.labels(layer="l1").inc()

        redis = await self._redis()
        if redis is None:
            return None

        namespaced_key = self._namespaced(key)
        try:
            payload = await redis.get(namespaced_key)
        except RedisError as exc:
            logger.warning("Redis get failed", extra={"error": str(exc)})
            CACHE_MISSES.labels(layer="l2").inc()
            return None

        if payload is None:
            CACHE_MISSES.labels(layer="l2").inc()
            return None

        try:
            value = pickle.loads(payload)
        except pickle.PickleError as exc:
            logger.error("Cache deserialization failed", extra={"error": str(exc)})
            CACHE_MISSES.labels(layer="l2").inc()
            return None

        CACHE_HITS.labels(layer="l2").inc()
        ttl = self.default_ttl
        try:
            ttl_response = await redis.ttl(namespaced_key)
        except RedisError as exc:
            logger.debug("Redis ttl failed", extra={"error": str(exc)})
        else:
            if ttl_response is not None and ttl_response >= 0:
                ttl = ttl_response
        await self.l1.set(key, value, ttl=max(ttl, 0))
        return value

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        ttl = self.default_ttl if ttl is None else ttl
        await self.l1.set(key, value, ttl)

        redis = await self._redis()
        if redis is None or ttl == 0:
            return

        payload = pickle.dumps(value)
        try:
            await redis.set(self._namespaced(key), payload, ex=ttl)
        except RedisError as exc:
            logger.warning("Redis set failed", extra={"error": str(exc)})

    async def invalidate(self, pattern: str = "*") -> None:
        await self.l1.delete_pattern(pattern)

        redis = await self._redis()
        if redis is None:
            return

        namespace_pattern = self._namespaced(pattern)
        try:
            async for key in redis.scan_iter(match=namespace_pattern):
                await redis.delete(key)
        except RedisError as exc:
            logger.warning("Redis invalidation failed", extra={"error": str(exc)})

    def cached(
        self,
        ttl: int | None = None,
        key_builder: Callable[..., str] | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Awaitable[Any]]]:
        """Decorator to cache the results of async functions."""

        def decorator(func: Callable[..., Awaitable[Any]]):
            if not asyncio.iscoroutinefunction(func):
                raise TypeError("CacheManager.cached decorator requires async functions")

            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                key = (
                    key_builder(*args, **kwargs) if key_builder else _build_key(func, args, kwargs)
                )
                cached_value = await self.get(key)
                if cached_value is not None:
                    return cached_value

                result = await func(*args, **kwargs)
                await self.set(key, result, ttl=ttl)
                return result

            return wrapper

        return decorator

    async def clear(self) -> None:
        await self.l1.clear()
        redis = await self._redis()
        if redis is None:
            return
        try:
            keys = [key async for key in redis.scan_iter(match=f"{self.namespace}:*")]
            if keys:
                await redis.delete(*keys)
        except RedisError as exc:
            logger.warning("Redis clear failed", extra={"error": str(exc)})


def _build_key(func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    return "::".join([func.__module__, func.__qualname__, repr(args), repr(sorted(kwargs.items()))])
