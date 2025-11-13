"""Prometheus metrics helpers and metric definitions."""

from __future__ import annotations

import time

from prometheus_client import (  # type: ignore
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    Summary,
    generate_latest,
)

from ..config import settings

NAMESPACE = settings.METRICS_NAMESPACE

REQUEST_COUNTER = Counter(
    "requests_total",
    "Total HTTP requests",
    labelnames=("method", "path", "status"),
    namespace=NAMESPACE,
)

REQUEST_LATENCY = Histogram(
    "request_latency_seconds",
    "Request latency in seconds",
    labelnames=("method", "path"),
    namespace=NAMESPACE,
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)

REQUEST_ERRORS = Counter(
    "request_errors_total",
    "Total HTTP request errors",
    labelnames=("method", "path", "status"),
    namespace=NAMESPACE,
)

CACHE_HITS = Counter(
    "cache_hits_total",
    "Cache hits per layer",
    labelnames=("layer",),
    namespace=NAMESPACE,
)

CACHE_MISSES = Counter(
    "cache_misses_total",
    "Cache misses per layer",
    labelnames=("layer",),
    namespace=NAMESPACE,
)

RATE_LIMIT_REJECTIONS = Counter(
    "rate_limit_rejections_total",
    "Requests rejected by the rate limiter",
    labelnames=("identity_type",),
    namespace=NAMESPACE,
)

DB_QUERY_DURATION = Summary(
    "db_query_duration_seconds",
    "Database query execution time",
    labelnames=("query",),
    namespace=NAMESPACE,
)

CACHE_SIZE = Gauge(
    "cache_l1_size",
    "Current size of the in-memory L1 cache",
    namespace=NAMESPACE,
)


def observe_request(method: str, path: str, status_code: int, duration: float) -> None:
    """Record metrics for a processed HTTP request."""

    REQUEST_COUNTER.labels(method=method, path=path, status=status_code).inc()
    REQUEST_LATENCY.labels(method=method, path=path).observe(duration)
    if status_code >= 500:
        REQUEST_ERRORS.labels(method=method, path=path, status=status_code).inc()


def metrics_response() -> tuple[bytes, str]:
    """Return serialized Prometheus metrics payload and content type."""

    payload = generate_latest()
    return payload, CONTENT_TYPE_LATEST


class QueryTimer:
    """Context manager helper to time database queries."""

    def __init__(self, label: str) -> None:
        self.label = label
        self._start: float | None = None

    def __enter__(self) -> QueryTimer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        if self._start is None:
            return
        duration = time.perf_counter() - self._start
        DB_QUERY_DURATION.labels(query=self.label).observe(duration)
