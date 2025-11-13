"""
DivineHaven FastAPI Application

Biblical text exploration API with graph and vector search capabilities.
Provides REST endpoints for:
- Full-text search (PostgreSQL FTS)
- Semantic search (pgvector embeddings)
- Hybrid search (RRF fusion)
- Cross-translation graph queries (Neo4j)
- Biblical text and metadata retrieval

Architecture:
    - Database: PostgreSQL with pgvector extension
    - Graph: Neo4j for cross-translation relationships
    - Embeddings: 768-D vectors from embeddinggemma
    - Search: DiskANN + FTS + RRF hybrid

Environment Configuration:
    All settings loaded from .env file via pydantic-settings.
    See backend/app/config.py for available configuration options.

API Endpoints:
    - /v1/healthz: Health check
    - /v1/verses/*: Verse and metadata retrieval
    - /v1/search/*: FTS, vector, and hybrid search
    - /v1/chunks/*: Chunk embedding retrieval and semantic search
    - /v1/graph/*: Cross-translation parallel verses
    - /v1/stats/*: Pipeline monitoring and metrics
    - /v1/assets/*: Asset CRUD, embeddings, and verse linking
    - /v1/analytics/*: Query telemetry and usage analytics

Interactive Documentation:
    - Swagger UI: http://localhost:8000/docs
    - ReDoc: http://localhost:8000/redoc
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import settings
from .db.neo4j import close_driver, init_driver
from .db.postgres_async import init_pool
from .dependencies.cache import get_cache_manager
from .middleware.auth import JWTAuthMiddleware
from .middleware.rate_limit import RateLimitMiddleware
from .middleware.request_logging import RequestLoggingMiddleware
from .routers import (
    analytics,
    assets,
    batch,
    chunks,
    graph,
    health,
    memory,
    monitoring,
    retrieval,
    search,
    stats,
    user_profiles,
    verses,
)
from .utils.logging import configure_logging
from .utils.observability import configure_tracing
from .utils.redis import close_redis


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan manager.

    Handles initialization and cleanup of shared resources:
        - PostgreSQL connection pool (asyncpg)
        - Redis connection (via cache manager)
        - Neo4j driver (graph database)
        - Cache manager initialization
        - Close Neo4j driver explicitly

    Args:
        app: FastAPI application instance

    Yields:
        None (context manager pattern for startup/shutdown)
    """
    # Startup: Initialize connection pools
    await init_pool(min_size=1, max_size=16)
    cache = get_cache_manager()

    await init_driver()

    try:
        # Application is running, yield control
        yield
    finally:
        # Shutdown: Clean up resources
        await close_driver()
        await cache.clear()
        await close_redis()


# FastAPI application instance
configure_logging(settings.LOG_LEVEL, service_name=settings.OTEL_SERVICE_NAME)

app = FastAPI(
    title="DivineHaven API",
    description="Biblical text exploration with graph and vector search",
    version="0.1.0",
    lifespan=lifespan,
    default_response_class=JSONResponse,
)

tracer_provider = configure_tracing(
    settings,
    service_name=settings.OTEL_SERVICE_NAME,
    service_version="0.1.0",
)

if tracer_provider is not None:
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(
            app,
            tracer_provider=tracer_provider,
            excluded_urls=settings.TRACING_EXCLUDED_PATHS,
        )
    except Exception:  # noqa: BLE001
        # Failing to instrument tracing should not prevent the API from starting
        import logging

        logging.getLogger(__name__).exception("failed_to_instrument_fastapi")

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers with API prefix
app.include_router(health.router, prefix=settings.API_PREFIX)
app.include_router(verses.router, prefix=settings.API_PREFIX)
app.include_router(search.router, prefix=settings.API_PREFIX)
app.include_router(stats.router, prefix=settings.API_PREFIX)
app.include_router(graph.router, prefix=settings.API_PREFIX)
app.include_router(retrieval.router, prefix=settings.API_PREFIX)
app.include_router(chunks.router, prefix=settings.API_PREFIX)
app.include_router(batch.router, prefix=settings.API_PREFIX)
app.include_router(analytics.router, prefix=settings.API_PREFIX)
app.include_router(assets.router, prefix=settings.API_PREFIX)
app.include_router(memory.router, prefix=settings.API_PREFIX)
app.include_router(user_profiles.router, prefix=settings.API_PREFIX)
app.include_router(monitoring.router)  # No prefix - uses /metrics directly

# Rate limiting and request logging middleware
exempt_paths = {
    "/",
    "/healthz",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/metrics",
    f"{settings.API_PREFIX}/healthz",
    "/docs/oauth2-redirect",
}

app.add_middleware(
    RateLimitMiddleware,
    redis_url=settings.REDIS_URL,
    limit=settings.RATE_LIMIT_REQUESTS,
    window_seconds=settings.RATE_LIMIT_WINDOW_SECONDS,
    exempt_paths=exempt_paths,
    enabled=settings.RATE_LIMIT_ENABLED,
)

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    JWTAuthMiddleware,
    secret_key=settings.JWT_SECRET_KEY,
    algorithm=settings.JWT_ALGORITHM,
    exempt_paths=exempt_paths,
)
