"""
Async PostgreSQL connection pooling for FastAPI.

Provides asyncpg connection pool management with proper initialization,
dependency injection for FastAPI routes, and automatic resource cleanup.

The connection pool is initialized once at application startup and reused
across all requests for optimal performance.
"""

from collections.abc import AsyncIterator

import asyncpg

from ..config import settings


def _normalize_dsn(dsn: str) -> str:
    """
    Normalize PostgreSQL DSN to standard format.

    Handles various DSN formats and normalizes them to postgresql://.
    Strips SQLAlchemy-style driver specifications (e.g., postgresql+psycopg://).

    Args:
        dsn: Input DSN in various formats

    Returns:
        Normalized DSN starting with postgresql://
    """
    # Remove SQLAlchemy driver specifications
    if dsn.startswith(("postgresql+", "postgres+")):
        dsn = "postgresql://" + dsn.split("://", 1)[1]

    # Normalize postgres:// to postgresql://
    if dsn.startswith("postgres://"):
        dsn = "postgresql://" + dsn.split("://", 1)[1]

    return dsn


# Normalized DSN for asyncpg
DSN = _normalize_dsn(settings.DATABASE_URL)

# Global connection pool (initialized at startup)
_pool: asyncpg.Pool | None = None


async def _init_conn(conn: asyncpg.Connection) -> None:
    """
    Initialize a new connection with application-specific settings.

    Called automatically for each new connection in the pool.
    Sets timeouts and optional pgvector/pgvectorscale query parameters.

    Args:
        conn: New asyncpg connection to initialize
    """
    await conn.execute(
        """
        SET application_name = 'divinehaven-fastapi';
        SET statement_timeout = '5s';
        SET idle_in_transaction_session_timeout = '5s';
        -- Uncomment if using pgvector HNSW fallback:
        -- SET hnsw.ef_search = 64;
        -- Uncomment if using pgvector IVF fallback:
        -- SET ivfflat.probes = 8;
        """
    )


async def init_pool(min_size: int = 1, max_size: int = 16) -> asyncpg.Pool:
    """
    Initialize the global asyncpg connection pool.

    Should be called once at application startup (in lifespan context).
    Subsequent calls return the existing pool without recreation.

    Args:
        min_size: Minimum number of connections to maintain in pool
        max_size: Maximum number of connections allowed in pool

    Returns:
        Initialized asyncpg connection pool

    Raises:
        asyncpg.PostgresError: If database connection fails
    """
    global _pool

    if _pool is None:
        _pool = await asyncpg.create_pool(
            dsn=DSN,
            min_size=min_size,
            max_size=max_size,
            max_inactive_connection_lifetime=60,  # Close idle connections after 60s
            init=_init_conn,
            statement_cache_size=1024,  # Cache prepared statements
        )

    return _pool


async def get_pg() -> AsyncIterator[asyncpg.Connection]:
    """
    FastAPI dependency for injecting PostgreSQL connections.

    Acquires a connection from the pool, yields it to the route handler,
    then automatically returns it to the pool when the request completes.

    Yields:
        asyncpg.Connection: Database connection for the current request

    Raises:
        RuntimeError: If connection pool has not been initialized

    Example:
        ```python
        @router.get("/verses/{verse_id}")
        async def get_verse(
            verse_id: str,
            pg: asyncpg.Connection = Depends(get_pg)
        ):
            row = await pg.fetchrow(
                "SELECT * FROM verse WHERE verse_id = $1",
                verse_id
            )
            return row
        ```
    """
    pool = await init_pool()

    if pool is None:
        raise RuntimeError(
            "PostgreSQL connection pool not initialized. "
            "Call init_pool() in application lifespan."
        )

    async with pool.acquire() as conn:
        yield conn
