"""
Async Neo4j connection management for FastAPI.

Provides async Neo4j driver initialization and session dependency injection
for FastAPI routes. The driver is initialized once at module load and reused
across all requests for optimal performance.

The session dependency handles automatic resource cleanup via async context
manager pattern, ensuring connections are properly returned to the pool.

Environment Variables (from config):
    NEO4J_URI: Neo4j bolt connection URI (e.g., bolt://localhost:7687)
    NEO4J_USER: Neo4j username for authentication
    NEO4J_PASSWORD: Neo4j password for authentication

Example Usage:
    ```python
    from fastapi import Depends
    from neo4j import AsyncSession
    from .db.neo4j import get_neo4j_session

    @router.get("/graph/verse/{verse_id}")
    async def get_verse_relationships(
        verse_id: str,
        session: AsyncSession = Depends(get_neo4j_session)
    ):
        query = '''
            MATCH (v:Verse {verse_id: $verse_id})-[r]->(related)
            RETURN v, r, related
        '''
        result = await session.run(query, verse_id=verse_id)
        records = await result.data()
        return records
    ```
"""

import asyncio
from collections.abc import AsyncIterator

from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncSession

from ..config import settings
from ..utils.logging import get_logger

logger = get_logger(__name__)

_driver: AsyncDriver | None = None
_driver_lock = asyncio.Lock()


async def init_driver(max_attempts: int = 3, initial_delay: float = 0.5) -> AsyncDriver:
    """Initialise the Neo4j driver with simple retry semantics."""

    global _driver
    if _driver is not None:
        return _driver

    async with _driver_lock:
        if _driver is not None:
            return _driver

        delay = initial_delay
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            candidate: AsyncDriver | None = None
            try:
                candidate = AsyncGraphDatabase.driver(
                    settings.NEO4J_URI,
                    auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
                )
                await candidate.verify_connectivity()
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "neo4j_driver_initialization_failed",
                    extra={"attempt": attempt, "error": str(exc)},
                )
                if candidate is not None:
                    await candidate.close()
                if attempt < max_attempts:
                    await asyncio.sleep(delay)
                    delay *= 2
                continue

            _driver = candidate
            logger.info("neo4j_driver_initialized", extra={"attempt": attempt})
            return candidate

        message = "Failed to initialise Neo4j driver"
        if last_error is not None:
            raise RuntimeError(message) from last_error
        raise RuntimeError(message)


async def close_driver() -> None:
    """Close the Neo4j driver if it was initialised."""

    global _driver
    if _driver is None:
        return
    await _driver.close()
    _driver = None


async def get_neo4j_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency that yields an active Neo4j session."""

    driver = await init_driver()
    async with driver.session() as session:
        yield session
