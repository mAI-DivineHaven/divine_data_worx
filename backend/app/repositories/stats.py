"""Statistics data access repository."""

from __future__ import annotations

import asyncpg

from ..models import EmbeddingCoverage


class StatsRepository:
    """Repository for analytics/statistics queries."""

    def __init__(self, conn: asyncpg.Connection) -> None:
        self._conn = conn

    async def embedding_coverage(self) -> list[EmbeddingCoverage]:
        """Return embedding coverage metrics per translation."""
        rows = await self._conn.fetch(
            """
            SELECT v.translation_code,
                   COUNT(*) AS verses,
                   COUNT(e.verse_id) AS embedded,
                   COUNT(*) - COUNT(e.verse_id) AS missing
            FROM verse v
            LEFT JOIN verse_embedding e ON e.verse_id = v.verse_id
            GROUP BY v.translation_code
            ORDER BY v.translation_code
            """
        )
        return [EmbeddingCoverage(**dict(r)) for r in rows]
