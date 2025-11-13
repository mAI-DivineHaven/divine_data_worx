"""Statistics service layer."""

from __future__ import annotations

import asyncpg

from ..models import EmbeddingCoverage
from ..repositories.stats import StatsRepository


class StatsService:
    """Provide analytics-related business logic."""

    def __init__(
        self,
        conn: asyncpg.Connection,
        repository: StatsRepository | None = None,
    ) -> None:
        self._repo = repository or StatsRepository(conn)

    async def embedding_coverage(self) -> list[EmbeddingCoverage]:
        """Return embedding coverage metrics."""
        return await self._repo.embedding_coverage()
