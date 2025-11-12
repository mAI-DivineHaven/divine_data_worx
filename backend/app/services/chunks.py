"""Service layer for chunk embedding search and retrieval."""

from __future__ import annotations

import asyncpg

from ..config import settings
from ..models import ChunkHit, ChunkSearchQuery, ChunkSearchResponse
from ..repositories import ChunkRepository

DEFAULT_CONTEXT_VERSES = 2


class ChunkService:
    """Provide high-level operations for chunk embeddings."""

    def __init__(self, conn: asyncpg.Connection):
        self.repo = ChunkRepository(conn)

    async def search(self, query: ChunkSearchQuery) -> ChunkSearchResponse:
        """Run a semantic chunk search and return a typed response."""

        limit = min(query.top_k, settings.PAGE_MAX)
        offset = query.offset

        rows = await self.repo.semantic_search(
            embedding=query.embedding,
            translation=query.translation,
            book_number=query.book_number,
            testament=query.testament,
            window_size=query.window_size,
            limit=limit,
            offset=offset,
            include_context=query.include_context,
            context_verses=DEFAULT_CONTEXT_VERSES,
        )

        total = await self.repo.count_chunks(
            translation=query.translation,
            book_number=query.book_number,
            testament=query.testament,
            window_size=query.window_size,
        )

        items = [self._record_to_hit(row) for row in rows]
        metadata = {
            "limit": limit,
            "offset": offset,
            "requested_top_k": query.top_k,
            "page_max": settings.PAGE_MAX,
            "context": query.include_context,
        }

        return ChunkSearchResponse(total=total, items=items, query_metadata=metadata)

    async def get_chunk(self, chunk_id: str, *, include_context: bool = False) -> ChunkHit | None:
        """Fetch a single chunk embedding by ID."""

        row = await self.repo.get_chunk(
            chunk_id,
            include_context=include_context,
            context_verses=DEFAULT_CONTEXT_VERSES,
        )
        if not row:
            return None
        return self._record_to_hit(row)

    @staticmethod
    def _record_to_hit(row: asyncpg.Record) -> ChunkHit:
        """Map a database record to the API response model."""

        return ChunkHit(
            chunk_id=row["chunk_id"],
            translation_code=row["translation_code"],
            book_number=row["book_number"],
            chapter_start=row["chapter_start"],
            verse_start=row["verse_start"],
            chapter_end=row["chapter_end"],
            verse_end=row["verse_end"],
            text=row["text"],
            score=float(row["score"]),
            window_size=row.get("window_size"),
            stride=row.get("stride"),
            context_before=row.get("context_before"),
            context_after=row.get("context_after"),
        )
