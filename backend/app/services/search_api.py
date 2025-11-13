"""Service layer used by API search endpoints.

The :class:`SearchApiService` encapsulates the validation and orchestration
required by the FastAPI routes so that they remain thin wrappers. Methods in
this class normalise query parameters, enforce dimensionality constraints, and
delegate to :class:`~backend.app.repositories.search.SearchRepository` for
database access.
"""

from __future__ import annotations

import asyncpg

from ..config import settings
from ..models import FTSQuery, HybridQuery, SearchHit, SearchResponse, VectorQuery
from ..repositories.search import SearchRepository


class SearchApiService:
    """Encapsulates validation and repository coordination for search."""

    def __init__(
        self,
        conn: asyncpg.Connection,
        repository: SearchRepository | None = None,
    ) -> None:
        """Initialise the service with a database connection.

        Args:
            conn: Active asyncpg connection bound to the request context.
            repository: Optional repository instance for dependency injection.
                Primarily used for testing.
        """

        self._repo = repository or SearchRepository(conn)

    async def full_text_search(self, body: FTSQuery) -> SearchResponse:
        """Execute full-text search with safe pagination parameters.

        Args:
            body: Validated full-text search request.

        Returns:
            SearchResponse containing ranked lexical matches.
        """
        limit = max(1, min(body.limit, settings.PAGE_MAX))
        offset = max(0, body.offset)
        total, items = await self._repo.search_fts(
            dictionary=settings.FTS_DICTIONARY,
            query=body.q,
            translation=body.translation,
            limit=limit,
            offset=offset,
        )
        hits = [SearchHit(**item) for item in items]
        return SearchResponse(total=total, items=hits)

    async def vector_search(self, body: VectorQuery) -> SearchResponse:
        """Execute vector similarity search with dimensional validation.

        Args:
            body: Validated vector search request.

        Returns:
            SearchResponse with semantic similarity scores.

        Raises:
            ValueError: If ``embedding`` length does not match ``dim``.
        """
        if len(body.embedding) != body.dim:
            raise ValueError(
                f"embedding length {len(body.embedding)} does not match dim {body.dim}"
            )
        limit = max(1, min(body.top_k, settings.PAGE_MAX))
        rows = await self._repo.search_vector(
            embedding=body.embedding,
            model=body.model,
            dim=body.dim,
            translation=body.translation,
            limit=limit,
        )
        hits = [SearchHit(**row) for row in rows]
        return SearchResponse(total=len(hits), items=hits)

    async def hybrid_search(self, body: HybridQuery) -> SearchResponse:
        """Execute hybrid search with validation rules.

        Args:
            body: Validated hybrid search request.

        Returns:
            SearchResponse produced by Reciprocal Rank Fusion.

        Raises:
            ValueError: If both ``q`` and ``embedding`` are missing or if the
                embedding size is inconsistent.
        """
        if body.embedding is None and body.q is None:
            raise ValueError("Provide at least q or embedding")
        if body.embedding is not None and len(body.embedding) != body.dim:
            raise ValueError(
                f"embedding length {len(body.embedding)} does not match dim {body.dim}"
            )
        top_k = max(1, min(body.top_k, settings.PAGE_MAX))
        rows = await self._repo.search_hybrid(
            embedding=body.embedding,
            model=body.model,
            dim=body.dim,
            query=body.q,
            dictionary=settings.FTS_DICTIONARY,
            translation=body.translation,
            fts_k=body.fts_k,
            vector_k=body.vector_k,
            k_rrf=body.k_rrf,
            top_k=top_k,
        )
        hits = [SearchHit(**row) for row in rows]
        return SearchResponse(total=len(hits), items=hits)
