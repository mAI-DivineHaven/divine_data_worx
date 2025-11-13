"""Chunk retrieval and semantic search endpoints."""

from __future__ import annotations

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Query

from ..db.postgres_async import get_pg
from ..models import ChunkHit, ChunkSearchQuery, ChunkSearchResponse
from ..services.chunks import ChunkService

router = APIRouter(prefix="/chunks", tags=["chunks"])


@router.post("/search", response_model=ChunkSearchResponse)
async def search_chunks(
    query: ChunkSearchQuery, conn: asyncpg.Connection = Depends(get_pg)
) -> ChunkSearchResponse:
    """Perform semantic chunk search with optional verse context."""

    """
    Perform semantic similarity search across chunk embeddings.

    Clips the requested `top_k` to the configured `settings.PAGE_MAX` to enforce
    consistent pagination limits.

    Example:
        ```bash
        curl -X POST "http://localhost:8000/v1/chunks/search" \\
          -H "Content-Type: application/json" \\
          -d '{"embedding": [0.01] * 768, "dim": 768, "translation": "NIV", "top_k": 10}'
        ```
    """

    service = ChunkService(conn)
    return await service.search(query)


@router.get("/{chunk_id}", response_model=ChunkHit)
async def get_chunk(
    chunk_id: str,
    include_context: bool = Query(False, description="Include verse context around the chunk"),
    conn: asyncpg.Connection = Depends(get_pg),
) -> ChunkHit:
    """Retrieve a single chunk embedding record by ID."""

    """
    Fetch a chunk embedding and expose its verse range metadata.

    Example:
        ```bash
        curl "http://localhost:8000/v1/chunks/NIV_43_3_16_chunk?include_context=true"
        ```
    """

    service = ChunkService(conn)
    chunk = await service.get_chunk(chunk_id, include_context=include_context)
    if not chunk:
        raise HTTPException(status_code=404, detail="chunk not found")
    return chunk
