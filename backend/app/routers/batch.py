"""Batch retrieval router for high-volume verse operations."""

from __future__ import annotations

import asyncpg
from fastapi import APIRouter, Depends

from ..db.postgres_async import get_pg
from ..models import (
    BatchVerseRequest,
    BatchVerseResponse,
    EmbeddingLookupRequest,
    EmbeddingLookupResponse,
    TranslationComparisonRequest,
    TranslationComparisonResponse,
)
from ..services.batch import BatchService

router = APIRouter(prefix="/batch", tags=["batch"])


@router.post("/verses", response_model=BatchVerseResponse)
async def fetch_batch_verses(
    payload: BatchVerseRequest, conn: asyncpg.Connection = Depends(get_pg)
) -> BatchVerseResponse:
    """
    Retrieve multiple verses in a single request.

    The request body accepts up to 500 verse IDs (enforced via Pydantic validation).
    Duplicates are de-duplicated server-side while preserving the original order in
    the response. Missing verse IDs are reported separately so consumers can
    distinguish partial successes.

    Example:
        ```bash
        curl -X POST http://localhost:8000/v1/batch/verses \
          -H "Content-Type: application/json" \
          -d '{"verse_ids": ["NIV:43:3:16", "ESV:19:23:1"]}'
        ```
    """

    service = BatchService(conn)
    return await service.fetch_verses(payload)


@router.post("/translations/compare", response_model=TranslationComparisonResponse)
async def compare_translations(
    payload: TranslationComparisonRequest,
    conn: asyncpg.Connection = Depends(get_pg),
) -> TranslationComparisonResponse:
    """
    Fetch the same verse across multiple translations.

    Clients provide canonical verse references (book/chapter/verse/suffix) and a
    list of translation codes. Up to 200 references and 25 translations are
    accepted per request. The response includes every requested translation in
    input order, tagging missing verses in the `missing_translations` list.

    Example:
        ```bash
        curl -X POST http://localhost:8000/v1/batch/translations/compare \
          -H "Content-Type: application/json" \
          -d '{
                "references": [{"book_number": 43, "chapter_number": 3, "verse_number": 16}],
                "translations": ["NIV", "ESV", "NET"]
              }'
        ```
    """

    service = BatchService(conn)
    return await service.compare_translations(payload)


@router.post("/embeddings", response_model=EmbeddingLookupResponse)
async def lookup_embeddings(
    payload: EmbeddingLookupRequest, conn: asyncpg.Connection = Depends(get_pg)
) -> EmbeddingLookupResponse:
    """
    Retrieve cached embeddings for verses.

    Supports up to 200 verse IDs per call with an optional `model` filter to
    select a specific embedding model. Missing embeddings are surfaced in the
    response for observability.

    Example:
        ```bash
        curl -X POST http://localhost:8000/v1/batch/embeddings \
          -H "Content-Type: application/json" \
          -d '{"verse_ids": ["NIV:43:3:16"], "model": "embeddinggemma"}'
        ```
    """

    service = BatchService(conn)
    return await service.lookup_embeddings(payload)
