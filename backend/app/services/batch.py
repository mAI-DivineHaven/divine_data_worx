"""Service layer orchestrating batch verse operations."""

from __future__ import annotations

import asyncpg

from ..models import (
    BatchVerseRequest,
    BatchVerseResponse,
    EmbeddingLookupRequest,
    EmbeddingLookupResponse,
    EmbeddingVector,
    TranslationComparisonItem,
    TranslationComparisonRequest,
    TranslationComparisonResponse,
    TranslationVerseEntry,
    Verse,
)
from ..repositories.batch import BatchRepository


class BatchService:
    """Business logic for high-volume verse retrieval and comparisons."""

    def __init__(self, conn: asyncpg.Connection) -> None:
        self.repo = BatchRepository(conn)

    async def fetch_verses(self, request: BatchVerseRequest) -> BatchVerseResponse:
        """Fetch multiple verses by their identifiers with a single query."""

        rows = await self.repo.fetch_verses_by_ids(request.verse_ids)
        unique_requested = list(dict.fromkeys(request.verse_ids))
        found_ids = {row["verse_id"] for row in rows}
        verses = [Verse(**dict(row)) for row in rows]
        missing = [vid for vid in unique_requested if vid not in found_ids]
        return BatchVerseResponse(verses=verses, missing_ids=missing)

    async def compare_translations(
        self, request: TranslationComparisonRequest
    ) -> TranslationComparisonResponse:
        """Compare verse texts across translations for canonical references."""

        rows = await self.repo.fetch_translation_comparisons(
            request.references, request.translations
        )
        translation_order = list(dict.fromkeys(request.translations))

        grouped: dict[int, dict[str, TranslationVerseEntry]] = {}
        for row in rows:
            idx = int(row["idx"])
            translation_code = row["translation_code"]
            grouped.setdefault(idx, {})[translation_code] = TranslationVerseEntry(
                translation_code=translation_code,
                verse_id=row["verse_id"],
                text=row["text"],
            )

        items: list[TranslationComparisonItem] = []
        for idx, reference in enumerate(request.references, start=1):
            translation_map = grouped.get(idx, {})
            translations: list[TranslationVerseEntry] = []
            missing_codes: list[str] = []

            for code in translation_order:
                entry = translation_map.get(code)
                if entry is None:
                    entry = TranslationVerseEntry(translation_code=code)
                    missing_codes.append(code)
                elif entry.verse_id is None:
                    missing_codes.append(code)
                translations.append(entry)

            items.append(
                TranslationComparisonItem(
                    reference=reference,
                    translations=translations,
                    missing_translations=missing_codes,
                )
            )

        return TranslationComparisonResponse(items=items)

    async def lookup_embeddings(self, request: EmbeddingLookupRequest) -> EmbeddingLookupResponse:
        """Return stored vector embeddings for a batch of verse identifiers."""

        rows = await self.repo.fetch_embeddings_by_ids(request.verse_ids, request.model)
        unique_requested = list(dict.fromkeys(request.verse_ids))
        found_ids = {row["verse_id"] for row in rows}

        results = [
            EmbeddingVector(
                verse_id=row["verse_id"],
                embedding=list(row["embedding"] or []),
                embedding_model=row["embedding_model"],
                embedding_dim=row["embedding_dim"],
            )
            for row in rows
        ]
        missing = [vid for vid in unique_requested if vid not in found_ids]

        return EmbeddingLookupResponse(results=results, missing_ids=missing)
