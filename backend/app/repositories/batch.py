"""Repository layer for batch-oriented verse queries."""

from __future__ import annotations

from collections.abc import Sequence

import asyncpg

from ..models import CanonicalVerseRef


class BatchRepository:
    """Data access methods for bulk verse, translation, and embedding retrieval."""

    def __init__(self, conn: asyncpg.Connection) -> None:
        self.conn = conn

    async def fetch_verses_by_ids(self, verse_ids: Sequence[str]) -> list[asyncpg.Record]:
        """Return verse rows for the provided verse IDs, preserving input order."""

        if not verse_ids:
            return []

        verse_ids_list = list(dict.fromkeys(verse_ids))
        rows = await self.conn.fetch(
            """
            SELECT
                translation_code,
                book_number,
                chapter_number,
                verse_number,
                suffix,
                verse_id,
                text
            FROM verse
            WHERE verse_id = ANY($1::text[])
            ORDER BY array_position($1::text[], verse_id)
            """,
            verse_ids_list,
        )
        return rows

    async def fetch_translation_comparisons(
        self,
        references: Sequence[CanonicalVerseRef],
        translations: Sequence[str],
    ) -> list[asyncpg.Record]:
        """Return translation comparison rows for the supplied references."""

        if not references or not translations:
            return []

        books = [ref.book_number for ref in references]
        chapters = [ref.chapter_number for ref in references]
        verses = [ref.verse_number for ref in references]
        suffixes = [ref.suffix or "" for ref in references]
        translation_list = list(dict.fromkeys(translations))

        rows = await self.conn.fetch(
            """
            WITH refs AS (
                SELECT
                    ref.idx,
                    ref.book_number,
                    ref.chapter_number,
                    ref.verse_number,
                    ref.suffix
                FROM unnest($1::int[], $2::int[], $3::int[], $4::text[]) WITH ORDINALITY AS ref(
                    book_number,
                    chapter_number,
                    verse_number,
                    suffix,
                    idx
                )
            ),
            requested_translations AS (
                SELECT unnest($5::text[]) AS translation_code
            )
            SELECT
                refs.idx,
                refs.book_number,
                refs.chapter_number,
                refs.verse_number,
                refs.suffix,
                requested_translations.translation_code,
                verse.verse_id,
                verse.text
            FROM refs
            CROSS JOIN requested_translations
            LEFT JOIN verse
                ON verse.translation_code = requested_translations.translation_code
               AND verse.book_number = refs.book_number
               AND verse.chapter_number = refs.chapter_number
               AND verse.verse_number = refs.verse_number
               AND verse.suffix = refs.suffix
            ORDER BY refs.idx, requested_translations.translation_code
            """,
            books,
            chapters,
            verses,
            suffixes,
            translation_list,
        )
        return rows

    async def fetch_embeddings_by_ids(
        self, verse_ids: Sequence[str], model: str | None
    ) -> list[asyncpg.Record]:
        """Return embedding rows for the specified verse IDs."""

        if not verse_ids:
            return []

        verse_ids_list = list(dict.fromkeys(verse_ids))

        if model:
            rows = await self.conn.fetch(
                """
                SELECT verse_id, embedding, embedding_model, embedding_dim
                FROM verse_embedding
                WHERE verse_id = ANY($1::text[])
                  AND embedding_model = $2
                ORDER BY array_position($1::text[], verse_id)
                """,
                verse_ids_list,
                model,
            )
        else:
            rows = await self.conn.fetch(
                """
                SELECT verse_id, embedding, embedding_model, embedding_dim
                FROM verse_embedding
                WHERE verse_id = ANY($1::text[])
                ORDER BY array_position($1::text[], verse_id)
                """,
                verse_ids_list,
            )
        return rows
