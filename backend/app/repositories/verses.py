"""Verse data access layer."""

from __future__ import annotations

import asyncpg

from ..models import Book, Chapter, Translation, Verse, VerseLite


class VerseRepository:
    """Repository providing verse and metadata queries."""

    def __init__(self, conn: asyncpg.Connection) -> None:
        self._conn = conn

    async def get_by_id(self, verse_id: str) -> Verse | None:
        """Fetch a verse by its identifier."""
        row = await self._conn.fetchrow(
            """
            SELECT translation_code, book_number, chapter_number,
                   verse_number, suffix, verse_id, text
            FROM verse
            WHERE verse_id = $1
            """,
            verse_id,
        )
        return Verse(**dict(row)) if row else None

    async def list_chapter_verses(
        self,
        *,
        translation: str,
        book_number: int,
        chapter_number: int,
        limit: int,
        offset: int,
    ) -> list[VerseLite]:
        """List verses in a chapter with pagination."""
        rows = await self._conn.fetch(
            """
            SELECT verse_id, text
            FROM verse
            WHERE translation_code = $1
              AND book_number = $2
              AND chapter_number = $3
            ORDER BY verse_number, suffix
            LIMIT $4 OFFSET $5
            """,
            translation,
            book_number,
            chapter_number,
            limit,
            offset,
        )
        return [VerseLite(**dict(r)) for r in rows]

    async def list_translations(self) -> list[Translation]:
        """Return all translation metadata."""
        rows = await self._conn.fetch(
            """
            SELECT translation_code, language, format
            FROM translation
            ORDER BY translation_code
            """
        )
        return [Translation(**dict(r)) for r in rows]

    async def list_books(self, *, translation: str) -> list[Book]:
        """Return all books for a translation."""
        rows = await self._conn.fetch(
            """
            SELECT translation_code, book_number, name, testament
            FROM book
            WHERE translation_code = $1
            ORDER BY book_number
            """,
            translation,
        )
        return [Book(**dict(r)) for r in rows]

    async def list_chapters(
        self,
        *,
        translation: str,
        book_number: int,
    ) -> list[Chapter]:
        """Return all chapters for a book in a translation."""
        rows = await self._conn.fetch(
            """
            SELECT translation_code, book_number, chapter_number
            FROM chapter
            WHERE translation_code = $1
              AND book_number = $2
            ORDER BY chapter_number
            """,
            translation,
            book_number,
        )
        return [Chapter(**dict(r)) for r in rows]
