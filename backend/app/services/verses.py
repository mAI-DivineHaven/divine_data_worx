"""Business logic for verse retrieval."""

from __future__ import annotations

import asyncpg

from ..models import Book, Chapter, Translation, Verse, VerseLite
from ..repositories.verses import VerseRepository


class VerseNotFoundError(LookupError):
    """Raised when a verse cannot be found."""


class VerseService:
    """Service layer for verse and metadata operations."""

    def __init__(
        self,
        conn: asyncpg.Connection,
        repository: VerseRepository | None = None,
    ) -> None:
        self._repo = repository or VerseRepository(conn)

    async def get_verse(self, verse_id: str) -> Verse:
        """Return a verse by ID, validating input."""
        if not verse_id:
            raise ValueError("verse_id must be provided")
        verse = await self._repo.get_by_id(verse_id)
        if verse is None:
            raise VerseNotFoundError(verse_id)
        return verse

    async def list_verses(
        self,
        *,
        translation: str,
        book_number: int,
        chapter_number: int,
        limit: int,
        offset: int,
    ) -> list[VerseLite]:
        """Return verses for a chapter with bounds checking."""
        if limit <= 0:
            raise ValueError("limit must be positive")
        if offset < 0:
            raise ValueError("offset must be non-negative")
        return await self._repo.list_chapter_verses(
            translation=translation,
            book_number=book_number,
            chapter_number=chapter_number,
            limit=limit,
            offset=offset,
        )

    async def list_translations(self) -> list[Translation]:
        """Return translation metadata."""
        return await self._repo.list_translations()

    async def list_books(self, translation: str) -> list[Book]:
        """Return books for translation with validation."""
        if not translation:
            raise ValueError("translation is required")
        return await self._repo.list_books(translation=translation)

    async def list_chapters(self, translation: str, book_number: int) -> list[Chapter]:
        """Return chapters for book with validation."""
        if not translation:
            raise ValueError("translation is required")
        if book_number <= 0:
            raise ValueError("book_number must be positive")
        return await self._repo.list_chapters(
            translation=translation,
            book_number=book_number,
        )
