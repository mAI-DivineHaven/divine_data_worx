"""
Verse and metadata retrieval router.

Provides REST endpoints for accessing biblical text and structural metadata:
- Individual verse retrieval by ID
- Chapter verse listing with pagination
- Translation metadata listing
- Book metadata per translation
- Chapter listing per book

All endpoints use async asyncpg connections for high performance.
Response models are defined in models.py for type safety and validation.

Example Usage:
    ```bash
    # Get specific verse
    curl http://localhost:8000/v1/verses/NIV_1_1_1_

    # List verses in Genesis 1 (NIV)
    curl "http://localhost:8000/v1/verses?translation=NIV&book_number=1&chapter_number=1"

    # List all translations
    curl http://localhost:8000/v1/verses/translations

    # List books in ESV
    curl "http://localhost:8000/v1/verses/books?translation=ESV"
    ```
"""

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Query

from ..db.postgres_async import get_pg
from ..models import Book, Chapter, Translation, Verse, VerseLite
from ..services.verses import VerseNotFoundError, VerseService

router = APIRouter(prefix="/verses", tags=["verses"])


def get_verse_service(conn: asyncpg.Connection = Depends(get_pg)) -> VerseService:
    """Dependency provider for VerseService."""
    return VerseService(conn)


# NOTE: Specific routes MUST be defined BEFORE the dynamic /{verse_id} route
# to prevent FastAPI from matching "translations", "books", etc. as verse IDs


@router.get("/translations", response_model=list[Translation])
async def list_translations(
    service: VerseService = Depends(get_verse_service),
) -> list[Translation]:
    """
    List all available Bible translations.

    Returns metadata for all translations in the database, ordered alphabetically
    by translation code.

    Args:
        service: VerseService dependency (injected)

    Returns:
        List of Translation objects with code, language, and format

    Example:
        ```bash
        curl http://localhost:8000/v1/verses/translations
        ```

        Response:
        ```json
        [
          {
            "translation_code": "ESV",
            "language": "en",
            "format": "json"
          },
          {
            "translation_code": "NIV",
            "language": "en",
            "format": "json"
          }
        ]
        ```
    """
    translations = await service.list_translations()
    return translations


@router.get("/books", response_model=list[Book])
async def list_books(
    translation: str = Query(..., description="Translation code (e.g., NIV)"),
    service: VerseService = Depends(get_verse_service),
) -> list[Book]:
    """
    List all books in a specific translation.

    Returns book metadata for a given translation in canonical order.
    Book names may vary across translations (e.g., "Psalms" vs "Psalm").

    Args:
        translation: Translation code (required)
        service: VerseService dependency (injected)

    Returns:
        List of Book objects ordered by book_number

    Example:
        ```bash
        curl "http://localhost:8000/v1/verses/books?translation=NIV"
        ```

        Response:
        ```json
        [
          {
            "translation_code": "NIV",
            "book_number": 1,
            "name": "Genesis",
            "testament": "Old"
          },
          {
            "translation_code": "NIV",
            "book_number": 2,
            "name": "Exodus",
            "testament": "Old"
          }
        ]
        ```
    """
    try:
        books = await service.list_books(translation)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return books


@router.get("/chapters", response_model=list[Chapter])
async def list_chapters(
    translation: str = Query(..., description="Translation code"),
    book_number: int = Query(..., description="Book number"),
    service: VerseService = Depends(get_verse_service),
) -> list[Chapter]:
    """
    List all chapters in a specific book.

    Returns chapter metadata for a given translation and book in sequential order.

    Args:
        translation: Translation code (required)
        book_number: Book number (required)
        service: VerseService dependency (injected)

    Returns:
        List of Chapter objects ordered by chapter_number

    Example:
        ```bash
        # Get all chapters in Genesis (NIV)
        curl "http://localhost:8000/v1/verses/chapters?translation=NIV&book_number=1"
        ```

        Response:
        ```json
        [
          {
            "translation_code": "NIV",
            "book_number": 1,
            "chapter_number": 1
          },
          {
            "translation_code": "NIV",
            "book_number": 1,
            "chapter_number": 2
          }
        ]
        ```
    """
    try:
        chapters = await service.list_chapters(translation, book_number)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return chapters


@router.get("/{verse_id}", response_model=Verse)
async def get_verse(
    verse_id: str,
    service: VerseService = Depends(get_verse_service),
) -> Verse:
    """
    Retrieve a single verse by its unique identifier.

    The verse_id follows the format: {translation}_{book}_{chapter}_{verse}_{suffix}
    Example: "NIV_1_1_1_" (Genesis 1:1 in NIV, no suffix)

    Args:
        verse_id: Unique verse identifier
        service: VerseService dependency (injected)

    Returns:
        Verse object with full metadata

    Raises:
        HTTPException: 404 if verse not found

    Example:
        ```bash
        curl http://localhost:8000/v1/verses/NIV_43_3_16_
        ```

        Response:
        ```json
        {
          "verse_id": "NIV_43_3_16_",
          "translation_code": "NIV",
          "book_number": 43,
          "chapter_number": 3,
          "verse_number": 16,
          "suffix": "",
          "text": "For God so loved the world..."
        }
        ```
    """
    try:
        verse = await service.get_verse(verse_id)
    except VerseNotFoundError:
        raise HTTPException(status_code=404, detail="verse not found") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return verse


@router.get("", response_model=list[VerseLite])
async def list_verses(
    translation: str = Query(..., description="Translation code (e.g., NIV, ESV)"),
    book_number: int = Query(
        ..., ge=1, le=200, description="Book number (1-66 for Protestant canon)"
    ),
    chapter_number: int = Query(..., ge=1, description="Chapter number"),
    limit: int = Query(100, ge=1, le=500, description="Maximum verses to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    service: VerseService = Depends(get_verse_service),
) -> list[VerseLite]:
    """
    List verses in a specific chapter with pagination.

    Retrieves verses from a specific translation, book, and chapter in canonical order.
    Returns lightweight VerseLite objects (verse_id and text only) for efficient transfer.

    Args:
        translation: Translation code (required)
        book_number: Book number 1-200 (1-66 for Protestant canon)
        chapter_number: Chapter number (>=1)
        limit: Max results per page (1-500, default 100)
        offset: Pagination offset (default 0)
        service: VerseService dependency (injected)

    Returns:
        List of VerseLite objects ordered by verse_number, suffix

    Example:
        ```bash
        # Get first 10 verses of John 3 (NIV)
        curl "http://localhost:8000/v1/verses?translation=NIV&book_number=43&chapter_number=3&limit=10"
        ```

        Response:
        ```json
        [
          {"verse_id": "NIV_43_3_1_", "text": "Now there was a Pharisee..."},
          {"verse_id": "NIV_43_3_2_", "text": "He came to Jesus at night..."}
        ]
        ```
    """
    try:
        verses = await service.list_verses(
            translation=translation,
            book_number=book_number,
            chapter_number=chapter_number,
            limit=limit,
            offset=offset,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return verses
