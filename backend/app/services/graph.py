"""Graph query helpers for Neo4j verse neighbourhood operations."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from neo4j import AsyncSession

from backend.etl.neo4j_client import _cv_key

from ..models import CanonicalVerse, GraphNeighborhood, Rendition


class GraphServiceError(Exception):
    """Base error for graph service operations."""


class InvalidVerseIdentifierError(GraphServiceError):
    """Raised when a verse identifier cannot be parsed."""


class VerseNeighborhoodNotFoundError(GraphServiceError):
    """Raised when a verse or canonical node does not exist."""


class GraphQueryService:
    """Service providing Neo4j graph neighbourhood queries."""

    CYPHER_BY_CVK: str = (
        "MATCH (cv:CV {cvk: $cvk})\n"
        "OPTIONAL MATCH (cv)<-[:RENDITION_OF]-(v:Verse)"
        "<-[:HAS_VERSE]-(ch:Chapter)<-[:HAS_CHAPTER]-(b:Book)"
        "<-[:HAS_BOOK]-(t:Translation)\n"
        "WITH cv, v, t\n"
        "ORDER BY t.code\n"
        "RETURN cv.cvk AS cvk,\n"
        "       cv.book_number AS book_number,\n"
        "       cv.chapter_number AS chapter_number,\n"
        "       cv.verse_number AS verse_number,\n"
        "       cv.suffix AS suffix,\n"
        "       collect(CASE WHEN v IS NULL THEN NULL ELSE {\n"
        "           verse_id: v.verse_id,\n"
        "           translation: t.code,\n"
        "           reference: v.reference,\n"
        "           text: v.text\n"
        "       } END) AS renditions"
    )

    CYPHER_BY_VERSE_ID: str = (
        "MATCH (v:Verse {verse_id: $verse_id})-[:RENDITION_OF]->(cv:CV)\n"
        "OPTIONAL MATCH (cv)<-[:RENDITION_OF]-(w:Verse)"
        "<-[:HAS_VERSE]-(ch:Chapter)<-[:HAS_CHAPTER]-(b:Book)"
        "<-[:HAS_BOOK]-(t:Translation)\n"
        "WITH cv, w, t\n"
        "ORDER BY t.code\n"
        "RETURN cv.cvk AS cvk,\n"
        "       cv.book_number AS book_number,\n"
        "       cv.chapter_number AS chapter_number,\n"
        "       cv.verse_number AS verse_number,\n"
        "       cv.suffix AS suffix,\n"
        "       collect(CASE WHEN w IS NULL THEN NULL ELSE {\n"
        "           verse_id: w.verse_id,\n"
        "           translation: t.code,\n"
        "           reference: w.reference,\n"
        "           text: w.text\n"
        "       } END) AS renditions"
    )

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def neighborhood_by_cvk(self, cvk: str) -> GraphNeighborhood:
        """Return the verse neighbourhood for a canonical verse key."""
        normalized_cvk = self._normalize_cvk(cvk)
        records = await self._run(self.CYPHER_BY_CVK, {"cvk": normalized_cvk})
        result = self._build_neighborhood(records)
        return result

    async def neighborhood_for_translation_verse(
        self, translation: str, verse_fragment: str
    ) -> GraphNeighborhood:
        """Return verse neighbourhood for a translation-specific verse."""
        verse_id, expected_cvk = self._compose_verse_id(translation, verse_fragment)
        records = await self._run(self.CYPHER_BY_VERSE_ID, {"verse_id": verse_id})
        result = self._build_neighborhood(records)
        if result.canonical.cvk != expected_cvk:
            raise VerseNeighborhoodNotFoundError(verse_id)
        if not any(r.translation.upper() == translation.upper() for r in result.renditions):
            raise VerseNeighborhoodNotFoundError(verse_id)
        return result

    async def _run(self, query: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        result = await self._session.run(query, **params)
        return await result.data()

    def _build_neighborhood(self, records: Iterable[dict[str, Any]]) -> GraphNeighborhood:
        first = next(iter(records), None)
        if not first:
            raise VerseNeighborhoodNotFoundError("neighborhood")

        try:
            canonical = CanonicalVerse(
                cvk=first["cvk"],
                book_number=int(first["book_number"]),
                chapter_number=int(first["chapter_number"]),
                verse_number=int(first["verse_number"]),
                suffix=(first.get("suffix") or ""),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise GraphServiceError("Malformed Neo4j response") from exc

        renditions_payload = [
            payload
            for payload in first.get("renditions", [])
            if payload and payload.get("verse_id")
        ]
        renditions = [
            Rendition(
                verse_id=payload["verse_id"],
                translation=payload["translation"],
                reference=payload.get("reference", ""),
                text=payload.get("text", ""),
            )
            for payload in renditions_payload
        ]
        renditions.sort(key=lambda rendition: rendition.translation)
        return GraphNeighborhood(canonical=canonical, renditions=renditions)

    def _normalize_cvk(self, cvk: str) -> str:
        parts = cvk.strip().split(":")
        if len(parts) < 3:
            raise InvalidVerseIdentifierError(cvk)
        suffix = parts[3] if len(parts) > 3 else ""
        try:
            book, chapter, verse = (int(parts[0]), int(parts[1]), int(parts[2]))
        except ValueError as exc:
            raise InvalidVerseIdentifierError(cvk) from exc
        return _cv_key(book, chapter, verse, suffix)

    def _compose_verse_id(self, translation: str, verse_fragment: str) -> tuple[str, str]:
        cleaned_translation = translation.strip()
        cleaned_fragment = verse_fragment.strip()
        if not cleaned_translation or not cleaned_fragment:
            raise InvalidVerseIdentifierError(verse_fragment)

        normalized_fragment = cleaned_fragment.replace(":", "_")
        if not normalized_fragment.upper().startswith(f"{cleaned_translation.upper()}_"):
            verse_id = f"{cleaned_translation}_{normalized_fragment}"
        else:
            verse_id = normalized_fragment

        parts = verse_id.split("_")
        if len(parts) < 4:
            raise InvalidVerseIdentifierError(verse_fragment)
        try:
            book = int(parts[1])
            chapter = int(parts[2])
            verse = int(parts[3])
        except ValueError as exc:
            raise InvalidVerseIdentifierError(verse_fragment) from exc
        suffix = parts[4] if len(parts) > 4 else ""
        cvk = _cv_key(book, chapter, verse, suffix)
        return verse_id, cvk
