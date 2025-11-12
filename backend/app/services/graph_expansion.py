"""Graph retrieval helpers for expanding search results via Neo4j."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from neo4j import AsyncResult, AsyncSession

from ..models import GraphExpansion, Rendition


class GraphExpansionService:
    """Service responsible for fetching parallel verse renditions."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def parallels_for(
        self, verse_ids: Sequence[str], *, limit: int | None = None
    ) -> dict[str, GraphExpansion]:
        """Return graph expansions for a collection of verse identifiers."""

        unique_ids = _dedupe_preserve_order(verse_ids)
        if not unique_ids:
            return {}

        cypher = """
        UNWIND $verse_ids AS vid
        MATCH (v:Verse {verse_id: vid})-[:RENDITION_OF]->(cv:CV)
        OPTIONAL MATCH (cv)<-[:RENDITION_OF]-(w:Verse)
        OPTIONAL MATCH (w)<-[:HAS_VERSE]-(ch:Chapter)<-[:HAS_CHAPTER]-(b:Book)<-[:HAS_BOOK]-(t:Translation)
        WITH vid AS source,
             cv.cvk AS cvk,
             w.verse_id AS verse_id,
             COALESCE(t.code, split(verse_id, "_")[0]) AS translation,
             w.reference AS reference,
             w.text AS text
        WHERE verse_id IS NOT NULL
        WITH source,
             cvk,
             verse_id,
             translation,
             reference,
             text
        ORDER BY translation, verse_id
        RETURN source,
               cvk,
               collect({
                 verse_id: verse_id,
                 translation: translation,
                 reference: reference,
                 text: text
               }) AS renditions
        """
        result: AsyncResult = await self._session.run(cypher, verse_ids=unique_ids)
        records = await result.data()

        expansions: dict[str, GraphExpansion] = {}
        for record in records:
            renditions_raw: list[dict] = record.get("renditions") or []
            if limit is not None and limit >= 0:
                renditions_raw = renditions_raw[:limit]
            renditions = [
                Rendition(
                    verse_id=item["verse_id"],
                    translation=item["translation"],
                    reference=item.get("reference", ""),
                    text=item.get("text", ""),
                )
                for item in renditions_raw
                if item.get("verse_id")
            ]
            expansions[record["source"]] = GraphExpansion(
                verse_id=record["source"],
                cvk=record.get("cvk"),
                renditions=renditions,
            )
        return expansions


def _dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered
