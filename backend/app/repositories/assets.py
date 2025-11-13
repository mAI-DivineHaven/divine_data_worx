"""Asset repository providing PostgreSQL data access helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import asyncpg

from ..models import Asset, AssetVerseLink


class AssetRepository:
    """Repository encapsulating SQL operations for asset tables."""

    def __init__(self, conn: asyncpg.Connection):
        self.conn = conn

    # ------------------------------------------------------------------
    # Asset CRUD
    # ------------------------------------------------------------------
    async def create(
        self,
        *,
        asset_id: str,
        media_type: str,
        title: str,
        description: str | None = None,
        text_payload: str | None = None,
        payload_json: dict[str, Any] | None = None,
        license: str | None = None,
        origin_url: str | None = None,
    ) -> Asset:
        row = await self.conn.fetchrow(
            """
            INSERT INTO asset (
                asset_id, media_type, title, description,
                text_payload, payload_json, license, origin_url
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING *
            """,
            asset_id,
            media_type,
            title,
            description,
            text_payload,
            payload_json,
            license,
            origin_url,
        )
        return self._row_to_asset(row)

    async def get_by_id(self, asset_id: str) -> Asset | None:
        row = await self.conn.fetchrow(
            "SELECT * FROM asset WHERE asset_id = $1",
            asset_id,
        )
        return self._row_to_asset(row) if row else None

    async def list_assets(
        self,
        *,
        limit: int,
        offset: int,
        media_type: str | None = None,
        search: str | None = None,
    ) -> tuple[int, list[Asset]]:
        filters: list[str] = []
        params: list[Any] = []

        if media_type:
            filters.append(f"media_type = ${len(params) + 1}")
            params.append(media_type)
        if search:
            filters.append(
                f"(title ILIKE ${len(params) + 1} OR description ILIKE ${len(params) + 1})"
            )
            params.append(f"%{search}%")

        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""

        count_sql = f"SELECT COUNT(*) FROM asset {where_clause}"
        total = int(await self.conn.fetchval(count_sql, *params) or 0)

        params_with_pagination = [*params, limit, offset]
        list_sql = (
            "SELECT * FROM asset "
            f"{where_clause} "
            "ORDER BY created_at DESC "
            f"LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}"
        )
        rows = await self.conn.fetch(list_sql, *params_with_pagination)
        items = [self._row_to_asset(row) for row in rows]
        return total, items

    async def update(
        self,
        asset_id: str,
        *,
        fields: dict[str, Any],
    ) -> Asset | None:
        if not fields:
            return await self.get_by_id(asset_id)

        assignments: list[str] = []
        values: list[Any] = [asset_id]
        for idx, (column, value) in enumerate(fields.items(), start=1):
            assignments.append(f"{column} = ${idx + 1}")
            values.append(value)

        sql = "UPDATE asset SET " + ", ".join(assignments) + " WHERE asset_id = $1 RETURNING *"
        row = await self.conn.fetchrow(sql, *values)
        return self._row_to_asset(row) if row else None

    async def delete(self, asset_id: str) -> int:
        result = await self.conn.execute(
            "DELETE FROM asset WHERE asset_id = $1",
            asset_id,
        )
        return self._parse_rowcount(result)

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------
    async def upsert_embedding(
        self,
        *,
        asset_id: str,
        embedding: Sequence[float],
        model: str,
        dim: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        await self.conn.execute(
            """
            INSERT INTO asset_embedding (
                asset_id, embedding, embedding_model, embedding_dim, metadata
            )
            VALUES ($1, $2::vector, $3, $4, $5)
            ON CONFLICT (asset_id) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                embedding_model = EXCLUDED.embedding_model,
                embedding_dim = EXCLUDED.embedding_dim,
                embedding_ts = now(),
                metadata = EXCLUDED.metadata
            """,
            asset_id,
            embedding,
            model,
            dim,
            metadata,
        )

    async def get_embedding_info(self, asset_id: str) -> dict[str, Any] | None:
        row = await self.conn.fetchrow(
            """
            SELECT asset_id, embedding_model, embedding_dim, embedding_ts, metadata
            FROM asset_embedding
            WHERE asset_id = $1
            """,
            asset_id,
        )
        return dict(row) if row else None

    async def delete_embedding(self, asset_id: str) -> int:
        result = await self.conn.execute(
            "DELETE FROM asset_embedding WHERE asset_id = $1",
            asset_id,
        )
        return self._parse_rowcount(result)

    async def search_by_embedding(
        self,
        *,
        embedding: Sequence[float],
        model: str,
        dim: int,
        limit: int,
    ) -> list[tuple[Asset, float]]:
        rows = await self.conn.fetch(
            """
            SELECT a.*, (1.0 - (ae.embedding <=> $1::vector))::float AS score
            FROM asset_embedding ae
            JOIN asset a ON a.asset_id = ae.asset_id
            WHERE ae.embedding_model = $2 AND ae.embedding_dim = $3
            ORDER BY ae.embedding <=> $1::vector ASC
            LIMIT $4
            """,
            embedding,
            model,
            dim,
            limit,
        )
        results: list[tuple[Asset, float]] = []
        for row in rows:
            record = dict(row)
            score = float(record.pop("score"))
            results.append((Asset(**record), score))
        return results

    # ------------------------------------------------------------------
    # Asset ↔ Verse linking
    # ------------------------------------------------------------------
    async def create_links(
        self,
        *,
        asset_id: str,
        verse_ids: Sequence[str],
        relation: str,
        chunk_id: str | None = None,
    ) -> int:
        rows = await self.conn.fetch(
            """
            INSERT INTO asset_link (asset_id, verse_id, chunk_id, relation)
            SELECT $1, verse_id, $4, $3
            FROM UNNEST($2::text[]) AS verse_id
            ON CONFLICT DO NOTHING
            RETURNING verse_id
            """,
            asset_id,
            list(verse_ids),
            relation,
            chunk_id,
        )
        return len(rows)

    async def delete_links(
        self,
        *,
        asset_id: str,
        verse_ids: Sequence[str] | None = None,
    ) -> int:
        if verse_ids:
            result = await self.conn.execute(
                "DELETE FROM asset_link WHERE asset_id = $1 AND verse_id = ANY($2::text[])",
                asset_id,
                list(verse_ids),
            )
        else:
            result = await self.conn.execute(
                "DELETE FROM asset_link WHERE asset_id = $1",
                asset_id,
            )
        return self._parse_rowcount(result)

    async def fetch_links(self, asset_id: str) -> list[AssetVerseLink]:
        rows = await self.conn.fetch(
            """
            SELECT
                al.verse_id,
                al.relation,
                al.chunk_id,
                v.translation_code,
                v.book_number,
                v.chapter_number,
                v.verse_number,
                v.suffix,
                v.text
            FROM asset_link al
            JOIN verse v ON v.verse_id = al.verse_id
            WHERE al.asset_id = $1
            ORDER BY v.translation_code, v.book_number, v.chapter_number, v.verse_number, v.suffix
            """,
            asset_id,
        )
        links: list[AssetVerseLink] = []
        for row in rows:
            record = dict(row)
            reference = (
                f"{record['translation_code']} "
                f"{record['book_number']}:{record['chapter_number']}:{record['verse_number']}{record['suffix']}"
            )
            links.append(
                AssetVerseLink(
                    verse_id=record["verse_id"],
                    relation=record["relation"],
                    chunk_id=record["chunk_id"],
                    translation_code=record["translation_code"],
                    book_number=record["book_number"],
                    chapter_number=record["chapter_number"],
                    verse_number=record["verse_number"],
                    suffix=record["suffix"],
                    text=record["text"],
                    reference=reference,
                )
            )
        return links

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_rowcount(command_tag: str) -> int:
        try:
            return int(command_tag.split(" ")[1])
        except (IndexError, ValueError):
            return 0

    @staticmethod
    def _row_to_asset(row: asyncpg.Record) -> Asset:
        return Asset(**dict(row))
