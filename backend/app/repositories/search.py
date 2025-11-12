"""Search query repository."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import asyncpg


class SearchRepository:
    """Repository providing SQL access for search endpoints."""

    def __init__(self, conn: asyncpg.Connection) -> None:
        self._conn = conn

    async def search_fts(
        self,
        *,
        dictionary: str,
        query: str,
        translation: str | None,
        limit: int,
        offset: int,
    ) -> tuple[int, list[dict[str, Any]]]:
        """Execute full-text search query."""
        where_extra = "AND v.translation_code = $3" if translation else ""
        param_count = 3 if translation else 2
        sql = f"""
        WITH ranked AS (
          SELECT v.verse_id, v.text,
                 ts_rank_cd(to_tsvector($1, v.text), plainto_tsquery($1, $2)) AS rank
          FROM verse v
          WHERE to_tsvector($1, v.text) @@ plainto_tsquery($1, $2)
          {where_extra}
          ORDER BY rank DESC
          LIMIT ${param_count + 1} OFFSET ${param_count + 2}
        )
        SELECT
          (SELECT COUNT(*) FROM verse v
            WHERE to_tsvector($1, v.text) @@ plainto_tsquery($1, $2)
            {where_extra}
          ) AS total,
          COALESCE(
            json_agg(
              json_build_object('verse_id', verse_id, 'text', text, 'score', rank)
              ORDER BY rank DESC
            ),
            '[]'::json
          ) AS items
        FROM ranked;
        """
        args: list[Any] = [dictionary, query]
        if translation:
            args.append(translation)
        args.extend([limit, offset])
        row = await self._conn.fetchrow(sql, *args)
        if not row:
            return 0, []
        total = int(row["total"] or 0)
        items: Sequence[Any] = row["items"] or []
        return total, list(items)

    async def search_vector(
        self,
        *,
        embedding: Sequence[float],
        model: str,
        dim: int,
        translation: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Execute vector similarity search."""
        where_parts = ["e.embedding_model = $1", "e.embedding_dim = $2"]
        params: list[Any] = [model, dim]
        if translation:
            where_parts.append(f"v.translation_code = ${len(params) + 1}")
            params.append(translation)
        where_clause = " AND ".join(where_parts)
        vec_param = f"${len(params) + 1}"
        limit_param = f"${len(params) + 2}"
        params.extend([list(embedding), limit])
        sql = f"""
        SELECT e.verse_id, v.text,
               (1.0 - (e.embedding <=> {vec_param}::vector))::numeric(6,4) AS score
        FROM verse_embedding e
        JOIN verse v ON v.verse_id = e.verse_id
        WHERE {where_clause}
        ORDER BY e.embedding <=> {vec_param}::vector ASC
        LIMIT {limit_param};
        """
        rows = await self._conn.fetch(sql, *params)
        return [
            {
                "verse_id": r["verse_id"],
                "text": r["text"],
                "score": float(r["score"]),
            }
            for r in rows
        ]

    async def search_hybrid(
        self,
        *,
        embedding: Sequence[float] | None,
        model: str,
        dim: int,
        query: str | None,
        dictionary: str,
        translation: str | None,
        fts_k: int,
        vector_k: int,
        k_rrf: int,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Execute hybrid Reciprocal Rank Fusion search."""
        params: list[Any] = []
        fts_cte = "SELECT NULL::text AS verse_id, NULL::bigint AS rnk WHERE FALSE"
        if query:
            fts_filter = f"AND v.translation_code = ${len(params) + 3}" if translation else ""
            param_dict = f"${len(params) + 1}"
            param_q1 = f"${len(params) + 2}"
            param_q2 = f"${len(params) + 2}"
            fts_limit = f"${len(params) + (4 if translation else 3)}"
            fts_cte = f"""
            SELECT v.verse_id,
                   ROW_NUMBER() OVER (
                     ORDER BY ts_rank_cd(
                       to_tsvector({param_dict}, v.text),
                       plainto_tsquery({param_dict}, {param_q1})
                     ) DESC
                   ) AS rnk
            FROM verse v
            WHERE to_tsvector({param_dict}, v.text) @@ plainto_tsquery({param_dict}, {param_q2})
            {fts_filter}
            LIMIT {fts_limit}
            """
            params.extend([dictionary, query])
            if translation:
                params.append(translation)
            params.append(fts_k)
        vec_cte = "SELECT NULL::text AS verse_id, NULL::bigint AS rnk WHERE FALSE"
        if embedding is not None:
            base = len(params)
            vec_param = f"${base + 1}"
            model_param = f"${base + 2}"
            dim_param = f"${base + 3}"
            trans_filter = f"AND v.translation_code = ${base + 4}" if translation else ""
            vec_limit = f"${base + (5 if translation else 4)}"
            vec_cte = f"""
            SELECT e.verse_id,
                   ROW_NUMBER() OVER (
                     ORDER BY e.embedding <=> {vec_param}::vector ASC
                   ) AS rnk
            FROM verse_embedding e
            JOIN verse v ON v.verse_id = e.verse_id
            WHERE e.embedding_model = {model_param}
              AND e.embedding_dim = {dim_param}
            {trans_filter}
            LIMIT {vec_limit}
            """
            params.extend([list(embedding), model, dim])
            if translation:
                params.append(translation)
            params.append(vector_k)
        k_rrf1 = f"${len(params) + 1}"
        k_rrf2 = f"${len(params) + 2}"
        final_limit = f"${len(params) + 3}"
        params.extend([k_rrf, k_rrf, top_k])
        sql = f"""
        WITH fts AS (
          {fts_cte}
        ),
        vec AS (
          {vec_cte}
        ),
        unioned AS (
          SELECT verse_id, 1.0 / ({k_rrf1} + rnk) AS score FROM fts WHERE verse_id IS NOT NULL
          UNION ALL
          SELECT verse_id, 1.0 / ({k_rrf2} + rnk) AS score FROM vec WHERE verse_id IS NOT NULL
        ),
        agg AS (
          SELECT verse_id, SUM(score)::numeric(6,4) AS score
          FROM unioned
          GROUP BY verse_id
        )
        SELECT a.verse_id, v.text, a.score
        FROM agg a
        JOIN verse v ON v.verse_id = a.verse_id
        ORDER BY a.score DESC
        LIMIT {final_limit};
        """
        rows = await self._conn.fetch(sql, *params)
        return [
            {
                "verse_id": r["verse_id"],
                "text": r["text"],
                "score": float(r["score"]),
            }
            for r in rows
        ]
