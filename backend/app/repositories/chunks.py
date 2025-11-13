"""Data access helpers for chunk embedding queries."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import asyncpg


class ChunkRepository:
    """Encapsulate SQL queries for chunk embeddings and context retrieval."""

    def __init__(self, conn: asyncpg.Connection):
        self.conn = conn

    @staticmethod
    def _build_filters(
        start_index: int,
        translation: str | None,
        book_number: int | None,
        testament: str | None,
        window_size: int | None,
    ) -> tuple[str, list[Any], int]:
        """Assemble a dynamic WHERE clause and positional parameters."""

        filters: list[str] = []
        params: list[Any] = []
        idx = start_index

        if translation:
            filters.append(f"ce.translation_code = ${idx}")
            params.append(translation)
            idx += 1

        if book_number is not None:
            filters.append(f"ce.book_number = ${idx}")
            params.append(book_number)
            idx += 1

        if testament:
            filters.append(f"b.testament = ${idx}")
            params.append(testament)
            idx += 1

        if window_size is not None:
            filters.append(f"ce.window_size = ${idx}")
            params.append(window_size)
            idx += 1

        where_clause = " AND ".join(filters) if filters else "TRUE"
        return where_clause, params, idx

    async def count_chunks(
        self,
        *,
        translation: str | None = None,
        book_number: int | None = None,
        testament: str | None = None,
        window_size: int | None = None,
    ) -> int:
        """Return the number of chunks matching the supplied filters."""

        where_clause, params, _ = self._build_filters(
            1, translation, book_number, testament, window_size
        )

        sql = f"""
        SELECT COUNT(*)
        FROM chunk_embedding ce
        JOIN book b ON ce.translation_code = b.translation_code
                   AND ce.book_number = b.book_number
        WHERE {where_clause};
        """

        row = await self.conn.fetchrow(sql, *params)
        return int(row["count"]) if row else 0

    async def semantic_search(
        self,
        *,
        embedding: Sequence[float],
        translation: str | None = None,
        book_number: int | None = None,
        testament: str | None = None,
        window_size: int | None = None,
        limit: int = 50,
        offset: int = 0,
        include_context: bool = False,
        context_verses: int = 2,
    ) -> list[asyncpg.Record]:
        """Execute a semantic similarity search across chunk embeddings."""

        where_clause, params, next_idx = self._build_filters(
            1, translation, book_number, testament, window_size
        )

        embed_param = f"${next_idx}"
        limit_param = f"${next_idx + 1}"
        offset_param = f"${next_idx + 2}"

        sql_context_select = ""
        sql_context_joins = ""
        final_params: list[Any] = list(params)

        final_params.extend([embedding, limit, offset])

        if include_context:
            context_param = f"${next_idx + 3}"
            sql_context_select = ", before_ctx.context_before, after_ctx.context_after"
            sql_context_joins = f"""
            LEFT JOIN LATERAL (
                SELECT verse_abs_index
                FROM verse
                WHERE translation_code = ce.translation_code
                  AND book_number = ce.book_number
                  AND chapter_number = ce.chapter_start
                  AND verse_number = ce.verse_start
                ORDER BY suffix
                LIMIT 1
            ) start_idx ON TRUE
            LEFT JOIN LATERAL (
                SELECT verse_abs_index
                FROM verse
                WHERE translation_code = ce.translation_code
                  AND book_number = ce.book_number
                  AND chapter_number = ce.chapter_end
                  AND verse_number = ce.verse_end
                ORDER BY suffix DESC
                LIMIT 1
            ) end_idx ON TRUE
            LEFT JOIN LATERAL (
                SELECT string_agg(v.text, ' ' ORDER BY v.verse_abs_index) AS context_before
                FROM verse v
                WHERE start_idx.verse_abs_index IS NOT NULL
                  AND v.translation_code = ce.translation_code
                  AND v.verse_abs_index BETWEEN GREATEST(start_idx.verse_abs_index - {context_param}, 0)
                                             AND start_idx.verse_abs_index - 1
            ) before_ctx ON TRUE
            LEFT JOIN LATERAL (
                SELECT string_agg(v.text, ' ' ORDER BY v.verse_abs_index) AS context_after
                FROM verse v
                WHERE end_idx.verse_abs_index IS NOT NULL
                  AND v.translation_code = ce.translation_code
                  AND v.verse_abs_index BETWEEN end_idx.verse_abs_index + 1
                                             AND end_idx.verse_abs_index + {context_param}
            ) after_ctx ON TRUE
            """
            final_params.append(context_verses)

        sql = f"""
        SELECT
            ce.chunk_id,
            ce.translation_code,
            ce.book_number,
            ce.chapter_start,
            ce.verse_start,
            ce.chapter_end,
            ce.verse_end,
            ce.text,
            ce.window_size,
            ce.stride,
            (1 - (ce.embedding <=> {embed_param}::vector))::numeric(6,4) AS score
            {sql_context_select}
        FROM chunk_embedding ce
        JOIN book b ON ce.translation_code = b.translation_code
                   AND ce.book_number = b.book_number
        {sql_context_joins}
        WHERE {where_clause}
        ORDER BY ce.embedding <=> {embed_param}::vector
        LIMIT {limit_param} OFFSET {offset_param};
        """

        return await self.conn.fetch(sql, *final_params)

    async def get_chunk(
        self,
        chunk_id: str,
        *,
        include_context: bool = False,
        context_verses: int = 2,
    ) -> asyncpg.Record | None:
        """Fetch a single chunk embedding record by its identifier."""

        params: list[Any] = [chunk_id]
        sql_context_select = ""
        sql_context_joins = ""

        if include_context:
            context_param = "$2"
            sql_context_select = ", before_ctx.context_before, after_ctx.context_after"
            sql_context_joins = f"""
            LEFT JOIN LATERAL (
                SELECT verse_abs_index
                FROM verse
                WHERE translation_code = ce.translation_code
                  AND book_number = ce.book_number
                  AND chapter_number = ce.chapter_start
                  AND verse_number = ce.verse_start
                ORDER BY suffix
                LIMIT 1
            ) start_idx ON TRUE
            LEFT JOIN LATERAL (
                SELECT verse_abs_index
                FROM verse
                WHERE translation_code = ce.translation_code
                  AND book_number = ce.book_number
                  AND chapter_number = ce.chapter_end
                  AND verse_number = ce.verse_end
                ORDER BY suffix DESC
                LIMIT 1
            ) end_idx ON TRUE
            LEFT JOIN LATERAL (
                SELECT string_agg(v.text, ' ' ORDER BY v.verse_abs_index) AS context_before
                FROM verse v
                WHERE start_idx.verse_abs_index IS NOT NULL
                  AND v.translation_code = ce.translation_code
                  AND v.verse_abs_index BETWEEN GREATEST(start_idx.verse_abs_index - {context_param}, 0)
                                             AND start_idx.verse_abs_index - 1
            ) before_ctx ON TRUE
            LEFT JOIN LATERAL (
                SELECT string_agg(v.text, ' ' ORDER BY v.verse_abs_index) AS context_after
                FROM verse v
                WHERE end_idx.verse_abs_index IS NOT NULL
                  AND v.translation_code = ce.translation_code
                  AND v.verse_abs_index BETWEEN end_idx.verse_abs_index + 1
                                             AND end_idx.verse_abs_index + {context_param}
            ) after_ctx ON TRUE
            """
            params.append(context_verses)

        sql = f"""
        SELECT
            ce.chunk_id,
            ce.translation_code,
            ce.book_number,
            ce.chapter_start,
            ce.verse_start,
            ce.chapter_end,
            ce.verse_end,
            ce.text,
            ce.window_size,
            ce.stride,
            1.0::numeric(6,4) AS score
            {sql_context_select}
        FROM chunk_embedding ce
        JOIN book b ON ce.translation_code = b.translation_code
                   AND ce.book_number = b.book_number
        {sql_context_joins}
        WHERE ce.chunk_id = $1
        LIMIT 1;
        """

        row = await self.conn.fetchrow(sql, *params)
        return row
