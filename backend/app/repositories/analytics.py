"""Analytics repository for aggregating search telemetry from PostgreSQL."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import asyncpg


class AnalyticsRepository:
    """Data access layer for analytics metrics sourced from `search_log`.

    The repository exposes aggregation helpers that are used by the
    :class:`~backend.app.services.analytics.AnalyticsService` to compute
    higher-level metrics. All queries rely on PostgreSQL/TimescaleDB features
    and are fully parameterised to prevent SQL injection.
    """

    def __init__(self, conn: asyncpg.Connection) -> None:
        """Initialise repository with an active asyncpg connection."""

        self.conn = conn

    async def fetch_query_summary(self, start: datetime, end: datetime) -> asyncpg.Record | None:
        """Return aggregate counts for queries within the window."""

        sql = """
            SELECT
                COUNT(*) AS total_queries,
                COUNT(DISTINCT user_id) FILTER (WHERE user_id IS NOT NULL) AS unique_users,
                AVG(latency_ms) AS avg_latency_ms
            FROM search_log
            WHERE ts >= $1 AND ts <= $2
        """
        return await self.conn.fetchrow(sql, start, end)

    async def fetch_mode_breakdown(
        self, start: datetime, end: datetime
    ) -> Sequence[asyncpg.Record]:
        """Return query counts grouped by search mode."""

        sql = """
            SELECT mode, COUNT(*) AS count
            FROM search_log
            WHERE ts >= $1 AND ts <= $2
            GROUP BY mode
            ORDER BY COUNT(*) DESC
        """
        return await self.conn.fetch(sql, start, end)

    async def fetch_top_queries(
        self, start: datetime, end: datetime, limit: int
    ) -> Sequence[asyncpg.Record]:
        """Return the most frequently executed queries within the window."""

        sql = """
            SELECT query, COUNT(*) AS count, MAX(ts) AS last_seen
            FROM search_log
            WHERE ts >= $1 AND ts <= $2
            GROUP BY query
            ORDER BY COUNT(*) DESC, MAX(ts) DESC
            LIMIT $3
        """
        return await self.conn.fetch(sql, start, end, limit)

    async def fetch_query_trend(
        self, start: datetime, end: datetime, interval: str
    ) -> Sequence[asyncpg.Record]:
        """Return time-series buckets for query volume trends."""

        sql = """
            SELECT date_trunc($3, ts) AS bucket_start, COUNT(*) AS count
            FROM search_log
            WHERE ts >= $1 AND ts <= $2
            GROUP BY bucket_start
            ORDER BY bucket_start
        """
        return await self.conn.fetch(sql, start, end, interval)

    async def fetch_translation_usage(
        self, start: datetime, end: datetime, limit: int
    ) -> Sequence[asyncpg.Record]:
        """Return query counts grouped by translation code."""

        sql = """
            SELECT translation_code, COUNT(*) AS count
            FROM search_log
            WHERE ts >= $1 AND ts <= $2
            GROUP BY translation_code
            ORDER BY COUNT(*) DESC
            LIMIT $3
        """
        return await self.conn.fetch(sql, start, end, limit)

    async def fetch_book_usage(
        self, start: datetime, end: datetime, limit: int
    ) -> Sequence[asyncpg.Record]:
        """Return book usage derived from top search results."""

        sql = """
            WITH first_hits AS (
                SELECT
                    split_part(hit.verse_id, '_', 2)::INT AS book_number
                FROM search_log sl
                CROSS JOIN LATERAL (
                    SELECT elem->>'verse_id' AS verse_id
                    FROM jsonb_array_elements(sl.results) AS elem
                    WHERE elem ? 'verse_id'
                    LIMIT 1
                ) AS hit
                WHERE sl.ts >= $1
                  AND sl.ts <= $2
                  AND sl.results IS NOT NULL
                  AND jsonb_typeof(sl.results) = 'array'
            ), counts AS (
                SELECT book_number, COUNT(*) AS count
                FROM first_hits
                WHERE book_number IS NOT NULL
                GROUP BY book_number
            )
            SELECT
                c.book_number,
                COALESCE(
                    (
                        SELECT b.name
                        FROM book b
                        WHERE b.book_number = c.book_number
                        ORDER BY CASE WHEN b.translation_code = 'NIV' THEN 0 ELSE 1 END,
                                 b.translation_code
                        LIMIT 1
                    ),
                    CONCAT('Book ', c.book_number)
                ) AS book_name,
                c.count
            FROM counts c
            ORDER BY c.count DESC
            LIMIT $3
        """
        return await self.conn.fetch(sql, start, end, limit)
