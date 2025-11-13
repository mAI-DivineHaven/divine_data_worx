"""Analytics service computing query telemetry and usage metrics."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import asyncpg

from ..models import (
    AnalyticsOverview,
    BookUsage,
    ModeCount,
    QueryCounts,
    QueryTrends,
    TopQuery,
    TranslationUsage,
    TrendPoint,
    UsageStats,
)
from ..repositories.analytics import AnalyticsRepository


class AnalyticsService:
    """High-level analytics computations built on top of :class:`AnalyticsRepository`."""

    def __init__(
        self,
        conn: asyncpg.Connection,
        *,
        top_queries: int = 10,
        top_translations: int = 10,
        top_books: int = 10,
    ) -> None:
        self.repo = AnalyticsRepository(conn)
        self.top_queries = top_queries
        self.top_translations = top_translations
        self.top_books = top_books

    async def overview(
        self,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        interval: str | None = None,
    ) -> AnalyticsOverview:
        """Return comprehensive analytics for the requested window."""

        window_start, window_end = self._resolve_window(start, end)
        bucket_interval = self._resolve_interval(interval, window_start, window_end)

        (
            summary_row,
            mode_rows,
            top_rows,
            trend_rows,
            translation_rows,
            book_rows,
        ) = await asyncio.gather(
            self.repo.fetch_query_summary(window_start, window_end),
            self.repo.fetch_mode_breakdown(window_start, window_end),
            self.repo.fetch_top_queries(window_start, window_end, self.top_queries),
            self.repo.fetch_query_trend(window_start, window_end, bucket_interval),
            self.repo.fetch_translation_usage(window_start, window_end, self.top_translations),
            self.repo.fetch_book_usage(window_start, window_end, self.top_books),
        )

        query_counts = self._build_query_counts(summary_row, mode_rows, top_rows)
        trends = self._build_trends(bucket_interval, trend_rows)
        usage = self._build_usage(summary_row, translation_rows, book_rows)

        return AnalyticsOverview(
            window_start=window_start,
            window_end=window_end,
            query_counts=query_counts,
            trends=trends,
            usage=usage,
        )

    async def trends(
        self,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        interval: str | None = None,
    ) -> QueryTrends:
        """Return trend data only."""

        window_start, window_end = self._resolve_window(start, end)
        bucket_interval = self._resolve_interval(interval, window_start, window_end)
        rows = await self.repo.fetch_query_trend(window_start, window_end, bucket_interval)
        return self._build_trends(bucket_interval, rows)

    async def usage(
        self,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> UsageStats:
        """Return usage metrics only."""

        window_start, window_end = self._resolve_window(start, end)
        summary_row = await self.repo.fetch_query_summary(window_start, window_end)
        translation_rows, book_rows = await asyncio.gather(
            self.repo.fetch_translation_usage(window_start, window_end, self.top_translations),
            self.repo.fetch_book_usage(window_start, window_end, self.top_books),
        )
        return self._build_usage(summary_row, translation_rows, book_rows)

    async def counts(
        self,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> QueryCounts:
        """Return query count summary only."""

        window_start, window_end = self._resolve_window(start, end)
        summary_row, mode_rows, top_rows = await asyncio.gather(
            self.repo.fetch_query_summary(window_start, window_end),
            self.repo.fetch_mode_breakdown(window_start, window_end),
            self.repo.fetch_top_queries(window_start, window_end, self.top_queries),
        )
        return self._build_query_counts(summary_row, mode_rows, top_rows)

    def _resolve_window(
        self, start: datetime | None, end: datetime | None
    ) -> tuple[datetime, datetime]:
        now = datetime.now(UTC)
        window_end = end.astimezone(UTC) if end else now
        window_start = start.astimezone(UTC) if start else window_end - timedelta(days=7)
        if window_start >= window_end:
            raise ValueError("start must be earlier than end")
        return window_start, window_end

    def _resolve_interval(self, interval: str | None, start: datetime, end: datetime) -> str:
        if interval:
            interval = interval.lower()
            if interval not in {"hour", "day"}:
                raise ValueError("interval must be 'hour' or 'day'")
            return interval
        delta = end - start
        if delta <= timedelta(days=2):
            return "hour"
        return "day"

    def _build_query_counts(
        self,
        summary_row: asyncpg.Record | None,
        mode_rows: Sequence[asyncpg.Record],
        top_rows: Sequence[asyncpg.Record],
    ) -> QueryCounts:
        total_queries = (
            int(summary_row["total_queries"]) if summary_row and summary_row["total_queries"] else 0
        )
        unique_users = (
            int(summary_row["unique_users"]) if summary_row and summary_row["unique_users"] else 0
        )
        avg_latency = (
            float(summary_row["avg_latency_ms"])
            if summary_row and summary_row["avg_latency_ms"] is not None
            else None
        )

        mode_breakdown: list[ModeCount] = []
        for row in mode_rows:
            count = int(row["count"]) if row["count"] is not None else 0
            percentage = (count / total_queries * 100.0) if total_queries else 0.0
            mode_breakdown.append(
                ModeCount(
                    mode=row["mode"],
                    count=count,
                    percentage=percentage,
                )
            )

        top_queries: list[TopQuery] = []
        for row in top_rows:
            if row["query"] is None:
                continue
            top_queries.append(
                TopQuery(
                    query=row["query"],
                    count=int(row["count"]) if row["count"] is not None else 0,
                    last_seen=row["last_seen"],
                )
            )

        return QueryCounts(
            total=total_queries,
            unique_users=unique_users,
            average_latency_ms=avg_latency,
            mode_breakdown=mode_breakdown,
            top_queries=top_queries,
        )

    def _build_trends(self, interval: str, rows: Sequence[asyncpg.Record]) -> QueryTrends:
        points: list[TrendPoint] = []
        for row in rows:
            bucket_start = row["bucket_start"]
            count = int(row["count"]) if row["count"] is not None else 0
            bucket_end = self._bucket_end(bucket_start, interval)
            points.append(
                TrendPoint(
                    bucket_start=bucket_start,
                    bucket_end=bucket_end,
                    count=count,
                )
            )
        return QueryTrends(interval=interval, points=points)

    def _build_usage(
        self,
        summary_row: asyncpg.Record | None,
        translation_rows: Sequence[asyncpg.Record],
        book_rows: Sequence[asyncpg.Record],
    ) -> UsageStats:
        total_queries = (
            int(summary_row["total_queries"]) if summary_row and summary_row["total_queries"] else 0
        )

        translation_stats: list[TranslationUsage] = []
        for row in translation_rows:
            count = int(row["count"]) if row["count"] is not None else 0
            percentage = (count / total_queries * 100.0) if total_queries else 0.0
            translation_stats.append(
                TranslationUsage(
                    translation_code=row["translation_code"],
                    count=count,
                    percentage=percentage,
                )
            )

        total_book_occurrences = sum(
            int(row["count"]) for row in book_rows if row["count"] is not None
        )
        book_stats: list[BookUsage] = []
        for row in book_rows:
            count = int(row["count"]) if row["count"] is not None else 0
            if row["book_number"] is None:
                continue
            percentage = (count / total_book_occurrences * 100.0) if total_book_occurrences else 0.0
            book_stats.append(
                BookUsage(
                    book_number=int(row["book_number"]),
                    book_name=row["book_name"],
                    count=count,
                    percentage=percentage,
                )
            )

        return UsageStats(translations=translation_stats, books=book_stats)

    @staticmethod
    def _bucket_end(start: datetime, interval: str) -> datetime:
        if interval == "hour":
            return start + timedelta(hours=1)
        if interval == "day":
            return start + timedelta(days=1)
        raise ValueError("Unsupported interval")
