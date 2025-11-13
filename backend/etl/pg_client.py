"""
PostgreSQL async client for ETL operations.

This module provides an asynchronous PostgreSQL client optimized for ETL (Extract, Transform, Load)
operations. It uses asyncpg for high-performance async database access with connection pooling
and server-side cursors for memory-efficient streaming of large datasets.

Key Features:
    - Connection pooling for efficient resource management
    - Server-side cursors for streaming large result sets without memory exhaustion
    - Type-safe with comprehensive type hints for LLM/agent tooling
    - Async context managers for proper resource cleanup

Example:
    ```python
    async with PgClient(dsn="postgresql://user:pass@host/db") as client:
        async for batch in client.iter_verses(batch_size=5000):
            process_batch(batch)  # Process verses in memory-efficient batches
    ```

Dependencies:
    - asyncpg: High-performance async PostgreSQL driver
"""

from collections.abc import AsyncIterator
from typing import Any

import asyncpg


class PgClient:
    """
    Asynchronous PostgreSQL client optimized for ETL operations with streaming support.

    This client uses asyncpg connection pooling and server-side cursors to efficiently
    stream large datasets without loading entire result sets into memory. It's designed
    for ETL pipelines that need to process biblical texts across multiple translations.

    Attributes:
        dsn: PostgreSQL connection string (format: postgresql://user:pass@host:port/database)
        pool: asyncpg connection pool (initialized on async context entry)
        min_pool_size: Minimum number of connections to maintain in pool
        max_pool_size: Maximum number of connections allowed in pool

    Example:
        ```python
        async with PgClient(dsn=settings.PG_DSN, max_pool_size=10) as client:
            # Stream verses efficiently
            async for verse_batch in client.iter_verses(batch_size=1000):
                await process_verses(verse_batch)

            # Get canonical verse keys
            async for book_num, chap_num, verse_num, suffix in client.iter_canonical_keys():
                print(f"{book_num}:{chap_num}:{verse_num}{suffix}")
        ```
    """

    def __init__(
        self,
        dsn: str,
        min_pool_size: int = 2,
        max_pool_size: int = 10,
    ) -> None:
        """
        Initialize the async PostgreSQL ETL client.

        Args:
            dsn: PostgreSQL connection string. Should follow format:
                 postgresql://username:password@hostname:port/database
                 Example: postgresql://postgres:secret@localhost:5432/divinehaven
            min_pool_size: Minimum number of connections to maintain in the pool.
                          Defaults to 2 for responsive startup.
            max_pool_size: Maximum number of connections allowed in the pool.
                          Defaults to 10 to balance concurrency and resource usage.

        Note:
            The connection pool is created lazily when entering the async context manager.
            This allows the class to be instantiated synchronously.
        """
        self.dsn: str = dsn
        self.min_pool_size: int = min_pool_size
        self.max_pool_size: int = max_pool_size
        self.pool: asyncpg.Pool | None = None

    async def __aenter__(self) -> "PgClient":
        """
        Async context manager entry - creates the connection pool.

        Returns:
            Self, with initialized connection pool ready for queries.

        Raises:
            asyncpg.PostgresError: If connection to database fails
            asyncpg.InvalidCatalogNameError: If database does not exist
        """
        self.pool = await asyncpg.create_pool(
            dsn=self.dsn,
            min_size=self.min_pool_size,
            max_size=self.max_pool_size,
            command_timeout=60,  # 60 second timeout for commands
        )
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: object | None,
    ) -> None:
        """
        Async context manager exit - closes the connection pool.

        Ensures all connections are properly closed and resources are freed.
        Exceptions are not suppressed - they propagate normally.

        Args:
            _exc_type: Exception type if an exception was raised in the context
            _exc_val: Exception value if an exception was raised
            _exc_tb: Exception traceback if an exception was raised
        """
        await self.close()

    async def close(self) -> None:
        """
        Explicitly close the connection pool and free all resources.

        This method can be called directly or is automatically called when
        exiting the async context manager. Safe to call multiple times.

        Example:
            ```python
            client = PgClient(dsn)
            await client.__aenter__()  # Or use: async with PgClient(dsn) as client
            try:
                # Use client...
                pass
            finally:
                await client.close()
            ```
        """
        if self.pool is not None:
            await self.pool.close()
            self.pool = None

    async def iter_verses(
        self,
        batch_size: int = 5000,
    ) -> AsyncIterator[list[dict[str, Any]]]:
        """
        Stream canonical verse rows across ALL translations with memory-efficient batching.

        This method uses a server-side cursor to stream verse data without loading the entire
        result set into memory. Verses are ordered consistently by book, chapter, verse, and
        translation for deterministic processing.

        Yielded dictionaries contain these keys:
            - verse_id (str): Unique verse identifier (e.g., "NIV|1|1|1|")
            - translation_code (str): Translation identifier (e.g., "NIV", "ESV")
            - book_number (int): Book number (1-66 for Protestant canon)
            - chapter_number (int): Chapter number within book
            - verse_number (int): Verse number within chapter
            - suffix (str): Verse suffix for split verses (e.g., "a", "b", or "")
            - text (str): The actual verse text content
            - book_name (str): Localized book name (e.g., "Genesis", "John")
            - testament (str): Either "Old" or "New"

        Args:
            batch_size: Number of verses to yield per batch. Larger batches improve
                       throughput but use more memory. Defaults to 5000 verses.

        Yields:
            List of dictionaries, each representing a verse with all required fields.
            Batches will be exactly batch_size except possibly the final batch.

        Example:
            ```python
            async with PgClient(dsn) as client:
                verse_count = 0
                async for batch in client.iter_verses(batch_size=1000):
                    verse_count += len(batch)
                    for verse in batch:
                        print(f"{verse['translation_code']} {verse['book_name']} "
                              f"{verse['chapter_number']}:{verse['verse_number']}")
                print(f"Processed {verse_count} total verses")
            ```

        Note:
            - Uses a named server-side cursor to minimize memory footprint
            - Results are consistently ordered for reproducible processing
            - Connection is held for the duration of iteration
        """
        if self.pool is None:
            raise RuntimeError(
                "PgClient must be used within async context manager (async with PgClient(...) as client)"
            )

        sql = """
            SELECT
              v.verse_id,
              v.translation_code,
              v.book_number,
              v.chapter_number,
              v.verse_number,
              v.suffix,
              v.text,
              b.name       AS book_name,
              b.testament  AS testament
            FROM verse v
            JOIN book b
              ON b.translation_code = v.translation_code
             AND b.book_number      = v.book_number
            ORDER BY
              v.book_number,
              v.chapter_number,
              v.verse_number,
              v.suffix,
              v.translation_code
        """

        async with self.pool.acquire() as conn:
            # Server-side cursor for memory-efficient streaming
            async with conn.transaction():
                cursor = await conn.cursor(sql)
                batch: list[dict[str, Any]] = []

                async for record in cursor:
                    # Convert asyncpg.Record to dict for easier consumption
                    verse_dict = {
                        "verse_id": record["verse_id"],
                        "translation_code": record["translation_code"],
                        "book_number": record["book_number"],
                        "chapter_number": record["chapter_number"],
                        "verse_number": record["verse_number"],
                        "suffix": record["suffix"],
                        "text": record["text"],
                        "book_name": record["book_name"],
                        "testament": record["testament"],
                    }
                    batch.append(verse_dict)

                    if len(batch) >= batch_size:
                        yield batch
                        batch = []

                # Yield remaining verses if any
                if batch:
                    yield batch

    async def list_distinct_references(self) -> list[str]:
        """
        Get list of ALL distinct canonical references across translations.

        Returns canonical reference strings like "Genesis 1:1", "John 3:16", etc.
        Useful for building cross-translation PARALLEL_TO relationships in graphs.

        Returns:
            List of reference strings, ordered by book, chapter, verse, suffix

        Example:
            ```python
            async with PgClient(dsn) as client:
                refs = await client.list_distinct_references()
                print(f"Found {len(refs)} unique references")
                # ['Genesis 1:1', 'Genesis 1:2', ..., 'Revelation 22:21']
            ```

        Note:
            This loads all references into memory at once. For very large datasets,
            consider using iter_canonical_keys() for streaming.
        """
        if self.pool is None:
            raise RuntimeError("PgClient must be used within async context manager")

        sql = """
            SELECT DISTINCT
                v.book_number,
                v.chapter_number,
                v.verse_number,
                v.suffix,
                b.name || ' ' || v.chapter_number || ':' || v.verse_number ||
                    CASE WHEN v.suffix = '' THEN '' ELSE v.suffix END AS reference
            FROM verse v
            JOIN book b
                ON v.translation_code = b.translation_code
               AND v.book_number = b.book_number
            ORDER BY
                v.book_number,
                v.chapter_number,
                v.verse_number,
                v.suffix
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql)
            return [row["reference"] for row in rows]

    async def iter_canonical_keys(self) -> AsyncIterator[tuple[int, int, int, str]]:
        """
        Stream DISTINCT canonical verse keys (book, chapter, verse, suffix).

        This method retrieves unique verse identifiers across all translations, useful for
        linking operations or ensuring verse coverage. Results are ordered consistently.

        Yields:
            Tuple of (book_number, chapter_number, verse_number, suffix) where:
                - book_number (int): Book number (1-66 for Protestant canon)
                - chapter_number (int): Chapter number within book
                - verse_number (int): Verse number within chapter
                - suffix (str): Verse suffix (empty string if no suffix)

        Example:
            ```python
            async with PgClient(dsn) as client:
                unique_verses = []
                async for book, chap, verse, sfx in client.iter_canonical_keys():
                    unique_verses.append(f"{book}:{chap}:{verse}{sfx}")
                print(f"Found {len(unique_verses)} unique verse locations")
            ```

        Use Cases:
            - Post-ingestion linking operations
            - Verse coverage validation across translations
            - Building verse reference indexes
            - Cross-reference resolution

        Note:
            - Returns deduplicated keys across all translations
            - Suffix is normalized to empty string if NULL in database
            - Ordered by book, chapter, verse, suffix for deterministic processing
        """
        if self.pool is None:
            raise RuntimeError(
                "PgClient must be used within async context manager (async with PgClient(...) as client)"
            )

        sql = """
            SELECT DISTINCT
                v.book_number,
                v.chapter_number,
                v.verse_number,
                v.suffix
            FROM verse v
            ORDER BY
                v.book_number,
                v.chapter_number,
                v.verse_number,
                v.suffix
        """

        async with self.pool.acquire() as conn:
            async for record in conn.cursor(sql):
                book_num = record["book_number"]
                chapter_num = record["chapter_number"]
                verse_num = record["verse_number"]
                suffix = record["suffix"] or ""  # Normalize NULL to empty string

                yield (book_num, chapter_num, verse_num, suffix)
