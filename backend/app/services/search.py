"""
DivineHaven Search Service

Implements semantic, lexical, and hybrid search with label-based optimizations.
Provides high-performance async search methods leveraging:
- DiskANN with label-based pre-filtering for semantic search
- Full-text search (FTS) with simple_unaccent for lexical matching
- Reciprocal Rank Fusion (RRF) for hybrid search combining ANN + FTS
- Context window retrieval using absolute verse indexing

All methods are async-first and designed for FastAPI route integration.
Uses asyncpg for connection pooling and prepared statement caching.

Performance Characteristics:
    - Semantic search: ~1-10ms with label filtering
    - Lexical search: ~5-20ms with FTS indexes
    - Hybrid search: ~10-30ms (parallel ANN + FTS, RRF fusion)
    - Context window: ~1-5ms using verse_abs_index

Example Usage:
    ```python
    from fastapi import Depends
    from asyncpg import Connection
    from .services.search import SearchService, SearchResult
    from .db.postgres_async import get_pg

    @router.post("/search/hybrid")
    async def hybrid_search(
        query: str,
        embedding: List[float],
        pg: Connection = Depends(get_pg)
    ):
        service = SearchService(pg)
        results = await service.hybrid_search(
            embedding=embedding,
            query=query,
            translation="NIV",
            limit=10
        )
        return {"results": [r.__dict__ for r in results]}
    ```
"""

from dataclasses import asdict, dataclass
from typing import Any

import asyncpg


@dataclass
class SearchResult:
    """
    Search result data structure containing verse information and relevance score.

    Attributes:
        verse_id: Unique verse identifier (format: translation_book_chapter_verse_suffix)
        translation_code: Translation identifier (e.g., "NIV", "ESV")
        book_number: Canonical book number (1-66 for Protestant canon)
        chapter_number: Chapter number within book
        verse_number: Verse number within chapter
        suffix: Verse suffix for split verses ("", "a", "b", etc.)
        text: Verse text content
        score: Relevance score (semantic: cosine similarity 0-1, FTS: ts_rank, hybrid: RRF)
        testament: Testament classification ("Old" or "New")
        book_name: Human-readable book name
    """

    verse_id: str
    translation_code: str
    book_number: int
    chapter_number: int
    verse_number: int
    suffix: str
    text: str
    score: float
    testament: str | None = None
    book_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class SearchService:
    """
    High-performance async search service for biblical text retrieval.

    Implements three search strategies:
        1. Semantic search: Vector similarity using DiskANN with label pre-filtering
        2. Lexical search: PostgreSQL full-text search with configurable dictionaries
        3. Hybrid search: Reciprocal Rank Fusion combining semantic + lexical

    Label-based Filtering:
        Uses pgvector's label support for efficient pre-filtering:
        - labels[1]: language (1=en, 2=es, 3=el, 4=he, 5=la, 6=fr, 7=de)
        - labels[2]: testament (1=Old, 2=New)
        - labels[3]: book_number (1-66)

    Args:
        conn: asyncpg connection from FastAPI dependency injection
    """

    def __init__(self, conn: asyncpg.Connection):
        """
        Initialize search service with database connection.

        Args:
            conn: Active asyncpg connection (typically from get_pg() dependency)
        """
        self.conn = conn

    @staticmethod
    def _lang_to_label(lang: str) -> int:
        """
        Convert language code to label integer for DiskANN filtering.

        Maps ISO 639-1 language codes to integer labels used in vector index.
        Must match the label generation in manifest_cli.py for consistency.

        Args:
            lang: Language code (e.g., "en", "es", "el")

        Returns:
            Integer label (1-7), or 0 if language not supported
        """
        mapping = {"en": 1, "es": 2, "el": 3, "he": 4, "la": 5, "fr": 6, "de": 7}
        return mapping.get(lang.lower(), 0)

    @staticmethod
    def _testament_to_label(testament: str) -> int:
        """
        Convert testament name to label integer for DiskANN filtering.

        Args:
            testament: Testament name (case-insensitive)
                      Accepts: "Old", "OT", "Hebrew" -> 1
                              "New", "NT", "Greek" -> 2

        Returns:
            Integer label (1 or 2), or 0 if invalid
        """
        t = testament.strip().lower()
        if t in ("old", "ot", "hebrew"):
            return 1
        elif t in ("new", "nt", "greek"):
            return 2
        return 0

    async def semantic_search(
        self,
        embedding: list[float],
        translation: str = "NIV",
        testament: str | None = None,
        books: list[int] | None = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        """
        Semantic search using vector similarity with label-based pre-filtering.

        Performs approximate nearest neighbor (ANN) search using pgvector's
        DiskANN index. Pre-filters by translation, testament, and book numbers
        using the label array for optimal performance.

        Args:
            embedding: Query embedding vector (must match PGVECTOR_DIM, typically 768)
            translation: Translation code for filtering (e.g., "NIV", "ESV")
            testament: Optional testament filter ("Old" or "New")
            books: Optional list of book numbers (1-66) to restrict search
            limit: Maximum number of results to return

        Returns:
            List of SearchResult objects ordered by cosine similarity (descending)

        Performance:
            Typical query time: 1-10ms with label pre-filtering
            Index used: diskann on verse_embedding.embedding with labels
        """
        # Build label filter conditions
        filters = ["v.translation_code = $1"]
        params: list[Any] = [translation]

        if testament:
            testament_label = self._testament_to_label(testament)
            if testament_label > 0:
                filters.append(f"e.labels[2] = ${len(params) + 1}")  # testament is labels[2]
                params.append(testament_label)

        if books:
            # Filter by multiple books using ANY
            filters.append(f"e.labels[3] = ANY(${len(params) + 1})")  # book_number is labels[3]
            params.append(books)

        # Construct query with dynamic placeholders
        where_clause = " AND ".join(filters)
        embed_param = f"${len(params) + 1}"
        limit_param = f"${len(params) + 2}"
        params.extend([embedding, limit])

        sql = f"""
        SELECT
            v.verse_id,
            v.translation_code,
            v.book_number,
            v.chapter_number,
            v.verse_number,
            v.suffix,
            v.text,
            b.name as book_name,
            b.testament,
            (1 - (e.embedding <=> {embed_param}::vector))::numeric(6,4) AS score
        FROM verse_embedding e
        JOIN verse v USING (verse_id)
        JOIN book b ON v.translation_code = b.translation_code
                   AND v.book_number = b.book_number
        WHERE {where_clause}
        ORDER BY e.embedding <=> {embed_param}::vector
        LIMIT {limit_param};
        """

        rows = await self.conn.fetch(sql, *params)

        return [
            SearchResult(
                verse_id=r["verse_id"],
                translation_code=r["translation_code"],
                book_number=r["book_number"],
                chapter_number=r["chapter_number"],
                verse_number=r["verse_number"],
                suffix=r["suffix"],
                text=r["text"],
                score=float(r["score"]),
                testament=r["testament"],
                book_name=r["book_name"],
            )
            for r in rows
        ]

    async def lexical_search(
        self,
        query: str,
        translation: str = "NIV",
        testament: str | None = None,
        books: list[int] | None = None,
        limit: int = 10,
        use_unaccent: bool = True,
    ) -> list[SearchResult]:
        """
        Full-text lexical search using PostgreSQL FTS.

        Uses PostgreSQL's text search capabilities with configurable dictionaries.
        The simple_unaccent configuration handles diacritics and is recommended
        for international text.

        Args:
            query: Search query string (will be converted to tsquery)
            translation: Translation code for filtering
            testament: Optional testament filter ("Old" or "New")
            books: Optional list of book numbers to restrict search
            limit: Maximum number of results
            use_unaccent: Use simple_unaccent config (handles diacritics) vs simple

        Returns:
            List of SearchResult objects ordered by FTS rank (descending)

        Performance:
            Typical query time: 5-20ms with GIN index on to_tsvector(text)
            Index used: verse_text_fts_idx (GIN index)

        Note:
            Uses plainto_tsquery for user-friendly query parsing (handles phrases)
        """
        # Build filters
        filters = ["v.translation_code = $1"]
        params: list[Any] = [translation]

        if testament:
            filters.append(f"b.testament = ${len(params) + 1}")
            params.append(testament.capitalize())

        if books:
            filters.append(f"v.book_number = ANY(${len(params) + 1})")
            params.append(books)

        # Choose FTS configuration
        config = "simple_unaccent" if use_unaccent else "simple"
        where_clause = " AND ".join(filters)

        config_param1 = f"${len(params) + 1}"
        query_param1 = f"${len(params) + 2}"
        config_param2 = f"${len(params) + 3}"
        query_param2 = f"${len(params) + 4}"
        limit_param = f"${len(params) + 5}"

        params.extend([config, query, config, query, limit])

        sql = f"""
        SELECT
            v.verse_id,
            v.translation_code,
            v.book_number,
            v.chapter_number,
            v.verse_number,
            v.suffix,
            v.text,
            b.name as book_name,
            b.testament,
            ts_rank(
                to_tsvector({config_param1}, v.text),
                plainto_tsquery({config_param1}, {query_param1})
            )::numeric(6,4) AS score
        FROM verse v
        JOIN book b ON v.translation_code = b.translation_code
                   AND v.book_number = b.book_number
        WHERE {where_clause}
          AND to_tsvector({config_param2}, v.text) @@ plainto_tsquery({config_param2}, {query_param2})
        ORDER BY score DESC
        LIMIT {limit_param};
        """

        rows = await self.conn.fetch(sql, *params)

        return [
            SearchResult(
                verse_id=r["verse_id"],
                translation_code=r["translation_code"],
                book_number=r["book_number"],
                chapter_number=r["chapter_number"],
                verse_number=r["verse_number"],
                suffix=r["suffix"],
                text=r["text"],
                score=float(r["score"]),
                testament=r["testament"],
                book_name=r["book_name"],
            )
            for r in rows
        ]

    async def hybrid_search(
        self,
        embedding: list[float],
        query: str,
        translation: str = "NIV",
        testament: str | None = None,
        books: list[int] | None = None,
        limit: int = 10,
        k: int = 60,
        topk_ann: int = 100,
        topk_fts: int = 100,
    ) -> list[SearchResult]:
        """
        Hybrid search using Reciprocal Rank Fusion (RRF) to combine ANN + FTS.

        Combines semantic (vector) and lexical (FTS) search for best retrieval
        performance. RRF algorithm fuses rankings from both methods using:

            RRF_score(d) = sum(1 / (k + rank_i(d)))

        Where rank_i(d) is the rank of document d in result set i.

        Args:
            embedding: Query embedding vector for semantic search
            query: Query text for lexical search
            translation: Translation code for filtering
            testament: Optional testament filter
            books: Optional book number filter list
            limit: Final number of results to return
            k: RRF constant (higher = more equal weighting, typical: 60)
            topk_ann: Top-K results to retrieve from ANN before fusion
            topk_fts: Top-K results to retrieve from FTS before fusion

        Returns:
            List of SearchResult objects ordered by RRF score (descending)

        Performance:
            Typical query time: 10-30ms (ANN + FTS in parallel, RRF fusion)
            Combines benefits of semantic understanding + exact keyword matching

        Algorithm Reference:
            "Reciprocal Rank Fusion outperforms Condorcet and individual Rank
            Learning Methods" - Cormack et al., SIGIR 2009
        """
        # Build label filters for ANN
        ann_filters = ["v.translation_code = $1"]
        fts_filters = ["v.translation_code = $2"]
        params: list[Any] = [translation, translation]

        if testament:
            testament_label = self._testament_to_label(testament)
            if testament_label > 0:
                ann_filters.append(f"e.labels[2] = ${len(params) + 1}")
                params.append(testament_label)
            fts_filters.append(f"b.testament = ${len(params) + 1}")
            params.append(testament.capitalize())

        if books:
            ann_filters.append(f"e.labels[3] = ANY(${len(params) + 1})")
            params.append(books)
            fts_filters.append(f"v.book_number = ANY(${len(params) + 1})")
            params.append(books)

        ann_where = " AND ".join(ann_filters)
        fts_where = " AND ".join(fts_filters)

        # Build parameter placeholders
        embed_param1 = f"${len(params) + 1}"
        embed_param2 = f"${len(params) + 2}"
        topk_ann_param = f"${len(params) + 3}"
        query_param1 = f"${len(params) + 4}"
        query_param2 = f"${len(params) + 5}"
        topk_fts_param = f"${len(params) + 6}"
        k_param1 = f"${len(params) + 7}"
        k_param2 = f"${len(params) + 8}"
        limit_param = f"${len(params) + 9}"

        params.extend(
            [
                embedding,
                embedding,
                topk_ann,  # ANN params
                query,
                query,
                topk_fts,  # FTS params
                k,
                k,  # RRF constants
                limit,
            ]
        )

        sql = f"""
        WITH ann AS (
            SELECT v.verse_id,
                   row_number() OVER (ORDER BY e.embedding <=> {embed_param1}::vector) AS rank
            FROM verse_embedding e
            JOIN verse v USING (verse_id)
            WHERE {ann_where}
            ORDER BY e.embedding <=> {embed_param2}::vector
            LIMIT {topk_ann_param}
        ),
        fts AS (
            SELECT v.verse_id,
                   row_number() OVER (ORDER BY ts_rank(
                       to_tsvector('simple_unaccent', v.text),
                       plainto_tsquery('simple_unaccent', {query_param1})
                   ) DESC) AS rank
            FROM verse v
            JOIN book b ON v.translation_code = b.translation_code
                        AND v.book_number = b.book_number
            WHERE {fts_where}
              AND to_tsvector('simple_unaccent', v.text)
                  @@ plainto_tsquery('simple_unaccent', {query_param2})
            LIMIT {topk_fts_param}
        ),
        combined AS (
            SELECT verse_id, 1.0/({k_param1} + rank) AS rrf_score FROM ann
            UNION ALL
            SELECT verse_id, 1.0/({k_param2} + rank) AS rrf_score FROM fts
        )
        SELECT
            v.verse_id,
            v.translation_code,
            v.book_number,
            v.chapter_number,
            v.verse_number,
            v.suffix,
            v.text,
            b.name as book_name,
            b.testament,
            SUM(c.rrf_score)::numeric(6,4) AS score
        FROM combined c
        JOIN verse v USING (verse_id)
        JOIN book b ON v.translation_code = b.translation_code
                   AND v.book_number = b.book_number
        GROUP BY v.verse_id, v.translation_code, v.book_number,
                 v.chapter_number, v.verse_number, v.suffix,
                 v.text, b.name, b.testament
        ORDER BY score DESC
        LIMIT {limit_param};
        """

        rows = await self.conn.fetch(sql, *params)

        return [
            SearchResult(
                verse_id=r["verse_id"],
                translation_code=r["translation_code"],
                book_number=r["book_number"],
                chapter_number=r["chapter_number"],
                verse_number=r["verse_number"],
                suffix=r["suffix"],
                text=r["text"],
                score=float(r["score"]),
                testament=r["testament"],
                book_name=r["book_name"],
            )
            for r in rows
        ]

    async def get_context_window(
        self,
        verse_id: str,
        context_before: int = 2,
        context_after: int = 2,
    ) -> list[SearchResult]:
        """
        Retrieve surrounding verses for context using absolute verse indexing.

        Uses the verse_abs_index column to efficiently fetch verses before and
        after a target verse within the same translation. Maintains canonical
        ordering across book boundaries.

        Args:
            verse_id: Target verse ID
            context_before: Number of verses to include before target
            context_after: Number of verses to include after target

        Returns:
            List of verses in canonical order (includes target verse)
            Score is always 0.0 (not a relevance-based search)

        Performance:
            Typical query time: 1-5ms using verse_abs_index btree index

        Example:
            ```python
            # Get John 3:16 with 2 verses before/after
            service = SearchService(conn)
            context = await service.get_context_window(
                verse_id="NIV_43_3_16_",
                context_before=2,
                context_after=2
            )
            # Returns: John 3:14, 3:15, 3:16, 3:17, 3:18
            ```
        """
        sql = """
        WITH target AS (
            SELECT translation_code, verse_abs_index
            FROM verse
            WHERE verse_id = $1
        )
        SELECT
            v.verse_id,
            v.translation_code,
            v.book_number,
            v.chapter_number,
            v.verse_number,
            v.suffix,
            v.text,
            b.name as book_name,
            b.testament,
            0.0::numeric(6,4) as score
        FROM verse v
        JOIN target t ON v.translation_code = t.translation_code
        JOIN book b ON v.translation_code = b.translation_code
                   AND v.book_number = b.book_number
        WHERE v.verse_abs_index BETWEEN
              t.verse_abs_index - $2 AND
              t.verse_abs_index + $3
        ORDER BY v.verse_abs_index;
        """

        rows = await self.conn.fetch(sql, verse_id, context_before, context_after)

        return [
            SearchResult(
                verse_id=r["verse_id"],
                translation_code=r["translation_code"],
                book_number=r["book_number"],
                chapter_number=r["chapter_number"],
                verse_number=r["verse_number"],
                suffix=r["suffix"],
                text=r["text"],
                score=float(r["score"]),
                testament=r["testament"],
                book_name=r["book_name"],
            )
            for r in rows
        ]
