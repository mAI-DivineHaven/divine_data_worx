"""
Neo4j Graph Database Client for DivineHaven ETL

Provides a synchronous Neo4j client optimized for batch graph operations.
Implements an idempotent graph schema for biblical text with multi-translation support.

Graph Schema:
    Nodes:
        - Translation: {code} - Translation identifier (e.g., "NIV", "ESV")
        - CanonBook: {number, testament} - Canonical book reference
        - Book: {translation, number, name} - Translation-specific book
        - Chapter: {translation, book_number, number} - Chapter within a translation
        - Verse: {verse_id, translation, reference, text} - Actual verse content
        - CV: {cvk, book_number, chapter_number, verse_number, suffix} - Canonical verse

    Relationships:
        - (:Translation)-[:HAS_BOOK]->(:Book)
        - (:Book)-[:TRANSLATES]->(:CanonBook)
        - (:Book)-[:HAS_CHAPTER]->(:Chapter)
        - (:Chapter)-[:HAS_VERSE]->(:Verse)
        - (:Verse)-[:RENDITION_OF]->(:CV)
        - (:Verse)-[:PARALLEL_TO {basis:'cvk'}]->(:Verse) - Cross-translation links

Example Usage:
    ```python
    client = Neo4jClient(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password"
    )
    client.init_constraints()
    cvks = client.merge_batch(verse_batch)
    client.link_parallels_for_cvks(cvks)
    client.close()
    ```
"""

from collections.abc import Iterable
from typing import Any, LiteralString, cast

from neo4j import Driver, GraphDatabase, Session


def _cv_key(book_number: int, chapter_number: int, verse_number: int, suffix: str) -> str:
    """
    Generate canonical verse key from components.

    Args:
        book_number: Book number (1-66 for Protestant canon)
        chapter_number: Chapter within book
        verse_number: Verse within chapter
        suffix: Verse suffix for split verses ("", "a", "b", etc.)

    Returns:
        Canonical key in format "book:chapter:verse:suffix" (e.g., "1:1:1:")
    """
    return f"{book_number}:{chapter_number}:{verse_number}:{suffix or ''}"


class Neo4jClient:
    """
    Synchronous Neo4j client for biblical text graph operations.

    Provides idempotent batch upserts for Translation/Book/Chapter/Verse nodes
    and efficient cross-translation parallel verse linking. Designed for ETL
    pipelines processing hundreds of thousands of verses.

    Attributes:
        driver: Neo4j driver instance for database connection

    Performance Notes:
        - Uses batch UNWIND for efficient bulk operations
        - Chunks large link operations to avoid memory issues
        - All operations are idempotent (safe to run multiple times)
    """

    def __init__(self, uri: str, user: str, password: str) -> None:
        """
        Initialize Neo4j connection.

        Args:
            uri: Neo4j bolt URI (e.g., "bolt://localhost:7687")
            user: Neo4j username
            password: Neo4j password
        """
        self.driver: Driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        """Close Neo4j driver and release resources."""
        self.driver.close()

    def init_constraints(self) -> None:
        """
        Initialize graph schema constraints and indexes.

        Creates uniqueness constraints for:
            - Translation codes
            - Canonical book numbers
            - Book (translation, number) pairs
            - Chapter (translation, book_number, number) triplets
            - Verse IDs
            - Canonical verse keys

        Also creates indexes on:
            - Verse references (for text search)
            - Book names (for lookups)

        Note:
            Drops legacy conflicting constraints if present.
            All operations use IF NOT EXISTS for idempotency.
        """
        self._drop_legacy_conflicting()

        constraints_and_indexes = [
            "CREATE CONSTRAINT translation_code IF NOT EXISTS FOR (t:Translation) REQUIRE t.code IS UNIQUE",
            "CREATE CONSTRAINT canon_book_num IF NOT EXISTS FOR (cb:CanonBook) REQUIRE cb.number IS UNIQUE",
            "CREATE CONSTRAINT book_by_txnum IF NOT EXISTS FOR (b:Book) REQUIRE (b.translation, b.number) IS UNIQUE",
            "CREATE CONSTRAINT chapter_by_triplet IF NOT EXISTS FOR (c:Chapter) REQUIRE (c.translation, c.book_number, c.number) IS UNIQUE",
            "CREATE CONSTRAINT verse_by_id IF NOT EXISTS FOR (v:Verse) REQUIRE v.verse_id IS UNIQUE",
            "CREATE CONSTRAINT cv_by_key IF NOT EXISTS FOR (cv:CV) REQUIRE cv.cvk IS UNIQUE",
            "CREATE INDEX verse_ref IF NOT EXISTS FOR (v:Verse) ON (v.reference)",
            "CREATE INDEX book_name IF NOT EXISTS FOR (b:Book) ON (b.name)",
        ]

        with self.driver.session() as session:
            for cypher_stmt in constraints_and_indexes:
                # Cast to LiteralString to satisfy neo4j type checker
                session.run(cast(LiteralString, cypher_stmt))

    def _drop_legacy_conflicting(self) -> None:
        """
        Remove legacy Book(name,number) uniqueness constraint if present.

        Old schema used (name, number) uniqueness which conflicts with our
        per-translation design where the same book number can have different
        names across translations.
        """
        query = """
        SHOW CONSTRAINTS
        YIELD name, labelsOrTypes, properties
        WHERE 'Book' IN labelsOrTypes AND properties = ['name','number']
        RETURN name
        """

        with self.driver.session() as session:
            constraint_names = [record["name"] for record in session.run(query)]
            for constraint_name in constraint_names:
                # Use parameterized query to avoid injection
                session.run("DROP CONSTRAINT $name IF EXISTS", name=constraint_name)

    def _session(self) -> Session:
        """
        Create a new Neo4j session.

        Returns:
            New Neo4j session instance
        """
        return self.driver.session()

    def merge_batch(self, batch: Iterable[dict[str, Any]]) -> set[str]:
        """
        Batch upsert Translation/Book/Chapter/Verse nodes with relationships.

        Creates or updates the complete graph structure for a batch of verses,
        including canonical verse (CV) nodes for cross-translation linking.

        Args:
            batch: Iterable of verse dictionaries containing:
                - translation_code: Translation identifier
                - book_number: Numeric book ID
                - chapter_number: Chapter number
                - verse_number: Verse number
                - suffix: Verse suffix ("" for normal verses)
                - book_name: Localized book name
                - testament: "Old" or "New"
                - verse_id: Unique verse identifier
                - text: Verse text content

        Returns:
            Set of canonical verse keys (CVKs) that were created/updated.
            Use these for subsequent parallel linking operations.

        Performance:
            Processes batches of ~5,000 verses efficiently using UNWIND.
            All operations are idempotent (safe to rerun).
        """
        rows: list[dict[str, Any]] = []
        cvks: set[str] = set()

        for record in batch:
            book_num = int(record["book_number"])
            chapter_num = int(record["chapter_number"])
            verse_num = int(record["verse_number"])
            suffix = (record.get("suffix") or "").strip()

            cvk = _cv_key(book_num, chapter_num, verse_num, suffix)
            cvks.add(cvk)

            # Human-friendly reference like "Genesis 1:1a"
            reference = f'{record["book_name"]} {chapter_num}:{verse_num}{suffix}'

            rows.append(
                {
                    "translation": record["translation_code"],
                    "book_number": book_num,
                    "chapter_number": chapter_num,
                    "verse_number": verse_num,
                    "suffix": suffix,
                    "book_name": record["book_name"],
                    "testament": record["testament"],
                    "verse_id": record["verse_id"],
                    "text": record["text"],
                    "reference": reference,
                    "cvk": cvk,
                }
            )

        cypher = """
        UNWIND $rows AS r
        // Translation node
        MERGE (t:Translation {code: r.translation})

        // Canonical book and translated book
        MERGE (cb:CanonBook {number: r.book_number})
          ON CREATE SET cb.testament = r.testament
          ON MATCH  SET cb.testament = coalesce(cb.testament, r.testament)

        MERGE (tb:Book {translation: r.translation, number: r.book_number})
          ON CREATE SET tb.name = r.book_name
          ON MATCH  SET tb.name = r.book_name

        MERGE (t)-[:HAS_BOOK]->(tb)
        MERGE (tb)-[:TRANSLATES]->(cb)

        // Chapter (translation-scoped)
        MERGE (ch:Chapter {
            translation: r.translation,
            book_number: r.book_number,
            number: r.chapter_number
        })
        MERGE (tb)-[:HAS_CHAPTER]->(ch)

        // Canonical verse node (shared across translations)
        MERGE (cv:CV {cvk: r.cvk})
          ON CREATE SET
            cv.book_number    = r.book_number,
            cv.chapter_number = r.chapter_number,
            cv.verse_number   = r.verse_number,
            cv.suffix         = r.suffix

        // Verse rendition (translation-specific)
        MERGE (v:Verse {verse_id: r.verse_id})
          ON CREATE SET v.translation = r.translation,
                        v.reference   = r.reference,
                        v.text        = r.text
          ON MATCH  SET v.translation = r.translation,
                        v.reference   = r.reference,
                        v.text        = r.text

        MERGE (ch)-[:HAS_VERSE]->(v)
        MERGE (v)-[:RENDITION_OF]->(cv)
        """

        with self._session() as session:
            session.run(cypher, rows=rows)

        return cvks

    def link_parallels_for_cvks(self, cvks: Iterable[str]) -> None:
        """
        Create PARALLEL_TO relationships between verses sharing canonical verse keys.

        For each canonical verse (CV), connects all translation renditions pairwise
        with bidirectional PARALLEL_TO edges. This enables cross-translation queries.

        Args:
            cvks: Iterable of canonical verse keys (from merge_batch)

        Performance Notes:
            - Processes in chunks of 2,000 keys to avoid memory issues
            - Uses id(a) < id(b) to avoid duplicate edges
            - MERGE ensures idempotency (safe to rerun)

        Example:
            For Genesis 1:1 with NIV, ESV, KJV translations:
            Creates: (NIV)-[:PARALLEL_TO]->(ESV)
                    (NIV)-[:PARALLEL_TO]->(KJV)
                    (ESV)-[:PARALLEL_TO]->(KJV)
        """
        unique_keys = list(set(cvks))
        if not unique_keys:
            return

        cypher = """
        UNWIND $kv AS k
        MATCH (cv:CV {cvk: k})
        WITH cv
        MATCH (cv)<-[:RENDITION_OF]-(a:Verse),
              (cv)<-[:RENDITION_OF]-(b:Verse)
        WHERE id(a) < id(b)
        MERGE (a)-[:PARALLEL_TO {basis:'cvk'}]->(b)
        """

        # Split into chunks to avoid oversized UNWIND payloads
        chunk_size = 2000
        with self._session() as session:
            for i in range(0, len(unique_keys), chunk_size):
                chunk = unique_keys[i : i + chunk_size]
                session.run(cypher, kv=chunk)
