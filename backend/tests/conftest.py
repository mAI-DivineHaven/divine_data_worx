"""
Pytest configuration and shared fixtures for DivineHaven API tests.

Provides:
- Test client setup with FastAPI TestClient
- Proper async event loop management
- Mock database fixtures for unit tests
- Real database fixtures for integration tests
- Comprehensive test logging
- Sample data factories
"""

import asyncio
import logging
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from backend.app.config import settings
from backend.app.db.neo4j import get_neo4j_session
from backend.app.db.postgres_async import get_pg
from backend.app.dependencies.cache import get_cache_manager
from backend.app.main import app

# ============================================================================
# Logging Configuration for Tests
# ============================================================================


def setup_test_logging():
    """Configure comprehensive logging for test runs."""
    logger = logging.getLogger("tests")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-25s | %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    log_path = Path("tests/test_run.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Configure app logger
    app_logger = logging.getLogger("backend")
    app_logger.setLevel(logging.WARNING)
    app_logger.handlers = [console_handler, file_handler]

    return logger


test_logger = setup_test_logging()


# ============================================================================
# Pytest Configuration Hooks
# ============================================================================


def pytest_configure(config):
    """Pytest configuration hook."""
    test_logger.info("=" * 80)
    test_logger.info("Starting DivineHaven API Test Suite")
    test_logger.info("=" * 80)

    # Register custom markers
    config.addinivalue_line("markers", "unit: Fast unit tests with mocked dependencies")
    config.addinivalue_line("markers", "integration: Integration tests with real database")


def pytest_sessionstart(session):
    """Called after the Session object has been created."""
    import sys

    test_logger.info(f"Python version: {sys.version.split()[0]}")
    test_logger.info(f"Test directory: {session.config.rootpath}")


def pytest_runtest_setup(item):
    """Called before each test runs."""
    test_logger.debug(f"Setting up test: {item.nodeid}")


def pytest_runtest_teardown(item, nextitem):
    """Called after each test completes."""
    test_logger.debug(f"Tearing down test: {item.nodeid}")


# ============================================================================
# Event Loop Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def event_loop():
    """
    Create event loop for the entire test session.

    This avoids the "Event loop is closed" error by providing
    a single event loop that persists across all tests.
    """
    test_logger.debug("Creating session-scoped event loop")
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    asyncio.set_event_loop(loop)

    yield loop

    test_logger.debug("Closing session event loop")
    loop.close()


# ============================================================================
# Mock Database Fixtures (for unit tests)
# ============================================================================


@pytest.fixture
def mock_pg_conn():
    """
    Mock PostgreSQL connection for unit tests.

    Returns an AsyncMock that can be configured for specific test needs.
    Use this for fast unit tests that don't need a real database.
    """
    test_logger.debug("Creating mock PostgreSQL connection")
    mock_conn = AsyncMock()

    asset_row = {
        "asset_id": "asset_mock",
        "media_type": "image/png",
        "title": "Mock Asset",
        "description": "Mock asset description",
        "text_payload": None,
        "payload_json": None,
        "license": None,
        "origin_url": None,
        "created_at": datetime.utcnow(),
    }
    embedding_row = {
        "asset_id": asset_row["asset_id"],
        "embedding_model": "embeddinggemma",
        "embedding_dim": 768,
        "embedding_ts": datetime.utcnow(),
        "metadata": None,
    }
    chunk_row = {
        "chunk_id": "chunk_mock",
        "translation_code": "NIV",
        "book_number": 1,
        "chapter_start": 1,
        "verse_start": 1,
        "chapter_end": 1,
        "verse_end": 2,
        "text": "In the beginning...",
        "window_size": 5,
        "stride": 1,
        "score": 0.95,
        "context_before": "Context before",
        "context_after": "Context after",
    }
    verse_row_primary = {
        "translation_code": "NIV",
        "book_number": 1,
        "chapter_number": 1,
        "verse_number": 1,
        "suffix": "",
        "verse_id": "NIV:1:1:1",
        "text": "In the beginning God created the heavens and the earth.",
    }
    verse_row_secondary = {
        "translation_code": "NIV",
        "book_number": 1,
        "chapter_number": 1,
        "verse_number": 2,
        "suffix": "",
        "verse_id": "NIV:1:1:2",
        "text": "Now the earth was formless and empty.",
    }
    verse_row_esv_primary = {
        "translation_code": "ESV",
        "book_number": 1,
        "chapter_number": 1,
        "verse_number": 1,
        "suffix": "",
        "verse_id": "ESV:1:1:1",
        "text": "In the beginning, God created the heavens and the earth.",
    }
    verse_row_esv_secondary = {
        "translation_code": "ESV",
        "book_number": 1,
        "chapter_number": 1,
        "verse_number": 2,
        "suffix": "",
        "verse_id": "ESV:1:1:2",
        "text": "The earth was without form and void.",
    }
    genesis_niv_texts = [
        "In the beginning God created the heavens and the earth.",
        "Now the earth was formless and empty.",
        "And God said, 'Let there be light,' and there was light.",
        "God saw that the light was good, and he separated the light from the darkness.",
        "God called the light 'day,' and the darkness he called 'night.'",
        "And there was evening, and there was morning—the first day.",
        "God said, 'Let there be a vault between the waters.'",
        "So God made the vault and separated the water under the vault from the water above it.",
        "God called the vault 'sky.'",
        "There was evening, and there was morning—the second day.",
    ]
    genesis_esv_texts = [
        "In the beginning, God created the heavens and the earth.",
        "The earth was without form and void, and darkness was over the face of the deep.",
        "God said, 'Let there be light,' and there was light.",
        "God saw that the light was good, and God separated the light from the darkness.",
        "God called the light Day, and the darkness he called Night.",
        "And there was evening and there was morning, the first day.",
        "God said, 'Let there be an expanse in the midst of the waters.'",
        "And God made the expanse and separated the waters that were under the expanse.",
        "God called the expanse Heaven.",
        "And there was evening and there was morning, the second day.",
    ]
    genesis_niv_verses = [
        {
            "translation_code": "NIV",
            "book_number": 1,
            "chapter_number": 1,
            "verse_number": idx + 1,
            "suffix": "",
            "verse_id": f"NIV:1:1:{idx + 1}",
            "text": text,
        }
        for idx, text in enumerate(genesis_niv_texts)
    ]
    genesis_esv_verses = [
        {
            "translation_code": "ESV",
            "book_number": 1,
            "chapter_number": 1,
            "verse_number": idx + 1,
            "suffix": "",
            "verse_id": f"ESV:1:1:{idx + 1}",
            "text": text,
        }
        for idx, text in enumerate(genesis_esv_texts)
    ]
    sample_verses = {
        **{row["verse_id"]: row for row in genesis_niv_verses},
        **{row["verse_id"]: row for row in genesis_esv_verses},
    }
    fts_items = [
        {
            "verse_id": "NIV:1:1:1",
            "text": "In the beginning God created the heavens and the earth.",
            "score": 0.98,
        },
        {"verse_id": "NIV:1:1:2", "text": "Now the earth was formless and empty.", "score": 0.95},
        {"verse_id": "NIV:1:1:3", "text": "God said, 'Let there be light.'", "score": 0.93},
        {"verse_id": "NIV:1:1:4", "text": "God saw that the light was good.", "score": 0.90},
        {
            "verse_id": "NIV:1:1:5",
            "text": "He separated the light from the darkness.",
            "score": 0.88,
        },
        {
            "verse_id": "ESV:1:1:1",
            "text": "In the beginning, God created the heavens and the earth.",
            "score": 0.86,
        },
        {"verse_id": "ESV:1:1:2", "text": "The earth was without form and void.", "score": 0.84},
        {
            "verse_id": "ESV:1:1:3",
            "text": "And God said, 'Let there be light,' and there was light.",
            "score": 0.82,
        },
        {"verse_id": "ESV:1:1:4", "text": "God saw that the light was good.", "score": 0.80},
        {
            "verse_id": "ESV:1:1:5",
            "text": "God separated the light from the darkness.",
            "score": 0.78,
        },
    ]
    vector_hits_by_translation = {
        None: [
            {
                "verse_id": verse_row_primary["verse_id"],
                "text": verse_row_primary["text"],
                "score": 0.91,
            },
            {
                "verse_id": verse_row_secondary["verse_id"],
                "text": verse_row_secondary["text"],
                "score": 0.89,
            },
            {
                "verse_id": verse_row_esv_primary["verse_id"],
                "text": verse_row_esv_primary["text"],
                "score": 0.87,
            },
            {
                "verse_id": verse_row_esv_secondary["verse_id"],
                "text": verse_row_esv_secondary["text"],
                "score": 0.85,
            },
        ],
        "NIV": [
            {
                "verse_id": verse_row_primary["verse_id"],
                "text": verse_row_primary["text"],
                "score": 0.91,
            },
            {
                "verse_id": verse_row_secondary["verse_id"],
                "text": verse_row_secondary["text"],
                "score": 0.89,
            },
        ],
        "ESV": [
            {
                "verse_id": verse_row_esv_primary["verse_id"],
                "text": verse_row_esv_primary["text"],
                "score": 0.88,
            },
            {
                "verse_id": verse_row_esv_secondary["verse_id"],
                "text": verse_row_esv_secondary["text"],
                "score": 0.86,
            },
        ],
    }
    translation_rows = [
        {
            "idx": 1,
            "book_number": 43,
            "chapter_number": 3,
            "verse_number": 16,
            "suffix": "",
            "translation_code": "NIV",
            "verse_id": "NIV:43:3:16",
            "text": "For God so loved the world",
        },
        {
            "idx": 1,
            "book_number": 43,
            "chapter_number": 3,
            "verse_number": 16,
            "suffix": "",
            "translation_code": "ESV",
            "verse_id": "ESV:43:3:16",
            "text": "For God so loved the world",
        },
    ]
    translations_data = [
        {"translation_code": "NIV", "language": "en", "format": "divine_haven.universal_v1"},
        {"translation_code": "ESV", "language": "en", "format": "divine_haven.universal_v1"},
        {"translation_code": "KJV", "language": "en", "format": "divine_haven.universal_v1"},
    ]
    books_data = [
        {"translation_code": "NIV", "book_number": 1, "name": "Genesis", "testament": "Old"},
        {"translation_code": "NIV", "book_number": 2, "name": "Exodus", "testament": "Old"},
        {"translation_code": "NIV", "book_number": 40, "name": "Matthew", "testament": "New"},
        {"translation_code": "ESV", "book_number": 1, "name": "Genesis", "testament": "Old"},
    ]
    chapters_data = [
        {"translation_code": "NIV", "book_number": 1, "chapter_number": 1},
        {"translation_code": "NIV", "book_number": 1, "chapter_number": 2},
        {"translation_code": "NIV", "book_number": 1, "chapter_number": 3},
        {"translation_code": "ESV", "book_number": 1, "chapter_number": 1},
    ]
    chapter_verse_map = {
        ("NIV", 1, 1): genesis_niv_verses,
        ("ESV", 1, 1): genesis_esv_verses,
    }
    mock_conn._translation_override = None
    verse_embedding_rows = [
        {
            "verse_id": verse_row_primary["verse_id"],
            "embedding": [0.1, 0.2, 0.3, 0.4],
            "embedding_model": "embeddinggemma",
            "embedding_dim": 4,
        },
        {
            "verse_id": verse_row_secondary["verse_id"],
            "embedding": [0.4, 0.3, 0.2, 0.1],
            "embedding_model": "embeddinggemma",
            "embedding_dim": 4,
        },
        {
            "verse_id": verse_row_esv_primary["verse_id"],
            "embedding": [0.2, 0.3, 0.4, 0.5],
            "embedding_model": "embeddinggemma",
            "embedding_dim": 4,
        },
        {
            "verse_id": verse_row_esv_secondary["verse_id"],
            "embedding": [0.5, 0.4, 0.3, 0.2],
            "embedding_model": "embeddinggemma",
            "embedding_dim": 4,
        },
    ]

    async def fetchrow_side_effect(query, *args, **kwargs):
        sql = str(query).upper()
        if "JSON_AGG" in sql or "PLAINTO_TSQUERY" in sql:
            translation = None
            if len(args) > 4:
                translation = args[2]
            limit = args[-2]
            offset = args[-1]
            filtered_items = [
                item
                for item in fts_items
                if (not translation or item["verse_id"].startswith(f"{translation}:"))
            ]
            paginated = filtered_items[offset : offset + limit]
            return create_mock_record({"total": len(filtered_items), "items": paginated})
        if "FROM SEARCH_LOG" in sql and "AVG" in sql:
            return create_mock_record(
                {
                    "total_queries": 24,
                    "unique_users": 6,
                    "avg_latency_ms": 112.5,
                }
            )
        if "FROM VERSE" in sql and "VERSE_ID = $1" in sql:
            verse_id = args[0] if args else None
            verse_data = sample_verses.get(verse_id)
            return create_mock_record(verse_data) if verse_data else None
        if "COUNT(" in sql:
            return create_mock_record({"count": 0})
        if "FROM CHUNK_EMBEDDING" in sql and "COUNT" not in sql:
            return create_mock_record(chunk_row)
        if "FROM ASSET_EMBEDDING" in sql:
            return create_mock_record(embedding_row)
        if "ASSET" in sql:
            return create_mock_record(asset_row)
        return None

    async def fetch_side_effect(query, *args, **kwargs):
        sql = str(query).upper()
        if "INSERT INTO ASSET_LINK" in sql:
            verse_ids = args[1] if len(args) > 1 else []
            return [create_mock_record({"verse_id": vid}) for vid in verse_ids]
        if "WITH REFS AS" in sql:
            return [create_mock_record(row) for row in translation_rows]
        if "VERSE_EMBEDDING" in sql:
            if "JOIN VERSE" in sql:
                translation = None
                if len(args) >= 5 and isinstance(args[2], str):
                    translation = args[2]
                hits = vector_hits_by_translation.get(translation, vector_hits_by_translation[None])
                limit = args[-1]
                return [create_mock_record(hit) for hit in hits[:limit]]
            verse_ids = args[0] if len(args) > 0 else []
            records = [
                create_mock_record(row)
                for row in verse_embedding_rows
                if (not verse_ids) or row["verse_id"] in verse_ids
            ]
            return records
        if "FROM VERSE" in sql and "ORDER BY VERSE_NUMBER" in sql:
            translation = args[0] if len(args) > 0 else None
            book_number = args[1] if len(args) > 1 else None
            chapter_number = args[2] if len(args) > 2 else None
            limit = args[3] if len(args) > 3 else 10
            offset = args[4] if len(args) > 4 else 0
            chapter_key = (translation, book_number, chapter_number)
            verses = chapter_verse_map.get(chapter_key, [])
            slice_verses = verses[offset : offset + limit]
            return [
                create_mock_record({"verse_id": row["verse_id"], "text": row["text"]})
                for row in slice_verses
            ]
        if "FROM VERSE" in sql and "VERSE_ID" in sql and "ANY" in sql:
            verse_ids = args[0] if len(args) > 0 else []
            records = []
            for vid in verse_ids:
                data = sample_verses.get(vid)
                if data:
                    records.append(create_mock_record(data))
            return records
        if "FROM TRANSLATION" in sql:
            override = getattr(mock_conn, "_translation_override", None)
            if override is not None:
                mock_conn._translation_override = None
                return [create_mock_record(row) for row in override]
            return [create_mock_record(row) for row in translations_data]
        if "FROM BOOK" in sql and "ORDER BY BOOK_NUMBER" in sql:
            translation = args[0] if len(args) > 0 else None
            filtered_books = [
                row
                for row in books_data
                if not translation or row["translation_code"] == translation
            ]
            return [create_mock_record(row) for row in filtered_books]
        if "FROM CHAPTER" in sql and "ORDER BY CHAPTER_NUMBER" in sql:
            translation = args[0] if len(args) > 0 else None
            book_number = args[1] if len(args) > 1 else None
            filtered_chapters = [
                row
                for row in chapters_data
                if (not translation or row["translation_code"] == translation)
                and (book_number is None or row["book_number"] == book_number)
            ]
            return [create_mock_record(row) for row in filtered_chapters]
        if "FROM CHUNK_EMBEDDING" in sql:
            record = chunk_row.copy()
            if "CONTEXT_BEFORE" not in sql:
                record.pop("context_before", None)
                record.pop("context_after", None)
            return [create_mock_record(record)]
        if "FROM ASSET_EMBEDDING" in sql and "SCORE" in sql:
            return [create_mock_record({**asset_row, "score": 0.9})]
        if "FROM ASSET" in sql and "ASSET_LINK" not in sql:
            return [create_mock_record(asset_row)]
        return []

    @asynccontextmanager
    async def fake_transaction():
        yield

    mock_conn.fetch = AsyncMock(side_effect=fetch_side_effect)
    mock_conn.fetchrow = AsyncMock(side_effect=fetchrow_side_effect)
    mock_conn.fetchval = AsyncMock(return_value=0)
    mock_conn.execute = AsyncMock(return_value="SELECT 1")
    mock_conn.close = AsyncMock()
    mock_conn.transaction = fake_transaction

    return mock_conn


@pytest.fixture
def mock_neo4j_session():
    """
    Mock Neo4j session for unit tests.

    Returns an AsyncMock configured for awaited Neo4j operations.
    """
    test_logger.debug("Creating mock Neo4j session")
    mock_session = AsyncMock()

    # Configure common async methods
    mock_result = AsyncMock()
    mock_result.data = AsyncMock(return_value=[])

    mock_session.run = AsyncMock(return_value=mock_result)
    mock_session.close = AsyncMock()

    return mock_session


@pytest.fixture
def override_db_dependencies(mock_pg_conn, mock_neo4j_session):
    """
    Override database dependencies with mocks for unit testing.

    This fixture automatically injects mocked database connections
    into the FastAPI dependency injection system.
    """
    test_logger.debug("Overriding database dependencies with mocks")

    previous_env = settings.APP_ENV
    previous_rate_limit = settings.RATE_LIMIT_ENABLED
    previous_secret = settings.JWT_SECRET_KEY
    previous_redis = settings.REDIS_URL

    settings.APP_ENV = "test"
    settings.RATE_LIMIT_ENABLED = False
    if not settings.JWT_SECRET_KEY:
        settings.JWT_SECRET_KEY = "test-secret"
    settings.REDIS_URL = ""
    get_cache_manager.cache_clear()

    from backend.app import main as app_main
    from backend.app.db import postgres_async

    original_init_pool = postgres_async.init_pool
    original_app_init_pool = app_main.init_pool
    postgres_async.init_pool = AsyncMock(return_value=AsyncMock())
    app_main.init_pool = AsyncMock(return_value=AsyncMock())

    async def mock_get_pg():
        yield mock_pg_conn

    def mock_get_neo4j():
        yield mock_neo4j_session

    app.dependency_overrides[get_pg] = mock_get_pg
    app.dependency_overrides[get_neo4j_session] = mock_get_neo4j

    yield mock_neo4j_session

    # Cleanup
    app.dependency_overrides.clear()
    settings.APP_ENV = previous_env
    settings.RATE_LIMIT_ENABLED = previous_rate_limit
    settings.JWT_SECRET_KEY = previous_secret
    settings.REDIS_URL = previous_redis
    postgres_async.init_pool = original_init_pool
    app_main.init_pool = original_app_init_pool
    test_logger.debug("Cleared database dependency overrides")


# ============================================================================
# Test Client Fixtures
# ============================================================================


@pytest.fixture
def client(override_db_dependencies) -> Generator[TestClient, None, None]:
    """
    Provide a TestClient for unit testing with mocked dependencies.

    This client has all database connections mocked, making tests fast
    and independent of external services. Perfect for testing business
    logic and API contracts.

    The override_db_dependencies fixture is automatically applied.
    """
    test_logger.debug("Creating TestClient with mocked dependencies")

    with TestClient(app, raise_server_exceptions=False) as test_client:
        test_logger.debug("TestClient ready")
        yield test_client

    test_logger.debug("TestClient closed")


@pytest.fixture
@pytest.mark.integration
def integration_client() -> Generator[TestClient, None, None]:
    """
    Provide a TestClient for integration testing with real database.

    This client connects to the actual database and services.
    Use this for end-to-end testing that validates the full stack.

    Note: Requires database to be running and accessible.
    """
    test_logger.debug("Creating integration TestClient")

    # Clear any mocks
    app.dependency_overrides.clear()

    with TestClient(app, raise_server_exceptions=False) as test_client:
        test_logger.debug("Integration TestClient ready")
        yield test_client

    test_logger.debug("Integration TestClient closed")


@pytest_asyncio.fixture
async def async_client(override_db_dependencies) -> AsyncGenerator[AsyncClient, None]:
    """
    Provide an AsyncClient for asynchronous unit testing with mocked dependencies.

    Uses ASGITransport to properly handle async app.
    Database dependencies are mocked for fast, isolated tests.
    """
    test_logger.debug("Creating AsyncClient with mocked dependencies")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as ac:
        test_logger.debug("AsyncClient ready")
        yield ac

    test_logger.debug("AsyncClient closed")


# ============================================================================
# Sample Test Data Fixtures
# ============================================================================


@pytest.fixture
def sample_verse_data():
    """Sample verse data for testing."""
    return {
        "verse_id": "NIV:1:1:1",
        "translation_code": "NIV",
        "book_number": 1,
        "chapter_number": 1,
        "verse_number": 1,
        "suffix": "",
        "text": "In the beginning God created the heavens and the earth.",
    }


@pytest.fixture
def sample_verses_list():
    """Sample list of verses for testing list endpoints."""
    return [
        {
            "verse_id": "NIV:1:1:1",
            "translation_code": "NIV",
            "book_number": 1,
            "chapter_number": 1,
            "verse_number": 1,
            "suffix": "",
            "text": "In the beginning God created the heavens and the earth.",
        },
        {
            "verse_id": "NIV:1:1:2",
            "translation_code": "NIV",
            "book_number": 1,
            "chapter_number": 1,
            "verse_number": 2,
            "suffix": "",
            "text": "Now the earth was formless and empty.",
        },
    ]


@pytest.fixture
def sample_translations_list():
    """Sample translations list for testing."""
    return [
        {"translation_code": "NIV", "language": "en", "format": "divine_haven.universal_v1"},
        {"translation_code": "ESV", "language": "en", "format": "divine_haven.universal_v1"},
        {"translation_code": "KJV", "language": "en", "format": "divine_haven.universal_v1"},
    ]


@pytest.fixture
def sample_books_list():
    """Sample books list for testing."""
    return [
        {"translation_code": "NIV", "book_number": 1, "name": "Genesis", "testament": "Old"},
        {"translation_code": "NIV", "book_number": 2, "name": "Exodus", "testament": "Old"},
        {"translation_code": "NIV", "book_number": 40, "name": "Matthew", "testament": "New"},
    ]


@pytest.fixture
def sample_translation_data():
    """Sample translation data for testing."""
    return {"translation_code": "NIV", "language": "en", "format": "divine_haven.universal_v1"}


@pytest.fixture
def sample_book_data():
    """Sample book data for testing."""
    return {"translation_code": "NIV", "book_number": 1, "name": "Genesis", "testament": "Old"}


@pytest.fixture
def sample_chunk_query():
    """Sample chunk search query for testing."""
    return {"embedding": [0.1] * 768, "model": "embeddinggemma", "dim": 768, "top_k": 10}


@pytest.fixture
def sample_asset_data():
    """Sample asset data for testing."""
    return {
        "media_type": "image/png",
        "title": "Test Image",
        "description": "A test image asset",
        "license": "CC-BY-4.0",
        "origin_url": "https://example.com/image.png",
    }


@pytest.fixture
def sample_batch_verse_request():
    """Sample batch verse request for testing."""
    return {"verse_ids": ["NIV:1:1:1", "NIV:1:1:2", "NIV:43:3:16"]}


# ============================================================================
# Helper Functions for Configuring Mocks
# ============================================================================


def create_mock_record(data: dict):
    """
    Create an asyncpg.Record-like object from a dictionary.

    This creates an object that behaves like asyncpg.Record, supporting
    both dict-style access (record['key']) and attribute access (record.key).

    Args:
        data: Dictionary containing the record data

    Returns:
        A mock object that mimics asyncpg.Record behavior
    """

    class MockRecord(dict):
        """Mock asyncpg.Record that supports both dict and attribute access."""

        def __init__(self, data):
            super().__init__(data)
            self.__dict__.update(data)

        def __getitem__(self, key):
            return super().__getitem__(key)

        def get(self, key, default=None):
            return super().get(key, default)

        def keys(self):
            return super().keys()

        def values(self):
            return super().values()

        def items(self):
            return super().items()

    return MockRecord(data)


def configure_mock_fetch(mock_conn, return_data: list[dict]):
    """
    Configure a mock connection to return specific data from fetch().

    Converts dictionaries to asyncpg.Record-like objects for compatibility
    with service layer code.

    Args:
        mock_conn: Mock PostgreSQL connection
        return_data: List of dictionaries to return from fetch()

    Example:
        configure_mock_fetch(mock_conn, [
            {"translation_code": "NIV", "language": "en"},
            {"translation_code": "ESV", "language": "en"}
        ])
    """
    mock_conn.fetch.return_value = [create_mock_record(data) for data in return_data]
    if hasattr(mock_conn, "_translation_override"):
        mock_conn._translation_override = return_data


def configure_mock_fetchrow(mock_conn, return_data: dict | None):
    """
    Configure a mock connection to return specific data from fetchrow().

    Converts dictionary to asyncpg.Record-like object for compatibility
    with service layer code.

    Args:
        mock_conn: Mock PostgreSQL connection
        return_data: Dictionary to return from fetchrow(), or None

    Example:
        configure_mock_fetchrow(mock_conn, {
            "verse_id": "NIV:1:1:1",
            "text": "In the beginning..."
        })
    """
    if return_data:
        mock_conn.fetchrow.return_value = create_mock_record(return_data)
    else:
        mock_conn.fetchrow.return_value = None


def configure_mock_fetchval(mock_conn, return_value):
    """
    Configure a mock connection to return a scalar value from fetchval().

    Args:
        mock_conn: Mock PostgreSQL connection
        return_value: Single value to return (string, int, etc.)

    Example:
        configure_mock_fetchval(mock_conn, 42)
    """
    mock_conn.fetchval.return_value = return_value


# ============================================================================
# Pytest Hooks for Enhanced Reporting
# ============================================================================


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Hook to log test results with detailed information."""
    outcome = yield
    report = outcome.get_result()

    if report.when == "call":
        if report.passed:
            test_logger.info(f"[PASS] {item.nodeid}")
        elif report.failed:
            test_logger.error(f"[FAIL] {item.nodeid}")
            if report.longrepr:
                # Log first few lines of error for quick diagnosis
                error_lines = str(report.longreprtext).split("\n")[:5]
                for line in error_lines:
                    if line.strip():
                        test_logger.error(f"  {line}")
        elif report.skipped:
            test_logger.warning(f"[SKIP] {item.nodeid}")


def pytest_sessionfinish(session, exitstatus):
    """Called after the entire test session finishes."""
    test_logger.info("=" * 80)
    test_logger.info("Test Session Complete")
    test_logger.info(f"Exit Status: {exitstatus}")
    test_logger.info("=" * 80)
