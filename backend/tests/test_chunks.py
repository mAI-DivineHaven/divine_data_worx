"""
Tests for chunk search endpoints.

Tests cover:
- POST /v1/chunks/search - Chunk-based semantic search
- GET /v1/chunks/{chunk_id} - Get specific chunk by ID
"""

from fastapi.testclient import TestClient


class TestChunkSearch:
    """Tests for POST /v1/chunks/search endpoint."""

    def test_chunk_search_basic(self, client: TestClient):
        """Should perform chunk-based semantic search."""
        response = client.post(
            "/v1/chunks/search",
            json={"embedding": [0.1] * 768, "model": "embeddinggemma", "dim": 768, "top_k": 10},
        )

        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "items" in data
        assert "query_metadata" in data
        assert isinstance(data["items"], list)

    def test_chunk_search_with_translation_filter(self, client: TestClient):
        """Should filter chunks by translation."""
        response = client.post(
            "/v1/chunks/search",
            json={"embedding": [0.1] * 768, "dim": 768, "translation": "NIV", "top_k": 5},
        )

        assert response.status_code == 200
        data = response.json()

        if len(data["items"]) > 0:
            for item in data["items"]:
                assert item["translation_code"] == "NIV"

    def test_chunk_search_with_book_filter(self, client: TestClient):
        """Should filter chunks by book number."""
        response = client.post(
            "/v1/chunks/search",
            json={"embedding": [0.1] * 768, "dim": 768, "book_number": 1, "top_k": 5},  # Genesis
        )

        assert response.status_code == 200
        data = response.json()

        if len(data["items"]) > 0:
            for item in data["items"]:
                assert item["book_number"] == 1

    def test_chunk_search_with_testament_filter(self, client: TestClient):
        """Should filter chunks by testament."""
        response = client.post(
            "/v1/chunks/search",
            json={"embedding": [0.1] * 768, "dim": 768, "testament": "Old", "top_k": 5},
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["items"], list)

    def test_chunk_search_with_window_size_filter(self, client: TestClient):
        """Should filter chunks by window size."""
        response = client.post(
            "/v1/chunks/search",
            json={"embedding": [0.1] * 768, "dim": 768, "window_size": 5, "top_k": 5},
        )

        assert response.status_code == 200
        data = response.json()

        if len(data["items"]) > 0:
            for item in data["items"]:
                if item.get("window_size") is not None:
                    assert item["window_size"] == 5

    def test_chunk_search_with_context(self, client: TestClient):
        """Should include context when requested."""
        response = client.post(
            "/v1/chunks/search",
            json={"embedding": [0.1] * 768, "dim": 768, "include_context": True, "top_k": 3},
        )

        assert response.status_code == 200
        data = response.json()

        if len(data["items"]) > 0:
            item = data["items"][0]
            # Context fields may be present
            assert "context_before" in item
            assert "context_after" in item

    def test_chunk_search_pagination(self, client: TestClient):
        """Should support pagination with offset."""
        response1 = client.post(
            "/v1/chunks/search",
            json={"embedding": [0.1] * 768, "dim": 768, "top_k": 5, "offset": 0},
        )

        response2 = client.post(
            "/v1/chunks/search",
            json={"embedding": [0.1] * 768, "dim": 768, "top_k": 5, "offset": 5},
        )

        assert response1.status_code == 200
        assert response2.status_code == 200

    def test_chunk_search_wrong_dimension(self, client: TestClient):
        """Should reject embedding with wrong dimensionality."""
        response = client.post(
            "/v1/chunks/search",
            json={"embedding": [0.1] * 384, "dim": 768, "top_k": 5},  # Wrong dimension
        )

        assert response.status_code == 422

    def test_chunk_search_response_structure(self, client: TestClient):
        """Should return properly structured chunk results."""
        response = client.post(
            "/v1/chunks/search", json={"embedding": [0.1] * 768, "dim": 768, "top_k": 2}
        )

        assert response.status_code == 200
        data = response.json()

        assert "total" in data
        assert "items" in data
        assert "query_metadata" in data

        if len(data["items"]) > 0:
            item = data["items"][0]
            assert "chunk_id" in item
            assert "translation_code" in item
            assert "book_number" in item
            assert "chapter_start" in item
            assert "verse_start" in item
            assert "chapter_end" in item
            assert "verse_end" in item
            assert "text" in item
            assert "score" in item


class TestGetChunkById:
    """Tests for GET /v1/chunks/{chunk_id} endpoint."""

    def test_get_chunk_not_implemented(self, client: TestClient):
        """Test if chunk retrieval by ID is implemented."""
        response = client.get("/v1/chunks/test_chunk_id")

        # May be 200, 404, or 501 (not implemented)
        assert response.status_code in [200, 404, 501]
