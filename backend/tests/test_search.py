"""
Tests for search endpoints (FTS, vector, hybrid).

Tests cover:
- POST /v1/search/fts - Full-text search
- POST /v1/search/vector - Semantic vector search
- POST /v1/search/hybrid - Hybrid RRF search
"""

from fastapi.testclient import TestClient


class TestFullTextSearch:
    """Tests for POST /v1/search/fts endpoint."""

    def test_fts_basic_search(self, client: TestClient):
        """Should return results for simple text query."""
        response = client.post("/v1/search/fts", json={"q": "beginning", "limit": 10})

        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "items" in data
        assert isinstance(data["items"], list)

    def test_fts_with_translation_filter(self, client: TestClient):
        """Should filter results by translation."""
        response = client.post(
            "/v1/search/fts", json={"q": "God", "translation": "NIV", "limit": 5}
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["items"], list)

        # All results should be from NIV if present
        if len(data["items"]) > 0:
            for item in data["items"]:
                assert "NIV" in item["verse_id"]

    def test_fts_pagination(self, client: TestClient):
        """Should respect limit and offset parameters."""
        response1 = client.post("/v1/search/fts", json={"q": "love", "limit": 5, "offset": 0})

        response2 = client.post("/v1/search/fts", json={"q": "love", "limit": 5, "offset": 5})

        assert response1.status_code == 200
        assert response2.status_code == 200

        data1 = response1.json()
        data2 = response2.json()

        # Should return different results
        if len(data1["items"]) > 0 and len(data2["items"]) > 0:
            first_ids = {item["verse_id"] for item in data1["items"]}
            second_ids = {item["verse_id"] for item in data2["items"]}
            assert first_ids != second_ids

    def test_fts_empty_query(self, client: TestClient):
        """Should handle empty query string."""
        response = client.post("/v1/search/fts", json={"q": "", "limit": 10})

        # Should return 422 validation error or 400
        assert response.status_code in [400, 422]

    def test_fts_limit_boundaries(self, client: TestClient):
        """Should enforce limit boundaries (1-500)."""
        # Test minimum
        response_min = client.post("/v1/search/fts", json={"q": "beginning", "limit": 1})
        assert response_min.status_code == 200

        # Test maximum
        response_max = client.post("/v1/search/fts", json={"q": "beginning", "limit": 500})
        assert response_max.status_code == 200

        # Test out of bounds
        response_invalid = client.post("/v1/search/fts", json={"q": "beginning", "limit": 1000})
        assert response_invalid.status_code == 422

    def test_fts_response_structure(self, client: TestClient):
        """Should return properly structured response."""
        response = client.post("/v1/search/fts", json={"q": "faith", "limit": 3})

        assert response.status_code == 200
        data = response.json()

        # Verify top-level structure
        assert "total" in data
        assert "items" in data
        assert isinstance(data["total"], int)
        assert isinstance(data["items"], list)

        # Verify item structure if results exist
        if len(data["items"]) > 0:
            item = data["items"][0]
            assert "verse_id" in item
            assert "text" in item
            assert "score" in item
            assert isinstance(item["score"], (int, float))


class TestVectorSearch:
    """Tests for POST /v1/search/vector endpoint."""

    def test_vector_search_basic(self, client: TestClient):
        """Should perform vector similarity search."""
        response = client.post(
            "/v1/search/vector",
            json={"embedding": [0.1] * 768, "model": "embeddinggemma", "dim": 768, "top_k": 5},
        )

        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "items" in data
        assert isinstance(data["items"], list)

    def test_vector_search_wrong_dimension(self, client: TestClient):
        """Should reject embedding with wrong dimensionality."""
        response = client.post(
            "/v1/search/vector",
            json={
                "embedding": [0.1] * 384,  # Wrong dimension
                "model": "embeddinggemma",
                "dim": 768,
                "top_k": 5,
            },
        )

        assert response.status_code == 422

    def test_vector_search_with_translation_filter(self, client: TestClient):
        """Should filter results by translation."""
        response = client.post(
            "/v1/search/vector",
            json={
                "embedding": [0.1] * 768,
                "model": "embeddinggemma",
                "dim": 768,
                "translation": "ESV",
                "top_k": 5,
            },
        )

        assert response.status_code == 200
        data = response.json()

        # All results should be from ESV if present
        if len(data["items"]) > 0:
            for item in data["items"]:
                assert "ESV" in item["verse_id"]

    def test_vector_search_top_k_limits(self, client: TestClient):
        """Should enforce top_k limits (1-500)."""
        # Valid top_k
        response_valid = client.post(
            "/v1/search/vector", json={"embedding": [0.1] * 768, "dim": 768, "top_k": 50}
        )
        assert response_valid.status_code == 200

        # Invalid top_k (too high)
        response_invalid = client.post(
            "/v1/search/vector", json={"embedding": [0.1] * 768, "dim": 768, "top_k": 1000}
        )
        assert response_invalid.status_code == 422

    def test_vector_search_response_structure(self, client: TestClient):
        """Should return properly structured response with similarity scores."""
        response = client.post(
            "/v1/search/vector", json={"embedding": [0.1] * 768, "dim": 768, "top_k": 3}
        )

        assert response.status_code == 200
        data = response.json()

        assert "total" in data
        assert "items" in data

        if len(data["items"]) > 0:
            item = data["items"][0]
            assert "verse_id" in item
            assert "text" in item
            assert "score" in item
            # Cosine similarity should be between 0 and 1
            assert 0 <= item["score"] <= 1


class TestHybridSearch:
    """Tests for POST /v1/search/hybrid endpoint."""

    def test_hybrid_search_both_modes(self, client: TestClient):
        """Should combine FTS and vector search results."""
        response = client.post(
            "/v1/search/hybrid",
            json={
                "q": "faith",
                "embedding": [0.1] * 768,
                "model": "embeddinggemma",
                "dim": 768,
                "vector_k": 50,
                "fts_k": 50,
                "k_rrf": 60,
                "top_k": 10,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "items" in data
        assert isinstance(data["items"], list)

    def test_hybrid_search_text_only(self, client: TestClient):
        """Should work with text query only."""
        response = client.post(
            "/v1/search/hybrid", json={"q": "love", "vector_k": 50, "fts_k": 50, "top_k": 10}
        )

        assert response.status_code == 200
        data = response.json()
        assert "items" in data

    def test_hybrid_search_vector_only(self, client: TestClient):
        """Should work with vector query only."""
        response = client.post(
            "/v1/search/hybrid",
            json={"embedding": [0.1] * 768, "dim": 768, "vector_k": 50, "fts_k": 50, "top_k": 10},
        )

        assert response.status_code == 200
        data = response.json()
        assert "items" in data

    def test_hybrid_search_no_query(self, client: TestClient):
        """Should reject request with neither text nor vector."""
        response = client.post("/v1/search/hybrid", json={"vector_k": 50, "fts_k": 50, "top_k": 10})

        # Should require at least one query type
        assert response.status_code in [400, 422]

    def test_hybrid_search_rrf_parameter(self, client: TestClient):
        """Should accept k_rrf parameter for RRF fusion."""
        response = client.post(
            "/v1/search/hybrid",
            json={
                "q": "peace",
                "embedding": [0.1] * 768,
                "dim": 768,
                "k_rrf": 100,  # Different RRF constant
                "top_k": 5,
            },
        )

        assert response.status_code == 200

    def test_hybrid_search_response_structure(self, client: TestClient):
        """Should return properly structured RRF-scored results."""
        response = client.post(
            "/v1/search/hybrid",
            json={"q": "hope", "embedding": [0.1] * 768, "dim": 768, "top_k": 5},
        )

        assert response.status_code == 200
        data = response.json()

        assert "total" in data
        assert "items" in data

        if len(data["items"]) > 0:
            item = data["items"][0]
            assert "verse_id" in item
            assert "text" in item
            assert "score" in item
            # RRF score should be positive
            assert item["score"] >= 0


class TestSearchEdgeCases:
    """Edge case tests for search endpoints."""

    def test_search_special_characters(self, client: TestClient):
        """Should handle special characters in search queries."""
        response = client.post("/v1/search/fts", json={"q": "God's", "limit": 5})

        # Should handle gracefully
        assert response.status_code in [200, 400]

    def test_search_very_long_query(self, client: TestClient):
        """Should handle very long search queries."""
        long_query = "word " * 1000

        response = client.post("/v1/search/fts", json={"q": long_query, "limit": 5})

        # Should handle gracefully (accept or reject, but not crash)
        assert response.status_code in [200, 400, 422]

    def test_search_unicode_characters(self, client: TestClient):
        """Should handle Unicode characters in queries."""
        response = client.post(
            "/v1/search/fts", json={"q": "αγάπη", "limit": 5}  # Greek for "love"
        )

        # Should handle gracefully
        assert response.status_code in [200, 400]
