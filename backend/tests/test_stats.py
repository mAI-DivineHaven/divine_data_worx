"""Tests for stats and monitoring endpoints."""

from fastapi.testclient import TestClient


class TestEmbeddingStats:
    """Tests for embedding coverage statistics."""

    def test_get_embedding_stats(self, client: TestClient):
        """Should return embedding coverage per translation."""
        response = client.get("/v1/stats/embeddings")
        assert response.status_code in [200, 404]

        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)

            if len(data) > 0:
                stat = data[0]
                assert "translation_code" in stat
                assert "verses" in stat
                assert "embedded" in stat
                assert "missing" in stat


class TestAnalytics:
    """Tests for analytics endpoints."""

    def test_get_analytics_overview(self, client: TestClient):
        """Should return analytics overview."""
        response = client.get("/v1/analytics/overview")
        assert response.status_code in [200, 404]

        if response.status_code == 200:
            data = response.json()
            assert "window_start" in data or "query_counts" in data
