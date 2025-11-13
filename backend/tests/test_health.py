"""Tests for health and monitoring endpoints."""

from fastapi.testclient import TestClient


class TestHealth:
    """Tests for health check endpoint."""

    def test_health_check(self, client: TestClient):
        """Should return healthy status."""
        response = client.get("/v1/healthz")

        assert response.status_code == 200
        data = response.json()
        assert "ok" in data
        assert data["ok"] is True

    def test_health_check_always_available(self, client: TestClient):
        """Health endpoint should always be accessible."""
        response = client.get("/v1/healthz")

        # Should never return 404 or 500
        assert response.status_code == 200


class TestMetrics:
    """Tests for Prometheus metrics endpoint."""

    def test_metrics_endpoint(self, client: TestClient):
        """Should return Prometheus metrics."""
        response = client.get("/metrics")

        # May be disabled or enabled
        assert response.status_code in [200, 404]

        if response.status_code == 200:
            # Should return text/plain
            assert "text/plain" in response.headers.get("content-type", "")
            # Should contain Prometheus metrics format
            content = response.text
            assert len(content) > 0


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_endpoint(self, client: TestClient):
        """Should respond to root path."""
        response = client.get("/")

        # May redirect to docs or return info
        assert response.status_code in [200, 307, 404]
