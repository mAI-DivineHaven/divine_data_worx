"""Tests for asset management endpoints."""

from fastapi.testclient import TestClient


class TestAssetCRUD:
    """Tests for asset CRUD operations."""

    def test_create_asset(self, client: TestClient):
        """Should create a new asset."""
        response = client.post(
            "/v1/assets/",
            json={
                "media_type": "image/png",
                "title": "Test Asset",
                "description": "Test description",
            },
        )
        assert response.status_code in [200, 201]
        if response.status_code in [200, 201]:
            data = response.json()
            assert "asset_id" in data

    def test_list_assets(self, client: TestClient):
        """Should list assets with pagination."""
        response = client.get("/v1/assets/?limit=10&offset=0")
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "items" in data

    def test_get_asset_by_id(self, client: TestClient):
        """Should get asset by ID or return 404."""
        response = client.get("/v1/assets/test_asset_id")
        assert response.status_code in [200, 404]

    def test_update_asset(self, client: TestClient):
        """Should update asset or return 404."""
        response = client.patch("/v1/assets/test_asset_id", json={"title": "Updated Title"})
        assert response.status_code in [200, 404]

    def test_delete_asset(self, client: TestClient):
        """Should delete asset or return 404."""
        response = client.delete("/v1/assets/test_asset_id")
        assert response.status_code in [204, 404]


class TestAssetSearch:
    """Tests for asset semantic search."""

    def test_asset_search(self, client: TestClient):
        """Should perform semantic search over assets."""
        response = client.post(
            "/v1/assets/search",
            json={"embedding": [0.1] * 768, "model": "embeddinggemma", "dim": 768, "top_k": 5},
        )
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "items" in data


class TestAssetEmbeddings:
    """Tests for asset embedding management."""

    def test_get_asset_embedding(self, client: TestClient):
        """Should get asset embedding or return 404."""
        response = client.get("/v1/assets/test_asset_id/embedding")
        assert response.status_code in [200, 404]

    def test_set_asset_embedding(self, client: TestClient):
        """Should set/generate asset embedding."""
        response = client.post(
            "/v1/assets/test_asset_id/embedding",
            json={"embedding": [0.1] * 768, "model": "embeddinggemma", "dim": 768},
        )
        assert response.status_code in [200, 404]

    def test_delete_asset_embedding(self, client: TestClient):
        """Should delete asset embedding."""
        response = client.delete("/v1/assets/test_asset_id/embedding")
        assert response.status_code in [204, 404]


class TestAssetVerseLinks:
    """Tests for asset-verse linking."""

    def test_create_asset_links(self, client: TestClient):
        """Should link asset to verses."""
        response = client.post(
            "/v1/assets/test_asset_id/links",
            json={"verse_ids": ["NIV:1:1:1"], "relation": "illustrates"},
        )
        assert response.status_code in [200, 404]

    def test_list_asset_links(self, client: TestClient):
        """Should list asset verse links."""
        response = client.get("/v1/assets/test_asset_id/links")
        assert response.status_code in [200, 404]

    def test_delete_asset_links(self, client: TestClient):
        """Should delete asset verse links."""
        response = client.delete("/v1/assets/test_asset_id/links?verse_ids=NIV:1:1:1")
        assert response.status_code in [200, 404]
