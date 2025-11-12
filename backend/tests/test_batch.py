"""
Tests for batch retrieval endpoints.

Tests cover:
- POST /v1/batch/verses - Batch verse retrieval
- POST /v1/batch/translations/compare - Translation comparison
- POST /v1/batch/embeddings - Batch embedding lookup
"""

from fastapi.testclient import TestClient


class TestBatchVerses:
    """Tests for POST /v1/batch/verses endpoint."""

    def test_batch_verses_basic(self, client: TestClient):
        """Should retrieve multiple verses in a single request."""
        response = client.post(
            "/v1/batch/verses", json={"verse_ids": ["NIV:1:1:1", "NIV:1:1:2", "ESV:1:1:1"]}
        )

        assert response.status_code == 200
        data = response.json()
        assert "verses" in data
        assert "missing_ids" in data
        assert isinstance(data["verses"], list)
        assert isinstance(data["missing_ids"], list)

    def test_batch_verses_deduplication(self, client: TestClient):
        """Should deduplicate verse IDs."""
        response = client.post(
            "/v1/batch/verses", json={"verse_ids": ["NIV:1:1:1", "NIV:1:1:1", "NIV:1:1:1"]}
        )

        assert response.status_code == 200
        data = response.json()

        # Should not return duplicates
        verse_ids = [v["verse_id"] for v in data["verses"]]
        assert len(verse_ids) == len(set(verse_ids))

    def test_batch_verses_missing_tracking(self, client: TestClient):
        """Should track missing verse IDs."""
        response = client.post(
            "/v1/batch/verses",
            json={"verse_ids": ["NIV:1:1:1", "INVALID:999:999:999"]},  # Exists  # Does not exist
        )

        assert response.status_code == 200
        data = response.json()

        # Should report missing IDs
        assert "missing_ids" in data
        assert "INVALID:999:999:999" in data["missing_ids"]

    def test_batch_verses_max_limit(self, client: TestClient):
        """Should enforce maximum of 500 verse IDs."""
        verse_ids = [f"NIV:1:1:{i}" for i in range(501)]

        response = client.post("/v1/batch/verses", json={"verse_ids": verse_ids})

        assert response.status_code == 422

    def test_batch_verses_empty_list(self, client: TestClient):
        """Should reject empty verse_ids list."""
        response = client.post("/v1/batch/verses", json={"verse_ids": []})

        assert response.status_code == 422

    def test_batch_verses_response_structure(self, client: TestClient):
        """Should return properly structured response."""
        response = client.post("/v1/batch/verses", json={"verse_ids": ["NIV:1:1:1"]})

        assert response.status_code == 200
        data = response.json()

        if len(data["verses"]) > 0:
            verse = data["verses"][0]
            assert "verse_id" in verse
            assert "translation_code" in verse
            assert "book_number" in verse
            assert "chapter_number" in verse
            assert "verse_number" in verse
            assert "suffix" in verse
            assert "text" in verse


class TestTranslationComparison:
    """Tests for POST /v1/batch/translations/compare endpoint."""

    def test_translation_comparison_basic(self, client: TestClient):
        """Should compare same verse across multiple translations."""
        response = client.post(
            "/v1/batch/translations/compare",
            json={
                "references": [
                    {"book_number": 1, "chapter_number": 1, "verse_number": 1, "suffix": ""}
                ],
                "translations": ["NIV", "ESV", "KJV"],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert isinstance(data["items"], list)
        assert len(data["items"]) == 1

    def test_translation_comparison_multiple_verses(self, client: TestClient):
        """Should compare multiple verses across translations."""
        response = client.post(
            "/v1/batch/translations/compare",
            json={
                "references": [
                    {"book_number": 1, "chapter_number": 1, "verse_number": 1, "suffix": ""},
                    {"book_number": 1, "chapter_number": 1, "verse_number": 2, "suffix": ""},
                ],
                "translations": ["NIV", "ESV"],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 2

    def test_translation_comparison_missing_translations(self, client: TestClient):
        """Should track missing translations."""
        response = client.post(
            "/v1/batch/translations/compare",
            json={
                "references": [
                    {"book_number": 1, "chapter_number": 1, "verse_number": 1, "suffix": ""}
                ],
                "translations": ["NIV", "INVALID_TRANSLATION"],
            },
        )

        assert response.status_code == 200
        data = response.json()

        if len(data["items"]) > 0:
            item = data["items"][0]
            assert "missing_translations" in item
            assert "INVALID_TRANSLATION" in item["missing_translations"]

    def test_translation_comparison_max_references(self, client: TestClient):
        """Should enforce maximum of 200 references."""
        references = [
            {"book_number": 1, "chapter_number": 1, "verse_number": i, "suffix": ""}
            for i in range(1, 202)
        ]

        response = client.post(
            "/v1/batch/translations/compare",
            json={"references": references, "translations": ["NIV"]},
        )

        assert response.status_code == 422

    def test_translation_comparison_max_translations(self, client: TestClient):
        """Should enforce maximum of 25 translations."""
        translations = [f"TRANS{i}" for i in range(26)]

        response = client.post(
            "/v1/batch/translations/compare",
            json={
                "references": [
                    {"book_number": 1, "chapter_number": 1, "verse_number": 1, "suffix": ""}
                ],
                "translations": translations,
            },
        )

        assert response.status_code == 422

    def test_translation_comparison_response_structure(self, client: TestClient):
        """Should return properly structured comparison."""
        response = client.post(
            "/v1/batch/translations/compare",
            json={
                "references": [
                    {"book_number": 43, "chapter_number": 3, "verse_number": 16, "suffix": ""}
                ],
                "translations": ["NIV", "ESV"],
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert "items" in data
        if len(data["items"]) > 0:
            item = data["items"][0]
            assert "reference" in item
            assert "translations" in item
            assert "missing_translations" in item
            assert isinstance(item["translations"], list)


class TestEmbeddingLookup:
    """Tests for POST /v1/batch/embeddings endpoint."""

    def test_embedding_lookup_basic(self, client: TestClient):
        """Should retrieve embeddings for multiple verses."""
        response = client.post(
            "/v1/batch/embeddings",
            json={"verse_ids": ["NIV:1:1:1", "NIV:1:1:2"], "model": "embeddinggemma"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "missing_ids" in data
        assert isinstance(data["results"], list)

    def test_embedding_lookup_missing_tracking(self, client: TestClient):
        """Should track verses without embeddings."""
        response = client.post(
            "/v1/batch/embeddings",
            json={"verse_ids": ["INVALID:999:999:999"], "model": "embeddinggemma"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "missing_ids" in data

    def test_embedding_lookup_max_limit(self, client: TestClient):
        """Should enforce maximum of 500 verse IDs."""
        verse_ids = [f"NIV:1:1:{i}" for i in range(501)]

        response = client.post(
            "/v1/batch/embeddings", json={"verse_ids": verse_ids, "model": "embeddinggemma"}
        )

        assert response.status_code == 422

    def test_embedding_lookup_response_structure(self, client: TestClient):
        """Should return properly structured embeddings."""
        response = client.post(
            "/v1/batch/embeddings", json={"verse_ids": ["NIV:1:1:1"], "model": "embeddinggemma"}
        )

        assert response.status_code == 200
        data = response.json()

        if len(data["results"]) > 0:
            result = data["results"][0]
            assert "verse_id" in result
            assert "embedding" in result
            assert "embedding_model" in result
            assert "embedding_dim" in result
            assert isinstance(result["embedding"], list)
