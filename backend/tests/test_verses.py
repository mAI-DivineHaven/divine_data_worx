"""
Tests for verse and metadata retrieval endpoints.

Tests cover:
- GET /v1/verses/{verse_id} - Single verse retrieval
- GET /v1/verses - List verses in chapter
- GET /v1/verses/translations - List all translations
- GET /v1/verses/books - List books in translation
- GET /v1/verses/chapters - List chapters in book
"""

from fastapi.testclient import TestClient


class TestGetVerseById:
    """Tests for GET /v1/verses/{verse_id} endpoint."""

    def test_get_existing_verse(self, client: TestClient):
        """Should return verse when valid verse_id exists."""
        response = client.get("/v1/verses/NIV:1:1:1")

        assert response.status_code == 200
        data = response.json()
        assert data["verse_id"] == "NIV:1:1:1"
        assert data["translation_code"] == "NIV"
        assert data["book_number"] == 1
        assert data["chapter_number"] == 1
        assert data["verse_number"] == 1
        assert "text" in data
        assert len(data["text"]) > 0

    def test_get_nonexistent_verse(self, client: TestClient):
        """Should return 404 when verse_id does not exist."""
        response = client.get("/v1/verses/INVALID:999:999:999")

        assert response.status_code == 404
        assert response.json()["detail"] == "verse not found"

    def test_get_verse_different_translations(self, client: TestClient):
        """Should return different text for same verse in different translations."""
        niv_response = client.get("/v1/verses/NIV:1:1:1")
        esv_response = client.get("/v1/verses/ESV:1:1:1")

        assert niv_response.status_code == 200
        assert esv_response.status_code == 200

        niv_data = niv_response.json()
        esv_data = esv_response.json()

        # Same verse location, different translations
        assert niv_data["book_number"] == esv_data["book_number"]
        assert niv_data["chapter_number"] == esv_data["chapter_number"]
        assert niv_data["verse_number"] == esv_data["verse_number"]
        # Text might be different or similar but worth checking they both exist
        assert len(niv_data["text"]) > 0
        assert len(esv_data["text"]) > 0


class TestListVerses:
    """Tests for GET /v1/verses (list verses in chapter)."""

    def test_list_verses_genesis_1(self, client: TestClient):
        """Should return verses from Genesis 1."""
        response = client.get(
            "/v1/verses",
            params={"translation": "NIV", "book_number": 1, "chapter_number": 1, "limit": 10},
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        assert len(data) <= 10

        # Verify structure of first verse
        first_verse = data[0]
        assert "verse_id" in first_verse
        assert "text" in first_verse
        assert "NIV" in first_verse["verse_id"]

    def test_list_verses_with_pagination(self, client: TestClient):
        """Should respect limit and offset parameters."""
        # Get first 5 verses
        response1 = client.get(
            "/v1/verses",
            params={
                "translation": "NIV",
                "book_number": 1,
                "chapter_number": 1,
                "limit": 5,
                "offset": 0,
            },
        )

        # Get next 5 verses
        response2 = client.get(
            "/v1/verses",
            params={
                "translation": "NIV",
                "book_number": 1,
                "chapter_number": 1,
                "limit": 5,
                "offset": 5,
            },
        )

        assert response1.status_code == 200
        assert response2.status_code == 200

        data1 = response1.json()
        data2 = response2.json()

        # Should return different verses
        assert len(data1) > 0
        assert len(data2) > 0
        if len(data1) > 0 and len(data2) > 0:
            assert data1[0]["verse_id"] != data2[0]["verse_id"]

    def test_list_verses_missing_required_params(self, client: TestClient):
        """Should return 422 when required parameters are missing."""
        response = client.get("/v1/verses")

        assert response.status_code == 422  # Validation error

    def test_list_verses_invalid_book_number(self, client: TestClient):
        """Should return 422 when book_number is out of range."""
        response = client.get(
            "/v1/verses",
            params={"translation": "NIV", "book_number": 999, "chapter_number": 1},  # Invalid
        )

        assert response.status_code == 422

    def test_list_verses_empty_chapter(self, client: TestClient):
        """Should return empty list when chapter has no verses (edge case)."""
        response = client.get(
            "/v1/verses",
            params={
                "translation": "NIV",
                "book_number": 1,
                "chapter_number": 999,  # Unlikely to exist
            },
        )

        # Should still return 200 with empty list, not 404
        assert response.status_code in [200, 400]
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)


class TestListTranslations:
    """Tests for GET /v1/verses/translations endpoint."""

    def test_list_translations_success(
        self, client: TestClient, mock_pg_conn, sample_translations_list
    ):
        """Should return list of all available translations."""
        # Configure mock to return sample translations
        from tests.conftest import configure_mock_fetch

        configure_mock_fetch(mock_pg_conn, sample_translations_list)

        response = client.get("/v1/verses/translations")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

        # Verify structure of first translation
        first_translation = data[0]
        assert "translation_code" in first_translation
        assert "language" in first_translation
        assert "format" in first_translation

    def test_list_translations_contains_common_translations(
        self, client: TestClient, mock_pg_conn, sample_translations_list
    ):
        """Should include common translations like NIV, ESV, KJV."""
        # Configure mock to return sample translations
        from tests.conftest import configure_mock_fetch

        configure_mock_fetch(mock_pg_conn, sample_translations_list)

        response = client.get("/v1/verses/translations")

        assert response.status_code == 200
        data = response.json()

        translation_codes = [t["translation_code"] for t in data]

        # Check for common translations
        common_translations = ["NIV", "ESV", "KJV"]
        for translation in common_translations:
            assert translation in translation_codes, f"{translation} should be in translations list"

    def test_list_translations_never_returns_404(self, client: TestClient, mock_pg_conn):
        """
        Critical: List endpoint should NEVER return 404.
        Should return empty array if no translations exist.
        """
        # Configure mock to return empty list
        from tests.conftest import configure_mock_fetch

        configure_mock_fetch(mock_pg_conn, [])

        response = client.get("/v1/verses/translations")

        assert response.status_code == 200, "List endpoint must return 200, not 404"
        data = response.json()
        assert isinstance(data, list), "Response must be an array"


class TestListBooks:
    """Tests for GET /v1/verses/books endpoint."""

    def test_list_books_success(self, client: TestClient):
        """Should return list of books for a translation."""
        response = client.get("/v1/verses/books", params={"translation": "NIV"})

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

        # Verify structure of first book
        first_book = data[0]
        assert "translation_code" in first_book
        assert "book_number" in first_book
        assert "name" in first_book
        assert "testament" in first_book
        assert first_book["testament"] in ["Old", "New"]

    def test_list_books_genesis_first(self, client: TestClient):
        """Should return Genesis as first book (book_number=1)."""
        response = client.get("/v1/verses/books", params={"translation": "NIV"})

        assert response.status_code == 200
        data = response.json()
        assert len(data) > 0

        genesis = next((b for b in data if b["book_number"] == 1), None)
        assert genesis is not None
        assert genesis["name"] in ["Genesis", "GENESIS"]
        assert genesis["testament"] == "Old"

    def test_list_books_missing_translation_param(self, client: TestClient):
        """Should return 422 when translation parameter is missing."""
        response = client.get("/v1/verses/books")

        assert response.status_code == 422

    def test_list_books_invalid_translation(self, client: TestClient):
        """Should return empty list or 400 for invalid translation."""
        response = client.get("/v1/verses/books", params={"translation": "INVALID_TRANSLATION"})

        # Should return 200 with empty list OR 400, but never 404
        assert response.status_code in [200, 400]
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)

    def test_list_books_never_returns_404(self, client: TestClient):
        """
        Critical: List endpoint should NEVER return 404.
        Should return empty array if no books exist.
        """
        response = client.get("/v1/verses/books", params={"translation": "NIV"})

        assert response.status_code == 200, "List endpoint must return 200, not 404"
        data = response.json()
        assert isinstance(data, list), "Response must be an array"


class TestListChapters:
    """Tests for GET /v1/verses/chapters endpoint."""

    def test_list_chapters_genesis(self, client: TestClient):
        """Should return list of chapters for Genesis."""
        response = client.get(
            "/v1/verses/chapters", params={"translation": "NIV", "book_number": 1}
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

        # Verify structure
        first_chapter = data[0]
        assert "translation_code" in first_chapter
        assert "book_number" in first_chapter
        assert "chapter_number" in first_chapter
        assert first_chapter["book_number"] == 1

    def test_list_chapters_missing_params(self, client: TestClient):
        """Should return 422 when required parameters are missing."""
        response = client.get("/v1/verses/chapters")

        assert response.status_code == 422

    def test_list_chapters_invalid_book(self, client: TestClient):
        """Should handle invalid book number gracefully."""
        response = client.get(
            "/v1/verses/chapters", params={"translation": "NIV", "book_number": 999}
        )

        # Should return 200 with empty list OR 400, but never 404
        assert response.status_code in [200, 400]
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)


class TestRouteOrdering:
    """Tests to verify route ordering is correct (regression tests)."""

    def test_translations_route_not_matched_as_verse_id(self, client: TestClient):
        """
        Regression test: /translations should not match /{verse_id} route.
        This was the original bug that caused 404 errors.
        """
        response = client.get("/v1/verses/translations")

        # Should return translations list, not "verse not found" error
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert "detail" not in data  # Should not be an error response

    def test_books_route_not_matched_as_verse_id(self, client: TestClient):
        """
        Regression test: /books should not match /{verse_id} route.
        This was the original bug that caused 404 errors.
        """
        response = client.get("/v1/verses/books?translation=NIV")

        # Should return books list, not "verse not found" error
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_chapters_route_not_matched_as_verse_id(self, client: TestClient):
        """Should correctly route to /chapters endpoint, not /{verse_id}."""
        response = client.get("/v1/verses/chapters?translation=NIV&book_number=1")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
