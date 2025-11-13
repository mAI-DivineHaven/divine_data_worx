"""Tests for graph verse neighbourhood endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from backend.app.services.graph import GraphQueryService


@pytest.fixture()
def sample_neighborhood() -> list[dict]:
    """Provide a canonical verse neighbourhood payload."""

    return [
        {
            "cvk": "43:3:16:",
            "book_number": 43,
            "chapter_number": 3,
            "verse_number": 16,
            "suffix": "",
            "renditions": [
                {
                    "verse_id": "NIV_43_3_16",
                    "translation": "NIV",
                    "reference": "John 3:16",
                    "text": "For God so loved the world",
                },
                {
                    "verse_id": "ESV_43_3_16",
                    "translation": "ESV",
                    "reference": "John 3:16",
                    "text": "For God so loved the world",
                },
            ],
        }
    ]


class TestGraphEndpoints:
    """Validate graph router behaviour and Cypher invocations."""

    def test_get_verse_neighborhood_executes_expected_cypher(
        self,
        client: TestClient,
        override_db_dependencies,
        sample_neighborhood: list[dict],
    ) -> None:
        mock_session = override_db_dependencies
        result = AsyncMock()
        result.data = AsyncMock(return_value=sample_neighborhood)
        mock_session.run.return_value = result

        response = client.get("/v1/graph/verse/43:3:16:")
        assert response.status_code == 200
        payload = response.json()
        assert payload["canonical"]["cvk"] == "43:3:16:"
        assert len(payload["renditions"]) == 2

        mock_session.run.assert_called_once()
        args, kwargs = mock_session.run.call_args
        assert args[0] == GraphQueryService.CYPHER_BY_CVK
        assert kwargs == {"cvk": "43:3:16:"}

    def test_parallel_neighborhood_normalizes_identifier(
        self,
        client: TestClient,
        override_db_dependencies,
        sample_neighborhood: list[dict],
    ) -> None:
        mock_session = override_db_dependencies
        result = AsyncMock()
        result.data = AsyncMock(return_value=sample_neighborhood)
        mock_session.run.return_value = result

        response = client.get("/v1/graph/parallel/NIV/43:3:16")
        assert response.status_code == 200
        payload = response.json()
        assert payload["canonical"]["cvk"] == "43:3:16:"

        args, kwargs = mock_session.run.call_args
        assert args[0] == GraphQueryService.CYPHER_BY_VERSE_ID
        assert kwargs == {"verse_id": "NIV_43_3_16"}

    def test_parallel_neighborhood_returns_404_when_missing(
        self,
        client: TestClient,
        override_db_dependencies,
    ) -> None:
        mock_session = override_db_dependencies
        result = AsyncMock()
        result.data = AsyncMock(return_value=[])
        mock_session.run.return_value = result

        response = client.get("/v1/graph/parallel/NIV/43_3_16")
        assert response.status_code == 404
        assert response.json()["detail"] == "verse not found"

    def test_parallel_neighborhood_rejects_invalid_identifier(
        self,
        client: TestClient,
        override_db_dependencies,
    ) -> None:
        mock_session = override_db_dependencies
        mock_session.run.reset_mock()

        response = client.get("/v1/graph/parallel/NIV/invalid")
        assert response.status_code == 400
        assert response.json()["detail"] == "invalid verse identifier"
        mock_session.run.assert_not_called()
