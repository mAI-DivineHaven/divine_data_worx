"""Golden query regression tests for hybrid retrieval + graph parallels."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from statistics import mean
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from backend.app.repositories.search import SearchRepository
from backend.app.services.graph import GraphQueryService

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DATASET = REPO_ROOT / "Analysis" / "golden" / "hybrid_graph_queries.json"
MIN_HIT_RATE = 0.9
MIN_MRR = 0.8


def _load_cases() -> list[dict]:
    if not GOLDEN_DATASET.exists():
        pytest.skip(f"Golden dataset missing: {GOLDEN_DATASET}")
    payload = json.loads(GOLDEN_DATASET.read_text())
    if not payload:
        pytest.skip("Golden dataset empty")
    return payload


@pytest.fixture(scope="module")
def golden_cases() -> list[dict]:
    return _load_cases()


@pytest.fixture(scope="module")
def hybrid_lookup(golden_cases: Sequence[dict]) -> dict[tuple[str, str], dict]:
    lookup: dict[tuple[str, str], dict] = {}
    for case in golden_cases:
        request = case["hybrid_request"]
        key = ((request.get("q") or ""), request.get("translation") or "")
        lookup[key] = case
    return lookup


async def _fake_hybrid(self, *, query, translation, **_: dict) -> list[dict]:
    raise AssertionError("hybrid_lookup patch not initialised")


@pytest.fixture
def patch_hybrid(monkeypatch: pytest.MonkeyPatch, hybrid_lookup: dict[tuple[str, str], dict]):
    async def fake_hybrid(self, *, query, translation, **kwargs):
        key = (query or "", translation or "")
        case = hybrid_lookup.get(key)
        if not case:
            raise AssertionError(f"No golden case for signature {key}")
        return case["expected_hybrid"]

    monkeypatch.setattr(SearchRepository, "search_hybrid", fake_hybrid)


def _compute_metrics(relevant: Sequence[str], retrieved: Sequence[str]) -> tuple[float, float]:
    rel_set = set(relevant)
    hit = 0.0
    rr = 0.0
    for idx, vid in enumerate(retrieved):
        if vid in rel_set:
            hit = 1.0
            rr = 1.0 / (idx + 1)
            break
    return hit, rr


def test_golden_hybrid_and_graph_alignment(
    client: TestClient,
    override_db_dependencies,
    golden_cases: Sequence[dict],
    patch_hybrid,
):
    mock_session = override_db_dependencies
    hits: list[float] = []
    mrrs: list[float] = []

    for case in golden_cases:
        graph_payload = case["graph"]["neo4j_payload"]
        result = AsyncMock()
        result.data = AsyncMock(return_value=[graph_payload])
        mock_session.run.return_value = result

        response = client.post("/v1/search/hybrid", json=case["hybrid_request"])
        assert response.status_code == 200, response.text
        body = response.json()
        retrieved_ids = [item["verse_id"] for item in body["items"]]
        expected_ids = [hit["verse_id"] for hit in case["expected_hybrid"]]
        assert retrieved_ids[: len(expected_ids)] == expected_ids

        hit, rr = _compute_metrics(case.get("relevant", []), retrieved_ids)
        hits.append(hit)
        mrrs.append(rr)

        graph_case = case["graph"]
        graph_response = client.get(
            f"/v1/graph/parallel/{graph_case['translation']}/{graph_case['verse_fragment']}"
        )
        assert graph_response.status_code == 200, graph_response.text
        graph_data = graph_response.json()
        assert graph_data["canonical"]["cvk"] == graph_payload["cvk"]
        translations = {rendition["translation"] for rendition in graph_data["renditions"]}
        expected_translations = set(graph_case["expected_translations"])
        assert expected_translations.issubset(translations)

    hit_rate = mean(hits)
    mrr = mean(mrrs)
    assert hit_rate >= MIN_HIT_RATE
    assert mrr >= MIN_MRR

    # ensure cypher invocations use expected templates
    expected_calls = [GraphQueryService.CYPHER_BY_VERSE_ID] * len(golden_cases)
    actual_calls = [call.args[0] for call in mock_session.run.call_args_list]
    assert actual_calls == expected_calls
