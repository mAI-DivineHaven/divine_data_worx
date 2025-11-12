"""Unit tests for manifest-driven validation utilities."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.validation import (
    ManifestMetadata,
    collect_verse_metrics,
    validate_embedding_completeness,
    validate_graph_edge_integrity,
    validate_verse_coverage,
)
from backend.validation.metrics import derive_metrics
from backend.validation.models import VerseMetrics


@pytest.fixture()
def sample_manifest(tmp_path: Path) -> ManifestMetadata:
    manifest = {
        "run_id": "test",
        "run_ts": "2024-01-01T00:00:00",
        "pipeline_version": "test",
        "source_version": "test",
        "translation_set": ["NIV", "ESV"],
        "languages": ["en"],
        "embedding_recipe": {
            "embedding_model": "test-model",
            "embedding_dim": 128,
            "chunking": {"granularity": "verse"},
        },
        "index_plan": {
            "hybrid": {
                "fusion": {"graph_expansion": {"enabled": True, "max_per_hit": 3, "weight": 0.1}}
            }
        },
        "batches": [],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return ManifestMetadata.from_path(path)


@pytest.fixture()
def translation_payload(tmp_path: Path) -> Path:
    payload = {
        "books": [
            {
                "number": 1,
                "name": "Genesis",
                "chapters": [
                    {
                        "number": 1,
                        "verses": [
                            {"number": 1, "suffix": "", "text": "In the beginning"},
                            {"number": 2, "suffix": "", "text": ""},
                        ],
                    },
                    {
                        "number": 2,
                        "verses": [
                            {"number": 1, "suffix": "", "text": "Thus the heavens"},
                        ],
                    },
                ],
            }
        ]
    }
    path = tmp_path / "niv.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_derive_metrics_counts_non_empty_texts(translation_payload: Path) -> None:
    payload = json.loads(translation_payload.read_text(encoding="utf-8"))
    metrics = derive_metrics("NIV", payload)
    assert metrics.verse_count == 3
    assert metrics.non_empty_text_count == 2
    assert "1:1:1:" in metrics.canonical_keys


def test_validate_verse_coverage_flags_missing_translation(
    sample_manifest: ManifestMetadata,
) -> None:
    metrics = {"NIV": VerseMetrics("NIV", verse_count=10, non_empty_text_count=10)}
    result = validate_verse_coverage(sample_manifest, metrics)
    assert not result.passed
    assert any("ESV" in msg for msg in result.errors)


def test_validate_embedding_completeness_detects_empty_text(
    sample_manifest: ManifestMetadata,
) -> None:
    metrics = {
        "NIV": VerseMetrics("NIV", verse_count=10, non_empty_text_count=8),
        "ESV": VerseMetrics("ESV", verse_count=5, non_empty_text_count=5),
    }
    result = validate_embedding_completeness(sample_manifest, metrics)
    assert not result.passed
    assert any("NIV" in msg for msg in result.errors)


def test_validate_graph_edge_integrity_identifies_missing_keys(
    sample_manifest: ManifestMetadata,
) -> None:
    base_keys = {"1:1:1:", "1:1:2:"}
    metrics = {
        "NIV": VerseMetrics("NIV", verse_count=2, non_empty_text_count=2, canonical_keys=base_keys),
        "ESV": VerseMetrics(
            "ESV",
            verse_count=2,
            non_empty_text_count=2,
            canonical_keys={"1:1:1:"},
        ),
    }
    result = validate_graph_edge_integrity(sample_manifest, metrics, max_missing_ratio=0.0)
    assert not result.passed
    assert any("ESV" in msg for msg in result.errors)


def test_collect_verse_metrics_handles_missing_files(
    sample_manifest: ManifestMetadata, tmp_path: Path
) -> None:
    metrics, warnings = collect_verse_metrics(sample_manifest, tmp_path)
    assert metrics == {}
    assert warnings
    assert "NIV" in warnings[0]
