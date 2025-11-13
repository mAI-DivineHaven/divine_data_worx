"""Validation routines executed against manifest-aware ingestion metadata."""

from __future__ import annotations

from collections.abc import Mapping

from .models import ManifestMetadata, ValidationResult, VerseMetrics


def validate_verse_coverage(
    manifest: ManifestMetadata,
    verse_metrics: Mapping[str, VerseMetrics],
    minimum_verses: int = 1,
) -> ValidationResult:
    """Ensure that each translation specified in the manifest has verse coverage."""

    errors = []
    for translation in manifest.translation_set:
        metrics = verse_metrics.get(translation)
        if not metrics:
            errors.append(
                f"Missing verse payload for translation '{translation}' declared in manifest"
            )
            continue
        if metrics.verse_count < minimum_verses:
            errors.append(
                f"Translation '{translation}' has insufficient verse coverage: {metrics.verse_count}"
            )
    passed = not errors
    return ValidationResult(
        name="verse_coverage",
        passed=passed,
        errors=errors,
    )


def validate_embedding_completeness(
    manifest: ManifestMetadata,
    verse_metrics: Mapping[str, VerseMetrics],
    allow_empty_ratio: float = 0.0,
) -> ValidationResult:
    """Confirm that verses contain text and are ready for embedding generation."""

    errors = []
    for translation in manifest.translation_set:
        metrics = verse_metrics.get(translation)
        if not metrics:
            errors.append(
                f"Cannot assess embedding completeness; missing verse metrics for '{translation}'"
            )
            continue
        missing = metrics.missing_text_count()
        if metrics.verse_count == 0:
            errors.append(f"Cannot embed translation '{translation}' because it has zero verses")
            continue
        ratio = missing / metrics.verse_count
        if ratio > allow_empty_ratio:
            errors.append(
                f"Translation '{translation}' has {missing} verses with empty text, "
                f"exceeding the allowed ratio of {allow_empty_ratio:.2%}"
            )
    passed = not errors
    return ValidationResult(
        name="embedding_completeness",
        passed=passed,
        errors=errors,
    )


def validate_graph_edge_integrity(
    manifest: ManifestMetadata,
    verse_metrics: Mapping[str, VerseMetrics],
    base_translation: str | None = None,
    max_missing_ratio: float = 0.02,
) -> ValidationResult:
    """Check that canonical verse coverage supports building cross-translation edges."""

    errors = []
    warnings = []

    if not manifest.graph_expansion_enabled:
        warnings.append("Graph expansion disabled in manifest; skipping strict edge checks")
        return ValidationResult("graph_edge_integrity", True, warnings=warnings)

    base = base_translation or manifest.translation_set[0]
    base_metrics = verse_metrics.get(base)
    if not base_metrics:
        errors.append(
            f"Base translation '{base}' missing from verse metrics; cannot validate graph edges"
        )
        return ValidationResult("graph_edge_integrity", False, errors=errors)

    if not base_metrics.canonical_keys:
        errors.append(
            f"Base translation '{base}' has no canonical verse keys; graph edges cannot be derived"
        )
        return ValidationResult("graph_edge_integrity", False, errors=errors)

    base_keys = base_metrics.canonical_keys
    for translation in manifest.translation_set:
        metrics = verse_metrics.get(translation)
        if not metrics:
            errors.append(
                f"Missing verse metrics for '{translation}' preventing graph linkage validation"
            )
            continue
        missing_keys = base_keys - metrics.canonical_keys
        if not base_keys:
            continue
        ratio = len(missing_keys) / len(base_keys)
        if ratio > max_missing_ratio:
            sample = sorted(missing_keys)[:5]
            preview = ", ".join(sample)
            errors.append(
                f"Translation '{translation}' is missing {len(missing_keys)} canonical keys (sample: {preview}) "
                f"exceeding allowed gap of {max_missing_ratio:.2%}"
            )
    passed = not errors
    return ValidationResult(
        name="graph_edge_integrity",
        passed=passed,
        errors=errors,
        warnings=warnings,
    )
