"""Utilities for loading translation payloads and deriving verse metrics."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from .models import ManifestMetadata, VerseMetrics

# Mapping sourced from scripts/shell_scripts/ingest_all.sh. The CLI allows overrides,
# but these defaults cover the canonical corpus shipped with the repository.
TRANSLATION_FILE_MAP: Mapping[str, str] = {
    "ASV": "en_asv.universal.json",
    "ESV": "en_esv.universal.json",
    "KJV": "en_kjv.universal.json",
    "NASB": "en_nasb.universal.json",
    "NASU": "en_nasu.universal.json",
    "NET": "en_net.universal.json",
    "NIRV": "en_nirv.universal.json",
    "NIV": "en_niv.universal.json",
    "NKJV": "en_nkjv.universal.json",
    "NLT": "en_nlt.universal.json",
    "NVI": "spanish_nvi_bible.universal.json",
    "RVR1960": "spanish_rvr1960_bible.universal.json",
    "LXX": "septuagint_lxx_complete.universal.json",
    "TNK": "he_tanakh.universal.json",
}


def resolve_translation_file(
    translation: str,
    corpus_dir: Path,
    overrides: Mapping[str, str] | None = None,
) -> Path | None:
    """Return the JSON file associated with a translation code if present."""

    mapping = dict(TRANSLATION_FILE_MAP)
    if overrides:
        mapping.update({k.upper(): v for k, v in overrides.items()})

    rel = mapping.get(translation.upper())
    if not rel:
        return None
    candidate = corpus_dir / rel
    if not candidate.exists():
        return None
    return candidate


def load_translation_file(path: Path | str) -> dict[str, object]:
    """Load a translation JSON payload."""

    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def derive_metrics(translation: str, payload: dict[str, object]) -> VerseMetrics:
    """Compute :class:`VerseMetrics` for a parsed translation payload."""

    books = payload.get("books", []) or []
    verse_count = 0
    non_empty = 0
    canonical_keys = set()

    for book in books:
        book_number = book.get("number")
        if book_number is None:
            continue
        chapters = book.get("chapters", []) or []
        for chapter in chapters:
            chapter_number = chapter.get("number")
            if chapter_number is None:
                continue
            verses = chapter.get("verses", []) or []
            for verse in verses:
                verse_number = verse.get("number")
                if verse_number is None:
                    continue
                suffix = verse.get("suffix") or ""
                text = verse.get("text") or ""
                key = f"{book_number}:{chapter_number}:{verse_number}:{suffix}"
                canonical_keys.add(key)
                verse_count += 1
                if text.strip():
                    non_empty += 1

    return VerseMetrics(
        translation=translation,
        verse_count=verse_count,
        non_empty_text_count=non_empty,
        canonical_keys=canonical_keys,
    )


def collect_verse_metrics(
    manifest: ManifestMetadata,
    corpus_dir: Path,
    overrides: Mapping[str, str] | None = None,
) -> tuple[dict[str, VerseMetrics], list[str]]:
    """Gather verse metrics for the translations enumerated in the manifest."""

    metrics: dict[str, VerseMetrics] = {}
    warnings: list[str] = []

    for translation in manifest.translation_set:
        file_path = resolve_translation_file(translation, corpus_dir, overrides=overrides)
        if not file_path:
            warnings.append(
                f"No corpus payload found for translation '{translation}' in {corpus_dir}"
            )
            continue
        try:
            payload = load_translation_file(file_path)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            warnings.append(f"Failed to parse JSON for translation '{translation}': {exc}")
            continue
        metrics[translation] = derive_metrics(translation, payload)

    return metrics, warnings
