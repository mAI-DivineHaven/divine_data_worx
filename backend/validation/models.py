"""Core dataclasses describing manifest metadata and validation results."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ManifestMetadata:
    """Subset of manifest.json fields required for validation routines."""

    translation_set: list[str]
    languages: list[str]
    embedding_model: str
    embedding_dim: int
    chunking_granularity: str
    graph_expansion_enabled: bool

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> ManifestMetadata:
        """Build a :class:`ManifestMetadata` instance from a manifest dictionary."""

        embedding = payload.get("embedding_recipe", {}) or {}
        chunking = embedding.get("chunking", {}) or {}
        index_plan = payload.get("index_plan", {}) or {}
        hybrid = index_plan.get("hybrid", {}) or {}
        fusion = hybrid.get("fusion", {}) or {}
        graph_opts = fusion.get("graph_expansion", {}) or {}

        translation_set = list(payload.get("translation_set", []) or [])
        languages = list(payload.get("languages", []) or [])

        if not translation_set:
            raise ValueError("manifest translation_set is empty; cannot run validations")

        if "embedding_model" not in embedding:
            raise ValueError("manifest missing embedding_recipe.embedding_model")
        if "embedding_dim" not in embedding:
            raise ValueError("manifest missing embedding_recipe.embedding_dim")
        if not chunking.get("granularity"):
            raise ValueError("manifest missing embedding_recipe.chunking.granularity")

        return cls(
            translation_set=translation_set,
            languages=languages,
            embedding_model=str(embedding["embedding_model"]),
            embedding_dim=int(embedding["embedding_dim"]),
            chunking_granularity=str(chunking["granularity"]),
            graph_expansion_enabled=bool(graph_opts.get("enabled", False)),
        )

    @classmethod
    def from_path(cls, path: Path | str) -> ManifestMetadata:
        """Load a manifest JSON file from disk."""

        manifest_path = Path(path)
        with manifest_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls.from_dict(data)


@dataclass(frozen=True)
class VerseMetrics:
    """Aggregated metrics derived from a translation JSON payload."""

    translation: str
    verse_count: int
    non_empty_text_count: int
    canonical_keys: set[str] = field(default_factory=set)

    def missing_text_count(self) -> int:
        """Return the number of verses with empty or whitespace-only payloads."""

        return max(self.verse_count - self.non_empty_text_count, 0)


@dataclass(frozen=True)
class ValidationResult:
    """Outcome container for an individual validation rule."""

    name: str
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def extend_errors(self, messages: Iterable[str]) -> ValidationResult:
        """Return a new :class:`ValidationResult` with additional error messages."""

        return ValidationResult(
            name=self.name,
            passed=False,
            errors=list(self.errors) + list(messages),
            warnings=list(self.warnings),
        )

    def add_warning(self, message: str) -> ValidationResult:
        """Return a copy with an appended warning message."""

        return ValidationResult(
            name=self.name,
            passed=self.passed,
            errors=list(self.errors),
            warnings=list(self.warnings) + [message],
        )
