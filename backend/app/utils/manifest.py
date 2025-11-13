"""Utility helpers for loading retrieval configuration from manifest.json."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Literal

from pydantic import BaseModel, Field

from ..config import settings


class GraphExpansionConfig(BaseModel):
    """Graph-based expansion options sourced from the manifest."""

    enabled: bool = False
    max_per_hit: int = Field(default=0, ge=0, le=50)
    weight: float = Field(default=0.0, ge=0.0)


class FusionConfig(BaseModel):
    """Fusion strategy configuration for hybrid retrieval."""

    method: Literal["rrf", "weighted_sum"] = "rrf"
    k: int = Field(default=60, ge=1)
    weight_vector: dict[str, float] | None = None
    graph_expansion: GraphExpansionConfig = Field(default_factory=GraphExpansionConfig)


class HybridConfig(BaseModel):
    """Hybrid retrieval configuration from the manifest."""

    vector_k: int = Field(default=50, ge=1)
    fts_k: int = Field(default=50, ge=1)
    fusion: FusionConfig = Field(default_factory=FusionConfig)


class IndexPlanConfig(BaseModel):
    """Subset of manifest index plan required by the API."""

    hybrid: HybridConfig = Field(default_factory=HybridConfig)


class ManifestConfig(BaseModel):
    """Minimal manifest representation for retrieval configuration."""

    index_plan: IndexPlanConfig = Field(default_factory=IndexPlanConfig)


_manifest_cache: ManifestConfig | None = None
_manifest_mtime: float | None = None
_manifest_path: Path | None = None
_manifest_generation: int = 0
_manifest_lock = Lock()


def load_manifest(path: str | None = None, *, force_reload: bool = False) -> ManifestConfig:
    """Load and validate the manifest file, caching the parsed model."""

    manifest_path = Path(path or settings.MANIFEST_PATH)

    global _manifest_cache, _manifest_mtime, _manifest_path, _manifest_generation

    with _manifest_lock:
        if not force_reload and _manifest_cache is not None and _manifest_path == manifest_path:
            try:
                current_mtime = manifest_path.stat().st_mtime
            except FileNotFoundError:
                # Bubble the error so callers can handle missing manifests explicitly.
                raise
            if _manifest_mtime == current_mtime:
                return _manifest_cache
        try:
            current_mtime = manifest_path.stat().st_mtime
            with manifest_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except FileNotFoundError:
            raise

        config = ManifestConfig.model_validate(data)
        _manifest_cache = config
        _manifest_path = manifest_path
        _manifest_mtime = current_mtime
        _manifest_generation += 1
        return config


def reset_manifest_cache() -> None:
    """Clear the manifest cache forcing the next access to reload from disk."""

    global _manifest_cache, _manifest_mtime, _manifest_path, _manifest_generation
    with _manifest_lock:
        _manifest_cache = None
        _manifest_mtime = None
        _manifest_path = None
        _manifest_generation += 1


def get_manifest_generation() -> int:
    """Return a monotonic generation counter that tracks manifest reloads."""

    # Ensure the manifest is loaded at least once to establish a baseline.
    load_manifest()
    return _manifest_generation


def get_hybrid_config() -> HybridConfig:
    """Convenience accessor for the manifest hybrid configuration."""

    return load_manifest().index_plan.hybrid
