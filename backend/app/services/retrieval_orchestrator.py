"""Composite retrieval orchestrator combining search and graph expansion."""

from __future__ import annotations

from dataclasses import dataclass

from ..models import (
    FusionInfo,
    GraphExpansion,
    GraphExpansionInfo,
    RetrievalHit,
    RetrievalQuery,
    RetrievalResponse,
    SearchHit,
)
from ..utils.manifest import get_hybrid_config, get_manifest_generation
from .graph_expansion import GraphExpansionService
from .search_api import SearchApiService


@dataclass(frozen=True)
class FusionStrategy:
    """Runtime fusion strategy resolved from the manifest."""

    method: str
    k_rrf: int
    vector_k: int
    fts_k: int
    graph_enabled: bool
    graph_max_per_hit: int
    graph_weight: float


_fusion_cache: FusionStrategy | None = None
_fusion_generation: int | None = None


def resolve_fusion_strategy() -> FusionStrategy:
    """Resolve the active fusion strategy from the manifest configuration."""

    global _fusion_cache, _fusion_generation

    generation = get_manifest_generation()
    if _fusion_cache is not None and _fusion_generation == generation:
        return _fusion_cache

    hybrid_cfg = get_hybrid_config()
    fusion_cfg = hybrid_cfg.fusion
    graph_cfg = fusion_cfg.graph_expansion
    _fusion_cache = FusionStrategy(
        method=fusion_cfg.method,
        k_rrf=fusion_cfg.k,
        vector_k=hybrid_cfg.vector_k,
        fts_k=hybrid_cfg.fts_k,
        graph_enabled=graph_cfg.enabled,
        graph_max_per_hit=graph_cfg.max_per_hit,
        graph_weight=graph_cfg.weight,
    )
    _fusion_generation = generation
    return _fusion_cache


class RetrievalOrchestrator:
    """Coordinate ranked retrieval by combining search and graph services."""

    def __init__(
        self,
        search_service: SearchApiService,
        graph_service: GraphExpansionService,
        fusion_strategy: FusionStrategy | None = None,
    ) -> None:
        self._search = search_service
        self._graph = graph_service
        self._fusion = fusion_strategy or resolve_fusion_strategy()

    async def retrieve(self, query: RetrievalQuery) -> RetrievalResponse:
        """Execute hybrid retrieval with optional graph expansion."""

        hybrid_query = self._build_hybrid_query(query)
        search_response = await self._search.hybrid_search(hybrid_query)

        graph_enabled = self._fusion.graph_enabled and query.include_parallels
        expansions: dict[str, GraphExpansion] = {}
        if graph_enabled:
            limit = min(
                query.parallel_limit,
                self._fusion.graph_max_per_hit,
            )
            expansions = await self._graph.parallels_for(
                [hit.verse_id for hit in search_response.items],
                limit=limit,
            )

        items: list[RetrievalHit] = []
        for hit in search_response.items:
            boosted_hit = self._apply_graph_boost(hit, expansions)
            items.append(
                RetrievalHit(
                    hit=boosted_hit,
                    parallels=expansions.get(hit.verse_id),
                )
            )

        fusion_info = FusionInfo(
            method=self._fusion.method,
            k=self._fusion.k_rrf,
            vector_k=self._fusion.vector_k,
            fts_k=self._fusion.fts_k,
            graph_expansion=GraphExpansionInfo(
                enabled=self._fusion.graph_enabled,
                max_per_hit=self._fusion.graph_max_per_hit,
                weight=self._fusion.graph_weight,
                applied=graph_enabled,
            ),
        )
        return RetrievalResponse(total=len(items), items=items, fusion=fusion_info)

    def _apply_graph_boost(
        self,
        hit: SearchHit,
        expansions: dict[str, GraphExpansion],
    ) -> SearchHit:
        """Apply a simple boost to the hit score based on graph coverage."""

        if not self._fusion.graph_enabled or self._fusion.graph_weight <= 0:
            return hit
        expansion = expansions.get(hit.verse_id)
        if not expansion or not expansion.renditions:
            return hit
        boost = self._fusion.graph_weight * len(expansion.renditions)
        boosted_score = round(hit.score + boost, 4)
        payload = hit.model_dump()
        payload["score"] = boosted_score
        return SearchHit(**payload)

    def _build_hybrid_query(self, query: RetrievalQuery) -> RetrievalQuery:
        """Normalise query parameters with manifest-backed defaults."""

        vector_k = query.vector_k if "vector_k" in query.model_fields_set else self._fusion.vector_k
        fts_k = query.fts_k if "fts_k" in query.model_fields_set else self._fusion.fts_k
        k_rrf = query.k_rrf if "k_rrf" in query.model_fields_set else self._fusion.k_rrf
        payload = query.model_dump()
        payload.update(
            {
                "vector_k": vector_k,
                "fts_k": fts_k,
                "k_rrf": k_rrf,
            }
        )
        return RetrievalQuery(**payload)
