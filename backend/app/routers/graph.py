"""Graph endpoints exposing Neo4j verse neighbourhood queries."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from neo4j import AsyncSession

from ..db.neo4j import get_neo4j_session
from ..models import GraphNeighborhood
from ..services.graph import (
    GraphQueryService,
    InvalidVerseIdentifierError,
    VerseNeighborhoodNotFoundError,
)

router = APIRouter(prefix="/graph", tags=["graph"])


def get_graph_service(session: AsyncSession = Depends(get_neo4j_session)) -> GraphQueryService:
    """Provide a graph query service bound to the request session."""

    return GraphQueryService(session)


@router.get("/verse/{cv_id}", response_model=GraphNeighborhood)
async def get_verse_neighborhood(
    cv_id: str,
    service: GraphQueryService = Depends(get_graph_service),
) -> GraphNeighborhood:
    """Return the canonical verse neighbourhood for the given CV identifier."""

    try:
        return await service.neighborhood_by_cvk(cv_id)
    except InvalidVerseIdentifierError as exc:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="invalid canonical verse identifier",
        ) from exc
    except VerseNeighborhoodNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="canonical verse not found",
        ) from exc


@router.get("/parallel/{translation}/{verse}", response_model=GraphNeighborhood)
async def get_parallel_neighborhood(
    translation: str,
    verse: str,
    service: GraphQueryService = Depends(get_graph_service),
) -> GraphNeighborhood:
    """Return the neighbourhood for a translation-specific verse identifier."""

    try:
        return await service.neighborhood_for_translation_verse(translation, verse)
    except InvalidVerseIdentifierError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="invalid verse identifier",
        ) from exc
    except VerseNeighborhoodNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="verse not found",
        ) from exc
