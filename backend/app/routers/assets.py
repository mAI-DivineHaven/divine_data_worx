"""FastAPI router exposing asset management endpoints.

These routes provide CRUD operations, embedding management, and verse linking
for assets stored in the platform. Each handler includes rich docstrings so
that autonomous agents or developers can introspect argument and response
behaviour programmatically.
"""

from __future__ import annotations

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Query, status

from ..config import settings
from ..db.postgres_async import get_pg
from ..models import (
    Asset,
    AssetCreate,
    AssetEmbeddingInfo,
    AssetEmbeddingRequest,
    AssetLinkListResponse,
    AssetLinkRequest,
    AssetLinkResponse,
    AssetListResponse,
    AssetSearchRequest,
    AssetSearchResponse,
    AssetUnlinkResponse,
    AssetUpdate,
)
from ..services.assets import AssetService
from ..services.embeddings import EmbeddingsService

router = APIRouter(prefix="/assets", tags=["assets"])

# Shared embeddings client reused across requests for deterministic behaviour.
embeddings_service = EmbeddingsService(model=settings.VECTOR_MODEL)


def get_asset_service(conn: asyncpg.Connection = Depends(get_pg)) -> AssetService:
    """Provide an :class:`AssetService` bound to the current request scope."""

    return AssetService(conn)


@router.get("/", response_model=AssetListResponse)
async def list_assets(
    limit: int = Query(20, ge=1),
    offset: int = Query(0, ge=0),
    media_type: str | None = Query(None, description="Filter by media type"),
    search: str | None = Query(None, description="Full-text search on title/description"),
    service: AssetService = Depends(get_asset_service),
) -> AssetListResponse:
    """Retrieve a paginated list of assets.

    Args:
        limit: Maximum number of assets to return (capped by ``PAGE_MAX``).
        offset: Pagination offset to start from.
        media_type: Optional media type filter such as ``"image"`` or
            ``"text/plain"``.
        search: Optional substring search over title and description fields.
        service: Asset service dependency injected by FastAPI.

    Returns:
        AssetListResponse with total hits and the selected page of assets.

    Example:
        ``curl "http://localhost:8000/v1/assets?limit=10&media_type=image"``
    """

    limit = min(limit, settings.PAGE_MAX)
    try:
        return await service.list_assets(
            limit=limit,
            offset=offset,
            media_type=media_type,
            search=search,
        )
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post("/", response_model=Asset, status_code=status.HTTP_201_CREATED)
async def create_asset(
    payload: AssetCreate,
    service: AssetService = Depends(get_asset_service),
) -> Asset:
    """Create a new asset record.

    Args:
        payload: Required media metadata for the asset.
        service: Asset service dependency injected by FastAPI.

    Returns:
        Asset instance for the newly created resource.

    Example:
        ``curl -X POST http://localhost:8000/v1/assets -d '{"media_type": "image", "title": "Map"}'``
    """

    try:
        return await service.create_asset(payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/{asset_id}", response_model=Asset)
async def get_asset(
    asset_id: str,
    service: AssetService = Depends(get_asset_service),
) -> Asset:
    """Fetch a single asset by its identifier.

    Args:
        asset_id: Identifier of the asset to fetch.
        service: Asset service dependency injected by FastAPI.

    Returns:
        Asset instance matching the provided ``asset_id``.
    """

    try:
        return await service.get_asset(asset_id)
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.patch("/{asset_id}", response_model=Asset)
async def update_asset(
    asset_id: str,
    payload: AssetUpdate,
    service: AssetService = Depends(get_asset_service),
) -> Asset:
    """Update mutable fields on an asset record.

    Args:
        asset_id: Identifier of the asset to mutate.
        payload: Partial payload containing fields to update.
        service: Asset service dependency injected by FastAPI.

    Returns:
        Updated Asset instance.
    """

    try:
        return await service.update_asset(asset_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.delete("/{asset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_asset(
    asset_id: str,
    service: AssetService = Depends(get_asset_service),
) -> None:
    """Delete an asset record.

    Args:
        asset_id: Identifier of the asset to delete.
        service: Asset service dependency injected by FastAPI.
    """

    try:
        await service.delete_asset(asset_id)
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post("/search", response_model=AssetSearchResponse)
async def search_assets(
    request: AssetSearchRequest,
    service: AssetService = Depends(get_asset_service),
) -> AssetSearchResponse:
    """Semantic search across assets using pgvector similarity.

    Args:
        request: Embedding search payload.
        service: Asset service dependency injected by FastAPI.

    Returns:
        AssetSearchResponse with ranked hits.
    """

    try:
        return await service.search_by_embedding(request)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/{asset_id}/embedding", response_model=AssetEmbeddingInfo)
async def get_asset_embedding(
    asset_id: str,
    service: AssetService = Depends(get_asset_service),
) -> AssetEmbeddingInfo:
    """Retrieve metadata about an asset's stored embedding.

    Args:
        asset_id: Identifier of the asset to inspect.
        service: Asset service dependency injected by FastAPI.

    Returns:
        AssetEmbeddingInfo describing the stored embedding.
    """

    try:
        return await service.get_embedding_info(asset_id)
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post("/{asset_id}/embedding", response_model=AssetEmbeddingInfo)
async def upsert_asset_embedding(
    asset_id: str,
    payload: AssetEmbeddingRequest,
    service: AssetService = Depends(get_asset_service),
) -> AssetEmbeddingInfo:
    """Provide or generate an embedding for an asset.

    Submit a vector directly using the ``embedding`` field, or set
    ``use_asset_text=true`` (optionally providing ``text``) to generate a
    vector using the configured embedding service.

    Args:
        asset_id: Identifier of the asset to augment.
        payload: Embedding payload describing either a vector or generation
            instructions.
        service: Asset service dependency injected by FastAPI.

    Returns:
        AssetEmbeddingInfo describing the persisted embedding metadata.
    """

    try:
        return await service.upsert_embedding(
            asset_id,
            payload,
            embeddings_service=embeddings_service,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc


@router.delete("/{asset_id}/embedding", status_code=status.HTTP_204_NO_CONTENT)
async def delete_asset_embedding(
    asset_id: str,
    service: AssetService = Depends(get_asset_service),
) -> None:
    """Delete the stored embedding for an asset.

    Args:
        asset_id: Identifier of the asset whose embedding to remove.
        service: Asset service dependency injected by FastAPI.
    """

    try:
        await service.delete_embedding(asset_id)
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post("/{asset_id}/links", response_model=AssetLinkResponse)
async def link_asset_to_verses(
    asset_id: str,
    payload: AssetLinkRequest,
    service: AssetService = Depends(get_asset_service),
) -> AssetLinkResponse:
    """Create relationships between an asset and one or more verses.

    Args:
        asset_id: Identifier of the asset to link.
        payload: Link metadata containing verse identifiers and relation type.
        service: Asset service dependency injected by FastAPI.

    Returns:
        AssetLinkResponse summarising the link operation.
    """

    try:
        return await service.link_asset(asset_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get("/{asset_id}/links", response_model=AssetLinkListResponse)
async def list_asset_links(
    asset_id: str,
    service: AssetService = Depends(get_asset_service),
) -> AssetLinkListResponse:
    """Return all verse links associated with an asset.

    Args:
        asset_id: Identifier of the asset whose links to retrieve.
        service: Asset service dependency injected by FastAPI.

    Returns:
        AssetLinkListResponse enumerating linked verses.
    """

    try:
        total, items = await service.list_links(asset_id)
        return AssetLinkListResponse(asset_id=asset_id, total=total, items=items)
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.delete("/{asset_id}/links", response_model=AssetUnlinkResponse)
async def unlink_asset_from_verses(
    asset_id: str,
    verse_id: list[str] | None = Query(None, description="Specific verse IDs to unlink"),
    service: AssetService = Depends(get_asset_service),
) -> AssetUnlinkResponse:
    """Remove verse links from an asset.

    Omitting the ``verse_id`` query parameter removes all links for the asset.

    Args:
        asset_id: Identifier of the asset to unlink.
        verse_id: Optional specific verse identifiers to remove.
        service: Asset service dependency injected by FastAPI.

    Returns:
        AssetUnlinkResponse detailing the number of removed links.
    """

    try:
        return await service.unlink_asset(asset_id, verse_ids=verse_id)
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
