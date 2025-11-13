"""Asset service implementing business logic for asset management."""

from __future__ import annotations

import uuid
from collections.abc import Sequence

import asyncpg

from ..models import (
    Asset,
    AssetCreate,
    AssetEmbeddingInfo,
    AssetEmbeddingRequest,
    AssetLinkRequest,
    AssetLinkResponse,
    AssetListResponse,
    AssetSearchHit,
    AssetSearchRequest,
    AssetSearchResponse,
    AssetUnlinkResponse,
    AssetUpdate,
    AssetVerseLink,
)
from ..repositories.assets import AssetRepository
from .embeddings import EmbeddingsService


class AssetService:
    """Coordinates asset CRUD, embedding, and linking operations."""

    def __init__(
        self,
        conn: asyncpg.Connection,
    ) -> None:
        """Create a new service instance bound to a database connection."""

        self.conn = conn
        self.repo = AssetRepository(conn)

    # ------------------------------------------------------------------
    # Asset CRUD
    # ------------------------------------------------------------------
    async def list_assets(
        self,
        *,
        limit: int,
        offset: int,
        media_type: str | None = None,
        search: str | None = None,
    ) -> AssetListResponse:
        """Return a paginated list of assets matching optional filters.

        Args:
            limit: Maximum number of assets to return.
            offset: Number of records to skip for pagination.
            media_type: Optional media type filter (e.g., ``"image"``).
            search: Optional text search across titles and descriptions.

        Returns:
            AssetListResponse containing the total count and returned assets.
        """

        total, items = await self.repo.list_assets(
            limit=limit,
            offset=offset,
            media_type=media_type,
            search=search,
        )
        return AssetListResponse(total=total, items=items)

    async def get_asset(self, asset_id: str) -> Asset:
        """Fetch a single asset by identifier.

        Args:
            asset_id: Primary key of the asset (``asset_<uuid>``).

        Returns:
            The requested Asset model instance.

        Raises:
            LookupError: If no asset exists with the provided identifier.
        """

        asset = await self.repo.get_by_id(asset_id)
        if not asset:
            raise LookupError(f"Asset not found: {asset_id}")
        return asset

    async def create_asset(self, payload: AssetCreate) -> Asset:
        """Insert a new asset after validating the payload.

        Args:
            payload: Structured data describing the asset to create.

        Returns:
            The newly persisted Asset instance.

        Raises:
            ValueError: If the media type or title is invalid.
        """

        self._validate_media_type(payload.media_type)
        if not payload.title or not payload.title.strip():
            raise ValueError("Asset title cannot be empty")

        asset_id = f"asset_{uuid.uuid4().hex[:16]}"
        asset = await self.repo.create(
            asset_id=asset_id,
            media_type=payload.media_type,
            title=payload.title.strip(),
            description=payload.description,
            text_payload=payload.text_payload,
            payload_json=payload.payload_json,
            license=payload.license,
            origin_url=payload.origin_url,
        )
        return asset

    async def update_asset(self, asset_id: str, payload: AssetUpdate) -> Asset:
        """Apply partial updates to an asset record.

        Args:
            asset_id: Identifier of the asset to update.
            payload: Fields to update on the asset.

        Returns:
            Updated Asset instance.

        Raises:
            ValueError: If provided fields fail validation rules.
            LookupError: If the asset does not exist.
        """

        fields = payload.model_dump(exclude_unset=True)
        if "media_type" in fields:
            self._validate_media_type(fields["media_type"])
        if "title" in fields and fields["title"] is not None:
            if not fields["title"].strip():
                raise ValueError("Asset title cannot be empty")
            fields["title"] = fields["title"].strip()

        asset = await self.repo.update(asset_id, fields=fields)
        if not asset:
            raise LookupError(f"Asset not found: {asset_id}")
        return asset

    async def delete_asset(self, asset_id: str) -> None:
        """Delete an asset by identifier.

        Args:
            asset_id: Identifier of the asset to remove.

        Raises:
            LookupError: If the asset does not exist.
        """

        deleted = await self.repo.delete(asset_id)
        if deleted == 0:
            raise LookupError(f"Asset not found: {asset_id}")

    # ------------------------------------------------------------------
    # Embedding operations
    # ------------------------------------------------------------------
    async def upsert_embedding(
        self,
        asset_id: str,
        request: AssetEmbeddingRequest,
        *,
        embeddings_service: EmbeddingsService | None = None,
    ) -> AssetEmbeddingInfo:
        """Store or generate an embedding for the specified asset.

        Args:
            asset_id: Identifier of the asset to augment.
            request: Embedding payload including vector or generation flags.
            embeddings_service: Optional override for the embedding provider.

        Returns:
            Metadata describing the persisted embedding.

        Raises:
            LookupError: If the asset is unknown.
            ValueError: If embedding dimensions are inconsistent or missing
                required text.
            RuntimeError: If the embedding generation fails downstream.
        """

        asset = await self.repo.get_by_id(asset_id)
        if not asset:
            raise LookupError(f"Asset not found: {asset_id}")

        vector: list[float] | None = None
        generated = False

        if request.embedding is not None:
            if len(request.embedding) != request.dim:
                raise ValueError(
                    f"Embedding length {len(request.embedding)} does not match dim {request.dim}"
                )
            vector = request.embedding
        else:
            text = request.text
            if request.use_asset_text:
                text = text or asset.text_payload
            if not text or not text.strip():
                raise ValueError("Cannot generate embedding without text payload")
            service = embeddings_service or EmbeddingsService(
                model=request.model,
            )
            try:
                vector = await service.embed_async(text)
            except Exception as exc:  # pragma: no cover - network failure handling
                raise RuntimeError(f"Failed to generate embedding: {exc}") from exc
            generated = True
            if len(vector) != request.dim:
                raise ValueError(
                    f"Generated embedding length {len(vector)} does not match dim {request.dim}"
                )

        await self.repo.upsert_embedding(
            asset_id=asset_id,
            embedding=vector,
            model=request.model,
            dim=request.dim,
            metadata=request.metadata,
        )
        info = await self.repo.get_embedding_info(asset_id)
        if not info:
            raise RuntimeError("Embedding upsert failed to persist metadata")
        return AssetEmbeddingInfo(
            asset_id=info["asset_id"],
            embedding_model=info["embedding_model"],
            embedding_dim=info["embedding_dim"],
            embedding_ts=info["embedding_ts"],
            metadata=info.get("metadata"),
            vector_length=len(vector) if vector is not None else None,
            generated=generated,
        )

    async def delete_embedding(self, asset_id: str) -> None:
        """Remove an embedding associated with an asset.

        Args:
            asset_id: Identifier of the asset whose embedding should be deleted.

        Raises:
            LookupError: If the embedding does not exist.
        """

        deleted = await self.repo.delete_embedding(asset_id)
        if deleted == 0:
            raise LookupError(f"Embedding not found for asset: {asset_id}")

    async def get_embedding_info(self, asset_id: str) -> AssetEmbeddingInfo:
        """Return metadata describing an asset's stored embedding.

        Args:
            asset_id: Identifier of the asset whose embedding metadata to read.

        Returns:
            AssetEmbeddingInfo describing the embedding configuration.

        Raises:
            LookupError: If the asset or its embedding cannot be found.
        """

        asset = await self.repo.get_by_id(asset_id)
        if not asset:
            raise LookupError(f"Asset not found: {asset_id}")
        info = await self.repo.get_embedding_info(asset_id)
        if not info:
            raise LookupError(f"Embedding not found for asset: {asset_id}")
        return AssetEmbeddingInfo(
            asset_id=info["asset_id"],
            embedding_model=info["embedding_model"],
            embedding_dim=info["embedding_dim"],
            embedding_ts=info["embedding_ts"],
            metadata=info.get("metadata"),
            vector_length=None,
            generated=False,
        )

    async def search_by_embedding(
        self,
        request: AssetSearchRequest,
    ) -> AssetSearchResponse:
        """Execute semantic similarity search over asset embeddings.

        Args:
            request: Search parameters containing the query vector and limits.

        Returns:
            AssetSearchResponse containing ranked matches.

        Raises:
            ValueError: If the provided embedding does not match ``dim``.
        """

        if len(request.embedding) != request.dim:
            raise ValueError(
                f"Embedding length {len(request.embedding)} does not match dim {request.dim}"
            )
        results = await self.repo.search_by_embedding(
            embedding=request.embedding,
            model=request.model,
            dim=request.dim,
            limit=request.top_k,
        )
        hits = [
            AssetSearchHit(
                asset_id=asset.asset_id,
                media_type=asset.media_type,
                title=asset.title,
                description=asset.description,
                origin_url=asset.origin_url,
                score=score,
            )
            for asset, score in results
        ]
        return AssetSearchResponse(total=len(hits), items=hits)

    # ------------------------------------------------------------------
    # Linking operations
    # ------------------------------------------------------------------
    async def link_asset(
        self,
        asset_id: str,
        request: AssetLinkRequest,
    ) -> AssetLinkResponse:
        """Create verse associations for an asset.

        Args:
            asset_id: Identifier of the asset to link.
            request: Details of the verses and relation metadata.

        Returns:
            AssetLinkResponse summarizing additions vs. duplicates.

        Raises:
            ValueError: If no verse IDs are provided or verses are missing.
            LookupError: If the asset does not exist.
        """

        if not request.verse_ids:
            raise ValueError("verse_ids cannot be empty")

        asset = await self.repo.get_by_id(asset_id)
        if not asset:
            raise LookupError(f"Asset not found: {asset_id}")

        unique_ids = list(dict.fromkeys(request.verse_ids))
        async with self.conn.transaction():
            rows = await self.conn.fetch(
                "SELECT verse_id FROM verse WHERE verse_id = ANY($1::text[])",
                unique_ids,
            )
            found = {row["verse_id"] for row in rows}
            missing = sorted(set(unique_ids) - found)
            if missing:
                raise ValueError(f"Verses not found: {', '.join(missing)}")

            added = await self.repo.create_links(
                asset_id=asset_id,
                verse_ids=unique_ids,
                relation=request.relation,
                chunk_id=request.chunk_id,
            )
        skipped = len(unique_ids) - added
        return AssetLinkResponse(asset_id=asset_id, added=added, skipped=skipped)

    async def unlink_asset(
        self,
        asset_id: str,
        verse_ids: Sequence[str] | None = None,
    ) -> AssetUnlinkResponse:
        """Remove verse associations for an asset.

        Args:
            asset_id: Identifier of the asset to unlink.
            verse_ids: Optional specific verse identifiers to remove. ``None``
                removes all links.

        Returns:
            AssetUnlinkResponse describing how many links were removed.

        Raises:
            LookupError: If the asset does not exist.
        """

        asset = await self.repo.get_by_id(asset_id)
        if not asset:
            raise LookupError(f"Asset not found: {asset_id}")
        removed = await self.repo.delete_links(asset_id=asset_id, verse_ids=verse_ids)
        return AssetUnlinkResponse(asset_id=asset_id, removed=removed)

    async def list_links(self, asset_id: str) -> tuple[int, list[AssetVerseLink]]:
        """Fetch all verse links for an asset.

        Args:
            asset_id: Identifier of the asset whose links to enumerate.

        Returns:
            Tuple of ``(total_count, links)`` where ``links`` are the enriched
            AssetVerseLink records.

        Raises:
            LookupError: If the asset does not exist.
        """

        asset = await self.repo.get_by_id(asset_id)
        if not asset:
            raise LookupError(f"Asset not found: {asset_id}")
        links = await self.repo.fetch_links(asset_id)
        return len(links), links

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_media_type(media_type: str) -> None:
        """Ensure the provided media type is recognised."""

        allowed = {"image", "audio", "video", "text", "document"}

        if not media_type:
            raise ValueError("media_type must be provided")

        if media_type in allowed:
            return

        # Accept standard MIME types by validating the top-level category.
        if "/" in media_type:
            top_level = media_type.split("/", 1)[0]
            if top_level in allowed:
                return

        raise ValueError(
            f"Invalid media_type. Must be a recognised media category ({', '.join(sorted(allowed))})"
        )
