from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# ============================================================================
# Domain Models - Core biblical text entities
# ============================================================================


class Verse(BaseModel):
    """Complete verse record with full metadata."""

    verse_id: str
    translation_code: str
    book_number: int
    chapter_number: int
    verse_number: int
    suffix: str
    text: str


class VerseLite(BaseModel):
    """Lightweight verse record with minimal data."""

    verse_id: str
    text: str


class Book(BaseModel):
    """Book metadata for a specific translation."""

    translation_code: str
    book_number: int
    name: str
    testament: Literal["Old", "New"]


class Chapter(BaseModel):
    """Chapter reference within a book and translation."""

    translation_code: str
    book_number: int
    chapter_number: int


class Translation(BaseModel):
    """Translation metadata."""

    translation_code: str
    language: str | None = None
    format: str | None = None


# ============================================================================
# Search Models - Query and response schemas
# ============================================================================


class FTSQuery(BaseModel):
    """Full-text search query parameters."""

    q: str
    translation: str | None = None
    limit: int = Field(50, ge=1, le=500)
    offset: int = Field(0, ge=0)

    @field_validator("q")
    @classmethod
    def validate_query(cls, value: str) -> str:
        """Ensure the search query contains non-whitespace characters."""
        if not value or not value.strip():
            raise ValueError("Query must not be empty")
        return value


class VectorQuery(BaseModel):
    """Semantic vector search query parameters."""

    embedding: list[float]
    model: str = "embeddinggemma"
    dim: int = 768
    translation: str | None = None
    top_k: int = Field(50, ge=1, le=500)

    @model_validator(mode="after")
    def validate_embedding_length(self) -> "VectorQuery":
        """Ensure the embedding length matches the expected dimensionality."""

        if len(self.embedding) != self.dim:
            raise ValueError(
                f"embedding length {len(self.embedding)} does not match dim {self.dim}"
            )
        return self


class HybridQuery(BaseModel):
    """Hybrid search query combining FTS and vector search."""

    q: str | None = None
    embedding: list[float] | None = None
    model: str = "embeddinggemma"
    dim: int = 768
    vector_k: int = Field(50, ge=1, le=500)
    fts_k: int = Field(50, ge=1, le=500)
    k_rrf: int = Field(60, ge=1, le=1000)
    top_k: int = Field(50, ge=1, le=500)
    translation: str | None = None


class RetrievalQuery(HybridQuery):
    """Extended hybrid query enabling graph expansion toggles."""

    include_parallels: bool = True
    parallel_limit: int = Field(3, ge=0, le=50)


class SearchHit(BaseModel):
    """Single search result with relevance score."""

    verse_id: str
    text: str
    score: float


class SearchResponse(BaseModel):
    """Search results container."""

    total: int
    items: list[SearchHit]


class GraphExpansion(BaseModel):
    """Graph expansion for a verse including parallel renditions."""

    verse_id: str
    cvk: str | None = None
    renditions: list["Rendition"] = Field(default_factory=list)


class GraphExpansionInfo(BaseModel):
    """Metadata describing graph expansion parameters."""

    enabled: bool
    max_per_hit: int
    weight: float
    applied: bool


class FusionInfo(BaseModel):
    """Fusion metadata returned with retrieval responses."""

    method: str
    k: int
    vector_k: int
    fts_k: int
    graph_expansion: GraphExpansionInfo


class RetrievalHit(BaseModel):
    """Combined retrieval result with optional graph expansion."""

    hit: SearchHit
    parallels: GraphExpansion | None = None


class RetrievalResponse(BaseModel):
    """Response envelope for orchestrated retrieval."""

    total: int
    items: list[RetrievalHit]
    fusion: FusionInfo


# ============================================================================
# Graph Models - Cross-translation relationships
# ============================================================================


class Rendition(BaseModel):
    """Single verse rendition in a specific translation."""

    verse_id: str
    translation: str
    reference: str
    text: str


class CanonicalVerse(BaseModel):
    """Canonical verse metadata for graph neighbourhood responses."""

    cvk: str
    book_number: int
    chapter_number: int
    verse_number: int
    suffix: str


class GraphNeighborhood(BaseModel):
    """Canonical verse node and its neighbouring renditions."""

    canonical: CanonicalVerse
    renditions: list[Rendition]


class ParallelsResponse(BaseModel):
    """Parallel verses across translations."""

    cvk: str
    renditions: list[Rendition]


# ============================================================================
# Translation Stats Models
# ============================================================================


class TranslationEmbeddingStats(BaseModel):
    """Embedding coverage metrics per translation.

    Attributes:
        translation_code: Translation identifier (e.g., KJV, NIV)
        verses: Total verse count
        embedded: Verses with embeddings generated
        missing: Verses without embeddings (verses - embedded)

    Use Cases:
        - Monitor embedding pipeline progress
        - Identify incomplete translations
        - Validate data ingestion completeness
        - Track ETL status for semantic search readiness
    """

    translation_code: str
    verses: int
    embedded: int
    missing: int


# Alias for backward compatibility
EmbeddingCoverage = TranslationEmbeddingStats


# ============================================================================
# Asset Models - Multi-modal resource management
# ============================================================================


class Asset(BaseModel):
    """Complete asset record with metadata and payload references."""

    asset_id: str
    media_type: str | None = None
    title: str | None = None
    description: str | None = None
    text_payload: str | None = None
    payload_json: dict[str, Any] | None = None
    license: str | None = None
    origin_url: str | None = None
    created_at: datetime | None = None


class AssetCreate(BaseModel):
    """Payload for creating a new asset record."""

    media_type: str
    title: str
    description: str | None = None
    text_payload: str | None = None
    payload_json: dict[str, Any] | None = None
    license: str | None = None
    origin_url: str | None = None


class AssetUpdate(BaseModel):
    """Partial update payload for asset records."""

    media_type: str | None = None
    title: str | None = None
    description: str | None = None
    text_payload: str | None = None
    payload_json: dict[str, Any] | None = None
    license: str | None = None
    origin_url: str | None = None


class AssetListResponse(BaseModel):
    """Paginated list response containing asset summaries."""

    total: int
    items: list[Asset]


class AssetEmbeddingRequest(BaseModel):
    """Request payload for providing or generating asset embeddings."""

    embedding: list[float] | None = None
    text: str | None = None
    use_asset_text: bool = False
    model: str = "embeddinggemma"
    dim: int = 768
    metadata: dict[str, Any] | None = None


class AssetEmbeddingInfo(BaseModel):
    """Metadata response describing the stored asset embedding."""

    asset_id: str
    embedding_model: str
    embedding_dim: int
    embedding_ts: datetime
    metadata: dict[str, Any] | None = None
    vector_length: int | None = None
    generated: bool = False


class AssetSearchRequest(BaseModel):
    """Semantic search request over stored asset embeddings."""

    embedding: list[float]
    model: str = "embeddinggemma"
    dim: int = 768
    top_k: int = Field(10, ge=1, le=200)


class AssetSearchHit(BaseModel):
    """Single asset search result with similarity score."""

    asset_id: str
    media_type: str | None = None
    title: str | None = None
    description: str | None = None
    origin_url: str | None = None
    score: float


class AssetSearchResponse(BaseModel):
    """Container for semantic asset search results."""

    total: int
    items: list[AssetSearchHit]


class AssetLinkRequest(BaseModel):
    """Request payload for linking an asset to verse references."""

    verse_ids: list[str] = Field(..., min_length=1)
    relation: str = Field("related", min_length=1, max_length=64)
    chunk_id: str | None = None


class AssetLinkResponse(BaseModel):
    """Response summarizing asset to verse link creation."""

    asset_id: str
    added: int
    skipped: int


class AssetUnlinkResponse(BaseModel):
    """Response summarizing link deletions for an asset."""

    asset_id: str
    removed: int


class AssetVerseLink(BaseModel):
    """Detailed representation of an asset-to-verse relationship."""

    verse_id: str
    relation: str | None = None
    chunk_id: str | None = None
    translation_code: str
    book_number: int
    chapter_number: int
    verse_number: int
    suffix: str
    text: str
    reference: str


class AssetLinkListResponse(BaseModel):
    """Collection response for asset verse links."""

    asset_id: str
    total: int
    items: list[AssetVerseLink]


# ============================================================================
# Batch Processing Models - High-volume verse operations
# ============================================================================


class CanonicalVerseRef(BaseModel):
    """Canonical verse reference without translation."""

    book_number: int
    chapter_number: int
    verse_number: int
    suffix: str = ""


class BatchVerseRequest(BaseModel):
    """Request payload for batch verse retrieval."""

    verse_ids: list[str] = Field(..., min_length=1, max_length=500)


class BatchVerseResponse(BaseModel):
    """Response containing requested verses and missing IDs."""

    verses: list[Verse]
    missing_ids: list[str]


class TranslationVerseEntry(BaseModel):
    """Single verse entry in a translation comparison."""

    translation_code: str
    verse_id: str | None = None
    text: str | None = None


class TranslationComparisonItem(BaseModel):
    """Comparison of a single verse across multiple translations."""

    reference: CanonicalVerseRef
    translations: list[TranslationVerseEntry]
    missing_translations: list[str]


class TranslationComparisonRequest(BaseModel):
    """Request for comparing verses across translations."""

    references: list[CanonicalVerseRef] = Field(..., min_length=1, max_length=200)
    translations: list[str] = Field(..., min_length=1, max_length=25)


class TranslationComparisonResponse(BaseModel):
    """Response containing translation comparisons."""

    items: list[TranslationComparisonItem]


class EmbeddingVector(BaseModel):
    """Embedding vector with metadata."""

    verse_id: str
    embedding: list[float]
    embedding_model: str
    embedding_dim: int


class EmbeddingLookupRequest(BaseModel):
    """Request for looking up verse embeddings."""

    verse_ids: list[str] = Field(..., min_length=1, max_length=500)
    model: str = "embeddinggemma"


class EmbeddingLookupResponse(BaseModel):
    """Response containing verse embeddings."""

    results: list[EmbeddingVector]
    missing_ids: list[str]


# ============================================================================
# Session Memory Models - Conversational context and citation trails
# ============================================================================


class SessionCitationBase(BaseModel):
    """Common fields shared by session citation payloads.

    Attributes:
        source_type: Domain-specific type that classifies the citation source
            (for example ``"verse"`` or ``"asset"``).
        source_id: Identifier for the source resource. The format is determined
            by ``source_type`` and is opaque to this layer.
        snippet: Optional text fragment or description highlighting the cited
            content.
        metadata: Optional arbitrary metadata supplied by agent clients to
            describe structured citation details.
    """

    source_type: str
    source_id: str
    snippet: str | None = None
    metadata: dict[str, Any] | None = None

    @field_validator("source_type", "source_id")
    @classmethod
    def _validate_non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("value must not be empty")
        return value


class SessionCitationCreate(SessionCitationBase):
    """Payload describing a citation reference attached to a message."""

    pass


class SessionCitation(SessionCitationBase):
    """Persisted citation record linked to a session message.

    Attributes:
        citation_id: Primary key for the citation record.
        message_id: Identifier of the parent :class:`SessionMessage`.
        created_at: Timestamp indicating when the citation was created.
    """

    citation_id: int
    message_id: int
    created_at: datetime


class SessionMessageAppendRequest(BaseModel):
    """Client payload for appending content to a conversation session.

    Attributes:
        role: The conversational role associated with the message. Supports the
            canonical OpenAI schema roles (``system``, ``user``, ``assistant``,
            ``tool``).
        content: Natural language content for the message.
        metadata: Optional arbitrary metadata attached to the message.
        citations: Optional collection of citation descriptors that should be
            linked to the message when persisted.
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    metadata: dict[str, Any] | None = None
    citations: list[SessionCitationCreate] = Field(default_factory=list)

    @field_validator("content")
    @classmethod
    def _validate_content(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("content must not be empty")
        return value


class SessionMessage(SessionMessageAppendRequest):
    """Message enriched with persistence metadata and attached citations.

    Attributes:
        message_id: Primary key for the message.
        session_id: Identifier of the session that owns the message.
        created_at: Timestamp for when the message was persisted.
        citations: Materialised citation instances for the message.
    """

    message_id: int
    session_id: str
    created_at: datetime
    citations: list[SessionCitation] = Field(default_factory=list)


class SessionMessageCreate(SessionMessageAppendRequest):
    """Internal payload for persisting a new session message.

    Attributes:
        session_id: Identifier of the session that the message belongs to.
    """

    session_id: str


class SessionMessageUpdate(BaseModel):
    """Mutation payload for session messages.

    Attributes:
        role: Optional conversational role override.
        content: Optional replacement text for the message body.
        metadata: Optional metadata override for the message.
        citations: Optional replacement citation descriptors for the message.
    """

    role: Literal["system", "user", "assistant", "tool"] | None = None
    content: str | None = None
    metadata: dict[str, Any] | None = None
    citations: list[SessionCitationCreate] | None = None

    @field_validator("content")
    @classmethod
    def _validate_content(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("content must not be empty")
        return value


class SessionContextResponse(BaseModel):
    """Paginated response containing a slice of session memory.

    Attributes:
        session_id: Identifier for the session whose context is returned.
        total: Total number of messages stored for the session.
        limit: Maximum number of messages included in this response.
        offset: Offset applied when retrieving the messages.
        items: Ordered list of :class:`SessionMessage` instances in the page.
    """

    session_id: str
    total: int
    limit: int
    offset: int
    items: list[SessionMessage]


# ============================================================================
# Chunk Search Models - Sliding window embeddings
# ============================================================================


class ChunkSearchQuery(BaseModel):
    """Chunk-based semantic search query parameters."""

    embedding: list[float] = Field(
        ..., min_length=768, max_length=4096, description="Query embedding vector"
    )
    model: str = Field(
        "embeddinggemma", description="Embedding model identifier used to generate the vector"
    )
    dim: int = Field(768, ge=1, le=4096, description="Embedding dimensionality")
    translation: str | None = Field(
        None, description="Restrict results to a specific translation code"
    )
    book_number: int | None = Field(
        None, ge=1, le=200, description="Restrict results to a canonical book number"
    )
    testament: Literal["Old", "New"] | None = Field(
        None, description="Restrict results by testament"
    )
    window_size: int | None = Field(None, ge=1, le=500, description="Filter by chunk window size")
    top_k: int = Field(50, ge=1, le=500, description="Maximum number of results to return")
    offset: int = Field(0, ge=0, description="Pagination offset for large result sets")
    include_context: bool = Field(
        False, description="Include concatenated verse context before and after the chunk"
    )

    @field_validator("embedding")
    @classmethod
    def validate_embedding_length(cls, v: list[float], info: Any) -> list[float]:
        """Ensure the embedding vector length matches the declared dimension."""

        dim = info.data.get("dim", 768)
        if len(v) != dim:
            raise ValueError(f"Embedding length {len(v)} does not match dim={dim}")
        return v


class ChunkHit(BaseModel):
    """Single chunk search result with verse range metadata."""

    chunk_id: str
    translation_code: str
    book_number: int
    chapter_start: int
    verse_start: int
    chapter_end: int
    verse_end: int
    text: str
    score: float = Field(..., ge=0.0, le=1.0, description="Cosine similarity score")
    window_size: int | None = Field(None, description="Size of the chunking window")
    stride: int | None = Field(None, description="Stride used when generating the chunk")
    context_before: str | None = Field(
        None, description="Concatenated verses before the chunk if context is requested"
    )
    context_after: str | None = Field(
        None, description="Concatenated verses after the chunk if context is requested"
    )


class ChunkSearchResponse(BaseModel):
    """Container for chunk search results and execution metadata."""

    total: int = Field(..., ge=0, description="Total matching chunks for the given filters")
    items: list[ChunkHit]
    query_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata describing how the query was executed (limit, offset, clipping)",
    )


# ============================================================================
# Analytics Models - Query telemetry and usage metrics
# ============================================================================


class ModeCount(BaseModel):
    """Search mode distribution within a time window."""

    mode: str | None
    count: int
    percentage: float


class TopQuery(BaseModel):
    """Frequently executed query with recent activity timestamp."""

    query: str
    count: int
    last_seen: datetime


class QueryCounts(BaseModel):
    """Aggregate query metrics for a time window."""

    total: int
    unique_users: int
    average_latency_ms: float | None
    mode_breakdown: list[ModeCount]
    top_queries: list[TopQuery]


class TrendPoint(BaseModel):
    """Time-series bucket representing query volume."""

    bucket_start: datetime
    bucket_end: datetime
    count: int


class QueryTrends(BaseModel):
    """Query trend time-series for analytics dashboards."""

    interval: Literal["hour", "day"]
    points: list[TrendPoint]


class TranslationUsage(BaseModel):
    """Usage metric for a translation within the time window."""

    translation_code: str | None
    count: int
    percentage: float


class BookUsage(BaseModel):
    """Usage metric summarising which books appear in top results."""

    book_number: int
    book_name: str
    count: int
    percentage: float


class UsageStats(BaseModel):
    """Aggregated usage statistics for translations and books."""

    translations: list[TranslationUsage]
    books: list[BookUsage]


class AnalyticsOverview(BaseModel):
    """Composite analytics payload used by analytics endpoints."""

    window_start: datetime
    window_end: datetime
    query_counts: QueryCounts
    trends: QueryTrends
    usage: UsageStats
