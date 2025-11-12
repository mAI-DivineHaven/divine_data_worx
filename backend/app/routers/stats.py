"""
Statistics and monitoring router.

Provides endpoints for database health metrics and embedding pipeline status.
Useful for monitoring data ingestion progress and identifying gaps in coverage.

Example Usage:
    ```bash
    # Get embedding coverage statistics
    curl http://localhost:8000/v1/stats/embedding_coverage
    ```

Response Format:
    ```json
    [
      {
        "translation_code": "ESV",
        "verses": 31102,
        "embedded": 31102,
        "missing": 0
      },
      {
        "translation_code": "NIV",
        "verses": 31103,
        "embedded": 28450,
        "missing": 2653
      }
    ]
    ```
"""

import asyncpg
from fastapi import APIRouter, Depends

from ..db.postgres_async import get_pg
from ..models import EmbeddingCoverage
from ..services.stats import StatsService

router = APIRouter(prefix="/stats", tags=["stats"])


def get_stats_service(conn: asyncpg.Connection = Depends(get_pg)) -> StatsService:
    """Dependency provider for StatsService."""
    return StatsService(conn)


@router.get("/embedding_coverage", response_model=list[EmbeddingCoverage])
async def embedding_coverage(
    service: StatsService = Depends(get_stats_service),
) -> list[dict]:
    """
    Get embedding coverage statistics per translation.

    Returns the number of verses with/without embeddings for each translation,
    enabling monitoring of the embedding pipeline progress and identification
    of gaps in semantic search coverage.

    Args:
        service: StatsService dependency (injected)

    Returns:
        List of coverage stats per translation, ordered alphabetically

    Response Format:
        ```json
        [
          {
            "translation_code": "ESV",
            "verses": 31102,
            "embedded": 31102,
            "missing": 0
          }
        ]
        ```

    Fields:
        - translation_code: Translation identifier
        - verses: Total verse count
        - embedded: Verses with embeddings
        - missing: Verses without embeddings (verses - embedded)

    Use Cases:
        - Monitor embedding pipeline progress
        - Identify incomplete translations
        - Validate data ingestion completeness
        - Track ETL pipeline status

    Performance:
        Uses LEFT JOIN with GROUP BY aggregation.
        Typical query time: 50-200ms depending on database size.
    """
    coverage = await service.embedding_coverage()
    return [c.dict() for c in coverage]
