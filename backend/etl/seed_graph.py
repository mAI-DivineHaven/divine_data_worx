"""
Neo4j Graph Seeding Script for DivineHaven

Seeds the Neo4j knowledge graph with biblical text data from PostgreSQL.
Creates Book, Chapter, and Verse nodes with relationships and cross-translation links.

This ETL script demonstrates async PostgreSQL streaming with synchronous Neo4j operations,
efficiently processing hundreds of thousands of verses without memory exhaustion.

Environment Variables:
    DATABASE_URL: PostgreSQL connection string
    NEO4J_URI: Neo4j bolt URI (default: bolt://localhost:7687)
    NEO4J_USER: Neo4j username (default: neo4j)
    NEO4J_PASSWORD: Neo4j password (default: password)
    MANIFEST_JSON: Optional path to manifest.json
    GRAPH_BATCH_SIZE: Verses per batch (default: 5000)
    GRAPH_LINK_MODE: Linking strategy - "per-batch" or "post" (default: per-batch)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from opentelemetry import trace
from prometheus_client import Counter, Histogram, start_http_server

# Ensure repository root is on the Python path when executed as a script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neo4j_client import Neo4jClient  # noqa: E402
from pg_client import PgClient  # noqa: E402

from backend.app.config import Settings  # noqa: E402
from backend.app.utils.logging import configure_logging, get_logger  # noqa: E402
from backend.app.utils.observability import configure_tracing  # noqa: E402

# Load environment variables from .env file
load_dotenv()

settings = Settings()

configure_logging(settings.LOG_LEVEL, service_name=f"{settings.OTEL_SERVICE_NAME}-etl")
logger = get_logger(__name__)

tracer_provider = configure_tracing(
    settings,
    service_name=f"{settings.OTEL_SERVICE_NAME}-etl",
    service_version="0.1.0",
)

tracer = trace.get_tracer("backend.etl.seed_graph")

# Configuration from environment with sensible defaults
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:Fr00pzPlz@localhost:5432/divinehaven",
)
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
MANIFEST_JSON = os.getenv("MANIFEST_JSON", "./manifest.json")
BATCH_SIZE = int(os.getenv("GRAPH_BATCH_SIZE", "5000"))
LINK_MODE = os.getenv("GRAPH_LINK_MODE", "per-batch")
ETL_METRICS_ENABLED = os.getenv("ETL_METRICS_ENABLED", "true").lower() == "true"
ETL_METRICS_PORT = int(os.getenv("ETL_METRICS_PORT", "9001"))

_METRICS_NAMESPACE = settings.METRICS_NAMESPACE

ETL_ROWS_TOTAL = Counter(
    "etl_rows_processed_total",
    "Total rows processed by the Neo4j seeding ETL",
    namespace=_METRICS_NAMESPACE,
)
ETL_BATCH_LATENCY = Histogram(
    "etl_batch_latency_seconds",
    "Time spent merging a batch into Neo4j",
    namespace=_METRICS_NAMESPACE,
    labelnames=("link_mode",),
    buckets=(0.25, 0.5, 1, 2, 5, 10, 20, 40, 60),
)
ETL_BATCHES_TOTAL = Counter(
    "etl_batches_total",
    "Number of batches processed",
    namespace=_METRICS_NAMESPACE,
    labelnames=("link_mode",),
)

if ETL_METRICS_ENABLED:
    start_http_server(ETL_METRICS_PORT)
    logger.info(
        "metrics_server_started",
        extra={"port": ETL_METRICS_PORT, "namespace": _METRICS_NAMESPACE},
    )


def read_manifest(path: str) -> dict | None:
    """
    Load and parse the manifest JSON file if it exists.

    The manifest provides metadata about the embedding pipeline run,
    including model information, pipeline version, and data sources.

    Args:
        path: File system path to manifest.json

    Returns:
        Parsed manifest dictionary with pipeline metadata, or None if
        file doesn't exist or fails to parse

    Example manifest structure:
        ```json
        {
            "run_id": "2024-01-15T10:30:00Z",
            "embedding_recipe": {
                "embedding_model": "embeddinggemma",
                "embedding_dim": 768
            }
        }
        ```
    """
    p = Path(path)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


async def run_async(batch_size: int, link_mode: str) -> None:
    """
    Main async pipeline for seeding Neo4j graph from PostgreSQL.

    Pipeline Modes:
        per-batch: Link PARALLEL_TO edges after each batch (real-time, slower)
        post: Link all PARALLEL_TO edges after verse upsert (faster, higher memory)

    Args:
        batch_size: Number of verses to process per batch
        link_mode: Linking strategy ("per-batch" or "post")

    Performance Notes:
        - Uses async PostgreSQL with server-side cursors for streaming
        - Neo4j operations run in thread pool to keep event loop responsive
        - Expected throughput: 5,000-10,000 verses/second depending on hardware
    """
    with tracer.start_as_current_span(
        "seed_graph.run",
        attributes={"batch_size": batch_size, "link_mode": link_mode},
    ):
        manifest = read_manifest(MANIFEST_JSON)
        if manifest:
            run_id = manifest.get("run_id", "?")
            model = manifest.get("embedding_recipe", {}).get("embedding_model", "?")
            logger.info(
                "manifest_loaded",
                extra={"run_id": run_id, "embedding_model": model},
            )

        # Initialize async Postgres client
        async with PgClient(DATABASE_URL) as pg:
            # Initialize sync Neo4j client (will run in thread pool)
            g = Neo4jClient(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
            loop = asyncio.get_running_loop()

            logger.info("initializing_constraints")
            await loop.run_in_executor(None, g.init_constraints)

            total_rows = 0
            touched_cvks: set[str] = set()

            logger.info(
                "starting_upserts",
                extra={"batch_size": batch_size, "link_mode": link_mode},
            )
            batch_count = 0

            async for batch in pg.iter_verses(batch_size=batch_size):
                batch_count += 1
                with tracer.start_as_current_span(
                    "seed_graph.merge_batch",
                    attributes={"batch_size": len(batch)},
                ):
                    with ETL_BATCH_LATENCY.labels(link_mode=link_mode).time():
                        cvks = await loop.run_in_executor(None, g.merge_batch, batch)

                batch_rows = len(batch)
                total_rows += batch_rows
                ETL_ROWS_TOTAL.inc(batch_rows)
                ETL_BATCHES_TOTAL.labels(link_mode=link_mode).inc()

                logger.info(
                    "batch_processed",
                    extra={
                        "batch_number": batch_count,
                        "rows": batch_rows,
                        "total_rows": total_rows,
                    },
                )

                if link_mode == "per-batch":
                    with tracer.start_as_current_span(
                        "seed_graph.link_parallels",
                        attributes={"cvk_count": len(cvks)},
                    ):
                        await loop.run_in_executor(None, g.link_parallels_for_cvks, cvks)
                else:
                    touched_cvks.update(cvks)

            if link_mode == "post" and touched_cvks:
                logger.info(
                    "linking_post_batches",
                    extra={"cvk_count": len(touched_cvks)},
                )
                with tracer.start_as_current_span(
                    "seed_graph.link_parallels_post",
                    attributes={"cvk_count": len(touched_cvks)},
                ):
                    await loop.run_in_executor(None, g.link_parallels_for_cvks, touched_cvks)

            # Clean shutdown
            g.close()

        logger.info("seed_complete", extra={"total_rows": total_rows})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Seed Neo4j knowledge graph from PostgreSQL biblical text data"
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Number of verses per batch (default: 5000)",
    )
    ap.add_argument(
        "--link-mode",
        choices=["per-batch", "post"],
        default=LINK_MODE,
        help="When to create PARALLEL_TO edges (default: per-batch)",
    )
    args = ap.parse_args()

    asyncio.run(run_async(args.batch_size, args.link_mode))
