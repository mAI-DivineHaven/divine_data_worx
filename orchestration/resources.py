"""Dagster resource definitions aligned with Docker deployment defaults."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from dagster import ConfigurableResource, InitResourceContext

from backend.validation import (
    ManifestMetadata,
    ValidationResult,
    collect_verse_metrics,
    validate_embedding_completeness,
    validate_graph_edge_integrity,
    validate_verse_coverage,
)
from manifest_cli import Manifest, load_manifest


@dataclass
class ManifestValidationReport:
    """Container returned by :class:`ManifestServiceResource.validate_manifest`."""

    manifest: Manifest
    results: dict[str, ValidationResult]
    warnings: list[str]

    def as_json(self) -> str:
        """Serialize the validation report for logging or dashboards."""

        payload = {
            "manifest": self.manifest.model_dump(mode="json"),
            "results": {
                name: {
                    "passed": result.passed,
                    "errors": result.errors,
                    "warnings": result.warnings,
                }
                for name, result in self.results.items()
            },
            "warnings": list(self.warnings),
        }
        return json.dumps(payload, sort_keys=True, indent=2)


class ManifestServiceResource(ConfigurableResource):
    """Loads and validates a manifest.json file using existing CLI helpers."""

    manifest_path: str = os.getenv("MANIFEST_PATH", "/app/manifest.json")
    corpus_dir: str = os.getenv("CORPUS_DIR", "/app/unified_json_bibles")

    def _load_metadata(self, manifest: Manifest) -> ManifestMetadata:
        data = manifest.model_dump(mode="python")
        return ManifestMetadata.from_dict(data)

    def load_manifest(self) -> Manifest:
        path = Path(self.manifest_path)
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found at {path}")
        return load_manifest(path)

    def validate_manifest(self) -> ManifestValidationReport:
        manifest = self.load_manifest()
        metadata = self._load_metadata(manifest)
        corpus_path = Path(self.corpus_dir)
        metrics, warnings = collect_verse_metrics(metadata, corpus_path)

        validations: dict[str, ValidationResult] = {}
        validations["verse_coverage"] = validate_verse_coverage(metadata, metrics)
        validations["embedding_completeness"] = validate_embedding_completeness(metadata, metrics)
        validations["graph_edge_integrity"] = validate_graph_edge_integrity(metadata, metrics)

        return ManifestValidationReport(manifest=manifest, results=validations, warnings=warnings)


class EmbeddingServiceResource(ConfigurableResource):
    """Lightweight HTTP client wrapper around the embedding generation endpoint."""

    endpoint_url: str = os.getenv("EMBEDDING_ENDPOINT", "http://backend:8000/api/embed/run")
    timeout: float = float(os.getenv("EMBEDDING_TIMEOUT", "30"))

    def generate_embeddings(self, manifest_payload: Mapping[str, object]) -> dict[str, object]:
        """Trigger embedding generation via HTTP.

        The actual embedding API may not exist in all environments. The resource
        therefore performs a best-effort POST request and returns metadata useful
        for downstream logging. Any network failures should surface as Dagster
        retries so operators can investigate.
        """

        import httpx  # imported lazily to keep orchestration import light

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                self.endpoint_url,
                json={"manifest": manifest_payload},
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            data = response.json()
        return {
            "status_code": response.status_code,
            "response": data,
        }


class PostgresResource(ConfigurableResource):
    """Runs pgvector index build statements against Postgres."""

    dsn: str = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg://postgres:Fr00pzPlz@db:5432/divinehaven",
    )
    schema: str = os.getenv("PGVECTOR_SCHEMA", "public")

    def _normalized_dsn(self) -> str:
        # Replace SQLAlchemy-style prefixes with psycopg-compatible scheme.
        if "+" in self.dsn:
            driver, remainder = self.dsn.split("+", maxsplit=1)
            if remainder.startswith("psycopg://"):
                return f"{driver}://{remainder.split('://', maxsplit=1)[1]}"
        return self.dsn

    def build_vector_indexes(
        self,
        manifest_payload: Mapping[str, object],
        statements: Iterable[str] | None = None,
    ) -> list[str]:
        """Create pgvector indexes guided by the manifest."""

        plan = list(statements or [])
        if not plan:
            embedding = manifest_payload.get("embedding_recipe", {}) or {}
            dim = int(embedding.get("embedding_dim", 768))
            plan.append(
                f"ALTER TABLE {self.schema}.embeddings " f"ALTER COLUMN vector TYPE vector({dim});"
            )
            plan.append(
                f"CREATE INDEX IF NOT EXISTS embeddings_vector_idx "
                f"ON {self.schema}.embeddings "
                f"USING ivfflat (vector vector_cosine_ops) WITH (lists = 2048);"
            )
            plan.append(f"ANALYZE {self.schema}.embeddings;")

        import psycopg

        normalized = self._normalized_dsn()
        executed: list[str] = []
        with psycopg.connect(normalized) as conn:
            conn.autocommit = True
            with conn.cursor() as cursor:
                for stmt in plan:
                    cursor.execute(stmt)
                    executed.append(stmt)
        return executed


class Neo4jResource(ConfigurableResource):
    """Seeds Neo4j with graph relationships derived from manifest metadata."""

    uri: str = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
    user: str = os.getenv("NEO4J_USER", "neo4j")
    password: str = os.getenv("NEO4J_PASSWORD", "password")

    def seed_graph(self, manifest_payload: Mapping[str, object]) -> int:
        """Upsert verse relationships and metadata into Neo4j."""

        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        created_count = 0
        query = """
        UNWIND $translations AS translation
        MERGE (t:Translation {code: translation})
        SET t.updated_at = timestamp()
        """
        translations = manifest_payload.get("translation_set", [])
        with driver.session() as session:
            session.execute_write(lambda tx: tx.run(query, translations=translations))
            created_count = len(translations)
        driver.close()
        return created_count


def build_resource_instances(context: InitResourceContext | None = None) -> dict[str, object]:
    """Return instantiated resources for Dagster Definitions."""

    return {
        "manifest_service": ManifestServiceResource(),
        "embedding_service": EmbeddingServiceResource(),
        "postgres": PostgresResource(),
        "neo4j": Neo4jResource(),
    }
