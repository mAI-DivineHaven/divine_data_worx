"""Reusable Dagster ops for DivineHaven orchestration."""

from __future__ import annotations

from datetime import timedelta

from dagster import In, Nothing, OpExecutionContext, Out, RetryPolicy, op

from .resources import ManifestValidationReport

DEFAULT_RETRY_POLICY = RetryPolicy(max_retries=3, delay=timedelta(minutes=5))


@op(
    required_resource_keys={"manifest_service"},
    retry_policy=DEFAULT_RETRY_POLICY,
    out=Out(dict, description="Validated manifest payload"),
)
def manifest_validation_op(context: OpExecutionContext) -> dict[str, object]:
    """Validate the manifest.json file and emit the manifest payload."""

    report: ManifestValidationReport = context.resources.manifest_service.validate_manifest()
    for name, result in report.results.items():
        if result.passed:
            context.log.info("Validation '%s' passed", name)
        else:
            context.log.error(
                "Validation '%s' failed: errors=%s warnings=%s",
                name,
                result.errors,
                result.warnings,
            )
    for warning in report.warnings:
        context.log.warning("Manifest validation warning: %s", warning)

    return report.manifest.model_dump(mode="python")


@op(
    required_resource_keys={"embedding_service"},
    retry_policy=DEFAULT_RETRY_POLICY,
    ins={"manifest_payload": In(dict)},
    out=Out(dict, description="Manifest payload propagated to downstream ops"),
)
def embedding_generation_op(
    context: OpExecutionContext, manifest_payload: dict[str, object]
) -> dict[str, object]:
    """Trigger embedding generation for the supplied manifest."""

    response = context.resources.embedding_service.generate_embeddings(manifest_payload)
    context.log.info("Embedding generation triggered: %s", response)
    return manifest_payload


@op(
    required_resource_keys={"postgres"},
    retry_policy=DEFAULT_RETRY_POLICY,
    ins={"manifest_payload": In(dict)},
    out=Out(dict, description="Manifest payload propagated to graph seeding"),
)
def pgvector_index_build_op(
    context: OpExecutionContext, manifest_payload: dict[str, object]
) -> dict[str, object]:
    """Build pgvector indexes based on manifest-provided embedding configuration."""

    executed = context.resources.postgres.build_vector_indexes(manifest_payload)
    for stmt in executed:
        context.log.info("Executed Postgres statement: %s", stmt)
    return manifest_payload


@op(
    required_resource_keys={"neo4j"},
    retry_policy=DEFAULT_RETRY_POLICY,
    ins={"manifest_payload": In(dict)},
    out=Out(Nothing),
)
def neo4j_seeding_op(context: OpExecutionContext, manifest_payload: dict[str, object]) -> None:
    """Seed the Neo4j graph based on manifest metadata."""

    created = context.resources.neo4j.seed_graph(manifest_payload)
    context.log.info("Seeded %s translations into Neo4j", created)
