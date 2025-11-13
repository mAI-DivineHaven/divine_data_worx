"""Dagster definitions for DivineHaven orchestration."""

from __future__ import annotations

from dagster import Definitions

from .jobs import (
    embedding_generation_job,
    full_data_refresh_job,
    manifest_validation_job,
    neo4j_seeding_job,
    pgvector_index_job,
)
from .resources import build_resource_instances
from .schedules import (
    embedding_generation_schedule,
    full_data_refresh_schedule,
    manifest_validation_schedule,
    neo4j_seeding_schedule,
    pgvector_index_schedule,
)

resource_instances = build_resource_instances()


defs = Definitions(
    jobs=[
        manifest_validation_job,
        embedding_generation_job,
        pgvector_index_job,
        neo4j_seeding_job,
        full_data_refresh_job,
    ],
    schedules=[
        manifest_validation_schedule,
        embedding_generation_schedule,
        pgvector_index_schedule,
        neo4j_seeding_schedule,
        full_data_refresh_schedule,
    ],
    resources=resource_instances,
)

__all__ = ["defs"]
