"""Dagster schedules for DivineHaven orchestration jobs."""

from __future__ import annotations

from dagster import DefaultScheduleStatus, ScheduleDefinition

from .jobs import (
    embedding_generation_job,
    full_data_refresh_job,
    manifest_validation_job,
    neo4j_seeding_job,
    pgvector_index_job,
)

manifest_validation_schedule = ScheduleDefinition(
    name="manifest_validation_hourly",
    cron_schedule="0 * * * *",
    job=manifest_validation_job,
    execution_timezone="UTC",
    default_status=DefaultScheduleStatus.RUNNING,
)


embedding_generation_schedule = ScheduleDefinition(
    name="embedding_generation_daily",
    cron_schedule="0 2 * * *",
    job=embedding_generation_job,
    execution_timezone="UTC",
    default_status=DefaultScheduleStatus.RUNNING,
)


pgvector_index_schedule = ScheduleDefinition(
    name="pgvector_index_weekly",
    cron_schedule="30 3 * * 1",
    job=pgvector_index_job,
    execution_timezone="UTC",
    default_status=DefaultScheduleStatus.RUNNING,
)


neo4j_seeding_schedule = ScheduleDefinition(
    name="neo4j_seeding_daily",
    cron_schedule="0 4 * * *",
    job=neo4j_seeding_job,
    execution_timezone="UTC",
    default_status=DefaultScheduleStatus.RUNNING,
)


full_data_refresh_schedule = ScheduleDefinition(
    name="full_data_refresh_weekly",
    cron_schedule="0 5 * * 1",
    job=full_data_refresh_job,
    execution_timezone="UTC",
    default_status=DefaultScheduleStatus.STOPPED,
)
