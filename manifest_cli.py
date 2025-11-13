#!/usr/bin/env python3
"""
DivineHaven — Embedding Run Manifest & DB Tools

A single-file, production-friendly CLI that:
  • Defines the Manifest schema (Pydantic v2)
  • Validates a manifest.json
  • Emits Postgres/Timescale + pgvector/pgvectorscale DDL
  • Plans vector + FTS indexes based on the manifest
  • Registers the manifest into Postgres (run_manifest table)
  • Ingests "universal" Bible JSONs (idempotent upsert)
  • Generates embeddings with Ollama (async batched / sync fallback)
  • Verifies ingestion quality (check-ingest)
  • Migrates embeddings table to composite PK (migrate-embeddings-pk)

Ollama embedding endpoint:
  - Prefer POST /api/embed (accepts text or list of text; returns "embeddings")
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import os
import re
import sys
import textwrap
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import typer
from pydantic import BaseModel, Field, ValidationError, field_validator

from backend.validation import (
    ManifestMetadata as ValidationManifest,
)
from backend.validation import (
    collect_verse_metrics,
    validate_embedding_completeness,
    validate_graph_edge_integrity,
    validate_verse_coverage,
)

try:
    import psycopg
except Exception:  # optional
    psycopg = None  # type: ignore

try:
    import aiohttp  # optional for async HTTP
except Exception:  # pragma: no cover
    aiohttp = None  # type: ignore

app = typer.Typer(add_completion=False, help="DivineHaven — Manifest & DB CLI")


# ---------------------------
# Models (Pydantic v2)
# ---------------------------
class Chunking(BaseModel):
    granularity: Literal["verse", "pericope", "sliding"] = "sliding"
    window_size: int | None = Field(default=None, ge=1)
    stride: int | None = Field(default=None, ge=1)

    @field_validator("stride")
    @classmethod
    def _stride_only_for_sliding(cls, v, info):
        data = info.data
        if data.get("granularity") == "sliding" and (v is None):
            raise ValueError("stride is required when granularity='sliding'")
        return v


class Preprocess(BaseModel):
    lower: bool = False
    strip_punct: bool = False
    collapse_ws: bool = True


class Filters(BaseModel):
    books_included: list[int] | None | Literal["all"] = "all"
    translations_included: list[str] | None = None


class EmbeddingRecipe(BaseModel):
    embedding_model: str
    embedding_dim: int = Field(ge=1)
    pooling: Literal["mean", "max", "cls"] = "mean"
    normalize: bool = True
    tokenizer: str | None = None
    truncation_strategy: Literal["none", "sliding", "head", "tail"] = "sliding"
    chunking: Chunking
    preprocess: Preprocess = Preprocess()
    filters: Filters = Filters()


class VectorScaleHNSW(BaseModel):
    type: Literal["hnsw"] = "hnsw"
    params: dict[str, int] = Field(
        default_factory=lambda: {"m": 32, "ef_construction": 200, "ef_search": 64}
    )


class VectorScaleIVF(BaseModel):
    type: Literal["ivf"] = "ivf"
    params: dict[str, int] = Field(default_factory=lambda: {"lists": 2048, "probes": 8})


VectorScaleCfg = VectorScaleHNSW | VectorScaleIVF


class FTSOptions(BaseModel):
    dictionary: Literal["simple", "english"] = "simple"
    stopwords: str | None = None


class GraphExpansionOptions(BaseModel):
    enabled: bool = False
    max_per_hit: int = Field(default=3, ge=0, le=50)
    weight: float = Field(default=0.0, ge=0.0)


class FusionOptions(BaseModel):
    method: Literal["rrf", "weighted_sum"] = "rrf"
    k: int = 60
    weight_vector: dict[str, float] | None = None  # if weighted_sum
    graph_expansion: GraphExpansionOptions = Field(default_factory=GraphExpansionOptions)


class HybridOptions(BaseModel):
    vector_k: int = 50
    fts_k: int = 50
    fusion: FusionOptions = FusionOptions()
    fts: FTSOptions = FTSOptions()


class IndexPlan(BaseModel):
    vectorscale: VectorScaleCfg = Field(default_factory=VectorScaleHNSW)
    hybrid: HybridOptions = Field(default_factory=HybridOptions)


class BatchEntry(BaseModel):
    input: str
    mapping: str
    sha256_input: str | None = None
    sha256_mapping: str | None = None
    item_count: int | None = None
    skipped_count: int | None = None


class Manifest(BaseModel):
    run_id: str
    run_ts: datetime
    pipeline_version: str
    source_version: str
    translation_set: list[str]
    languages: list[str]
    license: str | None = None
    operator: str | None = None

    embedding_recipe: EmbeddingRecipe
    index_plan: IndexPlan

    batches: list[BatchEntry]


# ---------------------------
# Utilities
# ---------------------------
def load_manifest(path: Path) -> Manifest:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return Manifest.model_validate(data)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _lang_label(lang: str | None) -> int:
    """Stable smallint mapping for labels[0]. Extend as needed."""
    if not lang:
        return 0
    table = {"en": 1, "es": 2, "el": 3, "he": 4, "la": 5, "fr": 6, "de": 7}
    return table.get(lang.lower(), 0)


def _testament_label(testament: str | None) -> int:
    """Stable smallint mapping for labels[1]."""
    if not testament:
        return 0
    t = testament.strip().lower()
    return 1 if t.startswith("old") else (2 if t.startswith("new") else 0)


def _vec_literal(vec: list[float], precision: int = 6) -> str:
    fmt = f"{{:.{precision}f}}"
    return "[" + ",".join(fmt.format(x) for x in vec) + "]"


# ---------------------------
# SQL Emitters
# ---------------------------
DDL_HEADER = """
-- DivineHaven baseline tables (Postgres 17 + TimescaleDB compatible)
-- Note: Enable TimescaleDB/vectors as appropriate in your DB:
--   CREATE EXTENSION IF NOT EXISTS timescaledb;
--   CREATE EXTENSION IF NOT EXISTS vector;       -- base pgvector
--   CREATE EXTENSION IF NOT EXISTS vectorscale;  -- advanced ANN (if installed)
""".strip()

CORE_DDL = """
-- Translations
CREATE TABLE IF NOT EXISTS translation (
  translation_code TEXT PRIMARY KEY,
  language        TEXT,
  format          TEXT NOT NULL,
  source_version  TEXT NOT NULL,
  created_at      TIMESTAMPTZ DEFAULT now()
);

-- Books
CREATE TABLE IF NOT EXISTS book (
  translation_code TEXT REFERENCES translation(translation_code),
  book_number      INT,
  name             TEXT NOT NULL,
  testament        TEXT CHECK (testament IN ('Old','New')),
  PRIMARY KEY (translation_code, book_number)
);

-- Chapters
CREATE TABLE IF NOT EXISTS chapter (
  translation_code TEXT,
  book_number      INT,
  chapter_number   INT,
  PRIMARY KEY (translation_code, book_number, chapter_number),
  FOREIGN KEY (translation_code, book_number) REFERENCES book(translation_code, book_number)
);

-- Verses (PK includes suffix; verse_id includes optional '|suffix')
CREATE TABLE IF NOT EXISTS verse (
  translation_code TEXT NOT NULL,
  book_number      INT  NOT NULL,
  chapter_number   INT  NOT NULL,
  verse_number     INT  NOT NULL,
  suffix           TEXT NOT NULL DEFAULT '',
  verse_id         TEXT GENERATED ALWAYS AS (
    translation_code || ':' || book_number || ':' || chapter_number || ':' ||
    verse_number || CASE WHEN suffix = '' THEN '' ELSE '|' || suffix END
  ) STORED,
  text             TEXT NOT NULL,
  words_json       JSONB,
  source_version   TEXT NOT NULL,
  ingest_run_id    TEXT,
  checksum         TEXT,
  created_at       TIMESTAMPTZ DEFAULT now(),
  PRIMARY KEY (translation_code, book_number, chapter_number, verse_number, suffix),
  UNIQUE (verse_id)
);

-- FTS index (idempotent)
CREATE INDEX IF NOT EXISTS verse_text_gin
  ON verse USING GIN (to_tsvector('simple', text));

-- Embeddings: verse-level (composite PK per model/dim)
CREATE TABLE IF NOT EXISTS verse_embedding (
  verse_id        TEXT NOT NULL REFERENCES verse(verse_id) ON DELETE CASCADE,
  embedding       vector(768),
  embedding_model TEXT NOT NULL,
  embedding_dim   INT  NOT NULL,
  embedding_ts    TIMESTAMPTZ DEFAULT now(),
  labels          SMALLINT[],
  metadata        JSONB,
  PRIMARY KEY (verse_id, embedding_model, embedding_dim)
);

-- Embeddings: sliding chunks (768-D)
CREATE TABLE IF NOT EXISTS chunk_embedding (
  chunk_id         TEXT PRIMARY KEY,
  translation_code TEXT NOT NULL,
  book_number      INT  NOT NULL,
  chapter_start    INT  NOT NULL,
  verse_start      INT  NOT NULL,
  chapter_end      INT  NOT NULL,
  verse_end        INT  NOT NULL,
  text             TEXT NOT NULL,
  embedding        vector(768),
  embedding_model  TEXT NOT NULL,
  embedding_dim    INT  NOT NULL,
  window_size      INT,
  stride           INT,
  embedding_ts     TIMESTAMPTZ DEFAULT now(),
  labels           SMALLINT[],
  metadata         JSONB
);

-- Assets + embeddings (768-D)
CREATE TABLE IF NOT EXISTS asset (
  asset_id     TEXT PRIMARY KEY,
  media_type   TEXT,
  title        TEXT,
  description  TEXT,
  text_payload TEXT,
  payload_json JSONB,
  license      TEXT,
  origin_url   TEXT,
  created_at   TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS asset_embedding (
  asset_id        TEXT PRIMARY KEY REFERENCES asset(asset_id),
  embedding       vector(768),
  embedding_model TEXT NOT NULL,
  embedding_dim   INT  NOT NULL,
  embedding_ts    TIMESTAMPTZ DEFAULT now(),
  metadata        JSONB
);

-- Linking assets to verses/chunks
CREATE TABLE IF NOT EXISTS asset_link (
  asset_id  TEXT REFERENCES asset(asset_id) ON DELETE CASCADE,
  verse_id  TEXT REFERENCES verse(verse_id) ON DELETE CASCADE,
  chunk_id  TEXT,
  relation  TEXT,
  PRIMARY KEY (asset_id, verse_id, chunk_id)
);

-- Run manifest registry
CREATE TABLE IF NOT EXISTS run_manifest (
  run_id     TEXT PRIMARY KEY,
  created_at TIMESTAMPTZ DEFAULT now(),
  manifest   JSONB NOT NULL
);

-- Conversational session memory
CREATE TABLE IF NOT EXISTS session_memory (
  message_id   BIGSERIAL PRIMARY KEY,
  session_id   TEXT NOT NULL,
  role         TEXT NOT NULL CHECK (role IN ('system','user','assistant','tool')),
  content      TEXT NOT NULL,
  metadata     JSONB DEFAULT '{}'::jsonb,
  created_at   TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS session_memory_session_idx
  ON session_memory (session_id, message_id);

-- Citation trails attached to session messages
CREATE TABLE IF NOT EXISTS session_citation (
  citation_id BIGSERIAL PRIMARY KEY,
  message_id  BIGINT NOT NULL REFERENCES session_memory(message_id) ON DELETE CASCADE,
  source_type TEXT NOT NULL,
  source_id   TEXT NOT NULL,
  snippet     TEXT,
  metadata    JSONB DEFAULT '{}'::jsonb,
  created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS session_citation_message_idx
  ON session_citation (message_id);
""".strip()


@dataclass
class VectorIndexPlanner:
    cfg: VectorScaleCfg

    def emit(
        self,
        table: str,
        column: str,
        with_labels: bool = True,
        index_name: str | None = None,
    ) -> str:
        """
        Return SQL to create an ANN index for a (table, column) pair.

        Primary: vectorscale DiskANN (cosine)
        Fallbacks (commented): pgvector HNSW / IVFFlat examples
        """
        idx = index_name or f"{table}_{column}_ann_idx"
        cols = (
            f"{column} vector_cosine_ops, labels" if with_labels else f"{column} vector_cosine_ops"
        )

        # Primary: Vectorscale DiskANN (cosine)
        diskann_sql = f"""-- DiskANN (vectorscale, cosine); memory_optimized for high recall
CREATE INDEX IF NOT EXISTS {idx}
  ON {table} USING diskann ({cols})
  WITH (storage_layout = 'memory_optimized');"""

        # Fallbacks (commented) — safe docs only
        if isinstance(self.cfg, VectorScaleHNSW):
            m = self.cfg.params.get("m", 32)
            efc = self.cfg.params.get("ef_construction", 200)
            efs = self.cfg.params.get("ef_search", 64)
            hnsw_sql = f"""
-- pgvector HNSW fallback (cosine):
-- CREATE INDEX IF NOT EXISTS {idx}
--   ON {table} USING hnsw ({column} vector_cosine_ops)
--   WITH (m = {m}, ef_construction = {efc});
-- At query time:
--   SET hnsw.ef_search = {efs};"""
            return diskann_sql + "\n" + hnsw_sql

        lists = self.cfg.params.get("lists", 2048) if isinstance(self.cfg, VectorScaleIVF) else 2048
        probes = self.cfg.params.get("probes", 8) if isinstance(self.cfg, VectorScaleIVF) else 8
        ivf_sql = f"""
-- pgvector IVFFlat fallback (cosine):
-- CREATE INDEX IF NOT EXISTS {idx}
--   ON {table} USING ivfflat ({column} vector_cosine_ops)
--   WITH (lists = {lists});
-- At query time:
--   SET ivfflat.probes = {probes};"""
        return diskann_sql + "\n" + ivf_sql


# ---------------------------
# CLI: Validate & Summary
# ---------------------------
@app.command()
def validate(manifest_path: Path = typer.Argument(..., exists=True, readable=True)) -> None:
    """Validate a manifest and print a compact summary."""
    try:
        manifest = load_manifest(manifest_path)
    except ValidationError as e:
        typer.secho("Manifest validation failed:\n", fg=typer.colors.RED, bold=True)
        typer.echo(e)
        raise typer.Exit(code=1)

    typer.secho("Manifest looks good ✔", fg=typer.colors.GREEN, bold=True)
    typer.echo(
        textwrap.dedent(
            """
            run_id:          {run_id}
            run_ts:          {run_ts}
            pipeline_version:{pipeline_version}
            source_version:  {source_version}
            translations:    {translations}
            languages:       {languages}
            batches:         {batches}

            embedding_model: {embedding_model}
            embedding_dim:   {embedding_dim}
            chunking:        {chunking}
            fusion:          {fusion}
            vectorscale:     {vectorscale}
            """
        )
        .strip()
        .format(
            run_id=manifest.run_id,
            run_ts=manifest.run_ts.isoformat(),
            pipeline_version=manifest.pipeline_version,
            source_version=manifest.source_version,
            translations=", ".join(manifest.translation_set),
            languages=", ".join(manifest.languages),
            batches=len(manifest.batches),
            embedding_model=manifest.embedding_recipe.embedding_model,
            embedding_dim=manifest.embedding_recipe.embedding_dim,
            chunking=manifest.embedding_recipe.chunking.model_dump(),
            fusion=manifest.index_plan.hybrid.fusion.model_dump(),
            vectorscale=manifest.index_plan.vectorscale.model_dump(),
        )
    )


@app.command("summary")
def summary_cmd(manifest_path: Path = typer.Argument(..., exists=True, readable=True)) -> None:
    """Print a richer summary with per-batch stats and inferred totals."""
    manifest = load_manifest(manifest_path)

    total_items = sum((b.item_count or 0) for b in manifest.batches)
    total_skipped = sum((b.skipped_count or 0) for b in manifest.batches)

    typer.secho("Embedding Run Summary", fg=typer.colors.CYAN, bold=True)
    typer.echo(
        f"run_id: {manifest.run_id} | batches: {len(manifest.batches)} | items: {total_items} | skipped: {total_skipped}"
    )
    typer.echo("")
    for i, b in enumerate(manifest.batches):
        typer.echo(
            f"[{i:04d}] input={b.input} mapping={b.mapping} items={b.item_count or '?'} skipped={b.skipped_count or 0}"
        )


# ---------------------------
# CLI: Emit DDL
# ---------------------------
@app.command("emit-ddl")
def emit_ddl(target: Literal["postgres"] = typer.Argument("postgres")) -> None:
    """Emit baseline DDL for Postgres/Timescale + pgvector/pgvectorscale."""
    if target != "postgres":
        raise typer.Exit(code=2)

    print(DDL_HEADER)
    print()
    print(CORE_DDL)


# ---------------------------
# CLI: Plan Indexes
# ---------------------------
@app.command("plan-indexes")
def plan_indexes(
    manifest_path: Path = typer.Argument(..., exists=True, readable=True),
    target: Literal["postgres"] = typer.Argument("postgres"),
) -> None:
    """Emit ANN / FTS index statements as guided by the manifest.index_plan."""
    if target != "postgres":
        raise typer.Exit(code=2)

    m = load_manifest(manifest_path)
    planner = VectorIndexPlanner(m.index_plan.vectorscale)

    fts_dict = m.index_plan.hybrid.fts.dictionary

    statements = [
        planner.emit("verse_embedding", "embedding", with_labels=True),
        planner.emit("chunk_embedding", "embedding", with_labels=True),
        planner.emit("asset_embedding", "embedding", with_labels=False),
        f"""-- FTS index (idempotent). To rebuild, drop then create concurrently.
CREATE INDEX IF NOT EXISTS verse_text_gin
  ON verse USING GIN (to_tsvector('{fts_dict}', text));
-- Rebuild (optional):
-- DROP INDEX CONCURRENTLY IF EXISTS verse_text_gin;
-- CREATE INDEX CONCURRENTLY IF NOT EXISTS verse_text_gin ON verse USING GIN (to_tsvector('{fts_dict}', text));""",
    ]

    typer.secho("-- Index Plan (apply in psql)", fg=typer.colors.CYAN, bold=True)
    print("\n\n".join(statements))


# ---------------------------
# CLI: Register manifest JSON
# ---------------------------
@app.command("register-run")
def register_run(
    manifest_path: Path = typer.Argument(..., exists=True, readable=True),
    dsn: str = typer.Argument(
        ..., help="Postgres DSN, e.g. postgresql://user:pass@host:5432/divinehaven"
    ),
) -> None:
    """Create run_manifest table (if missing) and upsert this manifest JSON into it."""
    if psycopg is None:
        typer.secho(
            "psycopg is not installed. `pip install psycopg[binary]`",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=3)

    manifest = load_manifest(manifest_path)

    sql = """
    CREATE TABLE IF NOT EXISTS run_manifest (
      run_id     TEXT PRIMARY KEY,
      created_at TIMESTAMPTZ DEFAULT now(),
      manifest   JSONB NOT NULL
    );

    INSERT INTO run_manifest (run_id, manifest)
    VALUES (%s, %s)
    ON CONFLICT (run_id) DO UPDATE SET manifest = EXCLUDED.manifest;
    """
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("BEGIN;")
            cur.execute(sql.split(";", 1)[0] + ";")  # create table
            cur.execute(
                sql.split(";", 1)[1] + ";",
                (manifest.run_id, json.dumps(manifest.model_dump(mode="json"))),
            )
            cur.execute("COMMIT;")

    typer.secho(
        f"Registered run_id={manifest.run_id} into run_manifest ✔",
        fg=typer.colors.GREEN,
        bold=True,
    )


# ---------------------------
# CLI: Checksums for batches
# ---------------------------
@app.command("checksum-batches")
def checksum_batches(
    manifest_path: Path = typer.Argument(..., exists=True, readable=True),
    base_dir: Path | None = typer.Option(None, help="Base directory to resolve batch paths"),
    write_back: bool = typer.Option(
        False, help="If true, write sha256 values back to manifest.json"
    ),
) -> None:
    """Compute sha256 for each batch input/mapping file; optionally write back into the manifest.json."""
    path = manifest_path
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    manifest = Manifest.model_validate(raw)

    root = base_dir or manifest_path.parent
    updated = False
    for b in manifest.batches:
        for attr in ("input", "mapping"):
            p = root / getattr(b, attr)
            if p.exists():
                digest = sha256_file(p)
                if attr == "input":
                    b.sha256_input = digest
                else:
                    b.sha256_mapping = digest
                updated = True
                typer.echo(f"{attr} {p} -> {digest}")
            else:
                typer.secho(f"WARN: missing file {p}", fg=typer.colors.YELLOW)

    if write_back and updated:
        with path.open("w", encoding="utf-8") as f:
            json.dump(manifest.model_dump(mode="json"), f, indent=2, ensure_ascii=False)
        typer.secho("Manifest updated with checksums ✔", fg=typer.colors.GREEN)


# ---------------------------
# CLI: Pre-ingest validations
# ---------------------------
@app.command("pre-ingest-validate")
def pre_ingest_validate(
    manifest_path: Path = typer.Option(
        Path("manifest.json"),
        "--manifest",
        "-m",
        help="Path to manifest.json describing the ingest run",
    ),
    corpus_dir: Path = typer.Option(
        Path("unified_json_bibles"),
        "--corpus-dir",
        "-c",
        help="Directory containing translation JSON payloads",
    ),
    base_translation: str | None = typer.Option(
        None,
        "--base-translation",
        help="Translation code to use as canonical baseline when checking graph edges",
    ),
    allow_empty_ratio: float = typer.Option(
        0.0,
        "--allow-empty-ratio",
        min=0.0,
        max=1.0,
        help="Maximum ratio of verses allowed to have empty text before failing embedding checks",
    ),
    max_graph_gap: float = typer.Option(
        0.02,
        "--max-graph-gap",
        min=0.0,
        max=1.0,
        help="Maximum tolerated ratio of missing canonical keys when validating graph edges",
    ),
) -> None:
    """Run static validations against the corpus before ingestion."""

    try:
        manifest = ValidationManifest.from_path(manifest_path)
    except Exception as exc:  # pragma: no cover - surfaced via CLI
        typer.secho(f"Failed to load manifest: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=2)

    metrics, warnings = collect_verse_metrics(manifest, corpus_dir)

    for warning in warnings:
        typer.secho(f"WARN: {warning}", fg=typer.colors.YELLOW)

    results = [
        validate_verse_coverage(manifest, metrics),
        validate_embedding_completeness(
            manifest,
            metrics,
            allow_empty_ratio=allow_empty_ratio,
        ),
        validate_graph_edge_integrity(
            manifest,
            metrics,
            base_translation=base_translation,
            max_missing_ratio=max_graph_gap,
        ),
    ]

    success = True
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        color = typer.colors.GREEN if result.passed else typer.colors.RED
        label = result.name.replace("_", " ").title()
        typer.secho(f"[{status}] {label}", fg=color)
        for warn in result.warnings:
            typer.secho(f"  - WARN: {warn}", fg=typer.colors.YELLOW)
        for err in result.errors:
            typer.secho(f"  - ERROR: {err}", fg=typer.colors.RED)
        success = success and result.passed

    raise typer.Exit(code=0 if success else 1)


# ---------------------------
# CLI: Ingest universal JSON (robust de-dup)
# ---------------------------
@app.command("ingest")
def ingest_bible(
    json_path: Path = typer.Option(..., "--json", help="Path to universal JSON file"),
    translation: str = typer.Option(..., "--translation", help="Translation code, e.g., NIV / TNK"),
    source_version: str = typer.Option(
        ..., "--source-version", help="e.g., divine_haven.universal_v1"
    ),
    dsn: str = typer.Option(
        "postgresql://postgres:postgres@localhost:5432/divinehaven",
        "--dsn",
        help="Postgres DSN",
    ),
    language: str | None = typer.Option(None, "--language"),
    batch_size: int = typer.Option(1000, "--batch-size", min=100, max=5000),
) -> None:
    """Ingest a universal JSON Bible into Postgres (idempotent upserts) with robust verse de-duplication."""
    if psycopg is None:
        typer.secho(
            "psycopg is not installed. `pip install psycopg[binary]`",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=3)

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not {"format", "translation", "books"}.issubset(data):
        typer.secho(
            "Input JSON does not match the universal schema (missing keys).",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=4)

    db_translation_code = translation
    file_translation_name = data.get("translation")
    lang = data.get("language") or language
    file_format = data.get("format") or file_translation_name or source_version

    def _alpha(n: int) -> str:
        s = ""
        while n > 0:
            n -= 1
            s = chr(97 + (n % 26)) + s
            n //= 26
        return s

    books = data["books"]

    with psycopg.connect(dsn) as conn:
        conn.execute("SET application_name = 'divinehaven-ingest';")
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO translation (translation_code, language, format, source_version)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (translation_code) DO UPDATE
                SET language = EXCLUDED.language,
                    format   = EXCLUDED.format,
                    source_version = EXCLUDED.source_version;
                """,
                (
                    db_translation_code,
                    lang,
                    file_format,
                    source_version,
                ),
            )

            book_rows = [
                (db_translation_code, b["number"], b["name"], b["testament"]) for b in books
            ]
            if book_rows:
                psycopg_rows = ",".join(["(%s,%s,%s,%s)"] * len(book_rows))
                cur.execute(
                    f"""
                    INSERT INTO book (translation_code, book_number, name, testament)
                    VALUES {psycopg_rows}
                    ON CONFLICT (translation_code, book_number) DO UPDATE
                    SET name = EXCLUDED.name, testament = EXCLUDED.testament;
                    """,
                    list(itertools.chain.from_iterable(book_rows)),
                )

            chapter_rows: list[tuple[str, int, int]] = []
            verse_rows: list[tuple[str, int, int, int, str, str, str, str]] = []

            for b in books:
                bnum = b["number"]
                for ch in b["chapters"]:
                    cnum = ch["number"]
                    chapter_rows.append((db_translation_code, bnum, cnum))

                    used_suffixes: dict[tuple[int, int, int], set] = {}
                    for v in ch["verses"]:
                        vnum = v["number"]
                        text_val = v.get("text", "")
                        sfx_in = (v.get("suffix") or "").strip()

                        key = (bnum, cnum, vnum)
                        if key not in used_suffixes:
                            used_suffixes[key] = set()

                        sfx = sfx_in
                        if sfx == "" and "" in used_suffixes[key]:
                            n = 1
                            cand = _alpha(n)
                            while cand in used_suffixes[key]:
                                n += 1
                                cand = _alpha(n)
                            sfx = cand
                        elif sfx != "" and sfx in used_suffixes[key]:
                            n = 1
                            cand = _alpha(n)
                            while cand in used_suffixes[key]:
                                n += 1
                                cand = _alpha(n)
                            sfx = cand

                        used_suffixes[key].add(sfx)

                        ck = hashlib.sha256(
                            f"{db_translation_code}|{bnum}|{cnum}|{vnum}|{sfx}|{text_val}".encode()
                        ).hexdigest()

                        verse_rows.append(
                            (
                                db_translation_code,
                                bnum,
                                cnum,
                                vnum,
                                text_val,
                                sfx,
                                file_format,
                                ck,
                            )
                        )

            if verse_rows:
                uniq: dict[
                    tuple[str, int, int, int, str],
                    tuple[str, int, int, int, str, str, str, str],
                ] = {}
                for row in verse_rows:
                    pk = (row[0], row[1], row[2], row[3], row[5])
                    uniq[pk] = row
                verse_rows = list(uniq.values())
                verse_rows.sort(key=lambda r: (r[0], r[1], r[2], r[3], r[5]))

            if chapter_rows:
                chapter_rows = sorted(set(chapter_rows))
                for i in range(0, len(chapter_rows), batch_size):
                    batch = chapter_rows[i : i + batch_size]
                    values = ",".join(["(%s,%s,%s)"] * len(batch))
                    cur.execute(
                        f"""
                        INSERT INTO chapter (translation_code, book_number, chapter_number)
                        VALUES {values}
                        ON CONFLICT (translation_code, book_number, chapter_number) DO NOTHING;
                        """,
                        list(itertools.chain.from_iterable(batch)),
                    )

            if verse_rows:
                for i in range(0, len(verse_rows), batch_size):
                    batch = verse_rows[i : i + batch_size]
                    values = ",".join(["(%s,%s,%s,%s,%s,%s,%s,%s)"] * len(batch))
                    cur.execute(
                        f"""
                        INSERT INTO verse (
                          translation_code, book_number, chapter_number, verse_number,
                          text, suffix, source_version, checksum
                        ) VALUES {values}
                        ON CONFLICT (translation_code, book_number, chapter_number, verse_number, suffix)
                        DO UPDATE SET text = EXCLUDED.text,
                                      suffix = EXCLUDED.suffix,
                                      source_version = EXCLUDED.source_version,
                                      checksum = EXCLUDED.checksum;
                        """,
                        list(itertools.chain.from_iterable(batch)),
                    )

        conn.commit()

    typer.secho(
        f"Ingested translation={db_translation_code} books={len(books)} OK",
        fg=typer.colors.GREEN,
        bold=True,
    )


# ---------------------------
# ✅ CLI: Verify ingest before embeddings
# ---------------------------
@app.command("check-ingest")
def check_ingest(
    dsn: str = typer.Option(
        "postgresql://postgres:postgres@localhost:5432/divinehaven",
        "--dsn",
        help="Postgres DSN",
    )
) -> None:
    """Run sanity checks to confirm the corpus loaded correctly."""
    if psycopg is None:
        typer.secho(
            "psycopg is not installed. `pip install psycopg[binary]`",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=3)

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            typer.secho("Extensions:", fg=typer.colors.CYAN, bold=True)
            cur.execute(
                "SELECT extname, extversion FROM pg_extension WHERE extname IN ('timescaledb','vector','vectorscale') ORDER BY 1;"
            )
            for r in cur.fetchall():
                typer.echo(f"  {r[0]} {r[1]}")

            typer.secho("\nVerse counts by translation:", fg=typer.colors.CYAN, bold=True)
            cur.execute(
                """
                SELECT translation_code, COUNT(*) AS verses
                FROM verse GROUP BY 1 ORDER BY 1;
            """
            )
            for r in cur.fetchall():
                typer.echo(f"  {r[0]} -> {r[1]}")

            typer.secho("\nOrphans (should be 0):", fg=typer.colors.CYAN, bold=True)
            cur.execute(
                """
                SELECT COUNT(*) FROM verse v
                LEFT JOIN book b  ON b.translation_code=v.translation_code AND b.book_number=v.book_number
                LEFT JOIN chapter c ON c.translation_code=v.translation_code AND c.book_number=v.book_number AND c.chapter_number=v.chapter_number
                WHERE b.translation_code IS NULL OR c.translation_code IS NULL;
            """
            )
            typer.echo(f"  {cur.fetchone()[0]}")  # type: ignore

            typer.secho(
                "\nDuplicate suffix within same ref (should be 0 rows):",
                fg=typer.colors.CYAN,
                bold=True,
            )
            cur.execute(
                """
                SELECT COUNT(*) FROM (
                  SELECT translation_code, book_number, chapter_number, verse_number, suffix, COUNT(*) AS n
                  FROM verse GROUP BY 1,2,3,4,5 HAVING COUNT(*) > 1
                ) s;
            """
            )
            typer.echo(f"  {cur.fetchone()[0]}")  # type: ignore

            typer.secho("\nEmpty verse text (should be 0):", fg=typer.colors.CYAN, bold=True)
            cur.execute("SELECT COUNT(*) FROM verse WHERE text IS NULL OR text='';")
            typer.echo(f"  {cur.fetchone()[0]}")  # type: ignore

            typer.secho(
                "\nEmbedding coverage (by translation):",
                fg=typer.colors.CYAN,
                bold=True,
            )
            cur.execute(
                """
                SELECT v.translation_code,
                       COUNT(*) AS verses,
                       COUNT(e.verse_id) AS embedded,
                       COUNT(*) - COUNT(e.verse_id) AS missing
                FROM verse v
                LEFT JOIN verse_embedding e
                  ON e.verse_id=v.verse_id
                GROUP BY 1 ORDER BY 1;
            """
            )
            rows = cur.fetchall()
            for t, verses, embedded, missing in rows:
                typer.echo(f"  {t}: verses={verses} embedded={embedded} missing={missing}")

            typer.secho("\nEmbedding model/dim mix:", fg=typer.colors.CYAN, bold=True)
            cur.execute(
                """
                SELECT embedding_model, embedding_dim, COUNT(*) AS rows
                FROM verse_embedding GROUP BY 1,2 ORDER BY 1,2;
            """
            )
            for r in cur.fetchall():
                typer.echo(f"  {r[0]} dim={r[1]} rows={r[2]}")

            typer.secho("\nLabels hygiene:", fg=typer.colors.CYAN, bold=True)
            cur.execute(
                """
                SELECT COUNT(*) FILTER (WHERE labels IS NULL) AS null_labels,
                       COUNT(*) FILTER (WHERE array_length(labels,1)=3) AS labels_ok,
                       COUNT(*) FILTER (WHERE array_length(labels,1) IS DISTINCT FROM 3) AS labels_bad
                FROM verse_embedding;
            """
            )
            r = cur.fetchone()
            typer.echo(f"  null={r[0]} ok={r[1]} bad={r[2]}")  # type: ignore

    typer.secho("\nChecks complete ✔", fg=typer.colors.GREEN, bold=True)


# ---------------------------
# ✨ CLI: Embed verses with Ollama (optimized)
# ---------------------------
async def _embed_batch_async(
    session: aiohttp.ClientSession,  # type: ignore
    api_base: str,
    model: str,
    texts: list[str],
    keep_alive: str = "5m",
    truncate: bool = True,
) -> list[list[float]]:
    url = api_base.rstrip("/") + "/api/embed"
    payload = {
        "model": model,
        "input": texts,
        "keep_alive": keep_alive,
        "truncate": truncate,
    }
    async with session.post(
        url, json=payload, timeout=aiohttp.ClientTimeout(total=120)  # type: ignore
    ) as resp:
        resp.raise_for_status()
        data = await resp.json()
        return data["embeddings"]


def _embed_batch_sync(
    api_base: str,
    model: str,
    texts: list[str],
    keep_alive: str = "5m",
    truncate: bool = True,
) -> list[list[float]]:
    import urllib.error
    import urllib.request

    url = api_base.rstrip("/") + "/api/embed"
    payload = json.dumps(
        {"model": model, "input": texts, "keep_alive": keep_alive, "truncate": truncate}
    ).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read().decode("utf-8"))["embeddings"]
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"Ollama embed error {e.code}: {e.read().decode('utf-8', 'ignore')}"
        ) from e


def _batched_indices(n: int, size: int) -> list[list[int]]:
    return [list(range(i, min(i + size, n))) for i in range(0, n, size)]


def _ensure_stage_and_copy_then_merge(conn, rows: list[tuple[Any, ...]]) -> None:
    """
    rows: (verse_id, embedding_text, embedding_model, embedding_dim, labels, metadata_json)
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS verse_embedding_stage(
              verse_id TEXT,
              embedding_text TEXT,
              embedding_model TEXT,
              embedding_dim INT,
              labels SMALLINT[],
              metadata JSONB
            );
        """
        )
        with cur.copy(
            """
            COPY verse_embedding_stage
            (verse_id, embedding_text, embedding_model, embedding_dim, labels, metadata)
            FROM STDIN WITH (FORMAT csv, DELIMITER E'\t', QUOTE E'\b')
        """
        ) as cp:
            for r in rows:
                cp.write_row(r)

        cur.execute(
            """
            INSERT INTO verse_embedding
              (verse_id, embedding, embedding_model, embedding_dim, labels, metadata)
            SELECT verse_id, embedding_text::vector, embedding_model, embedding_dim, labels, metadata
            FROM verse_embedding_stage
            ON CONFLICT (verse_id, embedding_model, embedding_dim) DO UPDATE
            SET embedding     = EXCLUDED.embedding,
                labels       = COALESCE(EXCLUDED.labels, verse_embedding.labels),
                metadata     = EXCLUDED.metadata,
                embedding_ts = now();

            TRUNCATE verse_embedding_stage;
        """
        )


async def _embed_worker(
    in_q,
    out_q,
    session,
    api_base,
    model,
    keep_alive,
    embedding_dim,
    use_labels,
    precision,
):
    while True:
        item = await in_q.get()
        if item is None:
            in_q.task_done()
            break
        idxs, rows = (
            item  # rows: (verse_id, text, language, testament, book_number, translation_code)
        )
        texts = [r[1] for r in rows]
        embs = await _embed_batch_async(session, api_base, model, texts, keep_alive)
        payload = []
        ts = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        for (verse_id, _t, lang, testament, book_no, tcode), vec in zip(rows, embs):
            if len(vec) != embedding_dim:
                raise RuntimeError(f"Dim mismatch for {verse_id}: {len(vec)} != {embedding_dim}")
            labels = None
            if use_labels:
                labels = [_lang_label(lang), _testament_label(testament), int(book_no)]
            meta = {
                "provider": "ollama",
                "model": model,
                "dim": embedding_dim,
                "translation": tcode,
                "lang": lang,
                "testament": testament,
                "keep_alive": keep_alive,
                "pipeline": "embed-verses",
                "ts_utc": ts,
            }
            payload.append(
                (
                    verse_id,
                    _vec_literal(vec, precision),
                    model,
                    embedding_dim,
                    labels,
                    json.dumps(meta),
                )
            )
        await out_q.put(payload)
        in_q.task_done()


async def _run_pipeline(
    dsn,
    api_base,
    model,
    batches,
    rows_all,
    workers,
    keep_alive,
    embedding_dim,
    use_labels,
    precision,
):
    in_q = asyncio.Queue(maxsize=workers * 4)
    out_q = asyncio.Queue(maxsize=workers * 4)

    timeout = aiohttp.ClientTimeout(total=120)  # type: ignore
    connector = aiohttp.TCPConnector(limit=workers * 2, ssl=False)  # type: ignore
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:  # type: ignore
        tasks = [
            asyncio.create_task(
                _embed_worker(
                    in_q,
                    out_q,
                    session,
                    api_base,
                    model,
                    keep_alive,
                    embedding_dim,
                    use_labels,
                    precision,
                )
            )
            for _ in range(workers)
        ]

        async def producer():
            for b in batches:
                rows = [rows_all[i] for i in b]
                await in_q.put((b, rows))
            for _ in tasks:
                await in_q.put(None)

        async def writer():
            total = 0
            with psycopg.connect(dsn) as conn:  # type: ignore
                conn.execute("SET application_name = 'divinehaven-embedder';")
                pending: list[tuple[Any, ...]] = []
                while True:
                    try:
                        payload = await asyncio.wait_for(out_q.get(), timeout=5)
                    except TimeoutError:
                        payload = None
                    if payload:
                        pending.extend(payload)
                        out_q.task_done()
                    # flush in healthy chunks
                    if pending and (
                        len(pending) >= workers * 32 or (payload is None and in_q.empty())
                    ):
                        _ensure_stage_and_copy_then_merge(conn, pending)
                        conn.commit()
                        total += len(pending)
                        pending.clear()
                        typer.echo(f"Upserted {total} embeddings so far…")
                    if payload is None and in_q.empty() and out_q.empty():
                        break

        await asyncio.gather(producer(), writer(), *tasks)


@app.command("embed-verses")
def embed_verses(
    dsn: str = typer.Option(
        "postgresql://postgres:postgres@localhost:5432/divinehaven",
        "--dsn",
        help="Postgres DSN",
    ),
    model: str = typer.Option("embeddinggemma", "--model", help="Ollama embedding model"),
    embedding_dim: int = typer.Option(768, "--dim", help="Target embedding dimension"),
    api_base: str = typer.Option(
        os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        "--api-base",
        help="Ollama base URL",
    ),
    translations: list[str] | None = typer.Option(
        None, "--translation", help="Filter to these translation codes (repeatable)"
    ),
    reembed: bool = typer.Option(
        False,
        "--reembed/--no-reembed",
        help="Re-embed even if an embedding exists for this model/dim",
    ),
    use_labels: bool = typer.Option(
        True,
        "--labels/--no-labels",
        help="Compute smallint[] labels [lang, testament, book_number]",
    ),
    select_limit: int | None = typer.Option(
        None, "--max-rows", help="Cap the number of verses to process"
    ),
    fetch_page: int = typer.Option(5000, "--fetch-page", help="Rows to fetch per DB page"),
    batch_size: int = typer.Option(32, "--batch", help="Texts per /api/embed call"),
    workers: int = typer.Option(6, "--workers", help="Concurrent HTTP batches (aiohttp)"),
    keep_alive: str = typer.Option("5m", "--keep-alive", help="Keep model in memory on Ollama"),
    write_mode: str = typer.Option(
        "staging-copy",
        "--write-mode",
        help="staging-copy (COPY to stage then MERGE) | direct (legacy batched INSERT)",
    ),
    precision: int = typer.Option(
        6, "--precision", help="Float precision for vector literal strings"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Don't write to DB; just count"),
) -> None:
    """
    Generate embeddings for verses via Ollama /api/embed (batched).
    Upserts into verse_embedding with composite key (verse_id, embedding_model, embedding_dim).
    """
    if psycopg is None:
        typer.secho(
            "psycopg is not installed. `pip install psycopg[binary]`",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=3)

    # Build selection SQL (exclude already-embedded for this model+dim)
    sel_base = """
        SELECT
          v.verse_id,
          v.text,
          t.language,
          b.testament,
          v.book_number,
          v.translation_code
        FROM verse v
        JOIN book b
          ON b.translation_code = v.translation_code
         AND b.book_number = v.book_number
        JOIN translation t
          ON t.translation_code = v.translation_code
    """
    where_parts: list[str] = []
    params: list[Any] = []

    # Build the LEFT JOIN first (if needed) so params are in correct order
    if not reembed:
        sel_base += " LEFT JOIN verse_embedding e ON e.verse_id = v.verse_id AND e.embedding_model = %s AND e.embedding_dim = %s\n"
        params.extend([model, embedding_dim])
        where_parts.append("(e.verse_id IS NULL)")

    # Then add translation filter
    if translations:
        # psycopg automatically converts Python list to PostgreSQL array for ANY()
        where_parts.append("v.translation_code = ANY(%s)")
        params.append(translations)

    where_sql = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""
    order_sql = (
        " ORDER BY v.translation_code, v.book_number, v.chapter_number, v.verse_number, v.suffix"
    )
    sql_select = sel_base + where_sql + order_sql

    vector_rows: list[tuple[str, str, str | None, str | None, int, str]] = []
    with psycopg.connect(dsn) as conn:
        conn.execute("SET application_name = 'divinehaven-embedder';")
        with conn.cursor() as cur:
            offset = 0
            while True:
                page_sql = sql_select + f" OFFSET {offset} LIMIT {fetch_page}"
                cur.execute(page_sql, params)  # type: ignore
                rows = cur.fetchall()
                if not rows:
                    break
                vector_rows.extend(rows)
                offset += fetch_page

    total_selected = len(vector_rows)
    typer.secho(f"Selected {total_selected} verse(s) to embed", fg=typer.colors.CYAN)

    if dry_run or total_selected == 0:
        if dry_run:
            typer.secho("Dry run complete (no DB writes).", fg=typer.colors.GREEN)
        return

    # Prepare batches for /api/embed
    batches = _batched_indices(total_selected, batch_size)

    if write_mode == "staging-copy" and aiohttp is not None:
        asyncio.run(
            _run_pipeline(
                dsn=dsn,
                api_base=api_base,
                model=model,
                batches=batches,
                rows_all=vector_rows,
                workers=workers,
                keep_alive=keep_alive,
                embedding_dim=embedding_dim,
                use_labels=use_labels,
                precision=precision,
            )
        )
        typer.secho("Pipeline finished ✔", fg=typer.colors.GREEN, bold=True)
        return

    # Fallback paths (direct batched INSERT)
    texts = [r[1] for r in vector_rows]

    async def _run_async_direct() -> None:
        connector = aiohttp.TCPConnector(limit=workers * 2, ssl=False)  # type: ignore
        async with aiohttp.ClientSession(  # type: ignore
            connector=connector, timeout=aiohttp.ClientTimeout(total=120)  # type: ignore
        ) as session:
            with psycopg.connect(dsn) as conn:  # type: ignore
                conn.execute("SET application_name = 'divinehaven-embedder';")
                with conn.cursor() as cur:
                    sem = asyncio.Semaphore(workers)

                    async def one(b):
                        async with sem:
                            idxs = b
                            embs = await _embed_batch_async(
                                session,
                                api_base,
                                model,
                                [texts[i] for i in idxs],
                                keep_alive,
                            )
                            upserts: list[tuple[Any, ...]] = []
                            ts = datetime.now(UTC).isoformat().replace("+00:00", "Z")
                            for j, irow in enumerate(idxs):
                                verse_id, _textv, lang, testament, book_no, tcode = vector_rows[
                                    irow
                                ]
                                vec = embs[j]
                                if len(vec) != embedding_dim:
                                    raise RuntimeError(
                                        f"Dimension mismatch for {verse_id}: {len(vec)} vs {embedding_dim}"
                                    )
                                labels = None
                                if use_labels:
                                    labels = [
                                        _lang_label(lang),
                                        _testament_label(testament),
                                        int(book_no),
                                    ]
                                meta = {
                                    "provider": "ollama",
                                    "model": model,
                                    "dim": embedding_dim,
                                    "translation": tcode,
                                    "lang": lang,
                                    "testament": testament,
                                    "keep_alive": keep_alive,
                                    "pipeline": "embed-verses",
                                    "ts_utc": ts,
                                }
                                upserts.append(
                                    (
                                        verse_id,
                                        _vec_literal(vec, precision),
                                        model,
                                        embedding_dim,
                                        labels,
                                        json.dumps(meta),
                                    )
                                )
                            if upserts:
                                values_sql = ",".join(
                                    ["(%s, %s::vector, %s, %s, %s, %s)"] * len(upserts)
                                )
                                cur.execute(
                                    f"""
                                    INSERT INTO verse_embedding
                                      (verse_id, embedding, embedding_model, embedding_dim, labels, metadata)
                                    VALUES {values_sql}
                                    ON CONFLICT (verse_id, embedding_model, embedding_dim) DO UPDATE
                                    SET embedding = EXCLUDED.embedding,
                                        labels   = COALESCE(EXCLUDED.labels, verse_embedding.labels),
                                        metadata = EXCLUDED.metadata,
                                        embedding_ts = now();
                                    """,
                                    list(itertools.chain.from_iterable(upserts)),
                                )

                    # write in waves
                    for k in range(0, len(batches), workers * 2):
                        wave = batches[k : k + workers * 2]
                        await asyncio.gather(*(one(b) for b in wave))
                        conn.commit()
                        typer.echo(f"Committed wave up to index {k + len(wave)}")

    def _run_sync_direct() -> None:
        with psycopg.connect(dsn) as conn:  # type: ignore
            conn.execute("SET application_name = 'divinehaven-embedder';")
            with conn.cursor() as cur:
                for b in batches:
                    idxs = b
                    embs = _embed_batch_sync(api_base, model, [texts[i] for i in idxs], keep_alive)
                    upserts: list[tuple[Any, ...]] = []
                    ts = datetime.utcnow().isoformat() + "Z"
                    for j, irow in enumerate(idxs):
                        verse_id, _textv, lang, testament, book_no, tcode = vector_rows[irow]
                        vec = embs[j]
                        if len(vec) != embedding_dim:
                            raise RuntimeError(
                                f"Dimension mismatch for {verse_id}: {len(vec)} vs {embedding_dim}"
                            )
                        labels = None
                        if use_labels:
                            labels = [
                                _lang_label(lang),
                                _testament_label(testament),
                                int(book_no),
                            ]
                        meta = {
                            "provider": "ollama",
                            "model": model,
                            "dim": embedding_dim,
                            "translation": tcode,
                            "lang": lang,
                            "testament": testament,
                            "keep_alive": keep_alive,
                            "pipeline": "embed-verses",
                            "ts_utc": ts,
                        }
                        upserts.append(
                            (
                                verse_id,
                                _vec_literal(vec, precision),
                                model,
                                embedding_dim,
                                labels,
                                json.dumps(meta),
                            )
                        )
                    if upserts:
                        values_sql = ",".join(["(%s, %s::vector, %s, %s, %s, %s)"] * len(upserts))
                        cur.execute(
                            f"""
                            INSERT INTO verse_embedding
                              (verse_id, embedding, embedding_model, embedding_dim, labels, metadata)
                            VALUES {values_sql}
                            ON CONFLICT (verse_id) DO UPDATE
                            SET embedding = EXCLUDED.embedding,
                                embedding_model = EXCLUDED.embedding_model,
                                embedding_dim = EXCLUDED.embedding_dim,
                                labels   = COALESCE(EXCLUDED.labels, verse_embedding.labels),
                                metadata = EXCLUDED.metadata,
                                embedding_ts = now();
                            """,
                            list(itertools.chain.from_iterable(upserts)),
                        )
                conn.commit()

    if aiohttp is not None and write_mode == "direct":
        asyncio.run(_run_async_direct())
    else:
        if write_mode != "staging-copy":
            typer.secho(
                "aiohttp not found or write-mode=direct; using synchronous path.",
                fg=typer.colors.YELLOW,
            )
        _run_sync_direct()

    typer.secho("Embedding run complete ✔", fg=typer.colors.GREEN, bold=True)


# ---------------------------
# 🧰 CLI: migrate embeddings PK (composite)
# ---------------------------
@app.command("migrate-embeddings-pk")
def migrate_embeddings_pk(
    dsn: str = typer.Argument(..., help="Postgres DSN"),
    truncate: bool = typer.Option(
        False,
        "--truncate/--no-truncate",
        help="TRUNCATE verse_embedding before altering",
    ),
) -> None:
    """Switch verse_embedding to composite PK (verse_id, embedding_model, embedding_dim)."""
    if psycopg is None:
        typer.secho(
            "psycopg is not installed. `pip install psycopg[binary]`",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=3)

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            if truncate:
                cur.execute("TRUNCATE TABLE verse_embedding;")
            cur.execute(
                "ALTER TABLE verse_embedding DROP CONSTRAINT IF EXISTS verse_embedding_pkey;"
            )
            cur.execute(
                """
                ALTER TABLE verse_embedding
                ADD PRIMARY KEY (verse_id, embedding_model, embedding_dim);
            """
            )
        conn.commit()
    typer.secho("Embeddings PK migrated ✔", fg=typer.colors.GREEN, bold=True)


# ---------------------------
# ✨ CLI: Scaffold a manifest.json
# ---------------------------
def _extract_key(p: Path) -> str:
    m = re.findall(r"(\d+)(?!.*\d)", p.name)
    return m[0] if m else p.stem


@app.command("init")
def manifest_init(
    out_path: Path = typer.Argument(
        Path("../manifest.json"), help="Where to write the new manifest"
    ),
    translations: list[str] = typer.Option(
        [
            "NIV",
            "ESV",
            "NLT",
            "KJV",
            "NKJV",
            "ASV",
            "NASU",
            "NIRV",
            "RVR1960",
            "NVI",
            "LXX",
            "TNK",
            "NASB",
            "NET",
        ],
        "--translations",
        help="Translation codes",
    ),
    languages: list[str] = typer.Option(["en", "es", "el", "he"], "--languages"),
    pipeline_version: str = typer.Option("embed-pipeline@1.2.0", "--pipeline-version"),
    source_version: str = typer.Option("divine_haven.universal_v1", "--source-version"),
    embedding_model: str = typer.Option("embeddinggemma", "--embedding-model"),
    embedding_dim: int = typer.Option(768, "--embedding-dim"),
    pooling: Literal["mean", "max", "cls"] = typer.Option("mean", "--pooling"),
    normalize: bool = typer.Option(True, "--normalize/--no-normalize"),
    tokenizer: str | None = typer.Option(None, "--tokenizer"),
    truncation_strategy: Literal["none", "sliding", "head", "tail"] = typer.Option(
        "sliding", "--truncation-strategy"
    ),
    granularity: Literal["verse", "pericope", "sliding"] = typer.Option("sliding", "--granularity"),
    window_size: int | None = typer.Option(128, "--window-size"),
    stride: int | None = typer.Option(32, "--stride"),
    lower: bool = typer.Option(False, "--lower/--no-lower", help="Preprocess: lowercase"),
    strip_punct: bool = typer.Option(False, "--strip-punct/--no-strip-punct"),
    collapse_ws: bool = typer.Option(True, "--collapse-ws/--no-collapse-ws"),
    vectorscale: Literal["hnsw", "ivf"] = typer.Option("hnsw", "--vectorscale"),
    m: int = typer.Option(32, "--hnsw-m"),
    ef_construction: int = typer.Option(200, "--hnsw-ef-construction"),
    ef_search: int = typer.Option(64, "--hnsw-ef-search"),
    lists: int = typer.Option(2048, "--ivf-lists"),
    probes: int = typer.Option(8, "--ivf-probes"),
    vector_k: int = typer.Option(50, "--vector-k"),
    fts_k: int = typer.Option(50, "--fts-k"),
    fusion: Literal["rrf", "weighted_sum"] = typer.Option("rrf", "--fusion"),
    fusion_k: int = typer.Option(60, "--fusion-k"),
    weight_vector: str | None = typer.Option(
        None,
        "--fusion-weights",
        help='JSON mapping for weighted_sum, e.g. \'{"vector":0.7,"fts":0.3}\'',
    ),
    fts_dictionary: Literal["simple", "english"] = typer.Option("simple", "--fts-dictionary"),
    stopwords: str | None = typer.Option(None, "--fts-stopwords"),
    discover_dir: Path | None = typer.Option(
        None, "--discover-dir", help="Scan dir for batches: input_glob + mapping_glob"
    ),
    input_glob: str = typer.Option("**/batch_input.*.jsonl", "--input-glob"),
    mapping_glob: str = typer.Option("**/mapping.*.jsonl", "--mapping-glob"),
    checksums: bool = typer.Option(True, "--checksums/--no-checksums"),
    operator: str | None = typer.Option("pipeline/init:embeddinggemma-768", "--operator"),
    overwrite: bool = typer.Option(False, "--overwrite/--no-overwrite"),
) -> None:
    """Create a fresh manifest.json with refined schema; optionally discover batch files."""
    if out_path.exists() and not overwrite:
        typer.secho(
            f"Refusing to overwrite existing {out_path}. Use --overwrite to replace.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    run_ts = datetime.utcnow().replace(microsecond=0)
    run_id = f"{run_ts.isoformat()}Z_{source_version}"

    recipe = EmbeddingRecipe(
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        pooling=pooling,
        normalize=normalize,
        tokenizer=tokenizer,
        truncation_strategy=truncation_strategy,
        chunking=Chunking(granularity=granularity, window_size=window_size, stride=stride),
        preprocess=Preprocess(lower=lower, strip_punct=strip_punct, collapse_ws=collapse_ws),
        filters=Filters(books_included="all"),
    )

    if vectorscale == "hnsw":
        vcfg: VectorScaleCfg = VectorScaleHNSW(
            params={"m": m, "ef_construction": ef_construction, "ef_search": ef_search}
        )
    else:
        vcfg = VectorScaleIVF(params={"lists": lists, "probes": probes})

    fusion_opts = FusionOptions(method=fusion, k=fusion_k)
    if fusion == "weighted_sum" and weight_vector:
        try:
            fusion_opts.weight_vector = json.loads(weight_vector)
        except Exception as e:
            typer.secho(f"Invalid --fusion-weights JSON: {e}", fg=typer.colors.RED)
            raise typer.Exit(code=2)

    index_plan = IndexPlan(
        vectorscale=vcfg,
        hybrid=HybridOptions(
            vector_k=vector_k,
            fts_k=fts_k,
            fusion=fusion_opts,
            fts=FTSOptions(dictionary=fts_dictionary, stopwords=stopwords),
        ),
    )

    batches: list[BatchEntry] = []
    if discover_dir:
        inputs = sorted(discover_dir.glob(input_glob))
        mappings = sorted(discover_dir.glob(mapping_glob))
        input_map = {_extract_key(p): p for p in inputs}
        mapping_map = {_extract_key(p): p for p in mappings}
        keys = sorted(set(input_map) & set(mapping_map))
        missing_in = sorted(set(mapping_map) - set(input_map))
        missing_map = sorted(set(input_map) - set(mapping_map))
        if missing_in:
            typer.secho(
                f"WARN: {len(missing_in)} mapping files without inputs: {len(missing_in)}",
                fg=typer.colors.YELLOW,
            )
        if missing_map:
            typer.secho(
                f"WARN: {len(missing_map)} input files without mappings: {len(missing_map)}",
                fg=typer.colors.YELLOW,
            )
        for k in keys:
            inp = input_map[k]
            mp = mapping_map[k]
            b = BatchEntry(
                input=str(inp.relative_to(discover_dir)),
                mapping=str(mp.relative_to(discover_dir)),
            )
            if checksums:
                try:
                    b.sha256_input = sha256_file(inp)
                    b.sha256_mapping = sha256_file(mp)
                except Exception as e:
                    typer.secho(
                        f"WARN: checksum failed for {inp} or {mp}: {e}",
                        fg=typer.colors.YELLOW,
                    )
            batches.append(b)
        typer.secho(
            f"Discovered {len(batches)} batch pairs in {discover_dir}",
            fg=typer.colors.GREEN,
        )

    manifest = Manifest(
        run_id=run_id,
        run_ts=run_ts,
        pipeline_version=pipeline_version,
        source_version=source_version,
        translation_set=translations,
        languages=languages,
        license="internal-use",
        operator=operator,
        embedding_recipe=recipe,
        index_plan=index_plan,
        batches=batches,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(manifest.model_dump(mode="json"), f, indent=2, ensure_ascii=False)

    typer.secho(f"Wrote {out_path} (run_id={run_id}) ✔", fg=typer.colors.GREEN, bold=True)


# ---------------------------
# Entrypoint
# ---------------------------
if __name__ == "__main__":
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)
