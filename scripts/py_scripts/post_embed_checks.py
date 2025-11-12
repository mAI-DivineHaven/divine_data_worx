#!/usr/bin/env python3
"""
Post-embedding quick checks without psql.

Usage examples:
  # use DSN from env
  export DSN="postgresql://postgres:postgres@localhost:5432/divinehaven"
  python post_embed_checks.py extensions
  python post_embed_checks.py table-shape --table verse_embedding
  python post_embed_checks.py verse-counts
  python post_embed_checks.py embed-coverage --model embeddinggemma --dim 768
  python post_embed_checks.py integrity
  python post_embed_checks.py analyze --table verse_embedding

  # ANN / FTS / Hybrid tests (NIV)
  python post_embed_checks.py ann-search --q "love your neighbor" --translation NIV
  python post_embed_checks.py fts-test --q "love & neighbor" --translation NIV
  python post_embed_checks.py hybrid --q "love your neighbor" --translation NIV

  # override DSN inline
  python post_embed_checks.py extensions --dsn postgresql://user:pass@host:5432/dbname
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, cast

import psycopg
import requests
from psycopg.rows import dict_row
from tabulate import tabulate


def get_conn(dsn: str | None) -> psycopg.Connection:
    dsn = dsn or os.getenv("DSN")
    if not dsn:
        print("ERROR: Provide a DSN via --dsn or DSN env var.", file=sys.stderr)
        sys.exit(2)
    return psycopg.connect(dsn, autocommit=True)


# ---------- Simple SELECT helpers ----------


def fetchall(
    conn: psycopg.Connection, sql: str, params: tuple[Any, ...] | None = None
) -> list[dict[str, Any]]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(cast(Any, sql), params or ())
        return list(cur.fetchall())


def fetchone(
    conn: psycopg.Connection, sql: str, params: tuple[Any, ...] | None = None
) -> dict[str, Any] | None:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(cast(Any, sql), params or ())
        return cur.fetchone()


def exec_sql(conn: psycopg.Connection, sql: str, params: tuple[Any, ...] | None = None) -> None:
    with conn.cursor() as cur:
        cur.execute(cast(Any, sql), params or ())


# ---------- Commands ----------


def cmd_extensions(conn: psycopg.Connection, args: argparse.Namespace) -> None:
    rows = fetchall(
        conn,
        """
        SELECT extname, extversion
        FROM pg_extension
        WHERE extname IN ('vector','vectorscale','timescaledb')
        ORDER BY 1;
        """,
    )
    print(tabulate(rows, headers="keys", tablefmt="github"))


def cmd_table_shape(conn: psycopg.Connection, args: argparse.Namespace) -> None:
    # approximate \d+ by reading columns + types + notnull + defaults
    sql = """
    SELECT a.attname AS column,
           pg_catalog.format_type(a.atttypid, a.atttypmod) AS type,
           NOT a.attnotnull AS is_nullable,
           pg_get_expr(ad.adbin, ad.adrelid) AS default
    FROM pg_attribute a
    JOIN pg_class c ON c.oid = a.attrelid
    JOIN pg_namespace n ON n.oid = c.relnamespace
    LEFT JOIN pg_attrdef ad ON ad.adrelid = a.attrelid AND ad.adnum = a.attnum
    WHERE c.relkind IN ('r','p') -- table/partitioned table
      AND n.nspname = COALESCE(%s, current_schema())
      AND c.relname = %s
      AND a.attnum > 0 AND NOT a.attisdropped
    ORDER BY a.attnum;
    """
    schema, table = split_table(args.table)
    rows = fetchall(conn, sql, (schema, table))
    if not rows:
        print(f"No table found: {args.table}")
        return

    print(f"Table: {args.table}")
    print(tabulate(rows, headers="keys", tablefmt="github"))

    # Indexes
    idx = fetchall(
        conn,
        """
        SELECT indexname, indexdef
        FROM pg_indexes
        WHERE schemaname = COALESCE(%s, current_schema()) AND tablename = %s
        ORDER BY 1;
    """,
        (schema, table),
    )
    if idx:
        print("\nIndexes")
        print(tabulate(idx, headers="keys", tablefmt="github"))

    # Row estimate
    est = fetchone(
        conn,
        """
        SELECT reltuples::bigint AS est_rows
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = COALESCE(%s, current_schema())
          AND c.relname = %s
    """,
        (schema, table),
    )
    if est:
        print(f"\nEstimated rows: {est['est_rows']:,}")


def cmd_verse_counts(conn: psycopg.Connection, args: argparse.Namespace) -> None:
    rows = fetchall(
        conn,
        """
        SELECT translation_code, COUNT(*) AS count
        FROM verse
        GROUP BY 1
        ORDER BY 1;
    """,
    )
    print(tabulate(rows, headers="keys", tablefmt="github"))


def cmd_embed_coverage(conn: psycopg.Connection, args: argparse.Namespace) -> None:
    rows = fetchall(
        conn,
        """
        SELECT v.translation_code,
               COUNT(*) AS verses,
               COUNT(e.verse_id) FILTER (WHERE e.embedding_model=%s AND e.embedding_dim=%s) AS embedded
        FROM verse v
        LEFT JOIN verse_embedding e USING (verse_id)
        GROUP BY 1
        ORDER BY 1;
    """,
        (args.model, args.dim),
    )
    print(tabulate(rows, headers="keys", tablefmt="github"))


def cmd_integrity(conn: psycopg.Connection, args: argparse.Namespace) -> None:
    null_result = fetchone(
        conn,
        "SELECT COUNT(*) AS null_vectors FROM verse_embedding WHERE embedding IS NULL;",
    )
    null_vectors = null_result["null_vectors"] if null_result else 0

    dims_result = fetchone(
        conn,
        "SELECT COUNT(*) AS wrong_dims FROM verse_embedding WHERE embedding_dim <> %s;",
        (args.dim,),
    )
    wrong_dims = dims_result["wrong_dims"] if dims_result else 0

    labels = fetchall(
        conn,
        """
        SELECT COALESCE(labels[1],0) AS lang,
               COALESCE(labels[2],0) AS test,
               COUNT(*) AS count
        FROM verse_embedding
        GROUP BY 1,2
        ORDER BY 1,2;
    """,
    )
    print(
        tabulate(
            [{"null_vectors": null_vectors, "wrong_dims": wrong_dims}],
            headers="keys",
            tablefmt="github",
        )
    )
    if labels:
        print("\nLabel distribution (lang, testament)")
        print(tabulate(labels, headers="keys", tablefmt="github"))


def cmd_analyze(conn: psycopg.Connection, args: argparse.Namespace) -> None:
    schema, table = split_table(args.table)
    q = f'ANALYZE {f"{schema}." if schema else ""}{table};'
    exec_sql(conn, q)
    print(f"ANALYZE done for {args.table}")


# ---------- ANN / FTS / Hybrid ----------


def call_ollama_embed(text: str, model: str, base_url: str) -> list[float]:
    url = base_url.rstrip("/") + "/api/embed"
    payload = {"model": model, "input": [text]}
    try:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        emb = data.get("embeddings", [[]])[0]
        if not emb:
            raise ValueError("Empty embedding returned.")
        return emb
    except Exception as e:
        raise RuntimeError(f"Ollama embed error: {e}") from e


def to_pgvector_literal(vec: list[float]) -> str:
    # pgvector text literal: '[v1, v2, ...]'
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def cmd_ann_search(conn, args):
    vec = call_ollama_embed(args.q, args.model, args.ollama)
    vec_lit = to_pgvector_literal(vec)  # returns string like "[1.0,2.0,...]"

    sql = """
    SELECT v.translation_code, v.book_number, v.chapter_number, v.verse_number, v.suffix,
           (1 - (e.embedding <=> %s::vector))::numeric(6,4) AS sim,
           v.text
    FROM verse_embedding e
    JOIN verse v USING (verse_id)
    WHERE v.translation_code = %s
    ORDER BY e.embedding <=> %s::vector
    LIMIT %s;
    """
    rows = fetchall(conn, sql, (vec_lit, args.translation, vec_lit, args.limit))
    print(f'Query: "{args.q}"  | translation={args.translation}  | top={args.limit}')
    if rows:
        print(tabulate(rows, headers="keys", tablefmt="github"))
    else:
        print("No results.")


def cmd_fts_test(conn: psycopg.Connection, args: argparse.Namespace) -> None:
    # Use 'simple' config as in your psql snippet
    rows = fetchall(
        conn,
        """
        SELECT verse_id, LEFT(text,120) AS snippet
        FROM verse
        WHERE translation_code=%s
          AND to_tsvector('simple', text) @@ to_tsquery('simple', %s)
        LIMIT %s;
    """,
        (args.translation, args.q, args.limit),
    )
    print(f'FTS: "{args.q}"  | translation={args.translation}  | top={args.limit}')
    if rows:
        print(tabulate(rows, headers="keys", tablefmt="github"))
    else:
        print("No results.")


def cmd_hybrid(conn, args):
    vec = call_ollama_embed(args.q, args.model, args.ollama)
    vec_lit = to_pgvector_literal(vec)

    sql = """
    WITH ann AS (
      SELECT v.verse_id, row_number() OVER (ORDER BY e.embedding <=> %s::vector) AS r
      FROM verse_embedding e
      JOIN verse v USING (verse_id)
      WHERE v.translation_code=%s
      LIMIT 100
    ),
    fts AS (
      SELECT v.verse_id,
             row_number() OVER (
               ORDER BY ts_rank_cd(to_tsvector('simple',v.text), plainto_tsquery('simple', %s)) DESC
             ) AS r
      FROM verse v
      WHERE v.translation_code=%s
        AND to_tsvector('simple',v.text) @@ plainto_tsquery('simple', %s)
      LIMIT 100
    ),
    u AS (
      SELECT verse_id, 1.0/(60+r) AS s FROM ann
      UNION ALL
      SELECT verse_id, 1.0/(60+r) AS s FROM fts
    )
    SELECT v.translation_code, v.book_number, v.chapter_number, v.verse_number, v.suffix,
           SUM(s) AS score, LEFT(v.text,180) AS snippet
    FROM u
    JOIN verse v USING (verse_id)
    GROUP BY v.translation_code, v.book_number, v.chapter_number, v.verse_number, v.suffix, v.text
    ORDER BY score DESC
    LIMIT %s;
    """
    rows = fetchall(
        conn,
        sql,
        (vec_lit, args.translation, args.q, args.translation, args.q, args.limit),
    )
    print(
        f'Hybrid (RRF): "{args.q}"  | translation={args.translation}  | model={args.model} | top={args.limit}'
    )
    if rows:
        print(tabulate(rows, headers="keys", tablefmt="github"))
    else:
        print("No results.")


# ---------- Utilities ----------


def split_table(qualified: str) -> tuple[str | None, str]:
    if "." in qualified:
        s, t = qualified.split(".", 1)
        return s, t
    return None, qualified


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Post-embedding DB sanity checks (no psql needed).")
    p.add_argument("--dsn", help="Postgres DSN. If omitted, uses DSN env var.")

    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("extensions", help="Check required extensions are present.")

    ts = sub.add_parser("table-shape", help="Describe table shape, indexes, row estimate.")
    ts.add_argument(
        "--table",
        default="verse_embedding",
        help="Table to describe (default: verse_embedding)",
    )

    sub.add_parser("verse-counts", help="Verse counts per translation.")

    ec = sub.add_parser("embed-coverage", help="Embedding coverage per translation.")
    ec.add_argument("--model", default="embeddinggemma")
    ec.add_argument("--dim", type=int, default=768)

    integ = sub.add_parser("integrity", help="Null vectors, wrong dims, label distribution.")
    integ.add_argument("--dim", type=int, default=768)

    an = sub.add_parser("analyze", help="Run ANALYZE after big upserts.")
    an.add_argument("--table", default="verse_embedding")

    ann = sub.add_parser("ann-search", help="ANN search smoke test by cosine distance.")
    ann.add_argument("--q", required=True, help="Query text to embed.")
    ann.add_argument("--translation", default="NIV")
    ann.add_argument("--model", default="embeddinggemma")
    ann.add_argument("--ollama", default="http://127.0.0.1:11434")
    ann.add_argument("--limit", type=int, default=10)

    fts = sub.add_parser("fts-test", help="FTS (lexical) test using to_tsvector/simple.")
    fts.add_argument(
        "--q",
        required=True,
        help="TS query, e.g. 'love & neighbor' or plainto syntax text.",
    )
    fts.add_argument("--translation", default="NIV")
    fts.add_argument("--limit", type=int, default=10)

    hy = sub.add_parser("hybrid", help="Simple hybrid (RRF) over ANN + FTS.")
    hy.add_argument("--q", required=True)
    hy.add_argument("--translation", default="NIV")
    hy.add_argument("--model", default="embeddinggemma")
    hy.add_argument("--ollama", default="http://127.0.0.1:11434")
    hy.add_argument("--limit", type=int, default=10)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    with get_conn(args.dsn) as conn:
        if args.cmd == "extensions":
            cmd_extensions(conn, args)
        elif args.cmd == "table-shape":
            cmd_table_shape(conn, args)
        elif args.cmd == "verse-counts":
            cmd_verse_counts(conn, args)
        elif args.cmd == "embed-coverage":
            cmd_embed_coverage(conn, args)
        elif args.cmd == "integrity":
            cmd_integrity(conn, args)
        elif args.cmd == "analyze":
            cmd_analyze(conn, args)
        elif args.cmd == "ann-search":
            cmd_ann_search(conn, args)
        elif args.cmd == "fts-test":
            cmd_fts_test(conn, args)
        elif args.cmd == "hybrid":
            cmd_hybrid(conn, args)
        else:
            parser.error("Unknown command")


if __name__ == "__main__":
    main()
