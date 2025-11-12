#!/usr/bin/env python3
"""Offline evaluation of hybrid retrieval against curated golden datasets."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

DEFAULT_DATASET = Path("Analysis/golden/hybrid_graph_queries.json")


@dataclass
class GoldenCase:
    """Container representing a single golden query scenario."""

    name: str
    relevant: Sequence[str]
    expected_ids: Sequence[str]

    @classmethod
    def from_payload(cls, payload: dict) -> GoldenCase:
        return cls(
            name=payload["name"],
            relevant=tuple(payload.get("relevant", [])),
            expected_ids=tuple(hit["verse_id"] for hit in payload.get("expected_hybrid", [])),
        )


def load_golden_cases(dataset_path: Path) -> list[GoldenCase]:
    """Load and normalise the curated golden dataset."""

    if not dataset_path.exists():
        raise FileNotFoundError(f"Golden dataset not found: {dataset_path}")

    payload = json.loads(dataset_path.read_text())
    return [GoldenCase.from_payload(item) for item in payload]


def load_predictions(
    predictions_path: Path | None, cases: Sequence[GoldenCase]
) -> dict[str, Sequence[str]]:
    """Resolve retrieval predictions either from file or default expected hits."""

    if predictions_path is None:
        return {case.name: case.expected_ids for case in cases}

    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    payload = json.loads(predictions_path.read_text())
    predictions: dict[str, Sequence[str]] = {}
    for item in payload:
        name = item.get("name")
        retrieved = item.get("retrieved", [])
        if not name:
            raise ValueError("Predictions entry missing 'name'")
        predictions[name] = tuple(retrieved)
    return predictions


def recall_at_k(relevant: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    """Compute binary recall@k (1 if any relevant item retrieved within top-k)."""

    if not relevant:
        return 1.0
    top_k = retrieved[:k]
    rel_set = set(relevant)
    return 1.0 if any(vid in rel_set for vid in top_k) else 0.0


def reciprocal_rank(relevant: Sequence[str], retrieved: Sequence[str]) -> float:
    """Compute reciprocal rank for the first relevant occurrence."""

    rel_set = set(relevant)
    for idx, vid in enumerate(retrieved):
        if vid in rel_set:
            return 1.0 / (idx + 1)
    return 0.0


def evaluate(
    cases: Sequence[GoldenCase],
    predictions: dict[str, Sequence[str]],
    k: int,
) -> tuple[float, float]:
    """Return (hit_rate@k, mean_reciprocal_rank)."""

    recalls: list[float] = []
    reciprocal_ranks: list[float] = []

    for case in cases:
        retrieved = predictions.get(case.name, ())
        recalls.append(recall_at_k(case.relevant, retrieved, k))
        reciprocal_ranks.append(reciprocal_rank(case.relevant, retrieved))

    return mean(recalls) if recalls else 0.0, mean(reciprocal_ranks) if reciprocal_ranks else 0.0


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to the curated golden dataset (JSON).",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Optional JSON file containing retrieval outputs to score.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Cut-off for hit-rate/recall calculation.",
    )
    parser.add_argument(
        "--min-hit-rate",
        type=float,
        default=0.9,
        help="Minimum acceptable hit-rate@k for pass/fail gating.",
    )
    parser.add_argument(
        "--min-mrr",
        type=float,
        default=0.8,
        help="Minimum acceptable mean reciprocal rank for pass/fail gating.",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    cases = load_golden_cases(args.dataset)
    predictions = load_predictions(args.predictions, cases)
    hit_rate, mrr = evaluate(cases, predictions, args.k)

    print(json.dumps({"hit_rate@k": hit_rate, "mrr": mrr, "k": args.k}, indent=2))

    if hit_rate < args.min_hit_rate:
        print(
            f"Hit-rate {hit_rate:.3f} fell below minimum threshold {args.min_hit_rate:.3f}",
            file=sys.stderr,
        )
        return 1

    if mrr < args.min_mrr:
        print(
            f"MRR {mrr:.3f} fell below minimum threshold {args.min_mrr:.3f}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
