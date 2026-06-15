#!/usr/bin/env python3
"""
scripts/ablation.py

Weight-tuning ablation script for the hybrid reranker.

Varies one weight at a time over the public query set to find better
fusion parameters.  Outputs a table of (weight_value, mean_NDCG@10).

This script tunes GENERAL WEIGHTS, not per-query label assignments.
Do NOT use public label IDs in runtime retrieve.py logic.

Usage:
    python scripts/ablation.py
"""
from __future__ import annotations

import sys
import copy
import time
from pathlib import Path

STUDENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDENT_ROOT))

from eval import mean_ndcg_at_k, load_query_file
from utils import PUBLIC_QUERIES_PATH
import retrieve   # import so we can modify _WEIGHTS in place


def evaluate(queries, ground_truth) -> float:
    """Run the full pipeline and return mean NDCG@10."""
    from main import run
    ranked = run(queries)
    return mean_ndcg_at_k(ranked, ground_truth)


def ablate_weight(
    key: str,
    values,
    queries,
    ground_truth,
    base_weights: dict,
):
    """
    Vary one weight across candidate values, keeping others fixed.
    Prints a result table.
    """
    print(f"\n=== Ablation: {key} ===")
    print(f"  {'Value':>8}   {'NDCG@10':>8}   {'Time (s)':>10}")
    print("  " + "-" * 32)

    best_val, best_score = None, -1.0
    for v in values:
        # Set the new weight; keep others as in base_weights
        retrieve._WEIGHTS = {**base_weights, key: v}
        t0 = time.perf_counter()
        score = evaluate(queries, ground_truth)
        elapsed = time.perf_counter() - t0
        print(f"  {v:>8.3f}   {score:>8.4f}   {elapsed:>10.2f}s")
        if score > best_score:
            best_score = score
            best_val   = v

    print(f"  Best: {key}={best_val} → NDCG@10={best_score:.4f}")
    return best_val, best_score


def main():
    rows         = load_query_file(PUBLIC_QUERIES_PATH)
    queries      = [r["query"]            for r in rows]
    ground_truth = [r["relevant_page_ids"] for r in rows]

    # Start from the defaults in retrieve.py
    base_weights = dict(retrieve._WEIGHTS)

    print("Public evaluation baseline:")
    base_score = evaluate(queries, ground_truth)
    print(f"  mean_ndcg@10 = {base_score:.4f}")

    # Ablation 1: dense_chunk weight
    best_dc, _ = ablate_weight(
        "dense_chunk",
        [0.15, 0.20, 0.25, 0.28, 0.32, 0.36, 0.40],
        queries, ground_truth, base_weights,
    )

    # Ablation 2: bm25_chunk weight
    updated = {**base_weights, "dense_chunk": best_dc}
    best_bc, _ = ablate_weight(
        "bm25_chunk",
        [0.15, 0.20, 0.24, 0.28, 0.32],
        queries, ground_truth, updated,
    )

    # Ablation 3: year_num_match weight
    updated2 = {**updated, "bm25_chunk": best_bc}
    ablate_weight(
        "year_num_match",
        [0.0, 0.02, 0.04, 0.06, 0.08, 0.10],
        queries, ground_truth, updated2,
    )

    print("\n=== Done ===")
    print("Apply best weights to retrieve._WEIGHTS in retrieve.py")
    print("Then re-run: python scripts/eval_public.py")


if __name__ == "__main__":
    main()
