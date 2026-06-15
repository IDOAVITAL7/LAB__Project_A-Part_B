#!/usr/bin/env python3
"""
scripts/inspect_failures.py

Per-query diagnostic tool for the Section B retrieval system.

For each public query:
  - Runs the full pipeline (loading artifacts once)
  - Shows the rank of each relevant page in the result
  - Shows which relevant pages are missing from the top-10
  - Classifies each failure as recall or ranking failure

Usage:
    python scripts/inspect_failures.py [--k 50]

Options:
    --k INT   Check candidate recall at this depth (default 50)
"""
from __future__ import annotations

import sys
import argparse
import json
from pathlib import Path

STUDENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDENT_ROOT))

from eval import ndcg_at_k, load_query_file
from main import run
from utils import PUBLIC_QUERIES_PATH, normalize_page_id


def parse_args():
    p = argparse.ArgumentParser(description="Inspect per-query retrieval failures")
    p.add_argument("--k", type=int, default=50,
                   help="Candidate depth for recall check (default: 50)")
    return p.parse_args()


def main():
    args = parse_args()
    k_recall = args.k

    rows        = load_query_file(PUBLIC_QUERIES_PATH)
    queries     = [r["query"] for r in rows]
    query_ids   = [r.get("query_id", f"q_{i:03d}") for i, r in enumerate(rows)]
    rel_sets    = [r["relevant_page_ids"] for r in rows]   # sets of int

    print(f"Running run() on {len(queries)} public queries...")
    ranked = run(queries)
    print()

    # Per-query analysis
    total_ndcg = 0.0
    recall_failures = 0
    rank_failures   = 0

    header = f"{'QID':<15}  {'NDCG@10':>8}  {'RANKS':^30}  {'STATUS':<15}  QUERY"
    print(header)
    print("-" * 130)

    for qid, query, relevant, result in zip(query_ids, queries, rel_sets, ranked):
        ndcg = ndcg_at_k(result, relevant)
        total_ndcg += ndcg

        # Find rank of each relevant page (1-indexed, deduped)
        seen:     set = set()
        rel_ranks: dict = {}
        for rank, pid in enumerate(result):
            if pid in seen:
                continue
            seen.add(pid)
            if pid in relevant:
                rel_ranks[pid] = rank + 1

        found_in_top_k  = {pid for pid in result[:k_recall] if pid in relevant}
        missing_top_k   = relevant - found_in_top_k
        missing_top_10  = relevant - set(result[:10])

        ranks_str = str(sorted(rel_ranks.values()))[:28]
        if ndcg >= 0.999:
            status = "PERFECT"
        elif missing_top_k:
            status = f"RECALL FAIL ({len(missing_top_k)} missing)"
            recall_failures += 1
        elif missing_top_10:
            status = f"RANK FAIL"
            rank_failures += 1
        else:
            status = "OK"

        print(f"{qid:<15}  {ndcg:>8.4f}  {ranks_str:<30}  {status:<15}  {query[:55]}")

        if missing_top_k:
            print(f"  ⚠ NOT IN TOP-{k_recall}: {missing_top_k}")
        elif missing_top_10:
            for pid in missing_top_10:
                rank_in_full = None
                for r, p in enumerate(result):
                    if p == pid:
                        rank_in_full = r + 1
                        break
                print(f"  ↓ pid={pid} retrieved at rank {rank_in_full} (outside top-10)")

    print()
    print(f"Mean NDCG@10       : {total_ndcg / len(queries):.4f}")
    print(f"Recall failures    : {recall_failures} (relevant page not in top-{k_recall})")
    print(f"Ranking failures   : {rank_failures} (retrieved but not in top-10)")
    print(f"Queries OK         : {len(queries) - recall_failures - rank_failures}")


if __name__ == "__main__":
    main()
