import copy
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval import load_query_file, mean_ndcg_at_k
from utils import PUBLIC_QUERIES_PATH
from index import load_all_artifacts
from embed import get_model
import retrieve

rows = load_query_file(PUBLIC_QUERIES_PATH)
queries = [r["query"] for r in rows]
ground_truth = [r["relevant_page_ids"] for r in rows]

print("Loading artifacts once...", flush=True)
t0 = time.perf_counter()
state = load_all_artifacts()
t1 = time.perf_counter()

cfg_key = "config" if "config" in state else "cfg" if "cfg" in state else None
if cfg_key is None:
    raise RuntimeError(f"Could not find config/cfg in state. keys={list(state.keys())}")

cfg = state[cfg_key]

print("Loading model once...", flush=True)
get_model()
t2 = time.perf_counter()

retrieve._STATE = state
base_time = t2 - t0
print(f"cfg_key={cfg_key}", flush=True)
print(f"base_load_model_time={base_time:.2f}s", flush=True)

configs = [
    ("current_safe", 64, 40, 80, 80, 100, 100),
    ("wider_bm25", 64, 50, 100, 100, 150, 200),
    ("wider_dense", 64, 50, 120, 160, 120, 120),
    ("wider_all", 64, 60, 120, 200, 200, 200),
    ("ef96_wider", 96, 50, 100, 120, 150, 200),
]

weights = [
    ("current", {
        "dense_chunk": 0.28, "bm25_chunk": 0.24, "dense_page": 0.16,
        "bm25_page": 0.14, "dense_title": 0.08, "title_overlap": 0.04,
        "year_num_match": 0.04, "source_bonus": 0.02,
    }),
    ("more_bm25_chunk", {
        "dense_chunk": 0.23, "bm25_chunk": 0.31, "dense_page": 0.14,
        "bm25_page": 0.14, "dense_title": 0.07, "title_overlap": 0.04,
        "year_num_match": 0.05, "source_bonus": 0.02,
    }),
    ("more_dense_chunk", {
        "dense_chunk": 0.35, "bm25_chunk": 0.20, "dense_page": 0.16,
        "bm25_page": 0.12, "dense_title": 0.07, "title_overlap": 0.04,
        "year_num_match": 0.04, "source_bonus": 0.02,
    }),
    ("more_bm25_page", {
        "dense_chunk": 0.26, "bm25_chunk": 0.22, "dense_page": 0.14,
        "bm25_page": 0.20, "dense_title": 0.07, "title_overlap": 0.04,
        "year_num_match": 0.05, "source_bonus": 0.02,
    }),
    ("title_year_boost", {
        "dense_chunk": 0.25, "bm25_chunk": 0.24, "dense_page": 0.14,
        "bm25_page": 0.13, "dense_title": 0.10, "title_overlap": 0.06,
        "year_num_match": 0.06, "source_bonus": 0.02,
    }),
]

original_cfg = copy.deepcopy(cfg)
original_weights = copy.deepcopy(retrieve._WEIGHTS)
results = []

for name, ef, kt, kp, kc, kbmp, kbmc in configs:
    cfg.clear()
    cfg.update(original_cfg)
    cfg.update({
        "use_bm25_chunk": True,
        "use_bm25_chunk_sqlite": True,
        "hnsw_ef_search": ef,
        "retrieval_top_k_title": kt,
        "retrieval_top_k_page": kp,
        "retrieval_top_k_chunk": kc,
        "retrieval_top_k_bm25_page": kbmp,
        "retrieval_top_k_bm25_chunk": kbmc,
    })

    if hasattr(state.get("chunk_index"), "hnsw"):
        state["chunk_index"].hnsw.efSearch = ef

    for w_name, w in weights:
        retrieve._WEIGHTS.clear()
        retrieve._WEIGHTS.update(w)

        s0 = time.perf_counter()
        ranked = retrieve.search_batch(queries)
        s1 = time.perf_counter()

        search_time = s1 - s0
        est_total = base_time + search_time
        ndcg = mean_ndcg_at_k(ranked, ground_truth)

        row = {
            "config": name,
            "weights": w_name,
            "ndcg": round(ndcg, 6),
            "search_time": round(search_time, 3),
            "estimated_total": round(est_total, 3),
        }
        results.append(row)

        print(
            f"{name:14s} {w_name:17s} "
            f"ndcg={ndcg:.4f} search={search_time:.2f}s est_total={est_total:.2f}s",
            flush=True,
        )

cfg.clear()
cfg.update(original_cfg)
retrieve._WEIGHTS.clear()
retrieve._WEIGHTS.update(original_weights)

out = ROOT / "tuning_results.tsv"
with out.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["config", "weights", "ndcg", "search_time", "estimated_total"],
        delimiter="\t",
    )
    writer.writeheader()
    writer.writerows(results)

print()
print("Top results under estimated_total < 55s:")
valid = [r for r in results if r["estimated_total"] < 55]
for r in sorted(valid, key=lambda x: (-x["ndcg"], x["estimated_total"]))[:10]:
    print(r)

print()
print("Top results overall:")
for r in sorted(results, key=lambda x: (-x["ndcg"], x["estimated_total"]))[:10]:
    print(r)

print(f"Saved: {out}")
