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


def main():
    rows = load_query_file(PUBLIC_QUERIES_PATH)
    queries = [r["query"] for r in rows]
    ground_truth = [r["relevant_page_ids"] for r in rows]

    print("Loading artifacts once...", flush=True)
    t0 = time.perf_counter()
    state = load_all_artifacts()
    get_model()
    t1 = time.perf_counter()

    cfg_key = "config" if "config" in state else "cfg"
    cfg = state[cfg_key]
    retrieve._STATE = state
    base_time = t1 - t0

    print(f"cfg_key={cfg_key}", flush=True)
    print(f"base_load_model_time={base_time:.2f}s", flush=True)

    original_cfg = copy.deepcopy(cfg)
    original_weights = copy.deepcopy(retrieve._WEIGHTS)

    weights = {
        "best_current": {
            "dense_chunk": 0.23, "bm25_chunk": 0.31, "dense_page": 0.14,
            "bm25_page": 0.14, "dense_title": 0.07, "title_overlap": 0.04,
            "year_num_match": 0.05, "source_bonus": 0.02,
        },
        "bm25_more": {
            "dense_chunk": 0.20, "bm25_chunk": 0.36, "dense_page": 0.13,
            "bm25_page": 0.14, "dense_title": 0.06, "title_overlap": 0.04,
            "year_num_match": 0.05, "source_bonus": 0.02,
        },
        "balanced_recall": {
            "dense_chunk": 0.25, "bm25_chunk": 0.30, "dense_page": 0.15,
            "bm25_page": 0.13, "dense_title": 0.07, "title_overlap": 0.04,
            "year_num_match": 0.04, "source_bonus": 0.02,
        },
    }

    config_sets = {
        "baseline": dict(kp=120, kc=160, kbmp=120, kbmc=120, kt=50, ef=64),
        "bm25_200": dict(kp=120, kc=160, kbmp=160, kbmc=200, kt=50, ef=64),
        "wide_recall": dict(kp=160, kc=240, kbmp=200, kbmc=240, kt=60, ef=64),
    }

    results = []
    for cfg_name, c in config_sets.items():
        for chunk_variants in [1, 2, 3, 5]:
            for bm25_variants in [False, True]:
                cfg.clear()
                cfg.update(original_cfg)
                cfg.update({
                    "use_bm25_chunk": True,
                    "use_bm25_chunk_sqlite": True,
                    "hnsw_ef_search": c["ef"],
                    "retrieval_top_k_title": c["kt"],
                    "retrieval_top_k_page": c["kp"],
                    "retrieval_top_k_chunk": c["kc"],
                    "retrieval_top_k_bm25_page": c["kbmp"],
                    "retrieval_top_k_bm25_chunk": c["kbmc"],
                    "chunk_query_variants": chunk_variants,
                    "bm25_query_variants": bm25_variants,
                })
                if hasattr(state.get("chunk_index"), "hnsw"):
                    state["chunk_index"].hnsw.efSearch = c["ef"]

                for w_name, w in weights.items():
                    retrieve._WEIGHTS.clear()
                    retrieve._WEIGHTS.update(w)
                    t2 = time.perf_counter()
                    ranked = retrieve.search_batch(queries)
                    t3 = time.perf_counter()
                    search_time = t3 - t2
                    est_total = base_time + search_time
                    ndcg = mean_ndcg_at_k(ranked, ground_truth)
                    row = {
                        "config": cfg_name,
                        "weights": w_name,
                        "chunk_variants": chunk_variants,
                        "bm25_variants": bm25_variants,
                        "ndcg": round(ndcg, 6),
                        "search_time": round(search_time, 3),
                        "estimated_total": round(est_total, 3),
                    }
                    results.append(row)
                    print(
                        f"{cfg_name:12s} {w_name:15s} "
                        f"cv={chunk_variants} bv={int(bm25_variants)} "
                        f"ndcg={ndcg:.4f} search={search_time:.2f}s est={est_total:.2f}s",
                        flush=True,
                    )

    cfg.clear()
    cfg.update(original_cfg)
    retrieve._WEIGHTS.clear()
    retrieve._WEIGHTS.update(original_weights)

    out = ROOT / "recall_experiment_results.tsv"
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["config", "weights", "chunk_variants", "bm25_variants", "ndcg", "search_time", "estimated_total"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nTop results under estimated_total < 55s:")
    valid = [r for r in results if r["estimated_total"] < 55]
    for r in sorted(valid, key=lambda x: (-x["ndcg"], x["estimated_total"]))[:15]:
        print(r)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
