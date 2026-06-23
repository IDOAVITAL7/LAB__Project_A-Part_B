#!/usr/bin/env python3
"""
export_rrf_cache.py
===================

Offline ranking cache exporter for optimize_rrf_reference.py.

This version intentionally reuses the same runtime artifacts and helper
functions as retrieve.py, instead of trying to read separate .npy/.sqlite files
directly. That makes the offline cache much closer to the online pipeline.

It does not modify eval_public.py or retrieve.py.

Expected usage from project root:

    python3 scripts/export_rrf_cache.py \
      --output-dir optimization_runs/rrf_reference/cache/export_pd200 \
      --pool-depth 200 \
      --top-k-title 40 \
      --hnsw-ef-search 64

Smoke test:

    python3 scripts/export_rrf_cache.py --pool-depth 200 --max-queries 3
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import time
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import the real project runtime helpers.
from embed import get_model
from index import load_all_artifacts
from utils import extract_years, extract_numbers, tokenize
import retrieve


def log(msg: str) -> None:
    print(f"[export_rrf_cache] {msg}", flush=True)


def resolve_queries_path(explicit: Optional[Path] = None) -> Path:
    if explicit is not None:
        return explicit

    candidates = [
        PROJECT_ROOT / "data" / "public_queries.json",
        PROJECT_ROOT / "public_queries.json",
        PROJECT_ROOT / "sample_data" / "public_queries.json",
    ]
    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        "Could not find public_queries.json. Tried:\n"
        + "\n".join(str(p) for p in candidates)
    )


def load_public_queries(path: Path, max_queries: Optional[int] = None) -> List[Dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected list in {path}, got {type(rows).__name__}")
    if max_queries is not None:
        rows = rows[:max_queries]
    return rows


def set_hnsw_ef_search(index: Any, ef_search: int) -> None:
    """Best-effort HNSW efSearch setter; harmless for flat indexes."""
    try:
        if hasattr(index, "hnsw"):
            index.hnsw.efSearch = int(ef_search)
    except Exception:
        pass


def rank_scores(scores: Dict[int, float], top_k: int) -> Tuple[List[str], Dict[str, float]]:
    """Convert {int page_id: score} to sorted ranking and string-keyed scores."""
    items = sorted(scores.items(), key=lambda x: (-float(x[1]), int(x[0])))[:top_k]
    ranking = [str(pid) for pid, _ in items]
    score_map = {str(pid): float(score) for pid, score in items}
    return ranking, score_map


def dense_chunk_raw_hits(
    index: Any,
    qvecs: np.ndarray,
    chunk_row2pid: List[int],
    pool_depth: int,
) -> List[Dict[str, Any]]:
    """
    Raw chunk-dense hits before page aggregation.

    Online retrieve.py uses only the original query vector for chunk search:
        _faiss_chunk_search(state["chunk_index"], qvecs[:1], ...)
    This exporter mirrors that behavior.
    """
    qv = np.ascontiguousarray(qvecs[:1], dtype=np.float32)
    raw_k = min(int(pool_depth) * 3, len(chunk_row2pid))
    scores_mat, idx_mat = index.search(qv, raw_k)

    hits: List[Dict[str, Any]] = []
    for rank, (score, row) in enumerate(zip(scores_mat[0], idx_mat[0])):
        if int(row) < 0:
            continue
        row_i = int(row)
        hits.append({
            "chunk_id": row_i,
            "page_id": str(int(chunk_row2pid[row_i])),
            "rank": int(rank),       # zero-based global rank
            "score": float(score),
        })
    return hits


def bm25_raw_doc_scores(
    query_tokens: List[str],
    postings: Any,
    stats: Dict[str, Any],
) -> Dict[int, float]:
    """
    Same BM25 scoring logic as retrieve._bm25_search, but returns row/doc scores
    before row->page aggregation. Works with both dict postings and SQLite postings.
    """
    k1 = float(stats["k1"])
    b = float(stats["b"])
    avgdl = float(stats["avgdl"])
    idf = stats["idf"]
    lens = stats["lengths"]

    doc_scores: Dict[int, float] = {}
    is_sqlite = hasattr(postings, "execute")

    for token in query_tokens:
        if is_sqlite:
            db_row = postings.execute(
                "SELECT idf, blob FROM postings WHERE token = ?",
                (token,),
            ).fetchone()
            if db_row is None:
                continue
            token_idf = float(db_row[0])
            if token_idf <= 0:
                continue
            plist = pickle.loads(zlib.decompress(db_row[1]))
        else:
            plist = postings.get(token)
            if not plist:
                continue
            token_idf = float(idf.get(token, 0.0))
            if token_idf <= 0:
                continue

        for row, tf in plist:
            row_i = int(row)
            dl = lens[row_i]
            norm_tf = (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * dl / avgdl))
            doc_scores[row_i] = doc_scores.get(row_i, 0.0) + token_idf * norm_tf

    return doc_scores


def bm25_chunk_raw_hits(
    query_tokens: List[str],
    postings: Any,
    stats: Dict[str, Any],
    chunk_row2pid: List[int],
    pool_depth: int,
) -> List[Dict[str, Any]]:
    """Raw chunk-BM25 hits before page aggregation, with global zero-based ranks."""
    doc_scores = bm25_raw_doc_scores(query_tokens, postings, stats)
    items = sorted(doc_scores.items(), key=lambda x: (-float(x[1]), int(x[0])))
    items = items[: min(int(pool_depth) * 3, len(items))]

    hits: List[Dict[str, Any]] = []
    for rank, (row, score) in enumerate(items):
        row_i = int(row)
        if row_i < 0 or row_i >= len(chunk_row2pid):
            continue
        hits.append({
            "chunk_id": row_i,
            "page_id": str(int(chunk_row2pid[row_i])),
            "rank": int(rank),        # zero-based global rank across retrieved chunks
            "score": float(score),
        })
    return hits


def title_overlap_channel(
    query_tokens: List[str],
    pages_by_id: Dict[int, Any],
    top_k_title: int,
) -> Tuple[List[str], Dict[str, float]]:
    """
    Token-overlap title channel, matching retrieve.py's title_overlap feature:
        len(query_tokens ∩ title_tokens) / len(title_tokens)
    """
    q_tok_set = set(query_tokens)
    scored: List[Tuple[int, float]] = []

    for pid, meta in pages_by_id.items():
        title_tokens = set(meta.get("title_tokens", []))
        if not title_tokens:
            continue
        overlap = len(q_tok_set & title_tokens)
        score = min(overlap / max(len(title_tokens), 1), 1.0)
        if score > 0:
            scored.append((int(pid), float(score)))

    scored.sort(key=lambda x: (-x[1], x[0]))
    scored = scored[:top_k_title]
    ranking = [str(pid) for pid, _ in scored]
    scores = {str(pid): float(score) for pid, score in scored}
    return ranking, scores


def compute_features_for_candidates(
    query: str,
    q_tokens: List[str],
    all_pids: set[str],
    pages_by_id: Dict[int, Any],
    source_presence: Dict[str, int],
) -> Dict[str, Dict[str, float]]:
    """Exact-match features matching retrieve.py as closely as possible."""
    q_years = set(extract_years(query))
    q_numbers = set(extract_numbers(query))
    q_tok_set = set(q_tokens)

    features: Dict[str, Dict[str, float]] = {}
    for pid_s in all_pids:
        try:
            pid_i = int(pid_s)
        except Exception:
            continue
        meta = pages_by_id.get(pid_i, {})

        page_years = set(meta.get("years", []))
        page_numbers = set(meta.get("numbers", []))
        year_num = 1.0 if (q_years & page_years) or (q_numbers & page_numbers) else 0.0

        page_title_toks = set(meta.get("title_tokens", []))
        overlap = len(q_tok_set & page_title_toks)
        title_ov = min(overlap / max(len(page_title_toks), 1), 1.0)

        # retrieve.py source_bonus = n_sources / 5.0
        src_bonus = float(source_presence.get(pid_s, 0)) / 5.0

        features[pid_s] = {
            "year_num_match": float(year_num),
            "title_overlap": float(title_ov),
            "source_bonus": float(src_bonus),
        }

    return features


def export_rrf_cache(
    output_dir: Path,
    pool_depth: int = 200,
    top_k_title: int = 40,
    hnsw_ef_search: int = 64,
    max_queries: Optional[int] = None,
    queries_path: Optional[Path] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "rrf_cache.json"

    q_path = resolve_queries_path(queries_path)
    rows = load_public_queries(q_path, max_queries=max_queries)
    queries = [str(r["query"]) for r in rows]

    log(f"Project root: {PROJECT_ROOT}")
    log(f"Queries path: {q_path}")
    log(f"Queries loaded: {len(rows)}")
    log("Loading artifacts via index.load_all_artifacts() ...")
    state = load_all_artifacts()
    model = get_model()

    set_hnsw_ef_search(state.get("chunk_index"), hnsw_ef_search)

    log("Generating query variants and embedding them in one batch ...")
    t0 = time.perf_counter()
    variant_vecs, spans = retrieve._encode_all_variants(queries, model)
    log(f"Embedding done in {time.perf_counter() - t0:.2f}s; vectors={variant_vecs.shape}")

    results: List[Dict[str, Any]] = []

    for qi, row in enumerate(rows):
        t_q = time.perf_counter()
        query = str(row["query"])
        qid = str(row.get("query_id", f"q{qi:03d}"))
        relevant = {str(int(pid)) for pid in row.get("relevant_page_ids", [])}

        start, end = spans[qi]
        qvecs = variant_vecs[start:end]
        q_tokens = retrieve._query_tokens_for_bm25(query)

        # Dense channels, using the same helper semantics as retrieve.py.
        title_scores_int = retrieve._faiss_page_search(
            state["title_index"], qvecs, state["title_row2pid"], top_k_title
        )
        page_scores_int = retrieve._faiss_page_search(
            state["page_index"], qvecs, state["page_row2pid"], pool_depth
        )
        chunk_hits = dense_chunk_raw_hits(
            state["chunk_index"], qvecs, state["chunk_row2pid"], pool_depth
        )

        page_dense_ranking, page_dense_scores = rank_scores(page_scores_int, pool_depth)

        # BM25 page channel mirrors retrieve._bm25_search.
        bm25_page_scores_int = retrieve._bm25_search(
            q_tokens,
            state["inv_pages"],
            state["bm25_page_stats"],
            state["page_row2pid"],
            pool_depth,
        )
        page_bm25_ranking, page_bm25_scores = rank_scores(bm25_page_scores_int, pool_depth)

        # Raw chunk BM25 before page aggregation.
        chunk_bm25_hits = bm25_chunk_raw_hits(
            q_tokens,
            state["inv_chunks"],
            state["bm25_chunk_stats"],
            state["chunk_row2pid"],
            pool_depth,
        )

        # Title overlap channel/features. This is token overlap, not dense title.
        title_overlap_ranking, title_overlap_scores = title_overlap_channel(
            q_tokens, state["pages_by_id"], top_k_title
        )

        all_pids: set[str] = set(page_dense_ranking)
        all_pids.update(h["page_id"] for h in chunk_hits)
        all_pids.update(page_bm25_ranking)
        all_pids.update(h["page_id"] for h in chunk_bm25_hits)
        all_pids.update(title_overlap_ranking)

        # Source presence count, aligned to retrieve.py's source_bonus denominator of 5.
        source_presence: Dict[str, int] = {}
        for pid in page_dense_ranking:
            source_presence[pid] = source_presence.get(pid, 0) + 1
        for pid in {h["page_id"] for h in chunk_hits}:
            source_presence[pid] = source_presence.get(pid, 0) + 1
        for pid in page_bm25_ranking:
            source_presence[pid] = source_presence.get(pid, 0) + 1
        for pid in {h["page_id"] for h in chunk_bm25_hits}:
            source_presence[pid] = source_presence.get(pid, 0) + 1
        for pid in title_overlap_ranking:
            source_presence[pid] = source_presence.get(pid, 0) + 1

        features = compute_features_for_candidates(
            query=query,
            q_tokens=q_tokens,
            all_pids=all_pids,
            pages_by_id=state["pages_by_id"],
            source_presence=source_presence,
        )

        # Complete labels: all retrieved pages get 0/1; relevant pages absent from cache
        # are still included so IDCG is computed from the true ground truth.
        labels: Dict[str, int] = {pid: (1 if pid in relevant else 0) for pid in all_pids}
        for pid in relevant:
            labels[pid] = 1

        results.append({
            "query_id": qid,
            "query": query,
            "labels": labels,
            "page_dense": {
                "ranking": page_dense_ranking,
                "scores": page_dense_scores,
            },
            "chunk_dense_raw_hits": chunk_hits,
            "page_bm25": {
                "ranking": page_bm25_ranking,
                "scores": page_bm25_scores,
            },
            "chunk_bm25_raw_hits": chunk_bm25_hits,
            "title_overlap": {
                "ranking": title_overlap_ranking,
                "scores": title_overlap_scores,
            },
            "features": features,
            # Helpful diagnostics; optimizer ignores this field.
            "diagnostics": {
                "dense_title_candidates": len(title_scores_int),
                "candidate_pages": len(all_pids),
                "relevant_pages": len(relevant),
                "retrieved_relevant_pages": sum(1 for pid in relevant if pid in all_pids),
            },
        })

        log(
            f"[{qi+1}/{len(rows)}] {qid}: "
            f"pd={len(page_dense_ranking)} cd={len(chunk_hits)} "
            f"pb={len(page_bm25_ranking)} cb={len(chunk_bm25_hits)} "
            f"to={len(title_overlap_ranking)} cand={len(all_pids)} "
            f"rel_hit={sum(1 for pid in relevant if pid in all_pids)}/{len(relevant)} "
            f"time={time.perf_counter() - t_q:.2f}s"
        )

    output_file.write_text(
        json.dumps(results, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    size_mb = output_file.stat().st_size / 1_048_576
    log(f"Cache written: {output_file} ({len(results)} queries, {size_mb:.1f} MB)")
    log("Next: python3 scripts/optimize_rrf_reference.py --stage 0 --preset exact_reference")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build offline RRF ranking cache for optimize_rrf_reference.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "optimization_runs" / "rrf_reference" / "cache",
        help="Directory to write rrf_cache.json.",
    )
    p.add_argument("--pool-depth", type=int, default=200)
    p.add_argument("--top-k-title", type=int, default=40)
    p.add_argument("--hnsw-ef-search", type=int, default=64)
    p.add_argument("--max-queries", type=int, default=None, help="Smoke-test limit.")
    p.add_argument("--queries-path", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    log("=" * 60)
    log("RRF Cache Exporter (LOCAL-ONLY)")
    log("=" * 60)
    log(f"output_dir:     {args.output_dir}")
    log(f"pool_depth:     {args.pool_depth}")
    log(f"top_k_title:    {args.top_k_title}")
    log(f"hnsw_ef_search: {args.hnsw_ef_search}")
    if args.max_queries:
        log(f"max_queries:    {args.max_queries}")
    log("")

    t0 = time.perf_counter()
    try:
        export_rrf_cache(
            output_dir=args.output_dir,
            pool_depth=args.pool_depth,
            top_k_title=args.top_k_title,
            hnsw_ef_search=args.hnsw_ef_search,
            max_queries=args.max_queries,
            queries_path=args.queries_path,
        )
    except Exception as exc:
        import traceback
        log(f"FATAL ERROR: {exc}")
        traceback.print_exc()
        sys.exit(1)

    log(f"Total time: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
