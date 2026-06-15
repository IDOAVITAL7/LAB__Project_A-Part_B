"""
index.py — Offline build pipeline and artifact loading for Section B.

Offline build (not timed at grading):
  Produces all artifacts under artifacts/ needed by retrieve.py.

Artifact loading (called by retrieve.py at query time):
  Loads all artifacts once into a dict; retrieve.py caches this globally.
"""
from __future__ import annotations

import json
import math
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

from chunk import build_all_records, TitleRecord, PageRecord, ChunkRecord
from embed import embed_texts
from utils import (
    ARTIFACTS_DIR,
    ensure_artifacts_dir,
    load_pages_sorted,
    tokenize,
    save_json_gz,
    load_json_gz,
)

# ─── Artifact filenames ───────────────────────────────────────────────────────
CFG_NAME          = "config.json"
BUILD_REPORT_NAME = "build_report.json"
PAGES_META_NAME   = "pages_meta.json"
TITLE_META_NAME   = "title_meta.json"
PAGE_META_NAME    = "page_meta.json"
CHUNK_META_NAME   = "chunk_meta.json"
TITLE_INDEX_NAME  = "title_index.faiss"
PAGE_INDEX_NAME   = "page_index.faiss"
CHUNK_INDEX_NAME  = "chunk_index.faiss"
BM25_PAGE_STATS   = "bm25_page_stats.json"
BM25_CHUNK_STATS  = "bm25_chunk_stats.json"
INV_PAGES_NAME    = "inverted_index_pages.json.gz"
INV_CHUNKS_NAME   = "inverted_index_chunks.json.gz"

DIM = 384   # all-MiniLM-L6-v2 output dimension


# ─── Main offline build function ──────────────────────────────────────────────

def build_index(
    *,
    entries_dir:   Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
) -> None:
    """
    Full offline build pipeline.
    Run once locally; artifacts committed to GitHub.
    Do not call this at query time.
    """
    out     = artifacts_dir or ensure_artifacts_dir()
    t_start = time.perf_counter()

    # ── Phase 1: Corpus loading ───────────────────────────────────────────────
    print("=== Phase 1: Loading corpus ===")
    pages = load_pages_sorted(entries_dir)
    print(f"  {len(pages):,} pages loaded (sorted by page_id)")

    # ── Phase 2: Build text records ───────────────────────────────────────────
    print("=== Phase 2: Building title / page / chunk records ===")
    title_recs, page_recs, chunk_recs = build_all_records(pages)
    print(f"  Title records : {len(title_recs):,}")
    print(f"  Page records  : {len(page_recs):,}")
    print(f"  Chunk records : {len(chunk_recs):,}")

    # ── Phase 3: Dense embeddings ─────────────────────────────────────────────
    print("=== Phase 3a: Embedding title texts ===")
    title_vecs = embed_texts([r.text for r in title_recs], batch_size=64)

    print("=== Phase 3b: Embedding page texts ===")
    page_vecs  = embed_texts([r.text for r in page_recs],  batch_size=64)

    print("=== Phase 3c: Embedding chunk texts ===")
    chunk_vecs = embed_texts([r.text for r in chunk_recs], batch_size=64)
    print(f"  Embedding shapes: title{title_vecs.shape}, "
          f"page{page_vecs.shape}, chunk{chunk_vecs.shape}")

    # ── Phase 4: Build FAISS indexes ─────────────────────────────────────────
    print("=== Phase 4: Building FAISS indexes ===")
    title_index = _flat_ip_index(title_vecs)
    page_index  = _flat_ip_index(page_vecs)
    chunk_index = _flat_ip_index(chunk_vecs)

    faiss.write_index(title_index, str(out / TITLE_INDEX_NAME))
    faiss.write_index(page_index,  str(out / PAGE_INDEX_NAME))
    faiss.write_index(chunk_index, str(out / CHUNK_INDEX_NAME))
    print(f"  FAISS indexes written")
    chunk_mb = _file_mb(out / CHUNK_INDEX_NAME)
    if chunk_mb > 90:
        print(f"  ⚠ chunk_index.faiss is {chunk_mb:.1f} MB → Git LFS required!")
    else:
        print(f"  chunk_index.faiss: {chunk_mb:.1f} MB (OK)")

    # ── Phase 5: BM25 inverted indexes ───────────────────────────────────────
    print("=== Phase 5: Building BM25 indexes ===")
    page_token_lists  = [tokenize(p["title"] + " " + p["content"]) for p in pages]
    chunk_token_lists = [tokenize(r.text) for r in chunk_recs]

    page_postings,  page_stats  = _build_bm25(page_token_lists)
    chunk_postings, chunk_stats = _build_bm25(chunk_token_lists)

    save_json_gz(page_postings,  out / INV_PAGES_NAME)
    save_json_gz(chunk_postings, out / INV_CHUNKS_NAME)
    (out / BM25_PAGE_STATS ).write_text(json.dumps(page_stats,  indent=2), encoding="utf-8")
    (out / BM25_CHUNK_STATS).write_text(json.dumps(chunk_stats, indent=2), encoding="utf-8")
    print(f"  Page  vocab: {len(page_postings):,} tokens")
    print(f"  Chunk vocab: {len(chunk_postings):,} tokens")

    # ── Phase 6: Metadata ─────────────────────────────────────────────────────
    print("=== Phase 6: Saving metadata ===")

    pages_meta = [
        {
            "page_id":      p["page_id"],
            "title":        p["title"],
            "title_tokens": tokenize(p["title"]),
            "years":        list(dict.fromkeys(_extract_years(p))),
            "numbers":      list(dict.fromkeys(_extract_numbers(p))),
        }
        for p in pages
    ]
    (out / PAGES_META_NAME).write_text(json.dumps(pages_meta), encoding="utf-8")

    (out / TITLE_META_NAME).write_text(
        json.dumps([{"row": r.row, "page_id": r.page_id} for r in title_recs]),
        encoding="utf-8")
    (out / PAGE_META_NAME).write_text(
        json.dumps([
            {"row": r.row, "page_id": r.page_id,
             "years": r.years, "numbers": r.numbers}
            for r in page_recs
        ]), encoding="utf-8")
    (out / CHUNK_META_NAME).write_text(
        json.dumps([
            {"row": r.row, "page_id": r.page_id, "chunk_idx": r.chunk_idx,
             "start_word": r.start_word, "end_word": r.end_word,
             "years": r.years, "numbers": r.numbers}
            for r in chunk_recs
        ]), encoding="utf-8")

    # ── Phase 7: Config and build report ─────────────────────────────────────
    cfg = _default_config()
    (out / CFG_NAME).write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    t_end = time.perf_counter()
    report = {
        "build_time_s":   round(t_end - t_start, 1),
        "num_pages":      len(pages),
        "num_chunks":     len(chunk_recs),
        "page_vocab":     len(page_postings),
        "chunk_vocab":    len(chunk_postings),
        "artifact_sizes": {
            p.name: round(_file_mb(p), 2)
            for p in sorted(out.iterdir()) if p.is_file()
        },
    }
    (out / BUILD_REPORT_NAME).write_text(json.dumps(report, indent=2), encoding="utf-8")

    elapsed = t_end - t_start
    print(f"\n{'='*50}")
    print(f"Build complete in {elapsed:.1f}s")
    print("Artifact sizes:")
    for name, mb in report["artifact_sizes"].items():
        flag = " ← GIT LFS NEEDED" if mb > 90 else ""
        print(f"  {name:<45} {mb:>8.2f} MB{flag}")
    print(f"{'='*50}")


# ─── FAISS helpers ────────────────────────────────────────────────────────────

def _flat_ip_index(vectors: np.ndarray) -> faiss.Index:
    """
    Build an exact inner-product FAISS index.

    Vectors MUST be L2-normalized before calling this.
    Inner product on unit-norm vectors == cosine similarity.
    embed_texts() always returns normalized vectors (normalize_embeddings=True).
    """
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)
    assert vectors.ndim == 2 and vectors.shape[1] == DIM, \
        f"Expected shape (N, {DIM}), got {vectors.shape}"
    index = faiss.IndexFlatIP(DIM)
    index.add(vectors)
    return index


def _hnsw_ip_index(
    vectors: np.ndarray,
    M: int = 32,
    ef_construction: int = 200,
) -> faiss.Index:
    """
    Build an HNSW approximate index for the chunk corpus.

    ONLY use as a fallback when chunk_index.faiss exceeds storage limits
    and Git LFS is unavailable.

    CRITICAL: Always use faiss.METRIC_INNER_PRODUCT.
    Default HNSW metric is L2 distance, which is wrong for normalized vectors
    and will silently produce incorrect rankings.
    """
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)
    index = faiss.IndexHNSWFlat(DIM, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = 128    # adjustable after loading
    index.add(vectors)
    return index


# ─── BM25 builder ─────────────────────────────────────────────────────────────

def _build_bm25(
    token_lists: List[List[str]],
    k1: float = 1.2,
    b:  float = 0.75,
) -> Tuple[Dict, Dict]:
    """
    Build offline BM25 inverted index.

    Parameters
    ----------
    token_lists : one list of tokens per document

    Returns
    -------
    postings : { token: [[row, tf], ...] }
    stats    : { N, avgdl, lengths, idf, k1, b }
    """
    N    = len(token_lists)
    lens = [len(toks) for toks in token_lists]
    avgdl = sum(lens) / N if N else 1.0

    # Document frequency
    df: Dict[str, int] = {}
    for toks in token_lists:
        for t in set(toks):    # set: count each token once per doc
            df[t] = df.get(t, 0) + 1

    # IDF: Robertson-Sparck Jones variant (always non-negative)
    idf: Dict[str, float] = {
        t: math.log(1.0 + (N - dft + 0.5) / (dft + 0.5))
        for t, dft in df.items()
    }

    # Postings: built in one pass over all documents
    postings: Dict[str, List] = {}
    for row, toks in enumerate(token_lists):
        tf_local: Dict[str, int] = {}
        for t in toks:
            tf_local[t] = tf_local.get(t, 0) + 1
        for t, freq in tf_local.items():
            if t not in postings:
                postings[t] = []
            postings[t].append([row, freq])

    stats = {
        "N":       N,
        "avgdl":   avgdl,
        "lengths": lens,
        "idf":     idf,
        "k1":      k1,
        "b":       b,
    }
    return postings, stats


# ─── Artifact loader (used by retrieve.py) ───────────────────────────────────

def load_all_artifacts(artifacts_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load everything retrieve.py needs into one dict.
    Called once per process; result is cached globally in retrieve.py.
    """
    root = artifacts_dir or ARTIFACTS_DIR

    # Configuration
    cfg = json.loads((root / CFG_NAME).read_text(encoding="utf-8"))

    # Metadata
    pages_meta  = json.loads((root / PAGES_META_NAME ).read_text(encoding="utf-8"))
    title_meta  = json.loads((root / TITLE_META_NAME ).read_text(encoding="utf-8"))
    page_meta   = json.loads((root / PAGE_META_NAME  ).read_text(encoding="utf-8"))
    chunk_row2pid_path = root / "chunk_row2pid.npy"
    if chunk_row2pid_path.exists():
        chunk_meta = []
    else:
        chunk_meta = json.loads((root / CHUNK_META_NAME).read_text(encoding="utf-8"))

    # FAISS indexes
    title_index = faiss.read_index(str(root / TITLE_INDEX_NAME))
    page_index  = faiss.read_index(str(root / PAGE_INDEX_NAME))
    chunk_index = faiss.read_index(str(root / CHUNK_INDEX_NAME))

    # Optionally increase efSearch on HNSW indexes
    for idx in (title_index, page_index, chunk_index):
        if hasattr(idx, "hnsw"):
            idx.hnsw.efSearch = cfg.get("hnsw_ef_search", 96)

    # BM25
    bm25_page_stats = json.loads((root / BM25_PAGE_STATS).read_text(encoding="utf-8"))
    inv_pages = load_json_gz(root / INV_PAGES_NAME)

    if cfg.get("use_bm25_chunk_sqlite", False):
        bm25_chunk_stats = json.loads((root / "bm25_chunk_stats_light.json").read_text(encoding="utf-8"))
        bm25_chunk_stats["lengths"] = np.load(root / "bm25_chunk_lengths.npy")
        bm25_chunk_stats["idf"] = {}
        sqlite_path = root / "bm25_chunks.sqlite"

        if cfg.get("bm25_chunk_sqlite_in_memory", True):
            disk_conn = sqlite3.connect(str(sqlite_path))
            disk_conn.execute("PRAGMA query_only=ON")

            mem_conn = sqlite3.connect(":memory:")
            disk_conn.backup(mem_conn)
            disk_conn.close()

            inv_chunks = mem_conn
        else:
            inv_chunks = sqlite3.connect(str(sqlite_path))

        inv_chunks.execute("PRAGMA query_only=ON")
        inv_chunks.execute("PRAGMA temp_store=MEMORY")
        inv_chunks.execute("PRAGMA cache_size=-200000")
    elif cfg.get("use_bm25_chunk", True):
        bm25_chunk_stats = json.loads((root / BM25_CHUNK_STATS).read_text(encoding="utf-8"))
        inv_chunks = load_json_gz(root / INV_CHUNKS_NAME)
    else:
        bm25_chunk_stats = {
            "N": 0,
            "avgdl": 1.0,
            "lengths": [],
            "idf": {},
            "k1": 1.2,
            "b": 0.75,
        }
        inv_chunks = {}

    # Precomputed row-to-page_id arrays (avoids dict lookup per FAISS result)
    title_row2pid = [int(r["page_id"]) for r in title_meta]
    page_row2pid  = [int(r["page_id"]) for r in page_meta]
    if chunk_row2pid_path.exists():
        chunk_row2pid = np.load(chunk_row2pid_path).astype(np.int64).tolist()
    else:
        chunk_row2pid = [int(r["page_id"]) for r in chunk_meta]

    # Dict: page_id → metadata for exact-match features in reranker
    pages_by_id: Dict[int, Any] = {
        m["page_id"]: m for m in pages_meta
    }

    return {
        "cfg":              cfg,
        "title_index":      title_index,
        "page_index":       page_index,
        "chunk_index":      chunk_index,
        "title_row2pid":    title_row2pid,
        "page_row2pid":     page_row2pid,
        "chunk_row2pid":    chunk_row2pid,
        "chunk_meta":       chunk_meta,
        "page_meta":        page_meta,
        "pages_by_id":      pages_by_id,
        "bm25_page_stats":  bm25_page_stats,
        "bm25_chunk_stats": bm25_chunk_stats,
        "inv_pages":        inv_pages,
        "inv_chunks":       inv_chunks,
    }


# ─── Legacy compatibility stubs ───────────────────────────────────────────────
# Kept so that the read-only scripts/build_index.py → main.build_offline_index()
# → index.build_index() chain still works.

INDEX_VECTORS_NAME = "index_vectors.npy"   # kept for reference, not used
INDEX_META_NAME    = "index_meta.json"


def load_index(artifacts_dir: Optional[Path] = None):
    """
    Legacy load function — not used by new pipeline.
    Returns (empty_array, []) to avoid breaking any stale references.
    """
    return np.zeros((0, DIM), dtype=np.float32), []


# ─── Private helpers ──────────────────────────────────────────────────────────

def _file_mb(path: Path) -> float:
    try:
        return path.stat().st_size / 1e6
    except FileNotFoundError:
        return 0.0


def _extract_years(page: Dict[str, Any]) -> List[str]:
    from utils import extract_years
    return extract_years(page["title"] + " " + page["content"])


def _extract_numbers(page: Dict[str, Any]) -> List[str]:
    from utils import extract_numbers
    return extract_numbers(page["title"] + " " + page["content"])


def _default_config() -> Dict[str, Any]:
    """Canonical config values — keep in sync with chunk.py constants."""
    return {
        "embedding_model":        "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_dim":          DIM,
        "normalize_embeddings":   True,
        "faiss_metric":           "inner_product",
        "page_truncation_words":  400,
        "chunk_size_words":       200,
        "chunk_step_words":       160,
        "title_prefix_in_chunks": True,
        "bm25_k1":                1.2,
        "bm25_b":                 0.75,
        "retrieval_top_k_title":  50,
        "retrieval_top_k_page":   150,
        "retrieval_top_k_chunk":  500,
        "retrieval_top_k_bm25_page":  200,
        "retrieval_top_k_bm25_chunk": 300,
        "schema_version": 1,
    }
