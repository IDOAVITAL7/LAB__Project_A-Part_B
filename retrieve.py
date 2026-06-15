"""
retrieve.py — Query-time hybrid retrieval (timed portion).

This module is the heart of the online pipeline. It must:
  1. Load all artifacts once (module-level cache).
  2. Generate query variants for all input queries.
  3. Call model.encode() exactly ONCE for all variants.
  4. Search FAISS indexes (title, page, chunk).
  5. Search BM25 inverted indexes (page, chunk).
  6. Merge candidates and compute feature vectors.
  7. Rerank deterministically with weighted fusion.
  8. Return top-10 unique page_ids per query as int.

Runtime contract: run(50 queries) must complete in ≤60s on GPU.
"""
from __future__ import annotations

import re
import pickle
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from embed import get_model
from index import load_all_artifacts
from utils import (
    K_EVAL,
    extract_years, extract_numbers,
    tokenize, top_k_items,
)

# ─── Module-level global cache ────────────────────────────────────────────────
# Populated on the first run() call; subsequent calls reuse without reloading.
_STATE: Optional[Dict[str, Any]] = None


def _get_state() -> Dict[str, Any]:
    global _STATE
    if _STATE is None:
        print("[retrieve] Loading artifacts into cache...")
        _STATE = load_all_artifacts()
        # Warm model (downloads weights if not yet cached; instant afterward)
        get_model()
        print("[retrieve] Cache ready.")
    return _STATE


# ─── Fusion weights ───────────────────────────────────────────────────────────
# Starting values — tune via ablation on public queries (scripts/ablation.py).
# Do NOT adjust these based on individual public query IDs.
_WEIGHTS: Dict[str, float] = {
    "dense_chunk":    0.20,
    "bm25_chunk":     0.36,
    "dense_page":     0.13,
    "bm25_page":      0.14,
    "dense_title":    0.06,
    "title_overlap":  0.04,
    "year_num_match": 0.05,
    "source_bonus":   0.02,
}


# ─── Main entry point ─────────────────────────────────────────────────────────

def search_batch(
    queries: List[str],
    *,
    top_k: int = K_EVAL,
    artifacts_dir: Optional[Path] = None,
) -> List[List[int]]:
    """
    Return ranked page_id lists for each query.
    Called by main.run(); this is the entire online pipeline.
    """
    state = _get_state()
    cfg   = state["cfg"]
    model = get_model()

    if not queries:
        return []

    # ── Step 1: One-shot batch encoding ───────────────────────────────────────
    # Generate ALL variants for ALL queries, then encode in ONE call.
    variant_vecs, spans = _encode_all_variants(queries, model)

    # ── Steps 2–6: Per-query retrieval and reranking ───────────────────────────
    results: List[List[int]] = []

    tk_title   = cfg.get("retrieval_top_k_title",       50)
    tk_page    = cfg.get("retrieval_top_k_page",        150)
    tk_chunk   = cfg.get("retrieval_top_k_chunk",       500)
    tk_bp      = cfg.get("retrieval_top_k_bm25_page",   200)
    tk_bc      = cfg.get("retrieval_top_k_bm25_chunk",  300)

    for q_idx, query in enumerate(queries):
        start, end = spans[q_idx]
        qvecs = variant_vecs[start:end]   # (n_variants, 384)

        # Dense retrieval
        title_scores = _faiss_page_search(
            state["title_index"], qvecs, state["title_row2pid"], tk_title)
        page_scores  = _faiss_page_search(
            state["page_index"],  qvecs, state["page_row2pid"],  tk_page)
        chunk_scores = _faiss_chunk_search(
            state["chunk_index"], qvecs[:1], state["chunk_row2pid"], tk_chunk)

        # BM25 retrieval (inverted index — no full-corpus scan)
        q_tokens = _query_tokens_for_bm25(query)
        bm25_page_scores  = _bm25_search(
            q_tokens, state["inv_pages"],  state["bm25_page_stats"],
            state["page_row2pid"],  tk_bp)
        bm25_chunk_scores = _bm25_search(
            q_tokens, state["inv_chunks"], state["bm25_chunk_stats"],
            state["chunk_row2pid"], tk_bc)

        # Rerank and return top-k
        ranked = _rerank(
            query=query,
            q_tokens=q_tokens,
            title_scores=title_scores,
            page_scores=page_scores,
            chunk_scores=chunk_scores,
            bm25_page_scores=bm25_page_scores,
            bm25_chunk_scores=bm25_chunk_scores,
            pages_by_id=state["pages_by_id"],
            top_k=max(top_k, 10),
        )
        results.append(ranked)

    return results


# ─── Query variant generation ─────────────────────────────────────────────────

_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "about", "what", "which",
    "who", "where", "when", "how", "did", "does", "was", "were",
    "is", "are", "that", "this", "it", "its", "their", "his", "her",
    "can", "do", "be", "have", "had", "has", "not", "no", "up",
    "also", "as", "so", "if", "he", "she", "they", "we", "you",
    "after", "before", "during", "than", "more", "most", "some",
    "links", "connect", "learned", "together", "both",
})


def _query_tokens_for_bm25(query: str) -> List[str]:
    """
    BM25 should not score stopwords online because frequent tokens create huge
    posting-list scans and weak ranking signals. Keep informative words and
    numeric tokens; fall back to raw tokens only if filtering removes everything.
    """
    toks = tokenize(query)
    filtered = [
        t for t in toks
        if (t not in _STOPWORDS and (len(t) >= 3 or t.isdigit()))
    ]
    return filtered or toks

_MULTI_PART_PREFIXES = re.compile(
    r"^(what links|how do|how did|what can be learned about)\s+",
    re.IGNORECASE,
)


def _generate_variants(query: str) -> List[str]:
    """
    Expand one query into ≤5 variants.

    Variant types:
    1. Original (always included)
    2. Cleaned lowercase (punctuation removed)
    3. Keyword-only (stopwords dropped)
    4. Sub-parts for multi-part queries ("What links A, B, and C")
    5. Number/year emphasis (numeric tokens moved to front)
    """
    variants: List[str] = [query]

    # Variant 2: cleaned lowercase
    cleaned = re.sub(r"[^\w\s]", " ", query.lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned and cleaned not in variants:
        variants.append(cleaned)

    # Variant 3: keyword-only
    kw = [w for w in cleaned.split() if w not in _STOPWORDS and len(w) >= 3]
    if kw and len(kw) < len(cleaned.split()):
        kw_str = " ".join(kw)
        if kw_str not in variants:
            variants.append(kw_str)

    # Variant 4: sub-parts for structural query templates
    body = _MULTI_PART_PREFIXES.sub("", query.lower()).rstrip("?").strip()
    if body != cleaned:
        parts = re.split(r",\s*| and ", body)
        for part in parts:
            part = part.strip()
            if len(part) >= 10 and part not in variants:
                variants.append(part)
                if len(variants) >= 5:
                    break

    # Variant 5: numeric/year emphasis — move digits to front
    nums = re.findall(r"\b[\d,]+\b", query)
    if nums:
        year_num_prefix = " ".join(n.replace(",", "") for n in nums)
        rest_kw = [w for w in kw if not re.match(r"^\d+$", w)]
        emphasis = (year_num_prefix + " " + " ".join(rest_kw[:6])).strip()
        if emphasis not in variants and len(variants) < 5:
            variants.append(emphasis)

    return variants[:5]   # hard cap


def _encode_all_variants(
    queries: List[str],
    model,
    batch_size: int = 64,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Flatten all query variants and encode in a single model.encode() call.

    Returns
    -------
    variant_vecs : shape (total_variants, 384), float32, L2-normalized
    spans        : list of (start, end) index pairs — one per query
                   variant_vecs[start:end] are the vectors for queries[i]
    """
    all_variants: List[str] = []
    spans: List[Tuple[int, int]] = []

    for q in queries:
        vs = _generate_variants(q)
        start = len(all_variants)
        all_variants.extend(vs)
        spans.append((start, len(all_variants)))

    if not all_variants:
        return np.zeros((0, 384), dtype=np.float32), spans

    vecs = model.encode(
        all_variants,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,   # cosine via inner product on normalized vecs
    )
    return np.asarray(vecs, dtype=np.float32), spans


# ─── FAISS retrieval ──────────────────────────────────────────────────────────

def _faiss_page_search(
    index:     Any,
    qvecs:     np.ndarray,   # (n_variants, 384)
    row2pid:   List[int],
    top_k:     int,
) -> Dict[int, float]:
    """
    Search a page-level FAISS index (title or page) with all query variants.
    Returns page_id → max_score over variants.
    """
    qv = np.ascontiguousarray(qvecs, dtype=np.float32)
    scores_mat, idx_mat = index.search(qv, top_k)

    page_scores: Dict[int, float] = {}
    for row_scores, row_idxs in zip(scores_mat, idx_mat):
        for score, faiss_row in zip(row_scores, row_idxs):
            if faiss_row < 0:
                continue
            pid = row2pid[int(faiss_row)]
            if score > page_scores.get(pid, -1e9):
                page_scores[pid] = float(score)
    return page_scores


def _faiss_chunk_search(
    index:         Any,
    qvecs:         np.ndarray,   # (n_variants, 384)
    chunk_row2pid: List[int],
    top_k:         int,
) -> Dict[int, float]:
    """
    Search the chunk FAISS index and aggregate to page_id level.
    Aggregation: max chunk score per page (a single highly relevant chunk
    is a strong enough signal).
    """
    qv = np.ascontiguousarray(qvecs, dtype=np.float32)
    scores_mat, idx_mat = index.search(qv, top_k)

    page_scores: Dict[int, float] = {}
    for row_scores, row_idxs in zip(scores_mat, idx_mat):
        for score, faiss_row in zip(row_scores, row_idxs):
            if faiss_row < 0:
                continue
            pid = chunk_row2pid[int(faiss_row)]
            if score > page_scores.get(pid, -1e9):
                page_scores[pid] = float(score)
    return page_scores


# ─── BM25 retrieval ──────────────────────────────────────────────────────────

def _bm25_search(
    query_tokens: List[str],
    postings:     Dict[str, List],   # token → [[row, tf], ...]
    stats:        Dict[str, Any],
    row2pid:      List[int],
    top_n:        int,
) -> Dict[int, float]:
    """
    Score documents that appear in at least one posting list.

    Complexity: O(Σ |postings[t]| for t in query_tokens)
    Not O(N_documents) — no full-corpus scan.
    """
    k1    = stats["k1"]
    b     = stats["b"]
    avgdl = stats["avgdl"]
    idf   = stats["idf"]
    lens  = stats["lengths"]

    doc_scores: Dict[int, float] = {}

    is_sqlite = hasattr(postings, "execute")

    for token in query_tokens:
        if is_sqlite:
            db_row = postings.execute(
                "SELECT idf, blob FROM postings WHERE token = ?",
                (token,)
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
            token_idf = idf.get(token, 0.0)
            if token_idf <= 0:
                continue

        for row, tf in plist:
            dl       = lens[row]
            norm_tf  = (tf * (k1 + 1)) / (tf + k1 * (1.0 - b + b * dl / avgdl))
            doc_scores[row] = doc_scores.get(row, 0.0) + token_idf * norm_tf

    # Aggregate row → page_id, keeping max score per page
    page_scores: Dict[int, float] = {}
    for row, score in doc_scores.items():
        pid = row2pid[row]
        if score > page_scores.get(pid, -1e9):
            page_scores[pid] = score

    return dict(top_k_items(page_scores, top_n))


# ─── Reranking ────────────────────────────────────────────────────────────────

def _normalize_scores(
    scores: Dict[int, float],
    pids:   List[int],
) -> Dict[int, float]:
    """
    Min-max normalize scores over the candidate set.
    Pages absent from a source receive 0.0 after normalization.
    """
    if not scores:
        return {p: 0.0 for p in pids}
    vals = [scores.get(p, 0.0) for p in pids]
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return {p: 0.0 for p in pids}
    span = hi - lo
    return {p: (scores.get(p, 0.0) - lo) / span for p in pids}


def _rerank(
    *,
    query:             str,
    q_tokens:          List[str],
    title_scores:      Dict[int, float],
    page_scores:       Dict[int, float],
    chunk_scores:      Dict[int, float],
    bm25_page_scores:  Dict[int, float],
    bm25_chunk_scores: Dict[int, float],
    pages_by_id:       Dict[int, Any],
    top_k:             int,
) -> List[int]:
    """
    Merge all retrieval sources and return a deterministic ranked list.
    """
    all_pids = list(
        set(title_scores)
        | set(page_scores)
        | set(chunk_scores)
        | set(bm25_page_scores)
        | set(bm25_chunk_scores)
    )
    if not all_pids:
        return []

    # Normalize each source over the full candidate set
    n_t  = _normalize_scores(title_scores,      all_pids)
    n_p  = _normalize_scores(page_scores,        all_pids)
    n_c  = _normalize_scores(chunk_scores,       all_pids)
    n_bp = _normalize_scores(bm25_page_scores,   all_pids)
    n_bc = _normalize_scores(bm25_chunk_scores,  all_pids)

    # Exact-match signals
    q_years   = set(extract_years(query))
    q_numbers = set(extract_numbers(query))
    q_tok_set = set(q_tokens)

    W = _WEIGHTS

    final_scores: Dict[int, float] = {}
    for pid in all_pids:
        meta = pages_by_id.get(pid, {})

        # Year / number exact match
        page_years   = set(meta.get("years",   []))
        page_numbers = set(meta.get("numbers", []))
        year_num = 1.0 if (q_years & page_years) or (q_numbers & page_numbers) else 0.0

        # Title token overlap (normalized by page title length)
        page_title_toks = set(meta.get("title_tokens", []))
        overlap = len(q_tok_set & page_title_toks)
        title_ov = min(overlap / max(len(page_title_toks), 1), 1.0)

        # Source count bonus (pages found by more sources are more likely relevant)
        n_sources = sum([
            pid in title_scores,
            pid in page_scores,
            pid in chunk_scores,
            pid in bm25_page_scores,
            pid in bm25_chunk_scores,
        ])
        src_bonus = n_sources / 5.0

        score = (
            W["dense_chunk"]    * n_c.get(pid,  0.0)
          + W["bm25_chunk"]     * n_bc.get(pid, 0.0)
          + W["dense_page"]     * n_p.get(pid,  0.0)
          + W["bm25_page"]      * n_bp.get(pid, 0.0)
          + W["dense_title"]    * n_t.get(pid,  0.0)
          + W["title_overlap"]  * title_ov
          + W["year_num_match"] * year_num
          + W["source_bonus"]   * src_bonus
        )
        final_scores[pid] = score

    ranked_pairs = top_k_items(final_scores, top_k)
    return [int(pid) for pid, _ in ranked_pairs]
