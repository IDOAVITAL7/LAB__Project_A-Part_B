# Section B — End-to-End Retrieval Pipeline

> **Project:** Project A - Section B Retrieval Pipeline  
> **Repository:** https://github.com/IDOAVITAL7/LAB__Project_A-Part_B  
> **Team Members:** Ido Avital, 207850280 | Eliezer Mashihov, 207476763  
> **Presentation Video:** https://technionmail-my.sharepoint.com/:f:/g/personal/avital_ido_campus_technion_ac_il/IgDqq-d1v3_CT7K8LbqX-8YvASHQV0FJO35h8ANzyCS_GTY?e=kWscmm
>
> **Final Public Validation Result:** `mean_ndcg@10 = 0.3310` | `query_phase_time = 35.85s`

---

## 1. Executive Summary

This repository implements an end-to-end retrieval pipeline for Section B of Project A.  
The system receives a batch of natural-language queries and returns, for each query, a ranked list of the most relevant page IDs.

The final submitted approach is a **tuned weighted hybrid retrieval pipeline**. It combines dense semantic retrieval, lexical BM25 retrieval, and deterministic reranking features.

Final verified result from a fresh clone:

```text
public_queries=50
mean_ndcg@10=0.3310
query_phase_time=35.85s
```

The final design was selected empirically. We tested a separate RRF-based fusion approach, but it underperformed the tuned weighted hybrid reranker on the public validation queries. Therefore, the final submission keeps the stronger weighted hybrid solution.

---

## 2. Quick Start

The repository is designed to run from a fresh clone without rebuilding the index.

```bash
git clone https://github.com/IDOAVITAL7/LAB__Project_A-Part_B.git
cd LAB__Project_A-Part_B

git lfs pull || true

python3 -m py_compile main.py chunk.py embed.py index.py retrieve.py utils.py
python3 scripts/eval_public.py
```

Expected output:

```text
public_queries=50
mean_ndcg@10=0.3310
query_phase_time≈35-36s
```

Important: the grader should not need to rebuild the index. The required runtime artifacts are already included under `artifacts/`.

---

## 3. Final Pipeline Architecture

### 3.1 High-level flow

```text
queries
  ↓
query variant generation
  ↓
single batched MiniLM embedding call
  ↓
candidate generation from multiple retrieval channels
  ├── dense title retrieval
  ├── dense full-page retrieval
  ├── dense chunk retrieval
  ├── page-level BM25 retrieval
  └── chunk-level BM25 retrieval
  ↓
candidate union
  ↓
score normalization and deterministic feature extraction
  ↓
weighted hybrid reranking
  ↓
top page IDs per query
```

### 3.2 Main runtime modules

| File | Role |
|---|---|
| `main.py` | Exposes the required `run(queries)` API. |
| `retrieve.py` | Implements the online retrieval and reranking pipeline. |
| `embed.py` | Loads and reuses the MiniLM sentence-transformer model. |
| `index.py` | Loads all prebuilt runtime artifacts from `artifacts/`. |
| `chunk.py` | Supports the offline chunking logic used to build the artifacts. |
| `utils.py` | Utility functions for tokenization, top-k selection, and exact-match extraction. |

---

## 4. Retrieval Channels

The final system combines five retrieval channels plus deterministic reranking features.

### 4.1 Dense semantic retrieval

Dense retrieval is based on:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The dense retrieval channels are:

| Channel | Artifact | Purpose |
|---|---|---|
| Dense title retrieval | `title_index.faiss` | Captures strong title-level semantic matches. |
| Dense page retrieval | `page_index.faiss` | Captures broad page-level semantic relevance. |
| Dense chunk retrieval | `chunk_index.faiss` | Captures focused passage-level relevance. |

### 4.2 Lexical BM25 retrieval

The lexical channels are:

| Channel | Artifact(s) | Purpose |
|---|---|---|
| Page-level BM25 | `inverted_index_pages.json.gz`, `bm25_page_stats.json` | Captures exact lexical matches at page level. |
| Chunk-level BM25 | `bm25_chunks.sqlite`, `bm25_chunk_lengths.npy`, `bm25_chunk_stats_light.json` | Captures exact lexical matches in smaller passages. |

BM25 is especially helpful for factual queries that contain names, years, numbers, or exact terminology.

### 4.3 Query variants

Each query is expanded into a small number of variants, such as:

1. original query,
2. cleaned lowercase query,
3. keyword-only query,
4. sub-parts for multi-part questions,
5. numeric/year-emphasis variant when relevant.

All variants for all input queries are embedded in a single batched model call.  
This reduces overhead and helps keep query-time runtime below the 60-second limit.

---

## 5. Weighted Hybrid Reranking

After candidate generation, the system creates a union of candidate page IDs from all retrieval channels and reranks them deterministically.

### 5.1 Normalized retrieval signals

The final reranker combines normalized scores from:

- dense title retrieval,
- dense page retrieval,
- dense chunk retrieval,
- page-level BM25,
- chunk-level BM25.

### 5.2 Deterministic exact-match features

The reranker also includes deterministic features:

| Feature | Meaning |
|---|---|
| `title_overlap` | Token overlap between query tokens and page-title tokens. |
| `year_num_match` | Exact match between years/numbers in the query and page metadata. |
| `source_bonus` | Rewards pages retrieved by multiple independent sources. |

The `year_num_match` feature was important because many queries include years, dates, quantities, or other exact identifiers.

### 5.3 Why weighted hybrid was selected

The tuned weighted hybrid reranker achieved the best verified public result:

```text
mean_ndcg@10=0.3310
```

It outperformed the tested RRF variants because it preserved the influence of deterministic exact-match signals while still benefiting from dense and BM25 retrieval.

---

## 6. Artifact Inventory

All required runtime artifacts are stored under `artifacts/`.

### 6.1 Configuration and build metadata

| Artifact | Size | Purpose |
|---|---:|---|
| `config.json` | 1,703 bytes | Runtime retrieval configuration and tuned fusion weights. |
| `build_report.json` | 544 bytes | Metadata from the offline index build. |

### 6.2 Dense FAISS indexes

| Artifact | Size | Purpose |
|---|---:|---|
| `title_index.faiss` | 41,585,709 bytes | Dense title-level FAISS index. |
| `page_index.faiss` | 41,585,709 bytes | Dense full-page FAISS index. |
| `chunk_index.faiss` | 695,387,122 bytes | Dense chunk-level FAISS index. Tracked with Git LFS. |

### 6.3 BM25 lexical artifacts

| Artifact | Size | Purpose |
|---|---:|---|
| `inverted_index_pages.json.gz` | 56,793,827 bytes | Compressed page-level BM25 inverted index. |
| `bm25_page_stats.json` | 24,866,910 bytes | Page-level BM25 statistics. |
| `bm25_chunks.sqlite` | 197,451,776 bytes | SQLite-backed chunk-level BM25 postings. Tracked with Git LFS. |
| `bm25_chunk_lengths.npy` | 1,538,480 bytes | Chunk length information for BM25 normalization. |
| `bm25_chunk_stats_light.json` | 73 bytes | Lightweight chunk-level BM25 statistics. |

### 6.4 Metadata and mappings

| Artifact | Size | Purpose |
|---|---:|---|
| `pages_meta.json` | 9,638,804 bytes | Page-level metadata used by the reranker. |
| `page_meta.json` | 13,468,769 bytes | Metadata for the dense page index. |
| `title_meta.json` | 904,543 bytes | Metadata for the dense title index. |
| `chunk_row2pid.npy` | 1,538,480 bytes | Mapping from chunk index row to parent page ID. |

### 6.5 Git LFS artifacts

The following large files are tracked through Git LFS:

```text
artifacts/bm25_chunks.sqlite
artifacts/chunk_index.faiss
```

Fresh clone validation confirmed that `git lfs pull` retrieves the required large files successfully.

---

## 7. Fresh Clone Validation

The final repository was validated from a clean clone using the following commands:

```bash
cd /tmp
rm -rf final_clone_test

git clone https://github.com/IDOAVITAL7/LAB__Project_A-Part_B.git final_clone_test
cd final_clone_test

git lfs pull || true

python3 -m py_compile main.py chunk.py embed.py index.py retrieve.py utils.py
python3 scripts/eval_public.py | tee fresh_clone_final_eval.log
```

Observed output:

```text
[retrieve] Loading artifacts into cache...
[embed] using GPU: Tesla M60
[retrieve] Cache ready.
public_queries=50
mean_ndcg@10=0.3310
query_phase_time=35.85s
```

This confirms that the repository can be graded from a fresh clone without local index rebuilding.

---

## 8. Notes for the Course Staff

- The repository includes prebuilt runtime artifacts under `artifacts/`.
- The query-time pipeline does not require rebuilding the index.
- Large artifacts are handled through Git LFS.
- The final approach is the tuned weighted hybrid pipeline.
- The RRF experiment was intentionally not selected because it underperformed the final weighted hybrid approach.
- The final result was validated from a clean clone.

