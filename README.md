# Section B – Hybrid Retrieval System

This repository contains the final Section B retrieval solution.

## Overview

The project implements an end-to-end retrieval system over the provided Wikipedia-style corpus.  
The final solution combines dense semantic retrieval, lexical retrieval, and lightweight reranking.

Main components:

- SentenceTransformer embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- FAISS dense retrieval
- HNSW FAISS index for chunk-level retrieval
- BM25 page-level retrieval
- SQLite-backed BM25 chunk retrieval for fast runtime access
- In-memory SQLite loading for stable query latency
- Hybrid reranking using dense, lexical, title, year/number, and source features

## Required API

The required API is implemented in `main.py`:

```python
run(queries: list[str]) -> list[list[int]]
```

For each query, the system returns a ranked list of 10 page IDs.

## Public Evaluation Result

Expected result on the provided public queries using the submitted artifacts:

```text
public_queries=50
mean_ndcg@10≈0.2794
query_phase_time≈34–35s
```

The runtime is safely below the 60-second limit on the provided GPU environment.

## How to Run the Public Evaluation

From the repository root:

```bash
python3 scripts/eval_public.py
```

## Runtime Artifacts

The repository includes prebuilt runtime artifacts under `artifacts/`.

Required runtime artifacts:

```text
config.json
build_report.json
pages_meta.json
title_meta.json
page_meta.json
title_index.faiss
page_index.faiss
chunk_index.faiss
chunk_row2pid.npy
bm25_page_stats.json
inverted_index_pages.json.gz
bm25_chunks.sqlite
bm25_chunk_lengths.npy
bm25_chunk_stats_light.json
```

Large artifacts are tracked with Git LFS:

```text
artifacts/chunk_index.faiss
artifacts/bm25_chunks.sqlite
```

## Notes About the Raw Corpus

The full raw Wikipedia corpus is not required at runtime.  
The submitted `run()` implementation uses the prebuilt artifacts.

The raw corpus directory is intentionally excluded from Git:

```text
data/Wikipedia Entries/
```

## Dependencies

The solution uses the allowed project dependencies:

```text
numpy
sentence-transformers
faiss / faiss-cpu
```

## Demo Video

Video link: TODO
