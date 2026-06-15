"""Shared paths and helpers for Section B.

This file extends the starter utils.py with:
  - tokenization (standard library only)
  - year / number extraction
  - gzip JSON helpers
  - top-k helper
  - robust page loading (handles control chars in JSON)
"""
from __future__ import annotations

import gzip
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

# ─── Paths ───────────────────────────────────────────────────────────────────
STUDENT_ROOT        = Path(__file__).resolve().parent
DATA_DIR            = STUDENT_ROOT / "data"
ENTRIES_DIR         = DATA_DIR / "Wikipedia Entries"
PUBLIC_QUERIES_PATH = DATA_DIR / "public_queries.json"
ARTIFACTS_DIR       = STUDENT_ROOT / "artifacts"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
K_EVAL = 10


# ─── Page-id normalization (kept from starter) ────────────────────────────────
def normalize_page_id(value: Any) -> int:
    """Coerce page_id from JSON (int or numeric string) to int."""
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    raise ValueError(f"Invalid page_id: {value!r}")


# ─── Tokenization (BM25 / exact features) ────────────────────────────────────
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)?")


def tokenize(text: str) -> List[str]:
    """
    Lowercase + comma removal + regex token extraction.

    Removing commas before tokenizing converts "1,456,779" to the single
    token "1456779", matching the same conversion done during build-time
    number extraction.  Consistency between index and query tokenization is
    essential — a mismatch silently kills BM25 recall.
    """
    cleaned = text.lower().replace(",", "")
    return _TOKEN_RE.findall(cleaned)


# ─── Year / number extraction ─────────────────────────────────────────────────
_YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20[0-2]\d)\b")
_NUM_RE  = re.compile(r"\b\d{4,}\b")   # 4+ digit numbers (years already covered)


def extract_years(text: str) -> List[str]:
    """Return all 4-digit year strings found in text."""
    return _YEAR_RE.findall(text)


def extract_numbers(text: str) -> List[str]:
    """
    Return 4+ digit numeric strings (population counts, IDs, etc.).
    Commas are stripped first so 1,456,779 → 1456779.
    """
    return _NUM_RE.findall(text.replace(",", ""))


# ─── Gzip JSON helpers ────────────────────────────────────────────────────────
def load_json_gz(path: Path) -> Any:
    """Load a gzip-compressed JSON file."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def save_json_gz(obj: Any, path: Path) -> None:
    """Save an object as gzip-compressed JSON (compact separators)."""
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f, separators=(",", ":"))


# ─── Top-k helper ─────────────────────────────────────────────────────────────
def top_k_items(scores: Dict[int, float], k: int) -> List[Tuple[int, float]]:
    """
    Return up to k (key, score) pairs sorted descending by score.
    Uses heapq.nlargest when len(scores) >> k for efficiency.
    """
    import heapq
    if not scores:
        return []
    if len(scores) <= k:
        return sorted(scores.items(), key=lambda x: -x[1])
    return heapq.nlargest(k, scores.items(), key=lambda x: x[1])


# ─── Corpus loading ───────────────────────────────────────────────────────────
def load_pages_sorted(entries_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Load all corpus JSON pages, sorted deterministically by page_id.

    Uses json.loads(raw, strict=False) to handle corpus files that contain
    literal ASCII control characters (several files in the corpus trigger
    JSONDecodeError with strict=True).

    Sorting by page_id ensures that FAISS row indices are stable across
    repeated builds on different machines.
    """
    root = entries_dir or ENTRIES_DIR
    if not root.is_dir():
        raise FileNotFoundError(
            f"Corpus directory not found: {root}\n"
            "Expected: data/Wikipedia Entries/ with one .json file per page."
        )

    pages = []
    failed = 0
    for path in sorted(root.glob("*.json")):
        try:
            raw  = path.read_text(encoding="utf-8", errors="replace")
            obj  = json.loads(raw, strict=False)
        except Exception as e:
            print(f"WARNING: skipping {path.name}: {e}")
            failed += 1
            continue
        pages.append({
            "page_id": normalize_page_id(obj.get("page_id", path.stem)),
            "title":   obj.get("title",   ""),
            "content": obj.get("content", ""),
        })

    if failed:
        print(f"WARNING: {failed} files could not be parsed and were skipped.")

    pages.sort(key=lambda p: p["page_id"])
    return pages


# ─── Kept from starter (used by read-only files) ─────────────────────────────
def load_public_queries(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load public_queries.json, normalizing relevant_page_ids to int."""
    path = path or PUBLIC_QUERIES_PATH
    rows = json.loads(path.read_text(encoding="utf-8"))
    for row in rows:
        row["relevant_page_ids"] = [
            normalize_page_id(pid) for pid in row["relevant_page_ids"]
        ]
    return rows


def iter_entries(entries_dir: Optional[Path] = None) -> Iterator[Dict[str, Any]]:
    """
    Yield one record per JSON file.
    Kept for compatibility with read-only scripts/build_index.py.
    Internally calls load_pages_sorted for robustness.
    """
    for page in load_pages_sorted(entries_dir):
        yield page


def entry_text(record: Dict[str, Any]) -> str:
    """Concatenate title + content into a single text string."""
    title   = record.get("title", "")
    content = record.get("content", "")
    if title:
        return f"{title}\n\n{content}".strip()
    return str(content).strip()


def ensure_artifacts_dir() -> Path:
    """Create artifacts/ if it does not exist, return Path."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR
