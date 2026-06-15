"""
chunk.py — Text representation and chunking for Section B.

Creates three types of retrieval records from each corpus page:
  TitleRecord  – title string only (short, entity-focused)
  PageRecord   – title + first PAGE_TRUNCATION_WORDS of content
  ChunkRecord  – title + overlapping word-window slice of content

All three types map back to page_id so results can be aggregated at scoring time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from utils import extract_years, extract_numbers


# ─── Configuration ─────────────────────────────────────────────────────────────
# Keep these in sync with config.json and the values used in index.py.

PAGE_TRUNCATION_WORDS = 400  # words fed to page-level embedding
CHUNK_SIZE_WORDS      = 200  # words per chunk
CHUNK_STEP_WORDS      = 160  # stride; overlap = CHUNK_SIZE_WORDS - CHUNK_STEP_WORDS = 40


# ─── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class TitleRecord:
    row:     int
    page_id: int
    title:   str
    text:    str   # just the title


@dataclass
class PageRecord:
    row:     int
    page_id: int
    title:   str
    text:    str          # title + first N words
    years:   List[str] = field(default_factory=list)
    numbers: List[str] = field(default_factory=list)


@dataclass
class ChunkRecord:
    row:        int
    page_id:    int
    chunk_idx:  int
    title:      str
    text:       str       # title + local window
    start_word: int
    end_word:   int
    years:      List[str] = field(default_factory=list)
    numbers:    List[str] = field(default_factory=list)


# ─── Record builders ──────────────────────────────────────────────────────────

def build_title_record(page: Dict[str, Any], row: int) -> TitleRecord:
    """One record per page: just its title."""
    return TitleRecord(
        row=row,
        page_id=page["page_id"],
        title=page["title"],
        text=page["title"].strip() or f"Page {page['page_id']}",
    )


def build_page_record(page: Dict[str, Any], row: int) -> PageRecord:
    """
    One record per page: title + first PAGE_TRUNCATION_WORDS words of content.

    Rationale: MiniLM truncates internally at ~256 sub-word tokens.
    Sending 400 words ensures we cover the most informative opening of the
    article without wasting compute on sub-words that will be silently cut.
    """
    words   = page["content"].split()
    snippet = " ".join(words[:PAGE_TRUNCATION_WORDS])
    title   = page["title"]
    text    = f"{title}. {snippet}".strip() if title else snippet.strip()
    full    = title + " " + page["content"]
    return PageRecord(
        row=row,
        page_id=page["page_id"],
        title=title,
        text=text,
        years=extract_years(full),
        numbers=extract_numbers(full),
    )


def build_chunk_records(
    page:       Dict[str, Any],
    start_row:  int,
    chunk_size: int = CHUNK_SIZE_WORDS,
    step:       int = CHUNK_STEP_WORDS,
) -> List[ChunkRecord]:
    """
    Split page content into overlapping word-window chunks.

    Text format: "{title}. {local window}"
    Prepending the title gives the embedding model entity context so
    passages like "averaged 24 points" are grounded to the right player.

    For very short pages (< chunk_size words), a single chunk is created.
    Empty content pages still produce one record (title only) so the page
    appears in candidate sets.
    """
    title = page["title"]
    words = page["content"].split()

    if not words:
        # Degenerate case: no content
        return [ChunkRecord(
            row=start_row,
            page_id=page["page_id"],
            chunk_idx=0,
            title=title,
            text=title.strip() or f"Page {page['page_id']}",
            start_word=0,
            end_word=0,
        )]

    chunks: List[ChunkRecord] = []
    row = start_row

    for start in range(0, len(words), step):
        end  = min(start + chunk_size, len(words))
        body = " ".join(words[start:end])
        text = f"{title}. {body}".strip() if title else body.strip()
        combined = title + " " + body

        chunks.append(ChunkRecord(
            row=row,
            page_id=page["page_id"],
            chunk_idx=len(chunks),
            title=title,
            text=text,
            start_word=start,
            end_word=end,
            years=extract_years(combined),
            numbers=extract_numbers(combined),
        ))
        row += 1
        if end == len(words):
            break

    return chunks


# ─── Corpus-wide builder ──────────────────────────────────────────────────────

def build_all_records(pages: List[Dict[str, Any]]):
    """
    Build all three record types for the full corpus.

    Parameters
    ----------
    pages : list of dicts with keys page_id, title, content
            Must be sorted by page_id for deterministic row indices.

    Returns
    -------
    title_records : List[TitleRecord]
    page_records  : List[PageRecord]
    chunk_records : List[ChunkRecord]

    Notes
    -----
    - title_records[i].row == i  (same row for title and page indexes)
    - page_records[i].row  == i
    - chunk_records have consecutive rows starting at 0 and are not
      aligned with page row numbers.
    """
    title_records: List[TitleRecord] = []
    page_records:  List[PageRecord]  = []
    chunk_records: List[ChunkRecord] = []

    chunk_row = 0
    for i, page in enumerate(pages):
        title_records.append(build_title_record(page, row=i))
        page_records.append(build_page_record(page, row=i))

        new_chunks = build_chunk_records(page, start_row=chunk_row)
        chunk_records.extend(new_chunks)
        chunk_row += len(new_chunks)

    return title_records, page_records, chunk_records


# ─── Legacy compatibility (used by read-only build_index.py indirectly) ───────
# The read-only scripts/build_index.py calls main.build_offline_index()
# which calls index.build_index(). As long as index.py imports from chunk.py
# correctly, the Chunk / chunk_corpus interface below is not needed.
# Kept as a stub for any code that references the original API.

from dataclasses import dataclass as _dc

@_dc
class Chunk:
    page_id:  int
    chunk_id: int
    text:     str


def chunk_entry(record: Dict[str, Any]) -> List[Chunk]:
    """Legacy stub — not used by the new pipeline."""
    from utils import entry_text
    return [Chunk(page_id=int(record["page_id"]), chunk_id=0, text=entry_text(record))]


def chunk_corpus(records: List[Dict[str, Any]]) -> List[Chunk]:
    """Legacy stub — not used by the new pipeline."""
    chunks: List[Chunk] = []
    for record in records:
        chunks.extend(chunk_entry(record))
    return chunks
