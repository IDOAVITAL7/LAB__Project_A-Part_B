#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"

REQUIRED = [
    "config.json",
    "build_report.json",
    "pages_meta.json",
    "title_meta.json",
    "page_meta.json",
    "title_index.faiss",
    "page_index.faiss",
    "chunk_index.faiss",
    "chunk_row2pid.npy",
    "bm25_page_stats.json",
    "inverted_index_pages.json.gz",
    "bm25_chunks.sqlite",
    "bm25_chunk_lengths.npy",
    "bm25_chunk_stats_light.json",
]

OPTIONAL_NOT_REQUIRED = [
    "chunk_meta.json",
    "inverted_index_chunks.json.gz",
    "bm25_chunk_stats.json",
    "chunk_index_flat_backup.faiss",
]

GITHUB_LIMIT_MB = 100.0
LARGE_WARN_MB = 50.0


def size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def main() -> int:
    print(f"Artifact directory: {ART}")
    if not ART.exists():
        print("ERROR: artifacts directory does not exist")
        return 1

    missing = []
    large = []

    print("\n=== Required runtime artifacts ===")
    for name in REQUIRED:
        path = ART / name
        if not path.exists():
            missing.append(name)
            print(f"  ✗  {name}")
            continue

        mb = size_mb(path)
        note = ""
        if mb > GITHUB_LIMIT_MB:
            note = "  ← Git LFS required"
            large.append((name, mb, True))
        elif mb > LARGE_WARN_MB:
            note = "  ← large, but under 100MB"
            large.append((name, mb, False))

        print(f"  ✓  {name:<38} {mb:8.2f} MB{note}")

    print("\n=== Optional artifacts that should usually NOT be committed ===")
    found_optional = False
    for name in OPTIONAL_NOT_REQUIRED:
        path = ART / name
        if path.exists():
            found_optional = True
            print(f"  !  {name:<38} {size_mb(path):8.2f} MB")
    if not found_optional:
        print("  ✓  none found")

    print("\n=== Additional files in artifacts ===")
    known = set(REQUIRED) | set(OPTIONAL_NOT_REQUIRED)
    extras = [p for p in ART.iterdir() if p.is_file() and p.name not in known]
    if extras:
        for path in sorted(extras):
            print(f"  +  {path.name:<38} {size_mb(path):8.2f} MB")
    else:
        print("  ✓  none")

    total_mb = sum(size_mb(p) for p in ART.iterdir() if p.is_file())

    print("\n=== Summary ===")
    print(f"  Total artifact size : {total_mb:.1f} MB")
    print(f"  Missing required    : {len(missing)}")
    print(f"  Files over 100 MB   : {sum(1 for _, _, over in large if over)}")

    if missing:
        print("\nERROR: Missing required runtime artifacts:")
        for name in missing:
            print(f"  - {name}")
        return 1

    lfs_required = [(name, mb) for name, mb, over in large if over]
    if lfs_required:
        print("\nGit LFS required for:")
        for name, mb in lfs_required:
            print(f"  - artifacts/{name} ({mb:.1f} MB)")
        print("\nRecommended Git LFS commands:")
        print("  git lfs install")
        for name, _ in lfs_required:
            print(f"  git lfs track 'artifacts/{name}'")
        print("  git add .gitattributes")

    print("\nOK: required runtime artifacts are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
