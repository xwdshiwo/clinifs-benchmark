"""
verify_integrity.py — scan checkpoint files under output/v2/ for integrity.

The scanner detects common corruption modes:
  1. NUL-byte corruption (file size > 0 but content is all \\0 bytes, a typical
     symptom after Windows crashes or power loss).
  2. JSON parse failures (truncated or encoding-corrupted files).
  3. CSV read failures (empty content or malformed structure).

Exit codes:
  0 = all checks passed
  1 = corruption detected

Usage:
  python tools/verify_integrity.py                   # scan all files
  python tools/verify_integrity.py --root path       # specify scan root
  python tools/verify_integrity.py --fix-empty-dirs  # delete bad task dirs
                                                     # so dispatcher can rerun
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Iterable

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def is_nul_bytes(path: str, sample_size: int = 4096) -> bool:
    """Return True if file size > 0 but the first `sample_size` bytes are all \\0.

    Reading just a sample is enough — NUL-corrupted files are always NUL from
    byte 0 onward (journaling data-block recovery), so a small prefix suffices.
    """
    try:
        sz = os.path.getsize(path)
    except OSError:
        return False
    if sz == 0:
        return False
    with open(path, "rb") as f:
        chunk = f.read(min(sample_size, sz))
    return chunk.count(b"\x00") == len(chunk)


def check_json(path: str) -> tuple[str, str]:
    """(status, detail). status ∈ {'ok', 'nul', 'parse', 'empty'}."""
    sz = os.path.getsize(path)
    if sz == 0:
        return ("empty", "0 bytes")
    if is_nul_bytes(path):
        return ("nul", f"{sz} B all-\\0")
    try:
        with open(path, "r", encoding="utf-8") as f:
            json.load(f)
        return ("ok", "")
    except Exception as e:
        return ("parse", str(e)[:80])


def check_csv(path: str) -> tuple[str, str]:
    sz = os.path.getsize(path)
    if sz == 0:
        return ("empty", "0 bytes")
    if is_nul_bytes(path):
        return ("nul", f"{sz} B all-\\0")
    # Don't import pandas just to validate CSV (slow). Check header-like content.
    with open(path, "rb") as f:
        first = f.read(256)
    # Require at least one comma and one newline (header plus one data row).
    if first.count(b",") == 0 or first.count(b"\n") == 0:
        return ("suspicious", f"no comma/newline in first 256B")
    return ("ok", "")


def walk_files(root: str, extensions: Iterable[str]) -> Iterable[str]:
    for dp, _, files in os.walk(root):
        for f in files:
            if any(f.endswith(e) for e in extensions):
                yield os.path.join(dp, f)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan output/v2 for NUL-byte / JSON / CSV corruption")
    parser.add_argument("--root", default=os.path.join(ROOT, "output", "v2"),
                        help="Scan root (default: output/v2)")
    parser.add_argument("--fix-empty-dirs", action="store_true",
                        help="DELETE task directories containing any bad file "
                             "(use with care; dispatcher will rerun them)")
    parser.add_argument("--quiet", action="store_true",
                        help="Only print summary")
    args = parser.parse_args()

    if not os.path.isdir(args.root):
        print(f"[error] root not found: {args.root}", file=sys.stderr)
        return 2

    if not args.quiet:
        print("=" * 72)
        print(f"Integrity scan: {args.root}")
        print("=" * 72)

    bad: list[tuple[str, str, str]] = []   # (path, status, detail)
    n_json = n_csv = 0

    for p in walk_files(args.root, (".json",)):
        n_json += 1
        status, detail = check_json(p)
        if status != "ok":
            bad.append((p, status, detail))
    for p in walk_files(args.root, (".csv",)):
        n_csv += 1
        status, detail = check_csv(p)
        if status != "ok":
            bad.append((p, status, detail))

    if not args.quiet:
        print(f"[scan] JSON: {n_json}   CSV: {n_csv}   bad: {len(bad)}")

    if not bad:
        if not args.quiet:
            print("\n" + "=" * 72)
            print("  ✅ INTEGRITY OK — all files parse cleanly.")
            print("=" * 72)
        else:
            print(f"✓ integrity OK  (JSON={n_json}, CSV={n_csv})")
        return 0

    # ------ Report problems ------
    print()
    print("=" * 72)
    print(f"  ❌ {len(bad)} BAD FILE(S)")
    print("=" * 72)
    by_status: dict[str, list[tuple[str, str]]] = {}
    for p, st, d in bad:
        by_status.setdefault(st, []).append((p, d))
    for st, items in sorted(by_status.items()):
        print(f"\n  [{st}]  {len(items)} file(s)")
        for p, d in items[:30]:
            rp = os.path.relpath(p, ROOT)
            print(f"    {rp}   ({d})")
        if len(items) > 30:
            print(f"    ... and {len(items) - 30} more")

    # ------ Aggregate affected task directories ------
    task_dirs = sorted(set(os.path.dirname(p) for p, _, _ in bad))
    print("\n" + "=" * 72)
    print(f"  Affected task directories: {len(task_dirs)}")
    print("=" * 72)
    for d in task_dirs:
        print("   " + os.path.relpath(d, ROOT))

    if args.fix_empty_dirs:
        print("\n" + "=" * 72)
        print("  --fix-empty-dirs: DELETING affected task directories")
        print("=" * 72)
        import shutil
        for d in task_dirs:
            shutil.rmtree(d)
            print(f"  deleted: {os.path.relpath(d, ROOT)}")
        print("\nNow re-run dispatcher — it will detect missing summary.json "
              "and rerun these tasks.")

    return 1


if __name__ == "__main__":
    sys.exit(main())
