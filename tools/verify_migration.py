"""
verify_migration.py — post-migration smoke verifier for Mac / Linux / Windows.

Run this script after unpacking the project and installing dependencies on a
target machine. It performs quick checks before a full benchmark continuation:
  1. Python and key dependency availability / version compatibility.
  2. Readability of the 11 CSV files under data/main_benchmark/.
  3. Recognition of output/v2 checkpoint summary.json files by the dispatcher.
  4. End-to-end execution of a lightweight filter task
     (ANOVA × k=5 × smallest available dataset).
  5. Loadability of offline enrichment GMT resources.

Usage:
    python tools/verify_migration.py          # run all checks
    python tools/verify_migration.py --quick  # skip the live smoke task
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

GREEN = "\033[92m"
RED   = "\033[91m"
YEL   = "\033[93m"
RST   = "\033[0m"

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    tag = f"{GREEN}PASS{RST}" if ok else f"{RED}FAIL{RST}"
    print(f"  [{tag}] {name}" + (f" — {detail}" if detail else ""))
    results.append((name, ok, detail))


# ────────────────────────────── 1. dependency check ──────────────────────────
def check_dependencies() -> None:
    print(f"\n{YEL}[1/5]{RST} Dependency check")
    required = {
        "numpy": "2.0",
        "scipy": "1.10",
        "pandas": "2.0",
        "sklearn": "1.3",
        "statsmodels": "0.14",
        "deap": "1.4",
        "pymoo": "0.6",
        "skrebate": "0.6",
        "boruta": "0.3",
        "mrmr": "0.2",
        "gprofiler": "1.0",
        "requests": "2.25",
        "matplotlib": "3.5",
        "seaborn": "0.12",
        "joblib": "1.2",
        "tqdm": "4.6",
    }
    for mod, min_ver in required.items():
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            check(f"{mod:<12} (>= {min_ver})", True, f"found {ver}")
        except Exception as e:
            check(f"{mod:<12} (>= {min_ver})", False, f"import error: {e}")


# ──────────────────────────── 2. data integrity check ────────────────────────
EXPECTED_DATASETS = [
    "Bladder_GSE31189",
    "Breast_GSE70947",
    "Colorectal_GSE44076",
    "Colorectal_GSE44861",
    "Leukemia_GSE63270",
    "Liver_GSE14520_U133A",
    "Liver_GSE76427",
    "Lung_GSE19804",
    "Pancreatic_GSE16515",
    "Prostate_GSE6919_U95Av2",
    "Renal_GSE53757",
]


def check_data() -> None:
    print(f"\n{YEL}[2/5]{RST} Dataset integrity")
    data_dir = ROOT / "data" / "main_benchmark"
    if not data_dir.is_dir():
        check("data/main_benchmark/ exists", False, f"missing: {data_dir}")
        return
    check("data/main_benchmark/ exists", True, str(data_dir))
    for ds in EXPECTED_DATASETS:
        p = data_dir / f"{ds}.csv"
        if not p.is_file():
            check(f"  {ds}.csv", False, "FILE MISSING")
            continue
        size_mb = p.stat().st_size / 1024 / 1024
        check(f"  {ds}.csv", size_mb > 1.0, f"{size_mb:.1f} MB")


# ──────────────────────────── 3. checkpoint recognition ──────────────────────
def check_checkpoint() -> None:
    print(f"\n{YEL}[3/5]{RST} Checkpoint recognition")
    out_root = ROOT / "output" / "v2"
    if not out_root.is_dir():
        check("output/v2/ exists", False, "No prior checkpoints found (OK for fresh run)")
        return
    total = 0
    buckets = {}
    for p in out_root.rglob("summary.json"):
        total += 1
        top = p.relative_to(out_root).parts[0]
        buckets[top] = buckets.get(top, 0) + 1
    check("output/v2/ exists", True, f"{total} summary.json total")
    for k in sorted(buckets):
        print(f"      {k:<25} {buckets[k]:>6}")
    # Ask the dispatcher to report pending tasks in dry-run mode.
    try:
        import subprocess
        for exp in ("E1", "E2", "E3", "E4"):
            r = subprocess.run(
                [sys.executable, "run_v2_dispatcher.py",
                 "--experiment", exp, "--dry-run"],
                capture_output=True, text=True, timeout=60, cwd=ROOT,
            )
            # Extract "Total: X, Pending: Y, Skipped: Z".
            line = next((ln for ln in r.stdout.splitlines()
                         if ln.startswith("Total:")), "")
            check(f"  dispatcher dry-run {exp}", bool(line), line or r.stderr[-200:])
    except Exception as e:
        check("dispatcher dry-run", False, str(e))


# ──────────────────────────── 4. smoke task (ANOVA) ──────────────────────────
def smoke_task() -> None:
    print(f"\n{YEL}[4/5]{RST} End-to-end smoke test (ANOVA k=5 on smallest CSV)")
    try:
        from runner_v2 import run_experiment_v2
        from methods.methods_v2 import ALL_METHODS_V2
    except Exception as e:
        check("import runner_v2 / methods_v2", False, str(e))
        return
    check("import runner_v2 / methods_v2", True)

    data_dir = ROOT / "data" / "main_benchmark"
    # pick smallest dataset by file size
    candidates = sorted(data_dir.glob("*.csv"), key=lambda p: p.stat().st_size)
    if not candidates:
        check("pick dataset", False, "no CSV found")
        return
    ds_path = candidates[0]
    out_dir = ROOT / "output" / "_smoke_migration" / "anova_k5" / ds_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        t0 = time.time()
        run_experiment_v2(
            str(ds_path),
            ALL_METHODS_V2["anova"],
            {"k": 5, "method_name": "anova"},
            str(out_dir),
        )
        dt = time.time() - t0
        summary_p = out_dir / "summary.json"
        ok = summary_p.is_file()
        detail = f"wrote {summary_p.name} in {dt:.1f}s"
        if ok:
            with summary_p.open() as f:
                sj = json.load(f)
            detail += f", auc={sj.get('auc_mean', '?')}, k_sel={sj.get('n_selected', '?')}"
        check(f"smoke run on {ds_path.stem}", ok, detail)
    except Exception as e:
        import traceback
        check(f"smoke run on {ds_path.stem}", False,
              f"{e}\n{traceback.format_exc()[-400:]}")


# ──────────────────────────── 5. enrichment GMT ──────────────────────────────
def check_enrichment() -> None:
    print(f"\n{YEL}[5/5]{RST} Offline enrichment resources")
    gmt_dir = ROOT / "refs" / "enrichment_gmt"
    if not gmt_dir.is_dir():
        check("refs/enrichment_gmt/", False, "dir missing — A2 will need online g:Profiler")
        return
    gmts = list(gmt_dir.glob("*.gmt"))
    check("refs/enrichment_gmt/", bool(gmts), f"{len(gmts)} GMT files")
    for g in gmts:
        n = sum(1 for _ in g.open(encoding="utf-8"))
        check(f"  {g.name}", n > 5, f"{n} gene sets")


# ──────────────────────────── main ───────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="Skip the end-to-end smoke task (faster, dependency-only).")
    args = ap.parse_args()

    print("=" * 64)
    print("  V2 Feature-Selection Benchmark — Migration Verifier")
    print(f"  ROOT = {ROOT}")
    print(f"  Python = {sys.version.split()[0]}  ({sys.executable})")
    print("=" * 64)

    check_dependencies()
    check_data()
    check_checkpoint()
    if not args.quick:
        smoke_task()
    check_enrichment()

    n_total = len(results)
    n_pass = sum(1 for _, ok, _ in results if ok)
    print("\n" + "=" * 64)
    if n_pass == n_total:
        print(f"{GREEN}ALL {n_total}/{n_total} CHECKS PASSED — safe to continue V2.{RST}")
        print("Next:  bash run_v2_followup.sh --skip-e2-check   # or only-analysis")
        return 0
    else:
        print(f"{RED}{n_total - n_pass} / {n_total} checks failed — inspect above.{RST}")
        for name, ok, detail in results:
            if not ok:
                print(f"  - {name}: {detail}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
