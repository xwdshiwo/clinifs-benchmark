"""
Parallel benchmark runner for slow methods.
Each worker process runs one (method x dataset) experiment independently.
Supports fold-level checkpointing (via runner.py).

Usage:
    python run_parallel.py [--workers N]
"""
import os
import sys
import json
import time
import logging
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

# ── paths ──────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(ROOT, "data", "main_benchmark")
OUTPUT_DIR = os.path.join(ROOT, "output", "main_benchmark")
LOG_DIR    = os.path.join(ROOT, "output", "parallel_logs")
os.makedirs(LOG_DIR, exist_ok=True)

SLOW_METHODS = ["mi", "elasticnet", "boruta", "ga", "bpso", "sfe", "mel"]

# ── worker initializer: limit inner parallelism per process ─────────────────
def _worker_init():
    os.environ["OMP_NUM_THREADS"]      = "1"
    os.environ["MKL_NUM_THREADS"]      = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["LOKY_MAX_CPU_COUNT"]   = "1"
    import warnings
    warnings.filterwarnings("ignore")


# ── single experiment worker ────────────────────────────────────────────────
def _run_one(args):
    """Run one (method_name, dataset_filename) experiment. Returns summary dict."""
    method_name, ds_filename = args
    t0 = time.time()

    # lazy imports inside worker to benefit from worker_init env vars
    sys.path.insert(0, ROOT)
    from src.runner import run_experiment
    from src.methods import ALL_METHODS

    method = ALL_METHODS[method_name]
    ds_path  = os.path.join(DATA_DIR, ds_filename)
    out_dir  = os.path.join(OUTPUT_DIR, method_name, ds_filename.replace(".csv", ""))
    log_file = os.path.join(LOG_DIR, f"{method_name}__{ds_filename.replace('.csv','')}.log")

    # redirect stdout for this worker to a per-experiment log
    import io, contextlib
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            result = run_experiment(ds_path, method, method_name, out_dir)
    except Exception as exc:
        with open(log_file, "w") as f:
            f.write(buf.getvalue())
            f.write(f"\nERROR: {exc}\n")
        return {
            "method": method_name, "dataset": ds_filename,
            "status": "ERROR", "error": str(exc),
            "elapsed": round(time.time() - t0, 1)
        }

    with open(log_file, "w") as f:
        f.write(buf.getvalue())

    elapsed = round(time.time() - t0, 1)
    return {
        "method":    method_name,
        "dataset":   ds_filename,
        "status":    "OK",
        "auc_mean":  round(result["auc_mean"], 4),
        "auc_std":   round(result["auc_std"],  4),
        "n_feat":    round(result["n_features_mean"], 1),
        "stability": round(result["stability"], 4),
        "elapsed":   elapsed,
    }


# ── discover pending experiments ────────────────────────────────────────────
def get_pending():
    datasets = sorted(
        f for f in os.listdir(DATA_DIR) if f.endswith(".csv")
    )
    pending = []
    for method in SLOW_METHODS:
        for ds in datasets:
            summary = os.path.join(OUTPUT_DIR, method, ds.replace(".csv", ""), "summary.json")
            if not os.path.exists(summary):
                pending.append((method, ds))
    return pending


# ── main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers (default: 4)")
    args = parser.parse_args()
    n_workers = min(args.workers, multiprocessing.cpu_count())

    pending = get_pending()
    total = len(pending)
    if total == 0:
        print("All experiments already complete.")
        return

    # sort: fast methods first so early results appear sooner
    time_est = {"sfe":0.5,"ga":3,"mel":4,"bpso":11,"mi":12,"elasticnet":12,"boruta":15}
    pending.sort(key=lambda x: time_est.get(x[0], 10))

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{'='*62}")
    print(f"Parallel Benchmark  [{ts}]")
    print(f"Workers: {n_workers}   Pending: {total} experiments")
    print(f"Methods: {SLOW_METHODS}")
    print(f"{'='*62}\n")

    # progress log
    progress_log = os.path.join(LOG_DIR, "progress.jsonl")
    done_count = 0
    errors = []
    start_wall = time.time()

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_worker_init
    ) as executor:
        futures = {executor.submit(_run_one, job): job for job in pending}

        for future in as_completed(futures):
            job = futures[future]
            try:
                r = future.result()
            except Exception as exc:
                r = {"method": job[0], "dataset": job[1],
                     "status": "ERROR", "error": str(exc)}

            done_count += 1
            wall = (time.time() - start_wall) / 60
            eta  = wall / done_count * (total - done_count) if done_count else 0

            if r["status"] == "OK":
                print(f"[{done_count:3d}/{total}] OK   "
                      f"{r['method']:12s} × {r['dataset'].replace('.csv',''):<28s} "
                      f"AUC={r['auc_mean']:.4f}±{r['auc_std']:.4f}  "
                      f"feat={r['n_feat']:6.1f}  stab={r['stability']:.3f}  "
                      f"t={r['elapsed']}s  wall={wall:.1f}min  ETA={eta:.1f}min")
            else:
                print(f"[{done_count:3d}/{total}] ERR  "
                      f"{r['method']:12s} × {r['dataset'].replace('.csv',''):<28s} "
                      f"→ {r.get('error','?')}")
                errors.append(r)

            # append to progress log
            with open(progress_log, "a") as f:
                f.write(json.dumps(r) + "\n")

    wall_total = (time.time() - start_wall) / 60
    print(f"\n{'='*62}")
    print(f"Done. {done_count} experiments in {wall_total:.1f} min")
    print(f"Errors: {len(errors)}")
    if errors:
        for e in errors:
            print(f"  ERROR: {e['method']} x {e['dataset']}: {e.get('error','?')}")
    print(f"Logs: {LOG_DIR}")
    print(f"{'='*62}")


if __name__ == "__main__":
    main()
