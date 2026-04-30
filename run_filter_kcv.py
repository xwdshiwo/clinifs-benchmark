"""
Parallel runner for filter methods with inner-CV k selection.
Methods: variance, anova, mi, mrmr, relieff
Output: output/main_benchmark_kcv/
"""
import os, sys, json, time, argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import StringIO

ROOT     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data", "main_benchmark")
OUT_DIR  = os.path.join(ROOT, "output", "main_benchmark_kcv")
LOG_DIR  = os.path.join(ROOT, "output", "parallel_logs")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

KCV_METHODS = ["variance", "anova", "mi", "mrmr", "relieff"]


def _worker_init():
    os.environ["OMP_NUM_THREADS"]      = "1"
    os.environ["MKL_NUM_THREADS"]      = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"]  = "1"


def _run_one(job):
    method_name, dataset_path, output_dir = job
    buf = StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        sys.path.insert(0, os.path.join(ROOT, "src"))
        from runner_kcv import run_experiment_kcv
        result = run_experiment_kcv(dataset_path, method_name, output_dir)
        sys.stdout = old
        return result, method_name, os.path.basename(dataset_path), None
    except Exception as e:
        sys.stdout = old
        return None, method_name, os.path.basename(dataset_path), str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()

    datasets = sorted(os.path.join(DATA_DIR, f)
                      for f in os.listdir(DATA_DIR) if f.endswith(".csv"))

    pending = []
    for m in KCV_METHODS:
        for dp in datasets:
            ds = os.path.basename(dp).replace(".csv", "")
            out = os.path.join(OUT_DIR, m, ds)
            if not os.path.exists(os.path.join(out, "summary.json")):
                pending.append((m, dp, out))

    t0 = time.time()
    total = len(pending)
    print("=" * 65)
    print(f"Filter k-CV Runner  [{time.strftime('%Y-%m-%d %H:%M:%S')}]")
    print(f"Workers: {args.workers}   Pending: {total} experiments")
    print(f"Methods: {KCV_METHODS}")
    print("=" * 65)

    done = errors = 0
    progress_log = os.path.join(LOG_DIR, "progress_kcv.jsonl")

    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_worker_init) as executor:
        futures = {executor.submit(_run_one, job): job for job in pending}
        for future in as_completed(futures):
            result, m, ds, err = future.result()
            done += 1
            wall  = (time.time() - t0) / 60
            remaining = total - done
            avg_t = (time.time() - t0) / max(done, 1)
            eta   = remaining * avg_t / args.workers / 60

            if err:
                errors += 1
                status = "ERR "
                line   = (f"[{done:3d}/{total}] {status} {m:12s} × "
                          f"{ds.replace('.csv',''):32s}  {err[:60]}")
            else:
                status = "OK  "
                auc    = result["auc_mean"]
                auc_s  = result["auc_std"]
                nf     = result["n_features_mean"]
                km     = result.get("k_mode", "?")
                line   = (f"[{done:3d}/{total}] {status} {m:12s} × "
                          f"{ds.replace('.csv',''):32s}  "
                          f"AUC={auc:.4f}±{auc_s:.4f}  "
                          f"k*={km}  feat={nf:.1f}  "
                          f"wall={wall:.1f}min  ETA={eta:.1f}min")

            print(line)

            log_entry = {"done": done, "total": total, "method": m,
                         "dataset": ds, "status": status.strip(),
                         "wall_min": round(wall, 1),
                         "result": result, "error": err}
            with open(progress_log, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

    wall_total = (time.time() - t0) / 60
    print(f"\n{'='*65}")
    print(f"Done. {total} experiments in {wall_total:.1f} min  Errors: {errors}")
    print("=" * 65)


if __name__ == "__main__":
    main()
