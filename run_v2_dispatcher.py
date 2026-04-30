"""
Unified V2 dispatcher: executes task lists in parallel with checkpointing.
Task format: dict with keys:
    experiment, method, dataset, config, output_subdir
"""
import os
import sys
import json
import time
import argparse
from io import StringIO
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data", "main_benchmark")
OUT_ROOT = os.path.join(ROOT, "output", "v2")
LOG_DIR  = os.path.join(ROOT, "output", "parallel_logs")
os.makedirs(OUT_ROOT, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

DATASETS = sorted(f.replace(".csv", "") for f in os.listdir(DATA_DIR)
                  if f.endswith(".csv"))


def _worker_init():
    os.environ["OMP_NUM_THREADS"]      = "1"
    os.environ["MKL_NUM_THREADS"]      = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"]  = "1"
    import warnings
    warnings.filterwarnings("ignore")


def _resolve_method(exp, method):
    """Return the callable function for a given experiment/method."""
    sys.path.insert(0, os.path.join(ROOT, "src"))
    if exp in ("E1", "E2", "E4"):
        from methods.methods_v2 import ALL_METHODS_V2
        return ALL_METHODS_V2[method]
    if exp == "E3":
        from methods.pipelines import PIPELINES
        return PIPELINES[method]
    raise ValueError(f"Unknown exp/method: {exp}/{method}")


def _worker_run(task):
    buf = StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        sys.path.insert(0, os.path.join(ROOT, "src"))
        from runner_v2 import run_experiment_v2

        exp    = task["experiment"]
        method = task["method"]
        ds     = task["dataset"]
        cfg    = task["config"]
        subdir = task["output_subdir"]

        dataset_path = os.path.join(DATA_DIR, ds + ".csv")
        output_dir = os.path.join(OUT_ROOT, subdir)

        method_fn = _resolve_method(exp, method)
        cfg_runtime = dict(cfg)
        cfg_runtime.setdefault("method_name", method)

        result = run_experiment_v2(dataset_path, method_fn, cfg_runtime,
                                    output_dir)
        sys.stdout = old
        return result, task, None
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        sys.stdout = old
        return None, task, f"{e}\n{tb[-400:]}"


def run_task_list(tasks, workers=5, log_file="progress_v2.jsonl"):
    t0 = time.time()
    total = len(tasks)
    print("=" * 75)
    print(f"V2 Dispatcher [{time.strftime('%Y-%m-%d %H:%M:%S')}]")
    print(f"Workers={workers}  Tasks={total}")
    print("=" * 75)

    done = errors = 0
    progress_log = os.path.join(LOG_DIR, log_file)

    with ProcessPoolExecutor(max_workers=workers,
                             initializer=_worker_init) as ex:
        futures = {ex.submit(_worker_run, t): t for t in tasks}
        for fut in as_completed(futures):
            result, task, err = fut.result()
            done += 1
            wall = (time.time() - t0) / 60
            avg  = (time.time() - t0) / max(done, 1)
            eta  = (total - done) * avg / workers / 60

            tag = (f"[{done:4d}/{total}] {task['experiment']} "
                   f"{task['method']:18s} × {task['dataset'][:28]:28s}"
                   f" cfg={task.get('config',{}).get('k','?')}")

            if err:
                errors += 1
                print(f"{tag}  ERR  {str(err)[:80]}")
            else:
                auc = result.get("auc_mean", -1)
                nf  = result.get("n_features_mean", -1)
                sn  = result.get("stability_nogueira", -1)
                print(f"{tag}  OK   AUC={auc:.4f}  n={nf:.0f}  "
                      f"stab={sn:.3f}  wall={wall:.1f}m  ETA={eta:.1f}m")

            with open(progress_log, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "done": done, "total": total,
                    "task": task, "error": err,
                    "wall_min": round(wall, 2),
                    "summary": (result if err is None
                                else None),
                }, ensure_ascii=False) + "\n")

    wall_total = (time.time() - t0) / 60
    print("=" * 75)
    print(f"DONE {total} tasks in {wall_total:.1f}min, {errors} errors")
    print("=" * 75)
    return errors == 0


# ───────── Task list builders ─────────

def build_e1_tasks():
    """E1: EA sparsity tuning."""
    methods = ["ga", "bpso"]          # SFE/MEL don't have alpha/beta; skip
    configs = [
        {"tag": "C_base",    "alpha": 0.9, "beta": 0.1},
        {"tag": "C_mid",     "alpha": 0.7, "beta": 0.3},
        {"tag": "C_high",    "alpha": 0.5, "beta": 0.5},
        {"tag": "C_extreme", "alpha": 0.3, "beta": 0.7},
    ]
    tasks = []
    for m in methods:
        extra = {"n_gen": 15, "pop_size": 20} if m == "ga" else \
                {"n_iter": 20, "n_particles": 20}
        for c in configs:
            for ds in DATASETS:
                cfg = {**extra, "alpha": c["alpha"], "beta": c["beta"]}
                if m == "ga":
                    cfg["max_features"] = None
                tasks.append({
                    "experiment": "E1",
                    "method": m,
                    "dataset": ds,
                    "config": cfg,
                    "output_subdir":
                        f"E1_ea_sparsity/{m}/{c['tag']}/{ds}",
                })
    return tasks


def build_e2_tasks():
    """E2: constrained k for all methods."""
    methods_all = ["variance", "anova", "mi", "mrmr", "relieff",
                   "l1_logistic", "elasticnet", "linearsvc_l1",
                   "boruta", "extratrees", "rfecv",
                   "ga", "bpso", "sfe", "mel"]
    k_grid = [3, 5, 10, 15, 20, 30, 50]
    tasks = []
    for m in methods_all:
        for k in k_grid:
            for ds in DATASETS:
                cfg = {"k": k}
                if m == "ga":
                    cfg.update({"alpha": 0.3, "beta": 0.7,
                                "n_gen": 15, "pop_size": 20,
                                "max_features": k})
                elif m == "bpso":
                    cfg.update({"alpha": 0.3, "beta": 0.7,
                                "n_iter": 20, "n_particles": 20})
                tasks.append({
                    "experiment": "E2",
                    "method": m,
                    "dataset": ds,
                    "config": cfg,
                    "output_subdir":
                        f"E2_constrained_k/{m}/k{k:02d}/{ds}",
                })
    return tasks


def build_e3_tasks():
    """E3: two-stage pipelines."""
    pipelines = ["P1_anova_ga", "P2_relieff_ga",
                 "P3_fusion_bpso", "P4_anova_bpso"]
    k_grid = [3, 5, 10, 15, 20, 30, 50]
    tasks = []
    for p in pipelines:
        for k in k_grid:
            for ds in DATASETS:
                if "ga" in p:
                    cfg = {"k": k, "prefilter": 200,
                           "alpha": 0.3, "beta": 0.7,
                           "n_gen": 15, "pop_size": 20}
                else:  # bpso
                    cfg = {"k": k, "prefilter": 200,
                           "alpha": 0.3, "beta": 0.7,
                           "n_iter": 20, "n_particles": 20}
                tasks.append({
                    "experiment": "E3",
                    "method": p,
                    "dataset": ds,
                    "config": cfg,
                    "output_subdir":
                        f"E3_pipelines/{p}/k{k:02d}/{ds}",
                })
    return tasks


def build_e4_tasks():
    """E4: EA multi-run consensus (10 seeds)."""
    methods = ["ga", "bpso"]
    seeds = list(range(10))
    K_TARGET = 10
    tasks = []
    for m in methods:
        for s in seeds:
            for ds in DATASETS:
                if m == "ga":
                    cfg = {"k": K_TARGET, "alpha": 0.3, "beta": 0.7,
                           "n_gen": 15, "pop_size": 20,
                           "max_features": K_TARGET,
                           "random_state": s}
                else:  # bpso
                    cfg = {"k": K_TARGET, "alpha": 0.3, "beta": 0.7,
                           "n_iter": 20, "n_particles": 20,
                           "random_state": s}
                tasks.append({
                    "experiment": "E4",
                    "method": m,
                    "dataset": ds,
                    "config": cfg,
                    "output_subdir":
                        f"E4_consensus/{m}/seed{s:02d}/{ds}",
                })
    return tasks


# ───────── estimators for scheduling ─────────

def estimate_runtime(task):
    """Rough cost estimate (in seconds) for sort order."""
    m = task["method"]
    cfg = task.get("config", {})
    k = cfg.get("k", 50)
    if m in ("variance", "anova"):
        return 5
    if m in ("mi",):
        return 15
    if m in ("mrmr",):
        return 30
    if m in ("relieff",):
        return 30
    if m in ("l1_logistic", "elasticnet", "linearsvc_l1", "extratrees"):
        return 20
    if m == "boruta":
        return 60
    if m == "rfecv":
        return 45
    if m == "ga":
        return 30
    if m == "bpso":
        return 150
    if m == "sfe":
        return 30
    if m == "mel":
        return 60
    if m.startswith("P1") or m.startswith("P2"):
        return 60
    if m.startswith("P3") or m.startswith("P4"):
        return 120
    return 30


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True,
                        choices=["E1", "E2", "E3", "E4", "ALL"])
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    builders = {"E1": build_e1_tasks, "E2": build_e2_tasks,
                "E3": build_e3_tasks, "E4": build_e4_tasks}
    if args.experiment == "ALL":
        tasks = []
        for b in builders.values():
            tasks.extend(b())
    else:
        tasks = builders[args.experiment]()

    # Filter out already-done tasks (checkpoint)
    pending = []
    for t in tasks:
        sp = os.path.join(OUT_ROOT, t["output_subdir"], "summary.json")
        if not os.path.exists(sp):
            pending.append(t)

    # Sort fast→slow
    pending.sort(key=estimate_runtime)

    print(f"Total: {len(tasks)}, Pending: {len(pending)}, "
          f"Skipped (checkpointed): {len(tasks) - len(pending)}")

    if args.dry_run:
        from collections import Counter
        ctr = Counter(t["method"] for t in pending)
        for k, v in ctr.most_common():
            print(f"  {k}: {v}")
        sys.exit(0)

    log_name = f"progress_v2_{args.experiment}.jsonl"
    run_task_list(pending, workers=args.workers, log_file=log_name)
