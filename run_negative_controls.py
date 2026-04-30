"""
Negative control experiments (S7).
Two controls per method × dataset:
  1. Label permutation: shuffle y labels, expect AUC ≈ 0.5
  2. Random gene baseline: randomly select k features (same k as main result)
Parallel execution, fold-level checkpointing.
"""
import os, sys, json, time, argparse, random
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import StringIO

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

# Representative methods covering all families
NEG_METHODS_NAMES = ["anova", "mi", "relieff", "l1_logistic",
                     "boruta", "rfecv", "ga", "mel"]

NEG_OUT = os.path.join(ROOT, "output", "negative_controls")
BENCH   = os.path.join(ROOT, "output", "main_benchmark")
DATA_DIR = os.path.join(ROOT, "data", "main_benchmark")
LOG_DIR  = os.path.join(ROOT, "output", "parallel_logs")
os.makedirs(NEG_OUT, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def _worker_init():
    os.environ["OMP_NUM_THREADS"]  = "1"
    os.environ["MKL_NUM_THREADS"]  = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"]  = "1"


def _run_permutation_control(job):
    """Run label-permuted version of one (method, dataset) experiment."""
    method_name, dataset_path, output_dir, seed = job
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            return json.load(f), method_name, os.path.basename(dataset_path), None

    buf = StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        from protocol import OUTER_CV, load_dataset, preprocess
        from metrics import compute_metrics, compute_stability
        from methods import ALL_METHODS
        from sklearn.linear_model import LogisticRegression

        method = ALL_METHODS[method_name]
        X, y, feature_names = load_dataset(dataset_path)
        total_features = X.shape[1]
        n_splits = getattr(OUTER_CV, "n_splits", 5)

        # Permute labels with fixed seed
        rng = np.random.RandomState(seed)
        y_perm = rng.permutation(y)

        fold_records = []
        all_selected = []

        for fold_idx, (train_idx, test_idx) in enumerate(OUTER_CV.split(X, y_perm)):
            rep  = fold_idx // n_splits
            fold = fold_idx % n_splits
            fold_path = os.path.join(output_dir, f"rep{rep}_fold{fold}.json")
            if os.path.exists(fold_path):
                with open(fold_path) as f:
                    rec = json.load(f)
                fold_records.append(rec)
                all_selected.append(rec["selected_features"])
                continue

            X_train_raw, X_test_raw = X[train_idx], X[test_idx]
            y_train, y_test = y_perm[train_idx], y_perm[test_idx]
            X_train, X_test, kept_idx = preprocess(X_train_raw, X_test_raw)

            t0 = time.time()
            try:
                proc_sel = method(X_train, y_train)
            except Exception as e:
                from sklearn.feature_selection import f_classif
                scores, _ = f_classif(X_train, y_train)
                proc_sel = np.argsort(np.nan_to_num(scores))[::-1][:10]
            runtime = time.time() - t0

            proc_sel = np.asarray(proc_sel, dtype=int)
            if len(proc_sel) == 0:
                proc_sel = np.arange(min(10, X_train.shape[1]))
            original_sel = kept_idx[proc_sel].tolist()

            X_tr = X_train[:, proc_sel]
            X_te = X_test[:, proc_sel]
            clf = LogisticRegression(C=1, max_iter=1000, random_state=42, solver="lbfgs")
            clf.fit(X_tr, y_train)
            y_pred = clf.predict(X_te)
            y_prob = clf.predict_proba(X_te)[:, 1]

            metrics = compute_metrics(y_test, y_pred, y_prob, original_sel, total_features)
            metrics.update({"rep": rep, "fold": fold, "runtime_sec": round(runtime, 2),
                            "selected_features": original_sel})
            with open(fold_path, "w") as ff:
                json.dump(metrics, ff)
            fold_records.append(metrics)
            all_selected.append(original_sel)

        stability = compute_stability(all_selected)
        aucs = [r["auc"] for r in fold_records]
        summary = {
            "method": method_name,
            "dataset": os.path.basename(dataset_path),
            "control": "label_permutation",
            "seed": seed,
            "n_folds": len(fold_records),
            "auc_mean": round(float(np.mean(aucs)), 4),
            "auc_std":  round(float(np.std(aucs)), 4),
            "n_features_mean": round(float(np.mean([r["n_features"] for r in fold_records])), 1),
            "stability": round(stability, 4),
        }
        with open(summary_path, "w") as ff:
            json.dump(summary, ff, indent=2)
        return summary, method_name, os.path.basename(dataset_path), None
    except Exception as e:
        return None, method_name, os.path.basename(dataset_path), str(e)
    finally:
        sys.stdout = old_stdout


def _run_random_baseline(job):
    """Randomly select k features (same k as main result), expect AUC ≈ 0.5."""
    method_name, dataset_path, output_dir, seed = job
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            return json.load(f), method_name, os.path.basename(dataset_path), None

    buf = StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        from protocol import OUTER_CV, load_dataset, preprocess
        from metrics import compute_metrics, compute_stability
        from sklearn.linear_model import LogisticRegression

        ds_name = os.path.basename(dataset_path).replace(".csv", "")
        main_sp = os.path.join(BENCH, method_name, ds_name, "summary.json")
        if not os.path.exists(main_sp):
            return None, method_name, ds_name, "no main summary"
        with open(main_sp) as f:
            main_s = json.load(f)
        k_target = int(round(main_s["n_features_mean"]))
        if k_target < 1:
            k_target = 10

        X, y, feature_names = load_dataset(dataset_path)
        total_features = X.shape[1]
        n_splits = getattr(OUTER_CV, "n_splits", 5)
        rng = np.random.RandomState(seed)

        fold_records = []
        all_selected = []

        for fold_idx, (train_idx, test_idx) in enumerate(OUTER_CV.split(X, y)):
            rep  = fold_idx // n_splits
            fold = fold_idx % n_splits
            fold_path = os.path.join(output_dir, f"rep{rep}_fold{fold}.json")
            if os.path.exists(fold_path):
                with open(fold_path) as f:
                    rec = json.load(f)
                fold_records.append(rec)
                all_selected.append(rec["selected_features"])
                continue

            X_train_raw, X_test_raw = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            X_train, X_test, kept_idx = preprocess(X_train_raw, X_test_raw)

            # Random feature selection
            k = min(k_target, X_train.shape[1])
            proc_sel = rng.choice(X_train.shape[1], size=k, replace=False)
            original_sel = kept_idx[proc_sel].tolist()

            X_tr = X_train[:, proc_sel]
            X_te = X_test[:, proc_sel]
            clf = LogisticRegression(C=1, max_iter=500, random_state=42, solver="lbfgs")
            clf.fit(X_tr, y_train)
            y_pred = clf.predict(X_te)
            y_prob = clf.predict_proba(X_te)[:, 1]

            metrics = compute_metrics(y_test, y_pred, y_prob, original_sel, total_features)
            metrics.update({"rep": rep, "fold": fold, "runtime_sec": 0.0,
                            "selected_features": original_sel})
            with open(fold_path, "w") as ff:
                json.dump(metrics, ff)
            fold_records.append(metrics)
            all_selected.append(original_sel)

        stability = compute_stability(all_selected)
        aucs = [r["auc"] for r in fold_records]
        summary = {
            "method": method_name,
            "dataset": os.path.basename(dataset_path),
            "control": "random_genes",
            "k_target": k_target,
            "seed": seed,
            "n_folds": len(fold_records),
            "auc_mean": round(float(np.mean(aucs)), 4),
            "auc_std":  round(float(np.std(aucs)), 4),
            "n_features_mean": float(k_target),
            "stability": round(stability, 4),
        }
        with open(summary_path, "w") as ff:
            json.dump(summary, ff, indent=2)
        return summary, method_name, os.path.basename(dataset_path), None
    except Exception as e:
        return None, method_name, os.path.basename(dataset_path), str(e)
    finally:
        sys.stdout = old_stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()

    datasets = sorted(os.path.join(DATA_DIR, f)
                      for f in os.listdir(DATA_DIR) if f.endswith(".csv"))

    # Build job lists
    perm_jobs, rand_jobs = [], []
    for m in NEG_METHODS_NAMES:
        for dp in datasets:
            ds = os.path.basename(dp).replace(".csv", "")
            perm_dir = os.path.join(NEG_OUT, "label_permutation", m, ds)
            rand_dir = os.path.join(NEG_OUT, "random_genes", m, ds)
            if not os.path.exists(os.path.join(perm_dir, "summary.json")):
                perm_jobs.append((m, dp, perm_dir, 2026))
            if not os.path.exists(os.path.join(rand_dir, "summary.json")):
                rand_jobs.append((m, dp, rand_dir, 2026))

    t0 = time.time()
    n_perm = len(perm_jobs)
    n_rand = len(rand_jobs)
    total  = n_perm + n_rand
    print("=" * 60)
    print(f"Negative Controls [{time.strftime('%Y-%m-%d %H:%M:%S')}]")
    print(f"  Label permutation: {n_perm} pending")
    print(f"  Random gene set:   {n_rand} pending")
    print(f"  Workers: {args.workers}")
    print("=" * 60)

    done = 0
    errors = 0

    # Run random baseline first (very fast)
    if rand_jobs:
        print(f"\n--- Random Gene Baseline ({n_rand} jobs) ---")
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=_worker_init) as ex:
            futures = {ex.submit(_run_random_baseline, j): j for j in rand_jobs}
            for fut in as_completed(futures):
                res, m, ds, err = fut.result()
                done += 1
                wall = (time.time() - t0) / 60
                if err:
                    errors += 1
                    print(f"[{done:3d}/{total}] ERR  {m:12s} × {ds:30s}  {err}")
                else:
                    print(f"[{done:3d}/{total}] rand {m:12s} × {ds:30s}  "
                          f"AUC={res['auc_mean']:.4f}  wall={wall:.1f}min")

    # Run label permutation (slower)
    if perm_jobs:
        print(f"\n--- Label Permutation ({n_perm} jobs) ---")
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=_worker_init) as ex:
            futures = {ex.submit(_run_permutation_control, j): j for j in perm_jobs}
            for fut in as_completed(futures):
                res, m, ds, err = fut.result()
                done += 1
                wall = (time.time() - t0) / 60
                remaining = total - done
                avg_t = (time.time() - t0) / max(done, 1)
                eta = remaining * avg_t / args.workers / 60
                if err:
                    errors += 1
                    print(f"[{done:3d}/{total}] ERR  {m:12s} × {ds:30s}  {err}")
                else:
                    print(f"[{done:3d}/{total}] perm {m:12s} × {ds:30s}  "
                          f"AUC={res['auc_mean']:.4f}  wall={wall:.1f}min  ETA={eta:.1f}min")

    print(f"\n{'='*60}")
    print(f"Done. {total} experiments in {(time.time()-t0)/60:.1f} min  Errors: {errors}")
    print("=" * 60)


if __name__ == "__main__":
    main()
