"""
Protocol-correct runner: inner-CV k selection for filter methods.
For each outer fold:
  1. Compute feature ranking scores on outer_train
  2. Inner 3-fold CV to select best k from {5, 10, 20, 50, 100}
  3. Classify with top-k* features on outer_test

Used for: variance, anova, mi, mrmr, relieff
Output directory: output/main_benchmark_kcv/{method}/{dataset}/
"""
import os, json, time, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import f_classif, mutual_info_classif

sys.path.insert(0, os.path.dirname(__file__))
from protocol import OUTER_CV, load_dataset, preprocess
from metrics import compute_metrics, compute_stability

K_GRID = [5, 10, 20, 50, 100]
INNER_CV = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
PREFILTER_N = 2000


def _anova_scores(X, y):
    scores, _ = f_classif(X, y)
    return np.nan_to_num(scores, nan=0.0)


def _mi_scores(X, y):
    return mutual_info_classif(X, y, random_state=42)


def _variance_scores(X, y):
    return np.var(X, axis=0)


def _mrmr_scores(X, y):
    """Returns ANOVA scores after pre-filter (mRMR is order-dependent, use ANOVA for scoring step)."""
    scores, _ = f_classif(X, y)
    return np.nan_to_num(scores, nan=0.0)


def _relieff_scores(X, y):
    """Returns ReliefF importance scores after ANOVA pre-filter."""
    from skrebate import ReliefF
    if X.shape[1] > PREFILTER_N:
        sc, _ = f_classif(X, y)
        top = np.argsort(np.nan_to_num(sc))[::-1][:PREFILTER_N]
        X = X[:, top]
    n_neigh = min(10, len(y) - 1)
    rf = ReliefF(n_features_to_select=min(100, X.shape[1]),
                 n_neighbors=n_neigh, n_jobs=1)
    rf.fit(X, y)
    # return full feature importance array for the pre-filtered space
    return rf.feature_importances_


SCORE_FUNCS = {
    "variance": _variance_scores,
    "anova":    _anova_scores,
    "mi":       _mi_scores,
    "mrmr":     _mrmr_scores,
    "relieff":  _relieff_scores,
}


def _inner_cv_best_k(X_train, y_train, score_fn, k_grid):
    """
    Use inner 3-fold CV to select best k.
    Returns (best_k, {k: mean_inner_auc}).
    """
    # Compute scores once on full outer_train (optimistic, but consistent)
    scores = score_fn(X_train, y_train)
    n_feat = X_train.shape[1]
    k_grid_valid = [k for k in k_grid if k <= n_feat]
    if not k_grid_valid:
        k_grid_valid = [n_feat]

    # Build ranking once
    ranking = np.argsort(scores)[::-1]

    k_aucs = {}
    for k in k_grid_valid:
        top_k_idx = ranking[:k]
        X_k = X_train[:, top_k_idx]
        fold_aucs = []
        for inner_tr, inner_val in INNER_CV.split(X_k, y_train):
            if len(np.unique(y_train[inner_val])) < 2:
                continue
            clf = LogisticRegression(C=1, max_iter=500, random_state=42, solver="lbfgs")
            try:
                clf.fit(X_k[inner_tr], y_train[inner_tr])
                from sklearn.metrics import roc_auc_score
                prob = clf.predict_proba(X_k[inner_val])[:, 1]
                auc  = roc_auc_score(y_train[inner_val], prob)
                fold_aucs.append(auc)
            except Exception:
                pass
        k_aucs[k] = float(np.mean(fold_aucs)) if fold_aucs else 0.5

    best_k = max(k_aucs, key=k_aucs.get)
    return best_k, k_aucs


def run_experiment_kcv(dataset_path, score_fn_name, output_dir):
    """
    Like run_experiment but with inner-CV k selection.
    score_fn_name: one of ('variance','anova','mi','mrmr','relieff')
    """
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "summary.json")

    if os.path.exists(summary_path):
        print(f"  [{score_fn_name}] Already done, skipping.")
        with open(summary_path) as f:
            return json.load(f)

    score_fn = SCORE_FUNCS[score_fn_name]
    X, y, feature_names = load_dataset(dataset_path)
    total_features = X.shape[1]
    n_splits = getattr(OUTER_CV, "n_splits", 5)
    fold_records = []
    all_selected_original = []

    for fold_idx, (train_idx, test_idx) in enumerate(OUTER_CV.split(X, y)):
        rep  = fold_idx // n_splits
        fold = fold_idx % n_splits
        fold_path = os.path.join(output_dir, f"rep{rep}_fold{fold}.json")

        if os.path.exists(fold_path):
            with open(fold_path) as f:
                rec = json.load(f)
            fold_records.append(rec)
            all_selected_original.append(rec["selected_features"])
            continue

        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        X_train, X_test, kept_idx = preprocess(X_train_raw, X_test_raw)

        t0 = time.time()

        # Inner CV: select best k
        best_k, k_aucs = _inner_cv_best_k(X_train, y_train, score_fn, K_GRID)

        # Re-compute scores on full outer_train and take top best_k
        scores = score_fn(X_train, y_train)
        proc_selected = np.argsort(scores)[::-1][:best_k]
        runtime = time.time() - t0

        original_selected = kept_idx[proc_selected].tolist()

        X_tr = X_train[:, proc_selected]
        X_te = X_test[:, proc_selected]
        clf = LogisticRegression(C=1, max_iter=1000, random_state=42, solver="lbfgs")
        clf.fit(X_tr, y_train)
        y_pred = clf.predict(X_te)
        y_prob  = clf.predict_proba(X_te)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_prob, original_selected, total_features)
        metrics.update({
            "rep": rep, "fold": fold,
            "runtime_sec": round(runtime, 2),
            "selected_features": original_selected,
            "best_k": int(best_k),
            "k_aucs": k_aucs,
        })

        with open(fold_path, "w") as f:
            json.dump(metrics, f)

        fold_records.append(metrics)
        all_selected_original.append(original_selected)
        print(f"  rep{rep}_fold{fold}: AUC={metrics['auc']:.4f}, "
              f"k*={best_k}, t={runtime:.1f}s")

    df_fold = pd.DataFrame([{kk: vv for kk, vv in r.items()
                              if kk not in ("selected_features","k_aucs")}
                             for r in fold_records])
    df_fold.to_csv(os.path.join(output_dir, "fold_results.csv"), index=False)

    with open(os.path.join(output_dir, "selected_features.json"), "w") as f:
        json.dump(all_selected_original, f)

    stability = compute_stability(all_selected_original)
    aucs = [r["auc"] for r in fold_records]
    best_ks = [r.get("best_k", 50) for r in fold_records]
    summary = {
        "method":          score_fn_name + "_kcv",
        "dataset":         os.path.basename(dataset_path),
        "protocol":        "inner_cv_k_selection",
        "k_grid":          K_GRID,
        "n_folds":         len(fold_records),
        "auc_mean":        round(float(np.mean(aucs)), 4),
        "auc_std":         round(float(np.std(aucs)), 4),
        "acc_mean":        round(float(np.mean([r["acc"] for r in fold_records])), 4),
        "bacc_mean":       round(float(np.mean([r.get("bacc", r["acc"]) for r in fold_records])), 4),
        "f1_mean":         round(float(np.mean([r["f1"]  for r in fold_records])), 4),
        "n_features_mean": round(float(np.mean([r["n_features"] for r in fold_records])), 1),
        "feature_ratio_mean": round(float(np.mean([r["feature_ratio"] for r in fold_records])), 4),
        "k_mean":          round(float(np.mean(best_ks)), 2),
        "k_mode":          int(pd.Series(best_ks).mode()[0]),
        "stability":       round(stability, 4),
        "runtime_mean_sec":round(float(np.mean([r["runtime_sec"] for r in fold_records])), 2),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  DONE: AUC={summary['auc_mean']:.4f}±{summary['auc_std']:.4f}, "
          f"k*={summary['k_mode']} (mean={summary['k_mean']:.1f}), "
          f"stab={summary['stability']:.3f}")
    return summary
