"""
Benchmark runner: standard nested CV (FS on full outer_train, evaluate on outer_test).
Supports RepeatedStratifiedKFold (5 splits × 5 repeats = 25 folds).
Fold files: rep{r}_fold{f}.json — supports per-fold resume.
"""
import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

import sys
sys.path.insert(0, os.path.dirname(__file__))
from protocol import OUTER_CV, load_dataset, preprocess
from metrics import compute_metrics, compute_stability


def run_experiment(dataset_path, method, method_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "summary.json")

    if os.path.exists(summary_path):
        print(f"  [{method_name}] Already done, skipping.")
        with open(summary_path) as f:
            return json.load(f)

    X, y, feature_names = load_dataset(dataset_path)
    total_features = X.shape[1]
    n_splits = getattr(OUTER_CV, "n_splits", 5)
    fold_records = []
    all_selected_original = []

    for fold_idx, (train_idx, test_idx) in enumerate(OUTER_CV.split(X, y)):
        rep = fold_idx // n_splits
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

        # Preprocess: zero-var removal + standardize (fit on train only)
        X_train, X_test, kept_idx = preprocess(X_train_raw, X_test_raw)

        # Feature selection on full preprocessed outer_train
        t0 = time.time()
        try:
            proc_selected = method(X_train, y_train)
        except Exception as e:
            print(f"  [{method_name}] rep{rep}_fold{fold} FS ERROR: {e} — using top-10 ANOVA")
            from sklearn.feature_selection import f_classif
            scores, _ = f_classif(X_train, y_train)
            proc_selected = np.argsort(np.nan_to_num(scores, nan=0.0))[::-1][:10]
        runtime = time.time() - t0

        proc_selected = np.asarray(proc_selected, dtype=int)
        if len(proc_selected) == 0:
            proc_selected = np.arange(min(10, X_train.shape[1]))

        # Map to original feature space for stability tracking
        original_selected = kept_idx[proc_selected].tolist()

        # Classify
        X_tr = X_train[:, proc_selected]
        X_te = X_test[:, proc_selected]
        clf = LogisticRegression(C=1, max_iter=1000, random_state=42, solver="lbfgs")
        clf.fit(X_tr, y_train)
        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_prob, original_selected, total_features)
        metrics.update({
            "rep": rep, "fold": fold,
            "runtime_sec": round(runtime, 2),
            "selected_features": original_selected,
        })

        with open(fold_path, "w") as f:
            json.dump(metrics, f)

        fold_records.append(metrics)
        all_selected_original.append(original_selected)
        print(f"  rep{rep}_fold{fold}: AUC={metrics['auc']:.4f}, "
              f"n_feat={metrics['n_features']}, t={runtime:.1f}s")

    # Save fold results CSV
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "selected_features"}
                       for r in fold_records])
    df.to_csv(os.path.join(output_dir, "fold_results.csv"), index=False)

    with open(os.path.join(output_dir, "selected_features.json"), "w") as f:
        json.dump(all_selected_original, f)

    stability = compute_stability(all_selected_original)
    aucs = [r["auc"] for r in fold_records]
    summary = {
        "method": method_name,
        "dataset": os.path.basename(dataset_path),
        "n_folds": len(fold_records),
        "auc_mean": round(float(np.mean(aucs)), 4),
        "auc_std":  round(float(np.std(aucs)), 4),
        "acc_mean": round(float(np.mean([r["acc"] for r in fold_records])), 4),
        "bacc_mean": round(float(np.mean([r.get("bacc", r["acc"]) for r in fold_records])), 4),
        "f1_mean":  round(float(np.mean([r["f1"] for r in fold_records])), 4),
        "n_features_mean": round(float(np.mean([r["n_features"] for r in fold_records])), 1),
        "feature_ratio_mean": round(float(np.mean([r["feature_ratio"] for r in fold_records])), 4),
        "stability": round(stability, 4),
        "runtime_mean_sec": round(float(np.mean([r["runtime_sec"] for r in fold_records])), 2),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  DONE: AUC={summary['auc_mean']:.4f}±{summary['auc_std']:.4f}, "
          f"n_feat={summary['n_features_mean']:.0f}, stability={summary['stability']:.3f}")
    return summary
