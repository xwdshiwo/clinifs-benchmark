"""
A1: Method-to-method feature overlap analysis.
For each dataset and k, compute pairwise Jaccard across methods.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from itertools import combinations

ROOT = os.path.dirname(os.path.abspath(__file__))
E2_DIR = os.path.join(ROOT, "output", "v2", "E2_constrained_k")
OUT_DIR = os.path.join(ROOT, "output", "v2", "A1_overlap")
os.makedirs(OUT_DIR, exist_ok=True)

DATASETS = sorted(f.replace(".csv", "") for f in
                  os.listdir(os.path.join(ROOT, "data", "main_benchmark"))
                  if f.endswith(".csv"))
METHODS = ["variance", "anova", "mi", "mrmr", "relieff",
           "l1_logistic", "elasticnet", "linearsvc_l1",
           "boruta", "extratrees", "rfecv",
           "ga", "bpso", "sfe", "mel"]
K_VALUES = [3, 5, 10, 15, 20, 30, 50]


def load_selected(method, k, dataset):
    """Return union of selected features across 25 folds (as a set)."""
    path = os.path.join(E2_DIR, method, f"k{k:02d}", dataset,
                         "selected_features.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        fold_sets = json.load(f)
    # Return list of sets (per fold)
    return [set(s) for s in fold_sets]


def consensus_set(fold_sets, threshold=0.8):
    """Features selected in >= threshold fraction of folds."""
    n_folds = len(fold_sets)
    counter = {}
    for s in fold_sets:
        for f in s:
            counter[f] = counter.get(f, 0) + 1
    return {f for f, c in counter.items() if c / n_folds >= threshold}


def jaccard(a, b):
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def compute_overlap_matrix(dataset, k, consensus_threshold=0.5):
    """For a given (dataset, k), compute method×method Jaccard matrix."""
    method_sets = {}
    for m in METHODS:
        fold_sets = load_selected(m, k, dataset)
        if fold_sets is None:
            continue
        method_sets[m] = consensus_set(fold_sets, consensus_threshold)

    if len(method_sets) < 2:
        return None, None

    methods_present = list(method_sets.keys())
    n = len(methods_present)
    mat = np.zeros((n, n))
    for i, mi in enumerate(methods_present):
        for j, mj in enumerate(methods_present):
            mat[i, j] = jaccard(method_sets[mi], method_sets[mj])

    df = pd.DataFrame(mat, index=methods_present, columns=methods_present)

    # Count cross-method consensus
    cross_counter = {}
    for m, s in method_sets.items():
        for f in s:
            cross_counter[f] = cross_counter.get(f, 0) + 1
    cross_df = pd.DataFrame({
        "feature": list(cross_counter.keys()),
        "n_methods": list(cross_counter.values())
    }).sort_values("n_methods", ascending=False)
    return df, cross_df


def main():
    all_overlap_rows = []
    for ds in DATASETS:
        for k in K_VALUES:
            mat, cross = compute_overlap_matrix(ds, k)
            if mat is None:
                continue
            mat.to_csv(os.path.join(OUT_DIR, f"overlap_{ds}_k{k}.csv"))
            if cross is not None:
                cross.to_csv(os.path.join(OUT_DIR, f"cross_consensus_{ds}_k{k}.csv"),
                             index=False)

            # Record aggregate stats
            upper = mat.where(np.triu(np.ones(mat.shape, dtype=bool), k=1))
            vals = upper.values[~np.isnan(upper.values)]
            all_overlap_rows.append({
                "dataset": ds, "k": k,
                "n_methods": mat.shape[0],
                "mean_jaccard":   float(np.mean(vals)) if len(vals) else None,
                "median_jaccard": float(np.median(vals)) if len(vals) else None,
                "max_jaccard":    float(np.max(vals)) if len(vals) else None,
                "min_jaccard":    float(np.min(vals)) if len(vals) else None,
                "n_features_cross3+": int((cross["n_methods"] >= 3).sum()) if cross is not None else 0,
                "n_features_cross5+": int((cross["n_methods"] >= 5).sum()) if cross is not None else 0,
            })
            print(f"  {ds:38s} k={k:3d}  mean_jac={all_overlap_rows[-1]['mean_jaccard']:.3f}  "
                  f"max={all_overlap_rows[-1]['max_jaccard']:.3f}  "
                  f"cross≥3: {all_overlap_rows[-1]['n_features_cross3+']}")

    if all_overlap_rows:
        pd.DataFrame(all_overlap_rows).to_csv(
            os.path.join(OUT_DIR, "overlap_summary.csv"), index=False)
        print(f"\nSaved: {os.path.join(OUT_DIR, 'overlap_summary.csv')}")


if __name__ == "__main__":
    main()
