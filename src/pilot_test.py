"""
Pilot test: 2-fold × 1 repeat on Lung_GSE19804 for ALL 15 methods.
Verifies pipeline correctness before full 5×5 run.
"""
import sys
import os
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from protocol import load_dataset, preprocess
from metrics import compute_metrics, compute_stability
from methods import ALL_METHODS

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "main_benchmark", "Lung_GSE19804.csv")
PILOT_CV = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)


def run_pilot_method(method_name, method_fn, X, y):
    total_features = X.shape[1]
    fold_records = []
    all_selected = []

    for fold_idx, (train_idx, test_idx) in enumerate(PILOT_CV.split(X, y)):
        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train, X_test, kept_idx = preprocess(X_train_raw, X_test_raw)

        t0 = time.time()
        try:
            proc_selected = method_fn(X_train, y_train)
        except Exception as e:
            return {"status": "ERROR", "error": str(e), "method": method_name}
        runtime = time.time() - t0

        proc_selected = np.asarray(proc_selected, dtype=int)
        if len(proc_selected) == 0:
            proc_selected = np.arange(min(10, X_train.shape[1]))

        original_selected = kept_idx[proc_selected].tolist()

        X_tr = X_train[:, proc_selected]
        X_te = X_test[:, proc_selected]
        clf = LogisticRegression(C=1, max_iter=1000, random_state=42, solver="lbfgs")
        clf.fit(X_tr, y_train)
        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_prob, original_selected, total_features)
        metrics["runtime_sec"] = round(runtime, 2)
        fold_records.append(metrics)
        all_selected.append(original_selected)

    stability = compute_stability(all_selected)
    aucs = [r["auc"] for r in fold_records]
    return {
        "status": "OK",
        "method": method_name,
        "auc": round(float(np.mean(aucs)), 4),
        "n_features": round(float(np.mean([r["n_features"] for r in fold_records])), 1),
        "stability": round(stability, 4),
        "runtime_sec": round(float(np.mean([r["runtime_sec"] for r in fold_records])), 2),
    }


if __name__ == "__main__":
    print("=== PILOT TEST: Lung_GSE19804 × all 15 methods (2-fold) ===\n")
    X, y, _ = load_dataset(DATA_PATH)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {int(y.sum())} pos\n")

    results = []
    for name, fn in ALL_METHODS.items():
        print(f"[{name}] running...", flush=True)
        t0 = time.time()
        res = run_pilot_method(name, fn, X, y)
        elapsed = time.time() - t0
        if res["status"] == "OK":
            print(f"  OK  AUC={res['auc']:.4f}  n_feat={res['n_features']:.0f}  "
                  f"stability={res['stability']:.3f}  t={elapsed:.1f}s")
            if res["auc"] < 0.4:
                print(f"  WARNING: AUC<0.4, possible label/pipeline issue!")
        else:
            print(f"  FAIL: {res['error']}")
        results.append({**res, "total_sec": round(elapsed, 1)})

    print("\n=== SUMMARY ===")
    ok = [r for r in results if r["status"] == "OK"]
    fail = [r for r in results if r["status"] != "OK"]
    print(f"PASS: {len(ok)}/15   FAIL: {len(fail)}/15")
    if fail:
        for r in fail:
            print(f"  FAILED: {r['method']} — {r.get('error','')}")

    with open(os.path.join(os.path.dirname(__file__), "..", "output", "pilot_results.json"), "w") as f:
        os.makedirs(os.path.join(os.path.dirname(__file__), "..", "output"), exist_ok=True)
        json.dump(results, f, indent=2)
    print("\nResults saved to output/pilot_results.json")
