"""
External validation runner.
Protocol:
  1. Run FS on full discovery set → selected features
  2. Train LR on full discovery, evaluate via 5-fold CV (internal AUC estimate)
  3. Train LR on full discovery, evaluate on held-out validation set
  4. Generalization drop = (disc_cv_auc - val_auc) / disc_cv_auc * 100%
"""
import sys
import os
import json
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
warnings.filterwarnings("ignore")

from protocol import load_dataset, preprocess
from methods import MAIN_METHODS

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "external_validation")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "external_validation")

PAIRS = [
    {
        "name": "E2_Lung",
        "discovery": "E2_Lung_discovery_aligned.csv",
        "validation": "E2_Lung_validation_aligned.csv",
    },
    {
        "name": "E3_Liver",
        "discovery": "E3_Liver_discovery_aligned.csv",
        "validation": "E3_Liver_validation_aligned.csv",
    },
    {
        "name": "E4_CRC",
        "discovery": "E4_CRC_44076_to_25070_discovery_aligned.csv",
        "validation": "E4_CRC_44076_to_25070_validation_aligned.csv",
    },
]


def run_external_pair(pair, method_name, method_fn, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, "result.json")
    if os.path.exists(result_path):
        with open(result_path) as f:
            return json.load(f)

    disc_path = os.path.join(DATA_DIR, pair["discovery"])
    val_path = os.path.join(DATA_DIR, pair["validation"])

    X_disc, y_disc, _ = load_dataset(disc_path)
    X_val, y_val, _   = load_dataset(val_path)

    total_features = X_disc.shape[1]

    # Preprocess discovery set (fit on full discovery)
    X_disc_proc, X_val_proc, kept_idx = preprocess(X_disc, X_val)

    # Feature selection on full preprocessed discovery
    t0 = time.time()
    try:
        proc_selected = method_fn(X_disc_proc, y_disc)
    except Exception as e:
        print(f"    FS ERROR: {e} — fallback to top-20 ANOVA")
        from sklearn.feature_selection import f_classif
        scores, _ = f_classif(X_disc_proc, y_disc)
        proc_selected = np.argsort(np.nan_to_num(scores))[::-1][:20]
    runtime = time.time() - t0

    proc_selected = np.asarray(proc_selected, dtype=int)
    if len(proc_selected) == 0:
        proc_selected = np.arange(min(10, X_disc_proc.shape[1]))

    original_selected = kept_idx[proc_selected].tolist()
    X_disc_sel = X_disc_proc[:, proc_selected]
    X_val_sel  = X_val_proc[:, proc_selected]

    # Internal 5-fold CV AUC on discovery
    clf = LogisticRegression(C=1, max_iter=1000, random_state=42, solver="lbfgs")
    cv_scores = cross_val_score(clf, X_disc_sel, y_disc, cv=5,
                                scoring="roc_auc", n_jobs=-1)
    disc_cv_auc = float(np.mean(cv_scores))

    # Train on full discovery, evaluate on validation
    clf.fit(X_disc_sel, y_disc)
    y_val_prob = clf.predict_proba(X_val_sel)[:, 1]
    val_auc = float(roc_auc_score(y_val, y_val_prob))

    gen_drop = (disc_cv_auc - val_auc) / max(disc_cv_auc, 1e-9) * 100

    result = {
        "pair":        pair["name"],
        "method":      method_name,
        "n_disc":      int(X_disc.shape[0]),
        "n_val":       int(X_val.shape[0]),
        "n_features":  len(original_selected),
        "total_features": total_features,
        "disc_cv_auc": round(disc_cv_auc, 4),
        "val_auc":     round(val_auc, 4),
        "gen_drop_pct": round(gen_drop, 2),
        "runtime_sec": round(runtime, 2),
    }
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"    disc_cv_auc={disc_cv_auc:.4f}  val_auc={val_auc:.4f}  "
          f"drop={gen_drop:.1f}%  n_feat={len(original_selected)}")
    return result


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = []
    total = len(PAIRS) * len(MAIN_METHODS)
    done = 0

    print(f"\n{'='*60}")
    print(f"External Validation: {len(PAIRS)} pairs × {len(MAIN_METHODS)} methods")
    print(f"{'='*60}\n")

    for pair in PAIRS:
        for method_name, method_fn in MAIN_METHODS.items():
            done += 1
            print(f"[{done}/{total}] {pair['name']} × {method_name}")
            out_dir = os.path.join(OUTPUT_DIR, pair["name"], method_name)
            res = run_external_pair(pair, method_name, method_fn, out_dir)
            if res:
                all_results.append(res)

    if all_results:
        df = pd.DataFrame(all_results)
        out_csv = os.path.join(OUTPUT_DIR, "external_validation_summary.csv")
        df.to_csv(out_csv, index=False)
        print(f"\n{'='*60}")
        print("EXTERNAL VALIDATION DONE")
        print(df[["pair", "method", "disc_cv_auc", "val_auc",
                   "gen_drop_pct", "n_features"]].to_string(index=False))
        print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
