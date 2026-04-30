"""
V2 runner with full result schema.
Saves per-fold: y_true, y_pred, y_prob, sample_ids, selected features,
metrics, runtime, EA convergence curves.
"""
import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from protocol import OUTER_CV, load_dataset, preprocess
from metrics_v2 import compute_full_metrics, aggregate_metrics_across_folds
from stability import nogueira_stability, kuncheva_avg, jaccard_avg


def _load_with_sample_ids(path):
    """Load X, y, feature_names, and sample_ids (from the first column)."""
    df = pd.read_csv(path)
    if "type" in df.columns:
        y = (~df["type"].str.contains("normal", case=False)).astype(int).values
        drop_cols = [c for c in ["samples", "type"] if c in df.columns]
        feat_df = df.drop(columns=drop_cols)
        X = feat_df.values
        feat_names = feat_df.columns.tolist()
        sample_ids = (df["samples"].astype(str).tolist()
                      if "samples" in df.columns
                      else [f"s{i}" for i in range(len(df))])
    elif "label" in df.columns:
        y = df["label"].values.astype(int)
        feat_df = df.drop(columns=["label"])
        X = feat_df.values
        feat_names = feat_df.columns.tolist()
        # Use index as sample_id
        sample_ids = [str(i) for i in df.index.tolist()]
    else:
        raise ValueError(f"Cannot detect format: {path}")
    return X, y, feat_names, sample_ids


def run_experiment_v2(dataset_path, method_fn, method_cfg, output_dir,
                     feature_name_col="probe", probe_to_gene=None):
    """
    Execute full benchmark run for one (method, dataset, config) combination.
    Saves per-fold JSON + summary.json with complete metrics and metadata.

    Args:
        dataset_path: CSV path
        method_fn:    callable accepting (X, y, **cfg) -> dict
        method_cfg:   kwargs dict (may include 'k', 'alpha', 'beta', etc.)
        output_dir:   where to save fold and summary results
        feature_name_col: description of feature name semantics
        probe_to_gene:  optional dict for probe→gene_symbol mapping
    """
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "summary.json")

    if os.path.exists(summary_path):
        with open(summary_path) as f:
            return json.load(f)

    X, y, feat_names, sample_ids = _load_with_sample_ids(dataset_path)
    total_features = X.shape[1]
    n_splits = getattr(OUTER_CV, "n_splits", 5)
    fold_records = []
    all_selected_orig = []

    for fold_idx, (train_idx, test_idx) in enumerate(OUTER_CV.split(X, y)):
        rep = fold_idx // n_splits
        fold = fold_idx % n_splits
        fold_path = os.path.join(output_dir, f"rep{rep}_fold{fold}.json")

        if os.path.exists(fold_path):
            with open(fold_path) as f:
                rec = json.load(f)
            fold_records.append(rec)
            all_selected_orig.append(rec["selected_features_orig"])
            continue

        X_tr_raw, X_te_raw = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        X_tr, X_te, kept_idx = preprocess(X_tr_raw, X_te_raw)

        # Feature selection — strip metadata-only keys before calling method_fn
        cfg_for_call = {k: v for k, v in method_cfg.items()
                         if k not in ("method_name",)}
        t0 = time.time()
        fallback_used = False
        fallback_reason = None
        try:
            fs_out = method_fn(X_tr, y_tr, **cfg_for_call)
            sel = np.asarray(fs_out["selected_idx"], dtype=int)
            scores = fs_out.get("scores", None)
            ea_info = fs_out.get("ea_info", {})
        except Exception as e:
            print(f"  !! FS FALLBACK TRIGGERED: {e} — using top-10 ANOVA")
            from sklearn.feature_selection import f_classif
            sc, _ = f_classif(X_tr, y_tr)
            sc = np.nan_to_num(sc, nan=0.0)
            sel = np.argsort(sc)[::-1][:10]
            scores = sc
            fallback_used = True
            fallback_reason = str(e)
            ea_info = {"error": str(e), "fallback": "anova_top10"}
        runtime_fs = time.time() - t0

        if len(sel) == 0:
            sel = np.arange(min(10, X_tr.shape[1]))

        # Map to original space
        orig_sel = kept_idx[sel].tolist()
        probe_ids = [feat_names[i] for i in orig_sel] if feat_names else None
        gene_symbols = ([probe_to_gene.get(p, None) for p in probe_ids]
                        if probe_to_gene is not None and probe_ids else None)

        # Classifier
        X_tr_sel = X_tr[:, sel]
        X_te_sel = X_te[:, sel]
        t1 = time.time()
        clf = LogisticRegression(C=1, max_iter=1000, random_state=42, solver="lbfgs")
        clf.fit(X_tr_sel, y_tr)
        y_prob = clf.predict_proba(X_te_sel)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        runtime_clf = time.time() - t1

        metrics = compute_full_metrics(y_te, y_pred, y_prob)

        rec = {
            "rep": rep, "fold": fold,
            "n_train": int(len(train_idx)),
            "n_test":  int(len(test_idx)),
            "selected_features_idx":  sel.tolist(),
            "selected_features_orig": orig_sel,
            "selected_probe_ids":     probe_ids,
            "selected_gene_symbols":  gene_symbols,
            "feature_scores": (
                None if scores is None
                else [float(scores[i]) for i in sel]
            ),
            "y_true": y_te.tolist(),
            "y_pred": y_pred.tolist(),
            "y_prob": [round(float(p), 6) for p in y_prob],
            "sample_ids_test": [sample_ids[i] for i in test_idx],
            "metrics": metrics,
            "runtime_fs_sec":  round(runtime_fs, 3),
            "runtime_clf_sec": round(runtime_clf, 3),
            "ea_info": ea_info,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "config": {k: (v if not isinstance(v, np.ndarray) else v.tolist())
                       for k, v in method_cfg.items()},
        }

        with open(fold_path, "w") as f:
            json.dump(rec, f)

        fold_records.append(rec)
        all_selected_orig.append(orig_sel)
        auc = metrics["auc"]
        print(f"  rep{rep}_fold{fold}: AUC={auc:.4f}  "
              f"n_feat={len(orig_sel)}  t_fs={runtime_fs:.1f}s")

    # Build predictions matrix (sample × fold long format)
    pred_rows = []
    for rec in fold_records:
        for sid, yt, yp, yp_prob in zip(rec["sample_ids_test"],
                                        rec["y_true"],
                                        rec["y_pred"],
                                        rec["y_prob"]):
            pred_rows.append({
                "sample_id": sid,
                "rep": rec["rep"], "fold": rec["fold"],
                "y_true": yt, "y_pred": yp, "y_prob": yp_prob,
            })
    pd.DataFrame(pred_rows).to_csv(
        os.path.join(output_dir, "predictions_long.csv"), index=False)

    # Fold-level results
    fold_df_rows = []
    for rec in fold_records:
        r = {"rep": rec["rep"], "fold": rec["fold"],
             "n_features": len(rec["selected_features_orig"]),
             "runtime_fs_sec": rec["runtime_fs_sec"]}
        r.update({k: v for k, v in rec["metrics"].items()
                 if k not in ("tp", "fp", "tn", "fn")})
        fold_df_rows.append(r)
    pd.DataFrame(fold_df_rows).to_csv(
        os.path.join(output_dir, "fold_results.csv"), index=False)

    # Selected features list
    with open(os.path.join(output_dir, "selected_features.json"), "w") as f:
        json.dump(all_selected_orig, f)

    # Feature frequency across folds
    feat_freq = {}
    for sel in all_selected_orig:
        for idx in sel:
            name = feat_names[idx] if feat_names and idx < len(feat_names) else str(idx)
            feat_freq[name] = feat_freq.get(name, 0) + 1

    # Stability
    n_folds = len(all_selected_orig)
    stab_nog = nogueira_stability(all_selected_orig, total_features)
    try:
        stab_kun = kuncheva_avg(all_selected_orig, total_features)
        if stab_kun is not None and (stab_kun != stab_kun):  # NaN check
            stab_kun = None
    except Exception:
        stab_kun = None
    stab_jac = jaccard_avg(all_selected_orig)

    agg = aggregate_metrics_across_folds(fold_records)
    runtime_fs_mean = float(np.mean([r["runtime_fs_sec"] for r in fold_records]))
    n_features_vals = [len(r["selected_features_orig"]) for r in fold_records]

    consensus_80 = {n: c for n, c in feat_freq.items() if c / n_folds >= 0.8}
    consensus_50 = {n: c for n, c in feat_freq.items() if c / n_folds >= 0.5}

    # Aggregate fallback usage across folds (critical for experiment integrity)
    fallback_flags = [r.get("fallback_used", False) for r in fold_records]
    fallback_reasons = [r.get("fallback_reason") for r in fold_records if r.get("fallback_used")]

    summary = {
        "method": method_cfg.get("method_name",
                                 method_fn.__name__.replace("method_", "")),
        "dataset": os.path.basename(dataset_path).replace(".csv", ""),
        "config": {k: (v if not isinstance(v, np.ndarray) else v.tolist())
                   for k, v in method_cfg.items()},
        "n_folds": n_folds,
        "total_features": total_features,
        "n_features_mean": round(float(np.mean(n_features_vals)), 2),
        "n_features_std":  round(float(np.std(n_features_vals)), 2),
        "stability_nogueira": round(stab_nog, 4),
        "stability_kuncheva": (round(stab_kun, 4) if stab_kun is not None else None),
        "stability_jaccard":  round(stab_jac, 4),
        "runtime_fs_mean_sec": round(runtime_fs_mean, 2),
        "feature_frequency_top20": dict(
            sorted(feat_freq.items(), key=lambda kv: -kv[1])[:20]),
        "consensus_features_80pct": consensus_80,
        "consensus_features_50pct_count": len(consensus_50),
        "fallback_used_any": bool(any(fallback_flags)),
        "fallback_used_fraction": round(sum(fallback_flags) / max(len(fallback_flags), 1), 3),
        "fallback_reasons": list(set(fallback_reasons))[:3] if fallback_reasons else [],
        **agg,
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  DONE: AUC={summary['auc_mean']:.4f}±{summary['auc_std']:.4f}  "
          f"n_feat={summary['n_features_mean']:.0f}  "
          f"stab={summary['stability_nogueira']:.3f}")
    return summary


if __name__ == "__main__":
    # Quick pilot
    import sys
    sys.path.insert(0, ".")
    from methods.methods_v2 import method_anova
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ds = os.path.join(root, "data", "main_benchmark", "Bladder_GSE31189.csv")
    out = os.path.join(root, "output", "v2", "_pilot", "anova_k10", "Bladder_GSE31189")
    run_experiment_v2(ds, method_anova, {"k": 10}, out)
