"""
E5: New external validation pairs (Renal, Lung, Prostate).
Runs 15 methods × 3 pairs, training on discovery and evaluating on validation.
"""
import os, sys, json, time
import numpy as np
import pandas as pd
from io import StringIO
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

ROOT = os.path.dirname(os.path.abspath(__file__))
EXT_DIR = os.path.join(ROOT, "data", "external_validation_v2")
OUT_ROOT = os.path.join(ROOT, "output", "v2", "E5_external_pairs")
os.makedirs(OUT_ROOT, exist_ok=True)

PAIRS = [
    ("E5_Renal_53757_to_66270",    "Renal"),
    ("E5_Lung_19804_to_27262",     "Lung"),
    ("E5_Prostate_6919_to_26910",  "Prostate"),
]
METHODS = ["variance", "anova", "mi", "mrmr", "relieff",
           "l1_logistic", "elasticnet", "linearsvc_l1",
           "boruta", "extratrees", "rfecv",
           "ga", "bpso", "sfe", "mel"]
K_TARGET = 20  # fixed k for external validation experiments
N_REPEATS = 5  # independent FS repeats on discovery


def _worker_init():
    for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS",
              "OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[k] = "1"
    import warnings
    warnings.filterwarnings("ignore")


def _run_pair_method(job):
    pair_tag, method_name = job
    out_dir = os.path.join(OUT_ROOT, pair_tag, method_name)
    summary_path = os.path.join(out_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            return json.load(f), pair_tag, method_name, None

    buf = StringIO(); old = sys.stdout; sys.stdout = buf
    try:
        os.makedirs(out_dir, exist_ok=True)
        sys.path.insert(0, os.path.join(ROOT, "src"))
        from methods.methods_v2 import ALL_METHODS_V2
        from metrics_v2 import compute_full_metrics

        disc_path = os.path.join(EXT_DIR, f"{pair_tag}_discovery.csv")
        val_path  = os.path.join(EXT_DIR, f"{pair_tag}_validation.csv")
        Xd = pd.read_csv(disc_path)
        Xv = pd.read_csv(val_path)
        yd = Xd["label"].values.astype(int)
        yv = Xv["label"].values.astype(int)
        Xd = Xd.drop(columns=["label"])
        Xv = Xv.drop(columns=["label"])

        # Ensure feature columns are aligned
        common_cols = [c for c in Xd.columns if c in Xv.columns]
        Xd = Xd[common_cols]; Xv = Xv[common_cols]
        feat_names = common_cols

        # Preprocess (fit on discovery only)
        vt = VarianceThreshold(threshold=0.0)
        Xd_p = vt.fit_transform(Xd.values)
        Xv_p = vt.transform(Xv.values)
        kept_idx = vt.get_support(indices=True)
        scaler = StandardScaler()
        Xd_p = scaler.fit_transform(Xd_p)
        Xv_p = scaler.transform(Xv_p)

        method_fn = ALL_METHODS_V2[method_name]

        # Method-specific config
        cfg = {"k": K_TARGET}
        if method_name == "ga":
            cfg.update({"alpha":0.3,"beta":0.7,"n_gen":15,"pop_size":20,
                        "max_features":K_TARGET})
        elif method_name == "bpso":
            cfg.update({"alpha":0.3,"beta":0.7,"n_iter":20,"n_particles":20})

        fs_results = []
        fs_sel_all = []
        for seed in range(N_REPEATS):
            cfg_i = dict(cfg)
            if method_name in ("ga","bpso","sfe","mel"):
                cfg_i["random_state"] = seed
            t0 = time.time()
            res = method_fn(Xd_p, yd, **cfg_i)
            sel = np.asarray(res["selected_idx"], dtype=int)
            orig = kept_idx[sel].tolist()
            fs_results.append({"seed":seed,
                                "sel_preproc": sel.tolist(),
                                "sel_orig": orig,
                                "probe_names": [feat_names[i] for i in orig],
                                "runtime_sec": round(time.time()-t0, 3)})
            fs_sel_all.append(orig)

        # Compute stable feature set (Jaccard-based: appear in >= N_REPEATS//2)
        from collections import Counter
        counter = Counter()
        for s in fs_sel_all:
            counter.update(s)
        threshold = max(1, N_REPEATS // 2)
        stable_idx_orig = [f for f, c in counter.items() if c >= threshold]
        if len(stable_idx_orig) == 0:
            # fallback to most-frequent top-K_TARGET
            stable_idx_orig = [f for f, _ in counter.most_common(K_TARGET)]

        # Map stable features back to preprocessed space
        idx_map = {o: p for p, o in enumerate(kept_idx)}
        stable_preproc = [idx_map[o] for o in stable_idx_orig if o in idx_map]

        # Train on discovery, test on validation using stable features
        clf = LogisticRegression(C=1, max_iter=1000, random_state=42, solver="lbfgs")
        clf.fit(Xd_p[:, stable_preproc], yd)
        y_prob_disc = clf.predict_proba(Xd_p[:, stable_preproc])[:, 1]
        y_pred_disc = (y_prob_disc >= 0.5).astype(int)
        y_prob_val  = clf.predict_proba(Xv_p[:, stable_preproc])[:, 1]
        y_pred_val  = (y_prob_val  >= 0.5).astype(int)

        disc_metrics = compute_full_metrics(yd, y_pred_disc, y_prob_disc)
        val_metrics  = compute_full_metrics(yv, y_pred_val,  y_prob_val)

        # Stability across N_REPEATS
        from stability import nogueira_stability
        # Need total features in preprocessed space — use kept_idx length
        stab = nogueira_stability(fs_sel_all, len(kept_idx))

        summary = {
            "pair": pair_tag,
            "method": method_name,
            "k_target": K_TARGET,
            "n_repeats": N_REPEATS,
            "n_stable_features": len(stable_idx_orig),
            "stable_probe_names": [feat_names[o] for o in stable_idx_orig],
            "stability_nogueira": round(stab, 4),
            "disc_auc":        disc_metrics["auc"],
            "val_auc":         val_metrics["auc"],
            "disc_acc":        disc_metrics["acc"],
            "val_acc":         val_metrics["acc"],
            "disc_f1":         disc_metrics["f1"],
            "val_f1":          val_metrics["f1"],
            "disc_bacc":       disc_metrics["bacc"],
            "val_bacc":        val_metrics["bacc"],
            "disc_mcc":        disc_metrics["mcc"],
            "val_mcc":         val_metrics["mcc"],
            "generalization_drop_auc": round(
                (disc_metrics["auc"] - val_metrics["auc"]) / max(disc_metrics["auc"], 1e-9), 4),
            "fs_results": fs_results,
            "val_y_true": yv.tolist(),
            "val_y_pred": y_pred_val.tolist(),
            "val_y_prob": [round(float(p), 6) for p in y_prob_val],
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=1)
        sys.stdout = old
        return summary, pair_tag, method_name, None
    except Exception as e:
        import traceback
        sys.stdout = old
        return None, pair_tag, method_name, f"{e}\n{traceback.format_exc()[-400:]}"


def main(workers=5):
    tasks = [(p, m) for p, _ in PAIRS for m in METHODS]
    t0 = time.time()
    print("=" * 75)
    print(f"E5 External Validation: {len(tasks)} experiments ({len(PAIRS)} pairs × {len(METHODS)} methods)")
    print("=" * 75)

    rows = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as ex:
        futures = {ex.submit(_run_pair_method, t): t for t in tasks}
        done = 0
        for fut in as_completed(futures):
            summary, pair, method, err = fut.result()
            done += 1
            wall = (time.time() - t0) / 60
            if err:
                print(f"[{done:2d}/{len(tasks)}] {pair}/{method}  ERR: {str(err)[:60]}")
            else:
                print(f"[{done:2d}/{len(tasks)}] {pair[:30]} {method:12s}  "
                      f"disc_AUC={summary['disc_auc']:.4f}  val_AUC={summary['val_auc']:.4f}  "
                      f"drop={summary['generalization_drop_auc']*100:+.1f}%  "
                      f"stab={summary['stability_nogueira']:.3f}  wall={wall:.1f}m")
                rows.append({
                    "pair": pair, "method": method,
                    "disc_auc": summary["disc_auc"], "val_auc": summary["val_auc"],
                    "drop": summary["generalization_drop_auc"],
                    "disc_bacc": summary["disc_bacc"], "val_bacc": summary["val_bacc"],
                    "stability": summary["stability_nogueira"],
                    "n_stable_features": summary["n_stable_features"],
                })

    if rows:
        pd.DataFrame(rows).to_csv(
            os.path.join(OUT_ROOT, "e5_summary.csv"), index=False)
        print(f"\nSaved: {os.path.join(OUT_ROOT, 'e5_summary.csv')}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=5)
    args = p.parse_args()
    main(workers=args.workers)
