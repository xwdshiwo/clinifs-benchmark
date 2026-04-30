"""
V2 enhanced metrics module.
Computes full metric suite including MCC, Brier, PR-AUC, calibration.
"""
import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, balanced_accuracy_score,
    matthews_corrcoef, brier_score_loss, average_precision_score,
    precision_score, recall_score, confusion_matrix
)


def compute_full_metrics(y_true, y_pred, y_prob):
    """Compute a comprehensive 12-metric suite from predictions."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = float("nan")
    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = float("nan")

    return {
        "auc":         round(auc, 4),
        "pr_auc":      round(pr_auc, 4),
        "acc":         round(float(accuracy_score(y_true, y_pred)), 4),
        "bacc":        round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "f1":          round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "precision":   round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":      round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "mcc":         round(float(matthews_corrcoef(y_true, y_pred)), 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "ppv":         round(ppv, 4),
        "npv":         round(npv, 4),
        "brier":       round(float(brier_score_loss(y_true, y_prob)), 4),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def aggregate_metrics_across_folds(fold_records):
    """Return mean/std/CI95 for all metric keys across folds."""
    metric_keys = [k for k in fold_records[0]["metrics"].keys()
                   if k not in ("tp", "fp", "tn", "fn")]
    out = {}
    for k in metric_keys:
        vals = [r["metrics"][k] for r in fold_records
                if r["metrics"][k] is not None and not np.isnan(r["metrics"][k])]
        if not vals:
            out[f"{k}_mean"] = None
            continue
        vals = np.asarray(vals, dtype=float)
        out[f"{k}_mean"] = round(float(vals.mean()), 4)
        out[f"{k}_std"]  = round(float(vals.std()), 4)
        out[f"{k}_ci95"] = [round(float(np.percentile(vals, 2.5)), 4),
                            round(float(np.percentile(vals, 97.5)), 4)]
    return out


if __name__ == "__main__":
    # Quick sanity
    y_true = [0, 1, 0, 1, 1, 0]
    y_pred = [0, 1, 0, 1, 0, 0]
    y_prob = [0.1, 0.9, 0.2, 0.8, 0.4, 0.3]
    m = compute_full_metrics(y_true, y_pred, y_prob)
    print(m)
