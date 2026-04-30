import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, balanced_accuracy_score
from itertools import combinations


def compute_metrics(y_true, y_pred, y_prob, selected_features, total_features):
    return {
        "auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "acc": round(float(accuracy_score(y_true, y_pred)), 4),
        "bacc": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "n_features": len(selected_features),
        "feature_ratio": round(len(selected_features) / max(total_features, 1), 6),
    }


def compute_stability(feature_lists):
    if len(feature_lists) < 2:
        return 0.0
    scores = []
    for a, b in combinations(feature_lists, 2):
        sa, sb = set(a), set(b)
        union = len(sa | sb)
        scores.append(len(sa & sb) / union if union > 0 else 1.0)
    return float(np.mean(scores))
