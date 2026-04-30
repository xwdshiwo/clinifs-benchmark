"""
V2 method interface — all methods support explicit k parameter.
Returns dict with 'selected_idx', 'scores' (optional), 'ea_info' (EA-only).
"""
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier


PREFILTER = 2000


def _anova_prefilter(X, y, n=PREFILTER):
    if X.shape[1] <= n:
        return X, np.arange(X.shape[1]), None
    scores, _ = f_classif(X, y)
    scores = np.nan_to_num(scores, nan=0.0)
    top_idx = np.argsort(scores)[::-1][:n]
    return X[:, top_idx], top_idx, scores


def _topk(scores, k, n_features):
    k = min(k, n_features)
    return np.argsort(scores)[::-1][:k]


# ════════════════════════ FILTER METHODS ════════════════════════

def method_variance(X, y, k):
    v = np.var(X, axis=0)
    return {"selected_idx": _topk(v, k, X.shape[1]),
            "scores": v}


def method_anova(X, y, k):
    scores, _ = f_classif(X, y)
    scores = np.nan_to_num(scores, nan=0.0)
    return {"selected_idx": _topk(scores, k, X.shape[1]),
            "scores": scores}


def method_mi(X, y, k):
    # Pre-filter to 2000 by ANOVA for speed (same rationale as mrmr/relieff).
    # For high-dim arrays, mutual_info_classif on 50K features is prohibitively slow.
    X_sub, pre_idx, _ = _anova_prefilter(X, y)
    scores_sub = mutual_info_classif(X_sub, y, random_state=42)
    sub_idx = _topk(scores_sub, k, X_sub.shape[1])
    scores_full = np.zeros(X.shape[1])
    scores_full[pre_idx] = scores_sub
    return {"selected_idx": pre_idx[sub_idx], "scores": scores_full}


def method_mrmr(X, y, k):
    from mrmr import mrmr_classif
    import contextlib, io
    X_sub, pre_idx, _ = _anova_prefilter(X, y)
    k_eff = min(k, X_sub.shape[1])
    feat_names = [f"f{i}" for i in range(X_sub.shape[1])]
    df = pd.DataFrame(X_sub, columns=feat_names)
    with contextlib.redirect_stderr(io.StringIO()):
        selected_names = mrmr_classif(X=df, y=pd.Series(y), K=k_eff)
    sub_idx = np.array([int(n[1:]) for n in selected_names])
    return {"selected_idx": pre_idx[sub_idx],
            "scores": None}


def method_relieff(X, y, k):
    from skrebate import ReliefF
    X_sub, pre_idx, _ = _anova_prefilter(X, y)
    k_eff = min(k, X_sub.shape[1])
    n_neighbors = min(10, len(y) - 1)
    rf = ReliefF(n_features_to_select=k_eff, n_neighbors=n_neighbors, n_jobs=1)
    rf.fit(X_sub, y)
    sub_idx = rf.top_features_[:k_eff]
    # Get scores for all pre-filtered features
    scores_full = np.zeros(X.shape[1])
    scores_full[pre_idx] = rf.feature_importances_
    return {"selected_idx": pre_idx[sub_idx],
            "scores": scores_full}


# ════════════════════════ EMBEDDED METHODS ════════════════════════

def method_l1_logistic(X, y, k):
    """L1-LR: C adjusted to produce ~k features, else top-k by |coef|."""
    clf = LogisticRegression(penalty="l1", C=0.1, solver="liblinear",
                              max_iter=1000, random_state=42)
    clf.fit(X, y)
    coef = np.abs(clf.coef_).ravel()
    # Take top-k by coefficient magnitude (regardless of whether they were
    # selected by the L1 threshold); this ensures exact k features
    idx = np.argsort(coef)[::-1][:k]
    return {"selected_idx": idx, "scores": coef}


def method_elasticnet(X, y, k):
    # Pre-filter to 2000 for saga solver speed
    X_sub, pre_idx, _ = _anova_prefilter(X, y)
    clf = LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5,
                              C=0.1, max_iter=1000, random_state=42, n_jobs=1)
    clf.fit(X_sub, y)
    coef = np.abs(clf.coef_).ravel()
    sub_idx = np.argsort(coef)[::-1][:min(k, len(coef))]
    scores_full = np.zeros(X.shape[1])
    scores_full[pre_idx] = coef
    return {"selected_idx": pre_idx[sub_idx], "scores": scores_full}


def method_linearsvc_l1(X, y, k):
    X_sub, pre_idx, _ = _anova_prefilter(X, y)
    clf = LinearSVC(penalty="l1", dual=False, C=0.1, max_iter=2000, random_state=42)
    clf.fit(X_sub, y)
    coef = np.abs(clf.coef_).ravel()
    sub_idx = np.argsort(coef)[::-1][:min(k, len(coef))]
    scores_full = np.zeros(X.shape[1])
    scores_full[pre_idx] = coef
    return {"selected_idx": pre_idx[sub_idx], "scores": scores_full}


def method_boruta(X, y, k):
    """Boruta returns a set; we take top-k by importance score within it (or overall)."""
    from boruta import BorutaPy
    X_sub, pre_idx, _ = _anova_prefilter(X, y)
    rf = RandomForestClassifier(n_estimators=50, max_depth=5, n_jobs=2, random_state=42)
    selector = BorutaPy(rf, n_estimators="auto", max_iter=50, verbose=0, random_state=42)
    selector.fit(X_sub, y)
    # Get importances
    rf.fit(X_sub, y)
    importances = rf.feature_importances_
    # Confirmed features ranked by importance, padded with tentative/rejected
    confirmed = np.where(selector.support_)[0]
    tentative = np.where(selector.support_weak_)[0]
    others = np.setdiff1d(np.arange(len(importances)),
                          np.union1d(confirmed, tentative))
    # Sort each group by importance then concat
    pool = list(confirmed[np.argsort(-importances[confirmed])])
    pool += list(tentative[np.argsort(-importances[tentative])])
    pool += list(others[np.argsort(-importances[others])])
    sub_idx = np.array(pool[:k])
    scores_full = np.zeros(X.shape[1])
    scores_full[pre_idx] = importances
    return {"selected_idx": pre_idx[sub_idx], "scores": scores_full}


def method_extratrees(X, y, k):
    X_sub, pre_idx, _ = _anova_prefilter(X, y)
    et = ExtraTreesClassifier(n_estimators=100, n_jobs=2, random_state=42)
    et.fit(X_sub, y)
    imp = et.feature_importances_
    sub_idx = np.argsort(imp)[::-1][:min(k, len(imp))]
    scores_full = np.zeros(X.shape[1])
    scores_full[pre_idx] = imp
    return {"selected_idx": pre_idx[sub_idx], "scores": scores_full}


def method_rfecv(X, y, k):
    """RFECV: run normal RFECV then take top-k by ranking."""
    from sklearn.feature_selection import RFECV
    X_sub, pre_idx, _ = _anova_prefilter(X, y)
    estimator = LogisticRegression(C=1, solver="liblinear",
                                    max_iter=500, random_state=42)
    selector = RFECV(estimator=estimator, step=0.1, cv=3,
                     scoring="roc_auc", n_jobs=-1, min_features_to_select=k)
    selector.fit(X_sub, y)
    # Use ranking_ to get top-k (rank 1 is best)
    ranking = selector.ranking_
    sub_idx = np.argsort(ranking)[:k]
    scores_full = np.zeros(X.shape[1])
    # Convert ranking to score: 1/rank
    scores_full[pre_idx] = 1.0 / ranking
    return {"selected_idx": pre_idx[sub_idx], "scores": scores_full}


# ════════════════════════ EA METHODS ════════════════════════

def method_ga(X, y, k=None, alpha=0.5, beta=0.5, n_gen=20, pop_size=30,
              max_features=None, random_state=42):
    """
    Custom GA with explicit sparsity penalty and max_features constraint.
    Fitness: alpha * (1 - AUC_cv) + beta * (|F|/n_features)
    """
    from .ga_custom import GAFeatureSelector
    X_sub, pre_idx, _ = _anova_prefilter(X, y)
    kmax = max_features if max_features is not None else k
    selector = GAFeatureSelector(
        alpha=alpha, beta=beta,
        pop_size=pop_size, n_gen=n_gen,
        max_features=kmax,
        random_state=random_state, verbose=False
    )
    selector.fit(X_sub, y)
    sub_idx = selector.best_features_
    # Truncate to exactly k if needed
    if k is not None and len(sub_idx) > k:
        # Take by feature_score (weighted by frequency in final population)
        if selector.feature_frequencies_ is not None:
            freqs = selector.feature_frequencies_[sub_idx]
            sub_idx = sub_idx[np.argsort(-freqs)[:k]]
        else:
            sub_idx = sub_idx[:k]
    elif k is not None and len(sub_idx) < k:
        # Pad with ANOVA-ranked features not already selected
        scores, _ = f_classif(X_sub, y)
        scores = np.nan_to_num(scores, nan=0.0)
        ranked = np.argsort(-scores)
        extra = [i for i in ranked if i not in set(sub_idx)][:k - len(sub_idx)]
        sub_idx = np.concatenate([sub_idx, np.array(extra, dtype=int)])
    return {
        "selected_idx": pre_idx[sub_idx],
        "scores": None,
        "ea_info": {
            "fitness_curve": selector.fitness_curve_.tolist(),
            "n_features_curve": selector.n_features_curve_.tolist(),
            "best_fitness": float(selector.best_fitness_),
            "n_gen": selector.n_gen,
            "alpha": alpha, "beta": beta, "max_features": kmax,
        }
    }


def method_bpso(X, y, k=None, alpha=0.5, beta=0.5, n_iter=30, n_particles=20,
                max_features=None, random_state=42):
    """BPSO with sparsity penalty and top-k truncation."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from algorithms.binary_pso import BinaryPSO
    X_sub, pre_idx, _ = _anova_prefilter(X, y)
    selector = BinaryPSO(
        n_particles=n_particles, max_iter=n_iter,
        w=0.9, c1=2.0, c2=2.0,
        alpha=alpha, beta=beta,
        k_neighbors=5, random_state=random_state, verbose=False
    )
    selector.fit(X_sub, y)
    sub_idx = selector.best_features_
    # Truncate to k by ANOVA score ranking among selected
    if k is not None and len(sub_idx) > k:
        scores, _ = f_classif(X_sub[:, sub_idx], y)
        scores = np.nan_to_num(scores, nan=0.0)
        sub_idx = sub_idx[np.argsort(-scores)[:k]]
    elif k is not None and len(sub_idx) < k:
        scores, _ = f_classif(X_sub, y)
        scores = np.nan_to_num(scores, nan=0.0)
        ranked = np.argsort(-scores)
        extra = [i for i in ranked if i not in set(sub_idx)][:k - len(sub_idx)]
        sub_idx = np.concatenate([sub_idx, np.array(extra, dtype=int)])
    return {
        "selected_idx": pre_idx[sub_idx],
        "scores": None,
        "ea_info": {
            "fitness_curve": selector.fitness_curve_.tolist(),
            "n_features_curve": selector.num_features_curve_.tolist(),
            "best_fitness": float(selector.best_fitness_),
            "alpha": alpha, "beta": beta,
        }
    }


def method_sfe(X, y, k=None, random_state=42):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from published_algorithms.sfe import SFE
    X_sub, pre_idx, _ = _anova_prefilter(X, y)
    selector = SFE(max_iter=60, random_state=random_state, verbose=False)
    selector.fit(X_sub, y)
    sub_idx = selector.best_features_
    if sub_idx is None or len(sub_idx) == 0:
        sub_idx = np.arange(min(10, X_sub.shape[1]))
    if k is not None and len(sub_idx) > k:
        scores, _ = f_classif(X_sub[:, sub_idx], y)
        scores = np.nan_to_num(scores, nan=0.0)
        sub_idx = sub_idx[np.argsort(-scores)[:k]]
    elif k is not None and len(sub_idx) < k:
        scores, _ = f_classif(X_sub, y)
        scores = np.nan_to_num(scores, nan=0.0)
        ranked = np.argsort(-scores)
        extra = [i for i in ranked if i not in set(sub_idx)][:k - len(sub_idx)]
        sub_idx = np.concatenate([sub_idx, np.array(extra, dtype=int)])
    return {"selected_idx": pre_idx[sub_idx], "scores": None, "ea_info": {}}


def method_mel(X, y, k=None, random_state=42):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from published_algorithms.mel import MEL
    X_sub, pre_idx, _ = _anova_prefilter(X, y)
    selector = MEL(n_particles=20, max_iter=50, random_state=random_state, verbose=False)
    selector.fit(X_sub, y)
    sub_idx = selector.best_features_
    if sub_idx is None or len(sub_idx) == 0:
        sub_idx = np.arange(min(10, X_sub.shape[1]))
    if k is not None and len(sub_idx) > k:
        scores, _ = f_classif(X_sub[:, sub_idx], y)
        scores = np.nan_to_num(scores, nan=0.0)
        sub_idx = sub_idx[np.argsort(-scores)[:k]]
    elif k is not None and len(sub_idx) < k:
        scores, _ = f_classif(X_sub, y)
        scores = np.nan_to_num(scores, nan=0.0)
        ranked = np.argsort(-scores)
        extra = [i for i in ranked if i not in set(sub_idx)][:k - len(sub_idx)]
        sub_idx = np.concatenate([sub_idx, np.array(extra, dtype=int)])
    return {"selected_idx": pre_idx[sub_idx], "scores": None, "ea_info": {}}


ALL_METHODS_V2 = {
    "variance":     method_variance,
    "anova":        method_anova,
    "mi":           method_mi,
    "mrmr":         method_mrmr,
    "relieff":      method_relieff,
    "l1_logistic":  method_l1_logistic,
    "elasticnet":   method_elasticnet,
    "linearsvc_l1": method_linearsvc_l1,
    "boruta":       method_boruta,
    "extratrees":   method_extratrees,
    "rfecv":        method_rfecv,
    "ga":           method_ga,
    "bpso":         method_bpso,
    "sfe":          method_sfe,
    "mel":          method_mel,
}

# EA methods that accept alpha/beta
EA_METHODS = ["ga", "bpso"]
