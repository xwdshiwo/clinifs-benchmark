"""
Two-stage pipelines: filter pre-selection → EA refinement.
Implements P1-P4 from the V2 plan.
"""
import numpy as np
from sklearn.feature_selection import f_classif, mutual_info_classif


def _filter_top(X, y, n_keep, method="anova"):
    """Return indices of top-n_keep features by a filter score."""
    if method == "anova":
        scores, _ = f_classif(X, y)
        scores = np.nan_to_num(scores, nan=0.0)
    elif method == "mi":
        scores = mutual_info_classif(X, y, random_state=42)
    elif method == "relieff":
        from skrebate import ReliefF
        rf = ReliefF(n_features_to_select=min(n_keep, X.shape[1]),
                     n_neighbors=min(10, len(y) - 1), n_jobs=-1)
        rf.fit(X, y)
        scores = rf.feature_importances_
    else:
        raise ValueError(method)
    return np.argsort(scores)[::-1][:n_keep], scores


def _multi_filter_fusion(X, y, n_keep=200):
    """Compute normalized rank average across 3 filters; keep top-n."""
    scores_a, _ = f_classif(X, y)
    scores_a = np.nan_to_num(scores_a, nan=0.0)
    scores_m = mutual_info_classif(X, y, random_state=42)

    from skrebate import ReliefF
    rf = ReliefF(n_features_to_select=min(200, X.shape[1]),
                 n_neighbors=min(10, len(y) - 1), n_jobs=-1)
    rf.fit(X, y)
    scores_r = rf.feature_importances_

    # Rank-based averaging (lower rank = better)
    def rank_norm(s):
        return np.argsort(np.argsort(-s)) / len(s)
    rank_a = rank_norm(scores_a)
    rank_m = rank_norm(scores_m)
    rank_r = rank_norm(scores_r)
    fused = (rank_a + rank_m + rank_r) / 3.0     # lower is better
    top = np.argsort(fused)[:n_keep]
    return top, -fused  # negate so higher=better


def pipeline_anova_ga(X, y, k, prefilter=200, alpha=0.3, beta=0.7,
                       n_gen=20, pop_size=30, random_state=42):
    """P1: ANOVA top-200 → custom GA with strong sparsity."""
    from .ga_custom import GAFeatureSelector
    top, scores = _filter_top(X, y, prefilter, "anova")
    X_sub = X[:, top]
    ga = GAFeatureSelector(alpha=alpha, beta=beta, pop_size=pop_size,
                            n_gen=n_gen, max_features=k,
                            random_state=random_state, verbose=False)
    ga.fit(X_sub, y)
    sub_idx = ga.best_features_
    if len(sub_idx) > k:
        freqs = ga.feature_frequencies_[sub_idx]
        sub_idx = sub_idx[np.argsort(-freqs)[:k]]
    elif len(sub_idx) < k:
        # Pad by ANOVA-ranked
        inner_scores, _ = f_classif(X_sub, y)
        inner_scores = np.nan_to_num(inner_scores, nan=0.0)
        ranked = np.argsort(-inner_scores)
        extra = [i for i in ranked if i not in set(sub_idx)][:k - len(sub_idx)]
        sub_idx = np.concatenate([sub_idx, np.array(extra, dtype=int)])
    return {
        "selected_idx": top[sub_idx],
        "scores": None,
        "ea_info": {
            "fitness_curve": ga.fitness_curve_.tolist(),
            "n_features_curve": ga.n_features_curve_.tolist(),
            "best_fitness": float(ga.best_fitness_),
            "prefilter": prefilter, "prefilter_method": "anova",
            "alpha": alpha, "beta": beta,
        }
    }


def pipeline_relieff_ga(X, y, k, prefilter=200, alpha=0.3, beta=0.7,
                         n_gen=20, pop_size=30, random_state=42):
    """P2: ReliefF top-200 → GA."""
    from .ga_custom import GAFeatureSelector
    top, scores = _filter_top(X, y, prefilter, "relieff")
    X_sub = X[:, top]
    ga = GAFeatureSelector(alpha=alpha, beta=beta, pop_size=pop_size,
                            n_gen=n_gen, max_features=k,
                            random_state=random_state, verbose=False)
    ga.fit(X_sub, y)
    sub_idx = ga.best_features_
    if len(sub_idx) > k:
        freqs = ga.feature_frequencies_[sub_idx]
        sub_idx = sub_idx[np.argsort(-freqs)[:k]]
    elif len(sub_idx) < k:
        inner_scores, _ = f_classif(X_sub, y)
        inner_scores = np.nan_to_num(inner_scores, nan=0.0)
        ranked = np.argsort(-inner_scores)
        extra = [i for i in ranked if i not in set(sub_idx)][:k - len(sub_idx)]
        sub_idx = np.concatenate([sub_idx, np.array(extra, dtype=int)])
    return {
        "selected_idx": top[sub_idx],
        "scores": None,
        "ea_info": {
            "fitness_curve": ga.fitness_curve_.tolist(),
            "n_features_curve": ga.n_features_curve_.tolist(),
            "best_fitness": float(ga.best_fitness_),
            "prefilter": prefilter, "prefilter_method": "relieff",
            "alpha": alpha, "beta": beta,
        }
    }


def pipeline_fusion_bpso(X, y, k, prefilter=200, alpha=0.3, beta=0.7,
                          n_iter=30, n_particles=20, random_state=42):
    """P3: Multi-filter fusion top-200 → BPSO."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from algorithms.binary_pso import BinaryPSO
    top, scores = _multi_filter_fusion(X, y, n_keep=prefilter)
    X_sub = X[:, top]
    pso = BinaryPSO(n_particles=n_particles, max_iter=n_iter,
                     w=0.9, c1=2.0, c2=2.0,
                     alpha=alpha, beta=beta, k_neighbors=5,
                     random_state=random_state, verbose=False)
    pso.fit(X_sub, y)
    sub_idx = pso.best_features_
    if len(sub_idx) > k:
        inner_scores, _ = f_classif(X_sub[:, sub_idx], y)
        inner_scores = np.nan_to_num(inner_scores, nan=0.0)
        sub_idx = sub_idx[np.argsort(-inner_scores)[:k]]
    elif len(sub_idx) < k:
        inner_scores, _ = f_classif(X_sub, y)
        inner_scores = np.nan_to_num(inner_scores, nan=0.0)
        ranked = np.argsort(-inner_scores)
        extra = [i for i in ranked if i not in set(sub_idx)][:k - len(sub_idx)]
        sub_idx = np.concatenate([sub_idx, np.array(extra, dtype=int)])
    return {
        "selected_idx": top[sub_idx],
        "scores": None,
        "ea_info": {
            "fitness_curve": pso.fitness_curve_.tolist(),
            "n_features_curve": pso.num_features_curve_.tolist(),
            "best_fitness": float(pso.best_fitness_),
            "prefilter": prefilter, "prefilter_method": "multi_fusion",
            "alpha": alpha, "beta": beta,
        }
    }


def pipeline_anova_bpso(X, y, k, prefilter=200, alpha=0.3, beta=0.7,
                         n_iter=30, n_particles=20, random_state=42):
    """P4: ANOVA top-200 → BPSO."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from algorithms.binary_pso import BinaryPSO
    top, scores = _filter_top(X, y, prefilter, "anova")
    X_sub = X[:, top]
    pso = BinaryPSO(n_particles=n_particles, max_iter=n_iter,
                     w=0.9, c1=2.0, c2=2.0,
                     alpha=alpha, beta=beta, k_neighbors=5,
                     random_state=random_state, verbose=False)
    pso.fit(X_sub, y)
    sub_idx = pso.best_features_
    if len(sub_idx) > k:
        inner_scores, _ = f_classif(X_sub[:, sub_idx], y)
        inner_scores = np.nan_to_num(inner_scores, nan=0.0)
        sub_idx = sub_idx[np.argsort(-inner_scores)[:k]]
    elif len(sub_idx) < k:
        inner_scores, _ = f_classif(X_sub, y)
        inner_scores = np.nan_to_num(inner_scores, nan=0.0)
        ranked = np.argsort(-inner_scores)
        extra = [i for i in ranked if i not in set(sub_idx)][:k - len(sub_idx)]
        sub_idx = np.concatenate([sub_idx, np.array(extra, dtype=int)])
    return {
        "selected_idx": top[sub_idx],
        "scores": None,
        "ea_info": {
            "fitness_curve": pso.fitness_curve_.tolist(),
            "n_features_curve": pso.num_features_curve_.tolist(),
            "best_fitness": float(pso.best_fitness_),
            "prefilter": prefilter, "prefilter_method": "anova",
            "alpha": alpha, "beta": beta,
        }
    }


PIPELINES = {
    "P1_anova_ga":     pipeline_anova_ga,
    "P2_relieff_ga":   pipeline_relieff_ga,
    "P3_fusion_bpso":  pipeline_fusion_bpso,
    "P4_anova_bpso":   pipeline_anova_bpso,
}
