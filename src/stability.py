"""
Stability metrics for feature selection benchmarks.

Implements:
- Nogueira et al. (2018) generalized stability (handles variable k)
- Kuncheva Consistency Index (fixed k)
- Jaccard pairwise (legacy)
"""
import numpy as np
from itertools import combinations


def nogueira_stability(subsets, n_total_features):
    """
    Nogueira et al. (2018) stability index.
    Handles variable subset sizes across folds.
    Range: (-inf, 1]. 0 = random, 1 = perfectly stable.

    Parameters
    ----------
    subsets : list of list/array of int
        Feature indices selected in each fold. Length = n_folds.
    n_total_features : int
        Total number of features in the feature space.

    Returns
    -------
    float : stability score
    """
    r = len(subsets)
    n = n_total_features
    if r < 2:
        return np.nan

    # Build binary matrix Z (r x n)
    Z = np.zeros((r, n), dtype=np.float32)
    for i, s in enumerate(subsets):
        for j in s:
            if 0 <= j < n:
                Z[i, j] = 1.0

    k_bar = Z.sum(axis=1).mean()   # mean subset size
    if k_bar == 0 or k_bar == n:
        return np.nan

    p_hat = Z.mean(axis=0)          # (n,) fraction of folds selecting each feature

    # Sample variance of p_hat
    var_sum = np.sum(p_hat * (1.0 - p_hat))

    numerator   = (r / (r - 1)) * (var_sum / n)
    denominator = (k_bar / n) * (1.0 - k_bar / n)

    if denominator == 0:
        return np.nan

    return float(1.0 - numerator / denominator)


def kuncheva_ci(s_i, s_j, n_total_features):
    """
    Kuncheva Consistency Index for two equally-sized subsets.
    Range: [-1, 1]. 0 = random, 1 = identical.
    """
    s_i, s_j = set(s_i), set(s_j)
    k = len(s_i)
    if k != len(s_j) or k == 0 or k == n_total_features:
        return np.nan
    n = n_total_features
    intersection = len(s_i & s_j)
    numerator   = intersection * n - k * k
    denominator = k * (n - k)
    return float(numerator / denominator)


def kuncheva_avg(subsets, n_total_features):
    """
    Average Kuncheva CI over all pairs.
    Only valid when all subsets have the same size.
    Falls back to NaN if sizes differ.
    """
    sizes = [len(s) for s in subsets]
    if len(set(sizes)) > 1:
        return np.nan
    r = len(subsets)
    if r < 2:
        return np.nan
    scores = [kuncheva_ci(subsets[i], subsets[j], n_total_features)
              for i, j in combinations(range(r), 2)]
    scores = [s for s in scores if not np.isnan(s)]
    return float(np.mean(scores)) if scores else np.nan


def jaccard_avg(subsets):
    """Average pairwise Jaccard similarity."""
    r = len(subsets)
    if r < 2:
        return np.nan
    sets = [set(s) for s in subsets]
    scores = []
    for i, j in combinations(range(r), 2):
        union = len(sets[i] | sets[j])
        if union == 0:
            continue
        scores.append(len(sets[i] & sets[j]) / union)
    return float(np.mean(scores)) if scores else 0.0


def compute_all_stability(subsets, n_total_features):
    """
    Compute all stability metrics for a list of feature subsets.
    Returns dict with nogueira, kuncheva, jaccard.
    """
    return {
        "nogueira": nogueira_stability(subsets, n_total_features),
        "kuncheva": kuncheva_avg(subsets, n_total_features),
        "jaccard":  jaccard_avg(subsets),
    }
