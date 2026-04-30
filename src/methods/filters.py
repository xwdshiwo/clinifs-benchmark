"""
Filter feature selection methods: F1-F5
All return top-K feature indices (in the preprocessed feature space).
K = min(50, n_features) for all filter methods.
For mRMR and ReliefF with n_features > 2000, ANOVA pre-filter to 2000 is applied first.
"""
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif, mutual_info_classif


K_FILTER = 50
PREFILTER_THRESH = 2000


def _topk(scores, n_features, k=K_FILTER):
    """Return indices of top-k scores. Handles ties by lower index preference."""
    k = min(k, n_features)
    return np.argsort(scores)[::-1][:k]


def _anova_prefilter(X, y, n=PREFILTER_THRESH):
    """Return (X_sub, top_idx) after ANOVA pre-filter to n features."""
    if X.shape[1] <= n:
        return X, np.arange(X.shape[1])
    scores, _ = f_classif(X, y)
    scores = np.nan_to_num(scores, nan=0.0)
    top_idx = np.argsort(scores)[::-1][:n]
    return X[:, top_idx], top_idx


def method_variance(X_train, y_train, k=K_FILTER):
    """F1: Top-K features by variance (after VarianceThreshold already removed zeros)."""
    variances = np.var(X_train, axis=0)
    return _topk(variances, X_train.shape[1], k)


def method_anova(X_train, y_train, k=K_FILTER):
    """F2: Top-K features by ANOVA F-score."""
    scores, _ = f_classif(X_train, y_train)
    scores = np.nan_to_num(scores, nan=0.0)
    return _topk(scores, X_train.shape[1], k)


def method_mi(X_train, y_train, k=K_FILTER):
    """F3: Top-K features by mutual information."""
    scores = mutual_info_classif(X_train, y_train, random_state=42)
    return _topk(scores, X_train.shape[1], k)


def method_mrmr(X_train, y_train, k=K_FILTER):
    """F4: mRMR with ANOVA pre-filter to 2000 if needed."""
    from mrmr import mrmr_classif
    X_sub, pre_idx = _anova_prefilter(X_train, y_train)
    k = min(k, X_sub.shape[1])
    feat_names = [f"f{i}" for i in range(X_sub.shape[1])]
    import tqdm as _tqdm, contextlib, io
    df = pd.DataFrame(X_sub, columns=feat_names)
    with contextlib.redirect_stderr(io.StringIO()):
        selected_names = mrmr_classif(X=df, y=pd.Series(y_train), K=k)
    sub_idx = np.array([int(n[1:]) for n in selected_names])
    return pre_idx[sub_idx]


def method_relieff(X_train, y_train, k=K_FILTER):
    """F5: ReliefF/MultiSURF with ANOVA pre-filter to 2000 if needed."""
    from skrebate import ReliefF
    X_sub, pre_idx = _anova_prefilter(X_train, y_train)
    k = min(k, X_sub.shape[1])
    n_neighbors = min(10, len(y_train) - 1)
    rf = ReliefF(n_features_to_select=k, n_neighbors=n_neighbors, n_jobs=-1)
    rf.fit(X_sub, y_train)
    sub_idx = rf.top_features_[:k]
    return pre_idx[sub_idx]
