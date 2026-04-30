"""
Supplementary reproduction track: W4 SFE, W5 MEL
Both use ANOVA pre-filter to 2000 features.
"""
import numpy as np
from sklearn.feature_selection import f_classif


PREFILTER_THRESH = 2000


def _anova_prefilter(X, y, n=PREFILTER_THRESH):
    if X.shape[1] <= n:
        return X, np.arange(X.shape[1])
    scores, _ = f_classif(X, y)
    scores = np.nan_to_num(scores, nan=0.0)
    top_idx = np.argsort(scores)[::-1][:n]
    return X[:, top_idx], top_idx


def method_sfe(X_train, y_train):
    """W4: SFE (Simple, Fast, Efficient) — supplementary reproduction track."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from published_algorithms.sfe import SFE
    X_sub, pre_idx = _anova_prefilter(X_train, y_train)
    selector = SFE(max_iter=60, random_state=42, verbose=False)
    selector.fit(X_sub, y_train)
    sub_idx = selector.best_features_
    if sub_idx is None or len(sub_idx) == 0:
        sub_idx = np.arange(min(10, X_sub.shape[1]))
    return pre_idx[sub_idx]


def method_mel(X_train, y_train):
    """W5: MEL (Multi-Task Evolutionary Learning) — supplementary reproduction track."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from published_algorithms.mel import MEL
    X_sub, pre_idx = _anova_prefilter(X_train, y_train)
    selector = MEL(
        n_particles=20, max_iter=50, random_state=42, verbose=False
    )
    selector.fit(X_sub, y_train)
    sub_idx = selector.best_features_
    if sub_idx is None or len(sub_idx) == 0:
        sub_idx = np.arange(min(10, X_sub.shape[1]))
    return pre_idx[sub_idx]
