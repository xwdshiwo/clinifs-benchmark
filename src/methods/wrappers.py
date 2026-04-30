"""
Wrapper feature selection methods: W1-W3 (main line)
All use ANOVA pre-filter to 2000 features if n_features > 2000.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import f_classif


PREFILTER_THRESH = 2000


def _anova_prefilter(X, y, n=PREFILTER_THRESH):
    if X.shape[1] <= n:
        return X, np.arange(X.shape[1])
    scores, _ = f_classif(X, y)
    scores = np.nan_to_num(scores, nan=0.0)
    top_idx = np.argsort(scores)[::-1][:n]
    return X[:, top_idx], top_idx


def method_rfecv(X_train, y_train):
    """W1: RFECV with LogisticRegression base estimator."""
    from sklearn.feature_selection import RFECV
    X_sub, pre_idx = _anova_prefilter(X_train, y_train)
    estimator = LogisticRegression(
        C=1, solver="liblinear", max_iter=500, random_state=42
    )
    selector = RFECV(
        estimator=estimator, step=0.1, cv=3,
        scoring="roc_auc", n_jobs=-1, min_features_to_select=5
    )
    selector.fit(X_sub, y_train)
    sub_idx = np.where(selector.support_)[0]
    if len(sub_idx) == 0:
        sub_idx = np.argsort(selector.ranking_)[:10]
    return pre_idx[sub_idx]


def method_ga(X_train, y_train):
    """W2: GAFeatureSelectionCV — genetic algorithm wrapper."""
    from sklearn_genetic import GAFeatureSelectionCV
    X_sub, pre_idx = _anova_prefilter(X_train, y_train)
    estimator = LogisticRegression(
        C=1, solver="liblinear", max_iter=300, random_state=42
    )
    evolved = GAFeatureSelectionCV(
        estimator=estimator,
        cv=3,
        scoring="roc_auc",
        population_size=20,
        generations=15,
        tournament_size=3,
        n_jobs=-1,
        verbose=False,
        keep_top_k=1,
        crossover_probability=0.8,
        mutation_probability=0.1,
    )
    evolved.fit(X_sub, y_train)
    sub_idx = np.where(evolved.support_)[0]
    if len(sub_idx) == 0:
        sub_idx = np.arange(min(10, X_sub.shape[1]))
    return pre_idx[sub_idx]


def method_bpso(X_train, y_train):
    """W3: Binary PSO feature selection."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from algorithms.binary_pso import BinaryPSO
    X_sub, pre_idx = _anova_prefilter(X_train, y_train)
    selector = BinaryPSO(
        n_particles=20, max_iter=50, w=0.9, c1=2.0, c2=2.0,
        alpha=0.9, beta=0.1, k_neighbors=5, random_state=42, verbose=False
    )
    selector.fit(X_sub, y_train)
    sub_idx = selector.best_features_
    if len(sub_idx) == 0:
        sub_idx = np.arange(min(10, X_sub.shape[1]))
    return pre_idx[sub_idx]
