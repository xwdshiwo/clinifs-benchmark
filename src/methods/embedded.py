"""
Embedded feature selection methods: E1-E5
E1, E2: L1/ElasticNet logistic regression — run on full feature set (sparse)
E3, E4, E5: Boruta / eBoruta / ExtraTrees — ANOVA pre-filter to 2000 if needed
"""
import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier


PREFILTER_THRESH = 2000


def _anova_prefilter(X, y, n=PREFILTER_THRESH):
    from sklearn.feature_selection import f_classif
    if X.shape[1] <= n:
        return X, np.arange(X.shape[1])
    scores, _ = f_classif(X, y)
    scores = np.nan_to_num(scores, nan=0.0)
    top_idx = np.argsort(scores)[::-1][:n]
    return X[:, top_idx], top_idx


def method_l1_logistic(X_train, y_train):
    """E1: L1 Logistic Regression sparse feature selection."""
    clf = LogisticRegression(
        penalty="l1", C=0.1, solver="liblinear", max_iter=1000, random_state=42
    )
    clf.fit(X_train, y_train)
    coef = np.abs(clf.coef_).ravel()
    idx = np.where(coef > 0)[0]
    if len(idx) == 0:
        idx = np.argsort(coef)[::-1][:10]
    return idx


def method_elasticnet(X_train, y_train):
    """E2: Elastic-Net logistic regression (saga solver, C=0.1 strong regularization)."""
    clf = LogisticRegression(
        penalty="elasticnet", solver="saga", l1_ratio=0.5,
        C=0.1, max_iter=2000, random_state=42, n_jobs=-1
    )
    clf.fit(X_train, y_train)
    coef = np.abs(clf.coef_).ravel()
    idx = np.where(coef > 0)[0]
    if len(idx) == 0:
        idx = np.argsort(coef)[::-1][:10]
    return idx


def method_boruta(X_train, y_train):
    """E3: BorutaPy all-relevant feature selection."""
    from boruta import BorutaPy
    X_sub, pre_idx = _anova_prefilter(X_train, y_train)
    rf = RandomForestClassifier(
        n_estimators=50, max_depth=5, n_jobs=-1, random_state=42
    )
    selector = BorutaPy(rf, n_estimators="auto", max_iter=50, verbose=0, random_state=42)
    selector.fit(X_sub, y_train)
    sub_idx = np.where(selector.support_)[0]
    if len(sub_idx) == 0:
        sub_idx = np.where(selector.support_weak_)[0]
    if len(sub_idx) == 0:
        sub_idx = np.argsort(selector.ranking_)[:10]
    return pre_idx[sub_idx]


def method_linearsvc_l1(X_train, y_train):
    """E4: LinearSVC with L1 penalty — sparse embedded selection (replaces eBoruta)."""
    X_sub, pre_idx = _anova_prefilter(X_train, y_train)
    clf = LinearSVC(
        penalty="l1", dual=False, C=0.1, max_iter=2000, random_state=42
    )
    selector = SelectFromModel(clf, prefit=False)
    selector.fit(X_sub, y_train)
    sub_idx = np.where(selector.get_support())[0]
    if len(sub_idx) == 0:
        clf.fit(X_sub, y_train)
        sub_idx = np.argsort(np.abs(clf.coef_).ravel())[::-1][:10]
    return pre_idx[sub_idx]


def method_extratrees(X_train, y_train, threshold="mean"):
    """E5: SelectFromModel with ExtraTreesClassifier."""
    X_sub, pre_idx = _anova_prefilter(X_train, y_train)
    et = ExtraTreesClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    selector = SelectFromModel(et, threshold=threshold)
    selector.fit(X_sub, y_train)
    sub_idx = np.where(selector.get_support())[0]
    if len(sub_idx) == 0:
        et.fit(X_sub, y_train)
        sub_idx = np.argsort(et.feature_importances_)[::-1][:10]
    return pre_idx[sub_idx]
