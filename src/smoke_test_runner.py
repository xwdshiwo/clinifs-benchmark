import sys
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from protocol import load_dataset, preprocess

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "main_benchmark", "Lung_GSE19804.csv")

def variance_threshold_method(X, y):
    # Already variance-filtered; just return all feature indices
    return np.arange(X.shape[1])

def main():
    X, y = load_dataset(DATA_PATH)
    total_features = X.shape[1]
    print(f"Dataset loaded: {X.shape[0]} samples, {total_features} features")

    outer_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train_outer, X_test_outer = X[train_idx], X[test_idx]
        y_train_outer, y_test_outer = y[train_idx], y[test_idx]

        # Inner CV feature selection
        inner_feature_counts = {}
        for _, inner_train_idx in inner_cv.split(X_train_outer, y_train_outer):
            X_inner = X_train_outer[inner_train_idx]
            y_inner = y_train_outer[inner_train_idx]
            X_inner_proc, _, kept_idx = preprocess(X_inner, X_inner)
            selected = variance_threshold_method(X_inner_proc, y_inner)
            for i in kept_idx[selected]:
                inner_feature_counts[int(i)] = inner_feature_counts.get(int(i), 0) + 1

        n_inner = inner_cv.n_splits
        selected_features = sorted(
            [k for k, v in inner_feature_counts.items() if v >= n_inner // 2 + 1]
        )
        if not selected_features:
            selected_features = sorted(inner_feature_counts, key=inner_feature_counts.get, reverse=True)[:10]

        # Outer fold evaluation
        X_train_proc, X_test_proc, kept_idx = preprocess(X_train_outer, X_test_outer)
        kept_set = {int(i): pos for pos, i in enumerate(kept_idx)}
        sel_in_proc = [kept_set[f] for f in selected_features if f in kept_set]
        if not sel_in_proc:
            sel_in_proc = list(range(X_train_proc.shape[1]))

        clf = LogisticRegression(C=1, max_iter=1000, random_state=42)
        clf.fit(X_train_proc[:, sel_in_proc], y_train_outer)
        y_prob = clf.predict_proba(X_test_proc[:, sel_in_proc])[:, 1]
        auc = roc_auc_score(y_test_outer, y_prob)

        print(f"Fold {fold_idx}: AUC={auc:.4f}, n_features={len(selected_features)}")

if __name__ == "__main__":
    main()
