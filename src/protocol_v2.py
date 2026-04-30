"""
Enhanced protocol for v2 experiments. Preserves sample IDs for downstream tracking.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

OUTER_CV = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
INNER_CV = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)


def load_dataset_v2(path):
    """
    Load dataset and return (X, y, feature_names, sample_ids).
    Auto-detects CuMiDa / aligned-external / MGRFE-canonical formats.
    """
    df = pd.read_csv(path)

    # Extract sample IDs
    if "samples" in df.columns:
        sample_ids = df["samples"].astype(str).tolist()
    elif df.columns[0].lower() in ("unnamed: 0", "sample", "id"):
        sample_ids = df.iloc[:, 0].astype(str).tolist()
    else:
        sample_ids = [f"s{i}" for i in range(len(df))]

    # Format A: CuMiDa (samples, type, feature1, ...)
    if "type" in df.columns:
        y = (~df["type"].str.contains("normal", case=False)).astype(int).values
        drop_cols = [c for c in ["samples", "type", "Unnamed: 0"] if c in df.columns]
        feat_df = df.drop(columns=drop_cols)
    elif "label" in df.columns:
        y = df["label"].values.astype(int)
        drop_cols = [c for c in ["label", "samples", "Unnamed: 0"] if c in df.columns]
        feat_df = df.drop(columns=drop_cols)
    else:
        raise ValueError(f"Cannot detect format (no 'type' or 'label' column): {path}")

    X = feat_df.values.astype(np.float32)
    feature_names = feat_df.columns.tolist()
    return X, y, feature_names, sample_ids


def preprocess(X_train, X_test):
    """Remove zero-variance features (fit on train) then standardize."""
    vt = VarianceThreshold(threshold=0.0)
    X_train = vt.fit_transform(X_train)
    X_test = vt.transform(X_test)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, vt.get_support(indices=True)
