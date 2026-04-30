import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

OUTER_CV = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
INNER_CV = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)


def load_dataset(path):
    """Auto-detect CuMiDa / aligned-external / MGRFE-canonical formats."""
    df = pd.read_csv(path)
    if "type" in df.columns:
        # CuMiDa format: samples, type, feature1, ...
        # tumor label varies (tumoral / HCC / AML / adenocarcinoma / ...)
        # normal label always contains 'normal'
        y = (~df["type"].str.contains("normal", case=False)).astype(int).values
        drop_cols = [c for c in ["samples", "type"] if c in df.columns]
        X = df.drop(columns=drop_cols).values
        feature_names = df.drop(columns=drop_cols).columns.tolist()
    elif "label" in df.columns:
        y = df["label"].values.astype(int)
        X = df.drop(columns=["label"]).values
        feature_names = df.drop(columns=["label"]).columns.tolist()
    else:
        raise ValueError(f"Cannot detect format (no 'type' or 'label' column): {path}")
    return X, y, feature_names


def preprocess(X_train, X_test):
    """Remove zero-variance features (fit on train) then standardize."""
    vt = VarianceThreshold(threshold=0.0)
    X_train = vt.fit_transform(X_train)
    X_test = vt.transform(X_test)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, vt.get_support(indices=True)
