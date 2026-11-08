"""Unfitted, fold-safe preprocessing for the other group members.

Example:
    X, y, metadata = load_training_bundle(ROOT)
    model = make_model_pipeline(LogisticRegression(max_iter=2000), kind="linear")
    search = GridSearchCV(model, grid, cv=shared_cv(metadata),
                          scoring="average_precision", n_jobs=-1)
    search.fit(X, y)  # Every preprocessing step is fitted within each fold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_preparation import BINARY, CONTINUOUS, ORDINAL


def shared_cv(metadata: pd.DataFrame) -> PredefinedSplit:
    if set(metadata["cv_fold"].unique()) != set(range(5)):
        raise ValueError("Expected all five shared training folds, numbered 0 to 4.")
    return PredefinedSplit(metadata["cv_fold"].to_numpy(dtype=int))


def make_preprocessor(kind: str = "linear") -> ColumnTransformer:
    if kind not in {"linear", "tree"}:
        raise ValueError("kind must be 'linear' or 'tree'")
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if kind == "linear":
        numeric_steps.append(("scaler", StandardScaler()))
    transformers = [
        ("numeric", Pipeline(numeric_steps), CONTINUOUS),
        ("binary", SimpleImputer(strategy="most_frequent"), BINARY),
    ]
    if kind == "linear":
        # One-hot coding avoids assuming equal effects between age/income/education bands.
        category_columns = ["Diabetes"] + ORDINAL
        categories = [[0, 1, 2], list(range(1, 6)), list(range(1, 14)),
                      list(range(1, 7)), list(range(1, 9))]
        transformers.append(("survey_categories", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(categories=categories, drop="first",
                                     handle_unknown="error", sparse_output=False)),
        ]), category_columns))
    else:
        transformers.extend([
            ("ordinal", SimpleImputer(strategy="most_frequent"), ORDINAL),
            ("diabetes", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(categories=[[0, 1, 2]],
                                         handle_unknown="error", sparse_output=False)),
            ]), ["Diabetes"]),
        ])
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)


def make_model_pipeline(estimator, kind: str = "linear") -> Pipeline:
    return Pipeline([("preprocess", make_preprocessor(kind)), ("model", estimator)])


def verify_preprocessor_on_training(X: pd.DataFrame, metadata: pd.DataFrame) -> dict:
    """Check shapes/finite values using one training CV fold, never the hold-out."""
    fit_idx, valid_idx = next(shared_cv(metadata).split())
    checks = {}
    for kind in ["linear", "tree"]:
        transformer = make_preprocessor(kind)
        fitted = transformer.fit_transform(X.iloc[fit_idx])
        validation = transformer.transform(X.iloc[valid_idx])
        if not (np.isfinite(fitted).all() and np.isfinite(validation).all()):
            raise AssertionError("Preprocessing created non-finite values.")
        assert len(transformer.get_feature_names_out()) == fitted.shape[1]
        checks[kind] = {"fitted_on_rows": len(fit_idx), "validated_on_rows": len(valid_idx),
                        "output_columns": fitted.shape[1], "finite": True}
    return checks
