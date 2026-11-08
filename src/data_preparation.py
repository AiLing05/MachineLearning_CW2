"""Audit the raw data and create one reproducible, profile-disjoint split.

No imputer, scaler, sampler, feature selector or classifier is fitted here.
Repeated survey profiles are retained, but cannot cross partition/fold boundaries.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

TARGET = "HeartDiseaseorAttack"
FEATURES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "Diabetes", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump",
    "AnyHealthcare", "NoDocbcCost", "GenHlth", "MentHlth", "PhysHlth",
    "DiffWalk", "Sex", "Age", "Education", "Income",
]
CONTINUOUS = ["BMI", "MentHlth", "PhysHlth"]
ORDINAL = ["GenHlth", "Age", "Education", "Income"]
BINARY = [c for c in FEATURES if c not in CONTINUOUS + ORDINAL + ["Diabetes"]]
SEED = 42
AGE_LABELS = {
    1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44",
    6: "45-49", 7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69",
    11: "70-74", 12: "75-79", 13: "80+",
}
ALLOWED = {c: {0, 1} for c in BINARY + [TARGET]}
ALLOWED.update({
    "Diabetes": {0, 1, 2}, "GenHlth": set(range(1, 6)),
    "Age": set(range(1, 14)), "Education": set(range(1, 7)),
    "Income": set(range(1, 9)),
    "MentHlth": set(range(31)), "PhysHlth": set(range(31)),
})


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if list(df.columns) != [TARGET] + FEATURES:
        raise ValueError("CSV schema or feature order differs from the agreed dataset.")
    if df.shape != (253680, 22):
        raise ValueError(f"Unexpected dataset size: {df.shape}")
    if not all(pd.api.types.is_numeric_dtype(df[c]) for c in df):
        raise ValueError("All supplied CSV columns must be numeric survey codes.")
    if df[TARGET].isna().any():
        raise ValueError("Missing target values must never be imputed.")
    issues = validation_table(df)
    if issues["invalid_values"].sum():
        raise ValueError("Unexpected survey codes detected; investigate before splitting.")
    # No missing values are expected in this specific, previously cleaned CSV.
    # Stop rather than silently changing this dataset version.
    if df.isna().any().any():
        raise ValueError("This CSV now contains missing values; review the source/version.")
    if not np.equal(df.to_numpy(), np.floor(df.to_numpy())).all():
        raise ValueError("Review non-integer values before choosing storage types.")
    return df


def validation_table(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for col in df:
        values = df[col].dropna()
        if col == "BMI":
            # Broad source screening, not a diagnosis or an outlier-removal rule.
            invalid = (~np.isfinite(values) | (values <= 0) | (values > 100)).sum()
            rule = "finite BMI greater than 0 and at most 100; review extremes separately"
        else:
            invalid = (~values.isin(ALLOWED[col])).sum()
            rule = ", ".join(map(str, sorted(ALLOWED[col])))
        records.append({
            "feature": col, "raw_dtype": str(df[col].dtype),
            "missing_values": int(df[col].isna().sum()),
            "invalid_values": int(invalid), "unique_values": int(values.nunique()),
            "minimum": float(values.min()), "maximum": float(values.max()),
            "allowed_values_or_check": rule,
        })
    return pd.DataFrame(records)


def duplicate_audit(df: pd.DataFrame) -> dict:
    profiles = df.groupby(FEATURES, sort=True, dropna=False)[TARGET].agg(["size", "nunique"])
    mixed = profiles["nunique"] > 1
    return {
        "duplicate_rows_beyond_first": int(df.duplicated().sum()),
        "rows_in_repeated_full_records": int(df.duplicated(keep=False).sum()),
        "distinct_full_records": int(len(df.drop_duplicates())),
        "distinct_feature_profiles": int(len(profiles)),
        "rows_beyond_first_feature_profile": int(len(df) - len(profiles)),
        "profiles_with_both_target_labels": int(mixed.sum()),
        "rows_in_profiles_with_both_target_labels": int(profiles.loc[mixed, "size"].sum()),
        "policy": "Retain records; keep identical 21-feature profiles together in all splits.",
    }


def prepare_partitions(df: pd.DataFrame, output_dir: Path, raw_path: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = df.astype("int16")  # Lossless conversion after validation.
    # Exact equality across all predictors; target is deliberately excluded.
    groups = df.groupby(FEATURES, sort=True, dropna=False).ngroup().to_numpy()
    X, y = prepared[FEATURES], prepared[TARGET]
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, test_idx = next(outer.split(X, y, groups))
    train = prepared.iloc[train_idx].reset_index(drop=True)
    test = prepared.iloc[test_idx].reset_index(drop=True)
    train_groups, test_groups = groups[train_idx], groups[test_idx]
    assert np.intersect1d(train_groups, test_groups).size == 0
    assert np.intersect1d(train_idx, test_idx).size == 0
    assert len(train_idx) + len(test_idx) == len(df)

    # Every model must use these SAME validation folds in the training partition.
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + 1)
    fold_ids = np.full(len(train), -1, dtype=np.int8)
    for fold, (fit_idx, valid_idx) in enumerate(inner.split(train[FEATURES], train[TARGET], train_groups)):
        assert np.intersect1d(train_groups[fit_idx], train_groups[valid_idx]).size == 0
        fold_ids[valid_idx] = fold
    assert (fold_ids >= 0).all()

    assignments = pd.DataFrame({
        "source_row_id": np.arange(len(df)), "partition": "test",
        "cv_fold": np.full(len(df), -1, dtype=np.int8), "profile_group": groups,
    })
    assignments.loc[train_idx, "partition"] = "train"
    assignments.loc[train_idx, "cv_fold"] = fold_ids
    assert assignments.groupby("profile_group")["partition"].nunique().max() == 1
    assert assignments.query("partition == 'train'").groupby("profile_group")["cv_fold"].nunique().max() == 1

    prepared.to_csv(output_dir / "heart_disease_prepared.csv", index=False)
    train.to_csv(output_dir / "train.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)
    assignments.to_csv(output_dir / "row_assignments.csv", index=False)
    manifest = {
        "target": TARGET, "positive_label": 1, "feature_columns": FEATURES,
        "random_state_outer": SEED, "random_state_cv": SEED + 1,
        "split_method": "First held-out fold of 5-fold StratifiedGroupKFold",
        "group_definition": "Exact equality of all 21 predictors, excluding target",
        "intended_test_fraction": 0.20, "actual_test_fraction": len(test) / len(df),
        "train_rows": len(train), "test_rows": len(test), "total_rows": len(df),
        "train_positive": int(train[TARGET].sum()), "test_positive": int(test[TARGET].sum()),
        "train_positive_rate": float(train[TARGET].mean()),
        "test_positive_rate": float(test[TARGET].mean()),
        "train_test_profile_overlap": 0, "cv_profile_overlap": 0,
        "cv_folds": 5, "test_cv_fold_sentinel": -1,
        "duplicates_removed": 0, "row_order": "Original source order within each partition",
        "stored_values": "Unscaled integer survey codes; no fitted preprocessing",
        "primary_selection_metric": "average_precision",
        "required_secondary_metrics": ["accuracy", "precision", "recall", "f1", "roc_auc"],
        "default_classification_threshold": 0.5,
        "test_use": "Final evaluation only after models, ensemble and thresholds are frozen",
        "pandas_version": pd.__version__, "sklearn_version": sklearn.__version__,
        "raw_sha256": sha256(raw_path),
        "files_sha256": {p.name: sha256(p) for p in sorted(output_dir.glob("*.csv"))},
    }
    (output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"prepared": prepared, "train": train, "test": test,
            "assignments": assignments, "manifest": manifest}


def load_training_bundle(root: Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load training data and aligned metadata WITHOUT opening the test file."""
    directory = root / "data" / "processed"
    manifest = json.loads((directory / "split_manifest.json").read_text(encoding="utf-8"))
    for name in ["train.csv", "row_assignments.csv"]:
        if sha256(directory / name) != manifest["files_sha256"][name]:
            raise ValueError(f"{name} changed after the shared split was created.")
    train = pd.read_csv(directory / "train.csv")
    metadata = pd.read_csv(directory / "row_assignments.csv")
    metadata = metadata.loc[metadata["partition"].eq("train")].reset_index(drop=True)
    assert len(train) == len(metadata) == manifest["train_rows"]
    return train[FEATURES], train[TARGET], metadata
