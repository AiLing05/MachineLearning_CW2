"""Common final evaluation for all four models (Member 4, Part G).

Every model is scored with the same code on the same hold-out test set, so the
comparison is fair. The test set is only opened here, after each model's
hyperparameters have been frozen using the shared cross-validation folds.

Each model is rebuilt from the best hyperparameters its member saved
(results/member2/*_best_parameters.csv, results/member3/random_forest_best_params.json)
and refitted on the full training set, so this notebook does not depend on
locally saved model files.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, average_precision_score,
    confusion_matrix, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.tree import DecisionTreeClassifier

from .data_preparation import FEATURES, TARGET, sha256
from .model_setup import make_model_pipeline

THRESHOLD = 0.5  # Agreed default in split_manifest.json; never tuned on the test set.
METRICS = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "Average_Precision"]


# ---------------------------------------------------------------- data loading
def load_test_bundle(root: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load the hold-out test set after checking it matches the shared split."""
    directory = root / "data" / "processed"
    manifest = json.loads((directory / "split_manifest.json").read_text(encoding="utf-8"))
    if sha256(directory / "test.csv") != manifest["files_sha256"]["test.csv"]:
        raise ValueError("test.csv changed after the shared split was created.")
    test = pd.read_csv(directory / "test.csv")
    assert len(test) == manifest["test_rows"]
    return test[FEATURES], test[TARGET]


# ------------------------------------------------------- rebuilding the models
def read_best_parameters(path: Path) -> dict:
    """Read a best-parameter CSV and restore Python types (int, None, str)."""
    table = pd.read_csv(path, keep_default_na=False)
    params = {}
    for name, value in zip(table["Hyperparameter"], table["Best Value"]):
        value = str(value).strip()
        if value in {"", "None", "nan"}:
            params[name] = None
            continue
        try:
            number = float(value)
            params[name] = int(number) if number.is_integer() else number
        except ValueError:
            params[name] = value
    return params


def optimised_logistic_regression(root: Path):
    params = read_best_parameters(
        root / "results" / "member2" / "logistic_regression_best_parameters.csv")
    model = make_model_pipeline(LogisticRegression(max_iter=2000, random_state=42),
                                kind="linear")
    return model.set_params(**params)


def optimised_decision_tree(root: Path):
    params = read_best_parameters(
        root / "results" / "member2" / "decision_tree_best_parameters.csv")
    model = make_model_pipeline(DecisionTreeClassifier(random_state=42), kind="tree")
    return model.set_params(**params)


def optimised_random_forest(root: Path):
    params = json.loads((root / "results" / "member3" / "random_forest_best_params.json")
                        .read_text(encoding="utf-8"))
    model = make_model_pipeline(RandomForestClassifier(random_state=42, n_jobs=-1),
                                kind="tree")
    return model.set_params(**params)


def soft_voting_ensemble(root: Path):
    """Member 3's ensemble: equal-weight soft voting of Logistic Regression + Random Forest."""
    return VotingClassifier(
        estimators=[("logistic_regression", optimised_logistic_regression(root)),
                    ("random_forest", optimised_random_forest(root))],
        voting="soft", weights=[1, 1], n_jobs=1)


# ------------------------------------------------------------------ evaluation
@dataclass
class ModelResult:
    name: str
    model: object
    fit_seconds: float
    predict_seconds: float
    train_scores: dict
    test_scores: dict
    test_probability: np.ndarray = field(repr=False)
    test_prediction: np.ndarray = field(repr=False)


def score(y_true, probability, threshold: float = THRESHOLD) -> dict:
    prediction = (probability >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, prediction),
        "Precision": precision_score(y_true, prediction, zero_division=0),
        "Recall": recall_score(y_true, prediction),
        "F1": f1_score(y_true, prediction),
        "ROC_AUC": roc_auc_score(y_true, probability),
        "Average_Precision": average_precision_score(y_true, probability),
    }


def evaluate_model(name, model, X_train, y_train, X_test, y_test,
                   already_fitted: bool = False) -> ModelResult:
    """Fit on the full training set (unless supplied fitted) and score on the test set."""
    fit_seconds = np.nan
    if not already_fitted:
        model = clone(model)
        start = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start

    start = time.perf_counter()
    test_probability = model.predict_proba(X_test)[:, 1]
    predict_seconds = time.perf_counter() - start

    train_probability = model.predict_proba(X_train)[:, 1]
    return ModelResult(
        name=name, model=model,
        fit_seconds=fit_seconds, predict_seconds=predict_seconds,
        train_scores=score(y_train, train_probability),
        test_scores=score(y_test, test_probability),
        test_probability=test_probability,
        test_prediction=(test_probability >= THRESHOLD).astype(int),
    )


# ------------------------------------------------------------ summary tables
def comparison_table(results: list[ModelResult]) -> pd.DataFrame:
    """One row per model: test metrics plus training/prediction time."""
    rows = []
    for r in results:
        row = {"Model": r.name, **r.test_scores,
               "Fit Time (s)": r.fit_seconds, "Predict Time (s)": r.predict_seconds}
        rows.append(row)
    return pd.DataFrame(rows).set_index("Model")


def generalisation_table(results: list[ModelResult], cv_paths: dict) -> pd.DataFrame:
    """Training vs cross-validation vs test scores, to check over/underfitting.

    cv_paths maps a model name to the member's cross-validation results CSV.
    """
    rows = []
    for r in results:
        cv = pd.Series(dtype=float)
        if r.name in cv_paths and Path(cv_paths[r.name]).exists():
            cv = read_cv_means(cv_paths[r.name])
        for metric in ["Accuracy", "F1", "ROC_AUC", "Average_Precision"]:
            rows.append({
                "Model": r.name, "Metric": metric,
                "Training": r.train_scores[metric],
                "CV Validation": cv.get(metric, np.nan),
                "Test": r.test_scores[metric],
                "Train - Test Gap": r.train_scores[metric] - r.test_scores[metric],
            })
    return pd.DataFrame(rows)


def read_cv_means(path) -> pd.Series:
    """Mean validation score per metric; members used slightly different column names."""
    table = pd.read_csv(path).set_index("Metric")
    column = "Mean" if "Mean" in table else "Mean Validation Score"
    return table[column]


def bootstrap_difference(y_test, probability_a, probability_b, metric,
                         n_boot: int = 1000, seed: int = 42) -> dict:
    """Paired bootstrap of metric(A) - metric(B) on the test set (95% interval)."""
    y = np.asarray(y_test)
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        differences.append(metric(y[idx], probability_a[idx])
                           - metric(y[idx], probability_b[idx]))
    differences = np.array(differences)
    return {"Difference": metric(y, probability_a) - metric(y, probability_b),
            "CI Lower": np.percentile(differences, 2.5),
            "CI Upper": np.percentile(differences, 97.5),
            "Share of Resamples > 0": (differences > 0).mean()}


def confusion_table(results: list[ModelResult], y_test) -> pd.DataFrame:
    rows = []
    for r in results:
        tn, fp, fn, tp = confusion_matrix(y_test, r.test_prediction).ravel()
        rows.append({"Model": r.name, "True Negative": tn, "False Positive": fp,
                     "False Negative": fn, "True Positive": tp,
                     "Specificity": tn / (tn + fp)})
    return pd.DataFrame(rows).set_index("Model")


# -------------------------------------------------------------------- figures
def plot_confusion_matrices(results, y_test, save_to: Path | None = None):
    fig, axes = plt.subplots(1, len(results), figsize=(4.5 * len(results), 4))
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, results):
        ConfusionMatrixDisplay.from_predictions(
            y_test, r.test_prediction, display_labels=["No HD", "HD"],
            cmap="Blues", values_format="d", colorbar=False, ax=ax)
        ax.set_title(r.name)
    fig.tight_layout()
    _save(fig, save_to)
    return fig


def plot_roc_curves(results, y_test, save_to: Path | None = None):
    fig, ax = plt.subplots(figsize=(6, 5))
    for r in results:
        fpr, tpr, _ = roc_curve(y_test, r.test_probability)
        ax.plot(fpr, tpr, label=f"{r.name} (AUC = {r.test_scores['ROC_AUC']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random guess")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate (Recall)",
           title="ROC Curves on the Test Set")
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, save_to)
    return fig


def plot_precision_recall_curves(results, y_test, save_to: Path | None = None):
    fig, ax = plt.subplots(figsize=(6, 5))
    for r in results:
        precision, recall, _ = precision_recall_curve(y_test, r.test_probability)
        ax.plot(recall, precision,
                label=f"{r.name} (AP = {r.test_scores['Average_Precision']:.3f})")
    ax.axhline(np.mean(y_test), color="k", linestyle="--", linewidth=1,
               label=f"Positive rate ({np.mean(y_test):.3f})")
    ax.set(xlabel="Recall", ylabel="Precision",
           title="Precision-Recall Curves on the Test Set")
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, save_to)
    return fig


def plot_metric_bars(table: pd.DataFrame, save_to: Path | None = None):
    ax = table[METRICS].plot(kind="bar", figsize=(10, 5), rot=0, width=0.8)
    ax.set(ylabel="Score", ylim=(0, 1), title="Test-Set Metrics by Model")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig = ax.get_figure()
    fig.tight_layout()
    _save(fig, save_to)
    return fig


def _save(fig, path: Path | None):
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
