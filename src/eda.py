"""Training-only exploratory analysis, tables and exportable figures."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import seaborn as sns

from .data_preparation import AGE_LABELS, FEATURES, TARGET, validation_table

BLUE, RED = "#23648A", "#BB4B4B"


def save_figure(fig, folder: Path, name: str) -> None:
    fig.savefig(folder / f"{name}.png", dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(folder / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_eda(raw: pd.DataFrame, bundle: dict, report_dir: Path) -> dict:
    tables, figures = report_dir / "tables", report_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    train, manifest = bundle["train"], bundle["manifest"]
    sns.set_theme(style="whitegrid", font_scale=1.0,
                  rc={"axes.spines.top": False, "axes.spines.right": False,
                      "axes.titleweight": "bold", "grid.alpha": 0.22})

    validation_table(raw).to_csv(tables / "raw_data_quality.csv", index=False)
    train.describe().T.rename_axis("feature").to_csv(tables / "training_summary_statistics.csv")
    train.groupby(TARGET)[["BMI", "MentHlth", "PhysHlth"]].agg(
        ["count", "mean", "median", "std", "min", "max"]
    ).to_csv(tables / "numeric_statistics_by_target.csv")
    class_table = train[TARGET].value_counts().sort_index().rename("count").to_frame()
    class_table["percentage"] = class_table["count"] / len(train) * 100
    class_table.rename_axis(TARGET).to_csv(tables / "training_class_distribution.csv")
    age = train.groupby("Age")[TARGET].agg(count="size", positives="sum", rate="mean").reset_index()
    age.insert(1, "age_band", age["Age"].map(AGE_LABELS))
    age.to_csv(tables / "age_target_rates.csv", index=False)

    rates = []
    for col in FEATURES:
        if col == "BMI":
            continue
        rate = train.groupby(col)[TARGET].agg(count="size", positives="sum", rate="mean").reset_index()
        rate = rate.rename(columns={col: "code"})
        rate.insert(0, "feature", col)
        rates.append(rate)
    pd.concat(rates, ignore_index=True).to_csv(tables / "categorical_target_rates.csv", index=False)
    joint = train.groupby(["Smoker", "HighBP"])[TARGET].agg(
        count="size", positives="sum", rate="mean").reset_index()
    joint.to_csv(tables / "smoking_blood_pressure_joint_rates.csv", index=False)

    correlations = train.corr(method="spearman")
    correlations.to_csv(tables / "training_spearman_correlations.csv")
    target_corr = correlations[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
    target_corr.rename("spearman_r").rename_axis("feature").to_csv(tables / "target_rank_associations.csv")
    q1, q3 = train["BMI"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    flags = (train["BMI"] < lower) | (train["BMI"] > upper)
    bmi_audit = {"q1": float(q1), "q3": float(q3), "iqr": float(iqr),
                 "lower_fence": float(lower), "upper_fence": float(upper),
                 "flagged_rows": int(flags.sum()), "flagged_percent": float(flags.mean() * 100),
                 "minimum": int(train["BMI"].min()), "maximum": int(train["BMI"].max()),
                 "removed_rows": 0,
                 "decision": "Retain valid extremes; IQR flags alone do not establish recording errors."}
    pd.DataFrame([bmi_audit]).to_csv(tables / "bmi_outlier_audit.csv", index=False)

    split_rows = []
    for name, frame in [("train", bundle["train"]), ("test", bundle["test"])]:
        split_rows.append({"partition": name, "rows": len(frame),
                           "negative": int(frame[TARGET].eq(0).sum()),
                           "positive": int(frame[TARGET].sum()),
                           "positive_percent": float(frame[TARGET].mean() * 100)})
    pd.DataFrame(split_rows).to_csv(tables / "split_summary.csv", index=False)
    metadata = bundle["assignments"].query("partition == 'train'").reset_index(drop=True)
    cv_summary = pd.DataFrame({"fold": metadata["cv_fold"], "target": train[TARGET]})
    cv_summary = cv_summary.groupby("fold")["target"].agg(rows="size", positive="sum", positive_rate="mean")
    cv_summary.to_csv(tables / "validation_fold_summary.csv")

    # 1. Class imbalance, training partition only.
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bars = ax.bar(["No reported history (0)", "Reported history (1)"], class_table["count"],
                  color=[BLUE, RED], width=0.58)
    ax.set(title="Reported heart disease or heart attack", ylabel="Training records")
    ax.set_ylim(0, class_table["count"].max() * 1.22)
    for bar, (_, row) in zip(bars, class_table.iterrows()):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+len(train)*0.018,
                f"{int(row['count']):,}\n{row['percentage']:.2f}%", ha="center", fontsize=11)
    fig.text(0.5, -0.01, f"Training partition only | n = {len(train):,} | Unweighted survey records",
             ha="center", fontsize=9, color="#555555")
    save_figure(fig, figures, "01_class_distribution")

    # 2. Age is a band, not a number of years.
    fig, ax = plt.subplots(figsize=(10, 5.0))
    bars = ax.bar(age["age_band"], age["rate"] * 100, color=BLUE, width=0.7)
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.set(title="Reported disease history by age band", xlabel="Age band in years",
           ylabel="Percentage within age band")
    ax.set_ylim(0, max(age["rate"] * 100) * 1.28)
    for bar, row in zip(bars, age.itertuples()):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.45,
                f"{row.rate*100:.1f}%", ha="center", fontsize=9)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.text(0.5, 0.012, "Training partition only | Rates are descriptive associations and are not survey-weighted",
             ha="center", fontsize=9, color="#555555")
    save_figure(fig, figures, "02_age_disease_rate")

    # 3. Full BMI range is visible; no silent trimming of the tail.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1.5, 1]})
    axes[0].hist(train["BMI"], bins=np.arange(11.5, 100.5, 1), color=BLUE, alpha=0.9)
    axes[0].axvline(upper, color=RED, linestyle="--", label=f"Upper IQR fence: {upper:.1f}")
    axes[0].set(title="BMI distribution", xlabel="BMI (kg/m2)", ylabel="Training records")
    axes[0].legend(frameon=False, fontsize=9)
    sns.boxplot(data=train, x=TARGET, y="BMI", hue=TARGET, palette=[BLUE, RED],
                legend=False, ax=axes[1], fliersize=1.5, linewidth=1)
    axes[1].set(title="BMI by reported disease history", xlabel="Reported history (0 or 1)", ylabel="BMI (kg/m2)")
    fig.tight_layout()
    fig.text(0.5, -0.02, "Training partition only | Valid extreme values retained", ha="center", fontsize=9)
    save_figure(fig, figures, "03_bmi_distribution")

    # 4. All input columns plus target; variable codes affect interpretation.
    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(correlations, mask=np.triu(np.ones_like(correlations, dtype=bool), k=1),
                cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                linewidths=0.25, square=True, cbar_kws={"shrink": 0.7, "label": "Spearman correlation"}, ax=ax)
    ax.set_title("Training data rank correlations", pad=16)
    ax.tick_params(axis="both", labelsize=9)
    fig.tight_layout()
    fig.text(0.5, -0.01, "Code-dependent descriptive associations | Correlation does not establish causation", ha="center", fontsize=10)
    save_figure(fig, figures, "04_spearman_heatmap")

    # 5. Separate smoking/BP marginal rates; not a causal effect estimate.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    for ax, feature, title in zip(axes, ["Smoker", "HighBP"], ["Lifetime smoking indicator", "High blood pressure indicator"]):
        values = train.groupby(feature)[TARGET].mean().reindex([0, 1]) * 100
        bars = ax.bar(["No (0)", "Yes (1)"], values, color=[BLUE, RED], width=0.6)
        ax.set(title=title, xlabel="Survey response")
        for bar, value in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2, value+0.4, f"{value:.2f}%", ha="center")
    axes[0].set_ylabel("Reported disease history within group")
    axes[0].yaxis.set_major_formatter(PercentFormatter(100))
    axes[0].set_ylim(0, 23)
    fig.tight_layout()
    fig.text(0.5, -0.02, "Training partition only | Smoking means at least 100 lifetime cigarettes", ha="center", fontsize=9)
    save_figure(fig, figures, "05_smoking_and_blood_pressure")

    # 6. Direction and magnitude of target associations.
    fig, ax = plt.subplots(figsize=(8, 7.2))
    ordered = target_corr.sort_values()
    ax.barh(ordered.index, ordered.values, color=[RED if v > 0 else BLUE for v in ordered])
    ax.axvline(0, color="#444444", linewidth=0.8)
    ax.set(title="Rank associations with reported disease history", xlabel="Spearman correlation with target")
    ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()
    fig.text(0.5, -0.01, "Training partition only | Association is not predictive importance or causation", ha="center", fontsize=9)
    save_figure(fig, figures, "06_target_associations")

    # 7. General health provides a second interpretable ordinal comparison.
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    health = train.groupby("GenHlth")[TARGET].mean().reindex(range(1, 6)) * 100
    bars = ax.bar(["Excellent", "Very good", "Good", "Fair", "Poor"], health, color=BLUE)
    for bar, value in zip(bars, health):
        ax.text(bar.get_x()+bar.get_width()/2, value+0.7, f"{value:.1f}%", ha="center")
    ax.set(title="Reported disease history by general health", ylabel="Percentage within health category")
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.set_ylim(0, health.max() * 1.22)
    fig.text(0.5, -0.01, "Training partition only | Current health may reflect consequences of prior disease", ha="center", fontsize=9)
    save_figure(fig, figures, "07_general_health")

    facts = {
        "raw_rows": len(raw), "raw_columns": len(raw.columns), "input_features": len(FEATURES),
        "raw_negative": int(raw[TARGET].eq(0).sum()), "raw_positive": int(raw[TARGET].sum()),
        "raw_positive_percent": float(raw[TARGET].mean() * 100),
        "missing_cells": int(raw.isna().sum().sum()),
        "invalid_cells": int(validation_table(raw)["invalid_values"].sum()),
        "raw_dtypes": {str(k): int(v) for k, v in raw.dtypes.astype(str).value_counts().items()},
        "training_rows": len(train), "test_rows": manifest["test_rows"],
        "training_positive_percent": float(train[TARGET].mean() * 100),
        "training_majority_baseline_accuracy": float(train[TARGET].eq(0).mean()),
        "training_constant_score_average_precision": float(train[TARGET].mean()),
        "training_bmi": {"mean": float(train["BMI"].mean()), "median": float(train["BMI"].median()),
                         "std": float(train["BMI"].std())},
        "bmi_outliers": bmi_audit,
        "top_target_rank_associations": {k: float(v) for k, v in target_corr.head(5).items()},
        "youngest_age_rate": float(age.iloc[0]["rate"]),
        "oldest_age_rate": float(age.iloc[-1]["rate"]),
        "smoking_rates": {str(k): float(v) for k, v in train.groupby("Smoker")[TARGET].mean().items()},
        "highbp_rates": {str(k): float(v) for k, v in train.groupby("HighBP")[TARGET].mean().items()},
    }
    (report_dir / "dataset_facts.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
    return facts
