"""Rebuild the data split, training-only EDA, tables and pipeline checks."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from download_dataset import download
from src.data_preparation import TARGET, duplicate_audit, load_raw, load_training_bundle, prepare_partitions
from src.eda import run_eda
from src.model_setup import verify_preprocessor_on_training


def main() -> None:
    raw_path = download()
    print("Validating raw data and investigating repeated profiles...", flush=True)
    raw = load_raw(raw_path)
    duplicates = duplicate_audit(raw)
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "duplicate_audit.json").write_text(json.dumps(duplicates, indent=2), encoding="utf-8")
    print(json.dumps(duplicates, indent=2), flush=True)
    print("Creating the held-out partition and five common training folds...", flush=True)
    bundle = prepare_partitions(raw, ROOT / "data" / "processed", raw_path)
    print("Generating training-only EDA figures and tables...", flush=True)
    facts = run_eda(raw, bundle, reports)
    X, y, metadata = load_training_bundle(ROOT)
    checks = verify_preprocessor_on_training(X, metadata)
    (reports / "pipeline_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(raw), "positive_percent": facts["raw_positive_percent"],
                      "train_rows": len(X), "test_rows": len(bundle["test"]),
                      "preprocessor_checks": checks}, indent=2), flush=True)


if __name__ == "__main__":
    main()
