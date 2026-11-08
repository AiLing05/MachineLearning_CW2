"""Download the group's specified CSV, preserving its commit and SHA-256."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
REPO = "doguilmak/Heart-Diseaseor-Attack-Classification"
FILENAME = "heart_disease_health_indicators_BRFSS2015.csv"
COMMIT = "ceb708fa31ead97d75da096ba85526f284d00864"
EXPECTED_SHA256 = "953fb23bb131736d7879539a071d3f6d1eaae8d4867ab05335d0ba89b94d5f99"


def download() -> Path:
    raw = ROOT / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    destination = raw / FILENAME
    metadata_path = raw / "provenance.json"
    if destination.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if digest != metadata["sha256"] or digest != EXPECTED_SHA256:
            raise ValueError("Raw dataset changed. Restore the recorded CSV before continuing.")
        print(f"Verified existing raw dataset: {destination.name}", flush=True)
        return destination

    session = requests.Session()
    session.headers["User-Agent"] = "6001CMD-member1-reproducible-analysis"
    commit = COMMIT
    url = f"https://raw.githubusercontent.com/{REPO}/{commit}/{FILENAME}"
    print("Downloading the CSV from the group's supplied public repository...", flush=True)
    response = session.get(url, timeout=120)
    response.raise_for_status()
    content = response.content
    if not content.startswith(b"HeartDiseaseorAttack,"):
        raise ValueError("Downloaded content does not have the expected CSV header.")
    if hashlib.sha256(content).hexdigest() != EXPECTED_SHA256:
        raise ValueError("The downloaded dataset does not match the frozen SHA-256.")
    destination.write_bytes(content)
    metadata = {
        "dataset": "Heart Disease Health Indicators BRFSS 2015",
        "group_supplied_url": f"https://github.com/{REPO}/blob/main/{FILENAME}",
        "download_url": url,
        "commit": commit,
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "original_survey": "CDC BRFSS 2015",
        "upstream_dataset": "https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset",
        "note": "This is a previously cleaned derivative, not the original CDC microdata.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({k: metadata[k] for k in ["commit", "sha256", "bytes"]}), flush=True)
    return destination


if __name__ == "__main__":
    download()
