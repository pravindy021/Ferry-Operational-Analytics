import hashlib
import json
from pathlib import Path

import joblib

from src.data.loader import load_ticket_data, regularize_intervals
from src.models.forecasting import train_forecaster


def dataset_digest(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def train_or_load(
    dataset_path: str | Path,
    artifact_path: str | Path,
    rolling_folds: int = 3,
    force: bool = False,
) -> tuple[dict, bool]:
    dataset_path = Path(dataset_path)
    artifact_path = Path(artifact_path)
    digest = dataset_digest(dataset_path)
    if artifact_path.is_file() and not force:
        artifact = joblib.load(artifact_path)
        if (
            artifact.get("schema_version") == 1
            and artifact.get("source_digest") == digest
            and artifact.get("rolling_folds") == rolling_folds
        ):
            return artifact, False

    data, quality = load_ticket_data(dataset_path)
    grid = regularize_intervals(data)
    artifact = train_forecaster(grid, rolling_folds=rolling_folds)
    artifact["source_digest"] = digest
    artifact["source_quality"] = quality
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_path, compress=3)

    report_path = artifact_path.with_name("model_evaluation.csv")
    artifact["evaluation"].to_csv(report_path, index=False)
    metadata_path = artifact_path.with_name("training_metadata.json")
    metadata_path.write_text(
        json.dumps(
            {
                "source_digest": digest,
                "latest_observation": str(artifact["latest_observation"]),
                "quality": quality,
                "validation_method": artifact["validation_method"],
                "target_horizons": artifact["horizons"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return artifact, True
