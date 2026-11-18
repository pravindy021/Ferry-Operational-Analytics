import argparse
from pathlib import Path

from src.models.artifacts import train_or_load


PROJECT_DIR = Path(__file__).resolve().parent
DATASET = PROJECT_DIR / "data" / "Toronto Island Ferry Ticket Counts(1).csv"
ARTIFACT = PROJECT_DIR / "data" / "model_artifacts" / "forecaster.joblib"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate direct short-term ferry demand forecasts."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even if a matching dataset artifact already exists.",
    )
    parser.add_argument(
        "--rolling-folds",
        type=int,
        default=3,
        help="Number of expanding rolling-origin validation folds (0 disables them).",
    )
    args = parser.parse_args()
    if args.rolling_folds < 0:
        parser.error("--rolling-folds cannot be negative.")

    print(f"Dataset: {DATASET}")
    print("Training chronological models; no missing intervals will be interpolated.")
    artifact, trained_now = train_or_load(
        DATASET,
        ARTIFACT,
        rolling_folds=args.rolling_folds,
        force=args.force,
    )
    if not trained_now:
        print(f"Reusing cached artifact: {ARTIFACT}")
    else:
        print(f"Saved trained models: {ARTIFACT}")
        print(f"Saved evaluation table: {ARTIFACT.with_name('model_evaluation.csv')}")
    print(f"Latest observed interval: {artifact['latest_observation']}")
    print(artifact["evaluation"].to_string(index=False))


if __name__ == "__main__":
    main()
