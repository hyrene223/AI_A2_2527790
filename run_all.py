"""Run the complete Musanze Harvest Intelligence assignment pipeline."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPECTED_STUDENT_ID = "25/27790"
EXPECTED_SEED = 7790
EXPECTED_CODE = "AI-A2-7790"

REQUIRED_OUTPUTS = [
    "data/AI_A2_2527790.csv",
    "data/AI_A2_2527790_clean.csv",
    "artifacts/data_fingerprint.txt",
    "artifacts/data_quality_report.json",
    "artifacts/regression_metrics.json",
    "artifacts/classification_metrics.json",
    "artifacts/clustering_report.json",
    "artifacts/test_predictions.csv",
    "models/regression_model.joblib",
    "models/classification_model.joblib",
    "models/clustering_model.joblib",
    "artifacts/plots/residuals.png",
    "artifacts/plots/confusion_matrix.png",
    "artifacts/plots/clusters.png",
]


def run_step(description, command):
    print(f"\n{'=' * 65}\n{description}\n{'=' * 65}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run the complete personalized assignment pipeline."
    )
    parser.add_argument("--student-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()

    if args.student_id != EXPECTED_STUDENT_ID:
        parser.error(f"Expected student ID {EXPECTED_STUDENT_ID}.")
    if args.seed != EXPECTED_SEED:
        parser.error(f"Expected seed {EXPECTED_SEED} for {EXPECTED_CODE}.")

    python = sys.executable

    run_step(
        "STEP 1/4: Generate reproducible synthetic data",
        [python, str(ROOT / "data" / "generate_data.py"),
         "--seed", str(args.seed)],
    )
    run_step(
        "STEP 2/4: Validate and clean the data",
        [python, str(ROOT / "data" / "clean_data.py")],
    )
    run_step(
        "STEP 3/4: Train and evaluate regression, classification, and clustering",
        [python, str(ROOT / "src" / "train_models.py")],
    )
    run_step(
        "STEP 4/4: Run automated tests",
        [python, "-m", "pytest", "-q"],
    )

    print("\nChecking required generated artifacts...")
    missing = [p for p in REQUIRED_OUTPUTS if not (ROOT / p).is_file()]
    if missing:
        print("ERROR: Required outputs are missing:")
        for path in missing:
            print(f"  - {path}")
        raise SystemExit(1)

    print(f"\nVerification code: {EXPECTED_CODE}")
    print(f"Student ID: {args.student_id}")
    print(f"Seed: {args.seed}")
    print(f"Required generated artifacts found: {len(REQUIRED_OUTPUTS)}")
    print("PIPELINE COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
