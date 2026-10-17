import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
FEATURES = [
    "rainfall_mm",
    "avg_temp_c",
    "soil_moisture_pct",
    "fertilizer_kg",
    "field_size_ha",
    "season",
    "cooperative_support",
]
VERIFICATION_CODE = "AI-A2-7790"


def validate_input(data):
    if not isinstance(data, dict):
        raise ValueError("Input JSON must be an object.")

    missing = [field for field in FEATURES if field not in data]
    extra = [field for field in data if field not in FEATURES]

    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")
    if extra:
        raise ValueError(f"Unexpected fields: {', '.join(extra)}")

    numeric_ranges = {
        "rainfall_mm": (600, 1400),
        "avg_temp_c": (12, 26),
        "soil_moisture_pct": (20, 80),
        "fertilizer_kg": (0, 250),
        "field_size_ha": (0.01, 4),
        "cooperative_support": (0, 1),
    }

    for field, (minimum, maximum) in numeric_ranges.items():
        value = data[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field} must be a number.")
        if not minimum <= value <= maximum:
            raise ValueError(
                f"{field} must be between {minimum} and {maximum}."
            )

    if data["cooperative_support"] not in (0, 1):
        raise ValueError("cooperative_support must be 0 or 1.")

    if data["season"] not in ("A", "B"):
        raise ValueError("season must be 'A' or 'B'.")

    return pd.DataFrame([{field: data[field] for field in FEATURES}])


def predict(data):
    row = validate_input(data)

    regression = joblib.load(MODELS / "regression_model.joblib")
    classifier = joblib.load(MODELS / "classification_model.joblib")
    clustering_bundle = joblib.load(MODELS / "clustering_model.joblib")

    harvest = float(regression.predict(row)[0])
    risk = int(classifier.predict(row)[0])

    probabilities = classifier.predict_proba(row)[0]
    classes = list(classifier.classes_)
    risk_probability = float(probabilities[classes.index(1)])

    cluster_features = clustering_bundle["features"]
    cluster_preprocessor = clustering_bundle["preprocessor"]
    cluster_model = clustering_bundle["model"]
    cluster_row = row[cluster_features]
    cluster_number = int(
        cluster_model.predict(cluster_preprocessor.transform(cluster_row))[0]
    )

    return {
        "verification_code": VERIFICATION_CODE,
        "predicted_harvest_tons": round(harvest, 3),
        "high_risk_prediction": risk,
        "high_risk_probability": round(risk_probability, 4),
        "cluster": cluster_number,
        "notice": (
            "Educational prediction from synthetic assignment data; "
            "not validated for real farming decisions."
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Musanze Harvest Intelligence prediction CLI"
    )
    parser.add_argument("--json", required=True, help="Path to input JSON file")
    args = parser.parse_args()

    try:
        with open(args.json, "r", encoding="utf-8-sig") as file:
            data = json.load(file)
        print(json.dumps(predict(data), indent=2))
    except (OSError, json.JSONDecodeError, ValueError, KeyError) as error:
        print(f"Prediction error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
