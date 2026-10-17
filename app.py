from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, field_validator

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
VERIFICATION_CODE = "AI-A2-7790"

FEATURES = [
    "rainfall_mm",
    "avg_temp_c",
    "soil_moisture_pct",
    "fertilizer_kg",
    "field_size_ha",
    "season",
    "cooperative_support",
]

app = FastAPI(
    title="Musanze Harvest Intelligence Service",
    description=(
        "Educational API for synthetic potato-cooperative harvest "
        "prediction, high-risk classification, and clustering."
    ),
    version="1.0.0",
)


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rainfall_mm: float = Field(ge=600, le=1400)
    avg_temp_c: float = Field(ge=12, le=26)
    soil_moisture_pct: float = Field(ge=20, le=80)
    fertilizer_kg: float = Field(ge=0, le=250)
    field_size_ha: float = Field(gt=0, le=4)
    season: StrictStr
    cooperative_support: StrictInt

    @field_validator("season")
    @classmethod
    def validate_season(cls, value):
        if value not in ("A", "B"):
            raise ValueError("season must be 'A' or 'B'")
        return value

    @field_validator("cooperative_support")
    @classmethod
    def validate_support(cls, value):
        if value not in (0, 1):
            raise ValueError("cooperative_support must be 0 or 1")
        return value


def load_models():
    required = [
        MODELS / "regression_model.joblib",
        MODELS / "classification_model.joblib",
        MODELS / "clustering_model.joblib",
    ]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing model files: {', '.join(missing)}")

    regression = joblib.load(required[0])
    classifier = joblib.load(required[1])
    clustering = joblib.load(required[2])
    return regression, classifier, clustering


@app.get("/health")
def health():
    try:
        load_models()
        return {"status": "ok", "verification_code": VERIFICATION_CODE}
    except (RuntimeError, OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/model-info")
def model_info():
    return {
        "service": "Musanze Harvest Intelligence Service",
        "verification_code": VERIFICATION_CODE,
        "regression_model": "Linear Regression",
        "classification_model": "Random Forest",
        "clustering_model": "K-Means",
        "input_features": FEATURES,
        "data_notice": "Models were trained on synthetic educational data.",
    }


@app.post("/predict")
def make_prediction(payload: PredictionInput):
    try:
        regression, classifier, clustering = load_models()
        row = pd.DataFrame([payload.model_dump()])[FEATURES]

        harvest = float(regression.predict(row)[0])
        risk = int(classifier.predict(row)[0])

        probabilities = classifier.predict_proba(row)[0]
        classes = list(classifier.classes_)
        risk_probability = float(probabilities[classes.index(1)])

        cluster_features = clustering["features"]
        transformed = clustering["preprocessor"].transform(
            row[cluster_features]
        )
        cluster_number = int(clustering["model"].predict(transformed)[0])

        return {
            "verification_code": VERIFICATION_CODE,
            "predicted_harvest_tons": round(harvest, 3),
            "high_risk_prediction": risk,
            "high_risk_probability": round(risk_probability, 4),
            "cluster": cluster_number,
            "notice": (
                "Educational prediction from synthetic data; "
                "not validated for real farming decisions."
            ),
        }
    except (RuntimeError, OSError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
