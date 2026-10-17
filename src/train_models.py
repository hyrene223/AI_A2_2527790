"""Train, compare, evaluate, and save the three ML tasks."""

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SEED = 7790
VERIFICATION_CODE = "AI-A2-7790"

DATA_FILE = ROOT / "data" / "AI_A2_2527790_clean.csv"
MODELS_DIR = ROOT / "models"
ARTIFACTS_DIR = ROOT / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"

NUMERIC = [
    "rainfall_mm",
    "avg_temp_c",
    "soil_moisture_pct",
    "fertilizer_kg",
    "field_size_ha",
    "cooperative_support",
]
CATEGORICAL = ["season"]

# Targets are deliberately excluded from model input features.
REGRESSION_FEATURES = NUMERIC + CATEGORICAL
CLASSIFICATION_FEATURES = NUMERIC + CATEGORICAL
CLUSTER_FEATURES = NUMERIC + CATEGORICAL


def make_preprocessor():
    """Create preprocessing that is fitted only on training data."""
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC),
        ("categorical", categorical_pipe, CATEGORICAL),
    ])


def write_json(path, content):
    """Write JSON while converting NumPy values to standard Python values."""
    path.write_text(
        json.dumps(content, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def main():
    """Run all three tasks and save evaluation artifacts."""
    for folder in [MODELS_DIR, ARTIFACTS_DIR, PLOTS_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Clean dataset not found: {DATA_FILE}. "
            "Run data/generate_data.py and data/clean_data.py first."
        )

    df = pd.read_csv(DATA_FILE)

    if df.isna().any().any():
        raise ValueError("Clean dataset still contains missing values.")
    if df.duplicated().any():
        raise ValueError("Clean dataset still contains duplicates.")
    if len(df) < 30:
        raise ValueError("Not enough rows for reliable train/test evaluation.")

    X = df[REGRESSION_FEATURES].copy()
    y_reg = df["yield_tons"].astype(float)
    y_cls = df["high_risk"].astype(int)

    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.20,
        random_state=SEED,
        stratify=y_cls,
    )

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_reg_train, y_reg_test = y_reg.iloc[train_idx], y_reg.iloc[test_idx]
    y_cls_train, y_cls_test = y_cls.iloc[train_idx], y_cls.iloc[test_idx]

    # ---------------------------------------------------------
    # 1. REGRESSION: estimate harvest quantity
    # ---------------------------------------------------------
    regression_candidates = {
        "mean_baseline": DummyRegressor(strategy="mean"),
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=200, random_state=SEED, min_samples_leaf=2
        ),
    }

    regression_results = {}
    regression_fitted = {}
    regression_predictions = {}

    for name, estimator in regression_candidates.items():
        model = Pipeline([
            ("preprocessor", make_preprocessor()),
            ("model", clone(estimator)),
        ])
        model.fit(X_train, y_reg_train)
        pred = model.predict(X_test)

        regression_results[name] = {
            "mae": float(mean_absolute_error(y_reg_test, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_reg_test, pred))),
            "r2": float(r2_score(y_reg_test, pred)),
        }
        regression_fitted[name] = model
        regression_predictions[name] = pred

    best_regression = min(
        ["linear_regression", "random_forest"],
        key=lambda name: regression_results[name]["rmse"],
    )
    reg_model = regression_fitted[best_regression]
    reg_pred = regression_predictions[best_regression]

    write_json(ARTIFACTS_DIR / "regression_metrics.json", {
        "student": "Nsabimana Irene",
        "registration_number": "25/27790",
        "verification_code": VERIFICATION_CODE,
        "target": "yield_tons",
        "test_rows": len(test_idx),
        "metrics": regression_results,
        "selected_model": best_regression,
        "selection_reason": "Lowest held-out test RMSE among candidate models.",
    })
    joblib.dump(reg_model, MODELS_DIR / "regression_model.joblib")

    residuals = y_reg_test.to_numpy() - reg_pred
    plt.figure(figsize=(7, 5))
    plt.scatter(reg_pred, residuals, alpha=0.75)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted harvest (tons)")
    plt.ylabel("Residual (actual - predicted)")
    plt.title("Regression Residuals")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "residuals.png", dpi=150)
    plt.close()

    # ---------------------------------------------------------
    # 2. CLASSIFICATION: predict high production risk
    # ---------------------------------------------------------
    classification_candidates = {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(
            max_iter=2000, random_state=SEED
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=SEED, min_samples_leaf=2
        ),
    }

    classification_results = {}
    classification_fitted = {}
    classification_predictions = {}
    classification_probabilities = {}

    for name, estimator in classification_candidates.items():
        model = Pipeline([
            ("preprocessor", make_preprocessor()),
            ("model", clone(estimator)),
        ])
        model.fit(X_train, y_cls_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

        result = {
            "precision": float(precision_score(
                y_cls_test, pred, zero_division=0
            )),
            "recall": float(recall_score(
                y_cls_test, pred, zero_division=0
            )),
            "f1": float(f1_score(
                y_cls_test, pred, zero_division=0
            )),
            "confusion_matrix": [
                [int(v) for v in row]
                for row in __import__(
                    "sklearn.metrics", fromlist=["confusion_matrix"]
                ).confusion_matrix(y_cls_test, pred).tolist()
            ],
            "roc_auc": float(roc_auc_score(y_cls_test, proba)),
        }
        classification_results[name] = result
        classification_fitted[name] = model
        classification_predictions[name] = pred
        classification_probabilities[name] = proba

    best_classifier = max(
        ["logistic_regression", "random_forest"],
        key=lambda name: classification_results[name]["f1"],
    )
    cls_model = classification_fitted[best_classifier]
    cls_pred = classification_predictions[best_classifier]
    cls_proba = classification_probabilities[best_classifier]

    write_json(ARTIFACTS_DIR / "classification_metrics.json", {
        "student": "Nsabimana Irene",
        "registration_number": "25/27790",
        "verification_code": VERIFICATION_CODE,
        "target": "high_risk",
        "test_rows": len(test_idx),
        "metrics": classification_results,
        "selected_model": best_classifier,
        "selection_reason": "Highest held-out test F1 among candidate models.",
        "positive_class": 1,
    })
    joblib.dump(cls_model, MODELS_DIR / "classification_model.joblib")

    ConfusionMatrixDisplay.from_predictions(
        y_cls_test, cls_pred, display_labels=["Low risk", "High risk"]
    )
    plt.title("Selected Classifier: Confusion Matrix")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    # ---------------------------------------------------------
    # 3. CLUSTERING: group farms by operating conditions
    # ---------------------------------------------------------
    cluster_preprocessor = make_preprocessor()
    cluster_matrix = cluster_preprocessor.fit_transform(
        df[CLUSTER_FEATURES]
    )
    if hasattr(cluster_matrix, "toarray"):
        cluster_matrix = cluster_matrix.toarray()

    cluster_scores = {}
    cluster_models = {}

    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        labels = km.fit_predict(cluster_matrix)
        score = silhouette_score(cluster_matrix, labels)
        cluster_scores[str(k)] = float(score)
        cluster_models[k] = km

    selected_k = max(
        range(2, 7), key=lambda k: cluster_scores[str(k)]
    )
    cluster_model = cluster_models[selected_k]
    cluster_labels = cluster_model.labels_

    profile_columns = [
        "rainfall_mm",
        "avg_temp_c",
        "soil_moisture_pct",
        "fertilizer_kg",
        "field_size_ha",
        "cooperative_support",
        "yield_tons",
    ]
    profile_df = df[profile_columns].copy()
    profile_df["cluster"] = cluster_labels
    profiles = profile_df.groupby("cluster").mean().round(3)
    cluster_sizes = profile_df["cluster"].value_counts().sort_index()

    cluster_report = {
        "student": "Nsabimana Irene",
        "registration_number": "25/27790",
        "verification_code": VERIFICATION_CODE,
        "features": CLUSTER_FEATURES,
        "silhouette_scores_by_k": cluster_scores,
        "selected_k": int(selected_k),
        "selection_reason": "Highest silhouette score among K=2 to K=6.",
        "cluster_sizes": {
            str(int(k)): int(v) for k, v in cluster_sizes.items()
        },
        "cluster_profiles": {
            str(int(idx)): {
                col: float(value) for col, value in row.items()
            }
            for idx, row in profiles.iterrows()
        },
        "interpretation_note": (
            "Compare cluster averages with overall averages to describe "
            "farm conditions. Clusters are exploratory, not risk labels."
        ),
    }
    write_json(ARTIFACTS_DIR / "clustering_report.json", cluster_report)

    joblib.dump({
        "preprocessor": cluster_preprocessor,
        "model": cluster_model,
        "features": CLUSTER_FEATURES,
        "selected_k": int(selected_k),
    }, MODELS_DIR / "clustering_model.joblib")

    coords = PCA(n_components=2).fit_transform(cluster_matrix)
    plt.figure(figsize=(7, 5))
    plt.scatter(
        coords[:, 0], coords[:, 1],
        c=cluster_labels, cmap="viridis", alpha=0.8
    )
    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.title(f"Farm Clusters (K={selected_k})")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "clusters.png", dpi=150)
    plt.close()

    # ---------------------------------------------------------
    # Test predictions for reproducibility and error analysis
    # ---------------------------------------------------------
    prediction_output = pd.DataFrame({
        "row_id": test_idx,
        "actual_yield_tons": y_reg_test.to_numpy(),
        "predicted_yield_tons": reg_pred,
        "yield_residual": residuals,
        "actual_high_risk": y_cls_test.to_numpy(),
        "predicted_high_risk": cls_pred,
        "high_risk_probability": cls_proba,
        "cluster": cluster_labels[test_idx],
    })
    prediction_output.to_csv(
        ARTIFACTS_DIR / "test_predictions.csv",
        index=False,
        float_format="%.4f",
    )

    write_json(ARTIFACTS_DIR / "training_summary.json", {
        "student": "Nsabimana Irene",
        "registration_number": "25/27790",
        "verification_code": VERIFICATION_CODE,
        "seed": SEED,
        "dataset_rows": int(len(df)),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "regression_selected_model": best_regression,
        "classification_selected_model": best_classifier,
        "selected_cluster_count": int(selected_k),
        "reproducibility": "All model random states fixed at 7790.",
        "intended_use": (
            "Classroom decision-support prototype using synthetic data only."
        ),
    })

    print("Training and evaluation completed.")
    print(f"Verification code: {VERIFICATION_CODE}")
    print(f"Training rows: {len(train_idx)}")
    print(f"Test rows: {len(test_idx)}")
    print("\nRegression metrics:")
    print(json.dumps(regression_results, indent=2))
    print(f"Selected regression model: {best_regression}")
    print("\nClassification metrics:")
    print(json.dumps(classification_results, indent=2))
    print(f"Selected classifier: {best_classifier}")
    print("\nSilhouette scores by K:")
    print(json.dumps(cluster_scores, indent=2))
    print(f"Selected K: {selected_k}")
    print(f"\nSaved models: {MODELS_DIR}")
    print(f"Saved reports and plots: {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
