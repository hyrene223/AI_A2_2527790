"""Validate, clean, and report on the personalized farm dataset."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "AI_A2_2527790.csv"
OUTPUT_FILE = ROOT / "data" / "AI_A2_2527790_clean.csv"
REPORT_FILE = ROOT / "artifacts" / "data_quality_report.json"

REQUIRED_COLUMNS = [
    "rainfall_mm",
    "avg_temp_c",
    "soil_moisture_pct",
    "fertilizer_kg",
    "field_size_ha",
    "season",
    "cooperative_support",
    "yield_tons",
    "high_risk",
]


def clean_dataset(df: pd.DataFrame):
    """Validate columns, remove duplicates, and impute missing values."""
    if list(df.columns) != REQUIRED_COLUMNS:
        raise ValueError("Dataset columns do not match the required schema.")

    rows_before = len(df)
    duplicates_before = int(df.duplicated().sum())
    missing_before = {
        column: int(count)
        for column, count in df.isna().sum().items()
        if count > 0
    }

    # Remove exact duplicate records before imputing missing values.
    cleaned = df.drop_duplicates().copy()
    rows_after_duplicates = len(cleaned)

    # Check valid values before imputing.
    if not cleaned["season"].dropna().isin(["A", "B"]).all():
        raise ValueError("Season must be A or B.")

    for column in [
        "rainfall_mm",
        "avg_temp_c",
        "soil_moisture_pct",
        "fertilizer_kg",
        "field_size_ha",
        "cooperative_support",
        "yield_tons",
        "high_risk",
    ]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="raise")

    if not cleaned["rainfall_mm"].dropna().between(600, 1400).all():
        raise ValueError("Rainfall is outside the expected range.")

    if not cleaned["avg_temp_c"].dropna().between(12, 26).all():
        raise ValueError("Temperature is outside the expected range.")

    if not cleaned["soil_moisture_pct"].dropna().between(20, 80).all():
        raise ValueError("Soil moisture is outside the expected range.")

    if not cleaned["fertilizer_kg"].dropna().between(0, 250).all():
        raise ValueError("Fertilizer is outside the expected range.")

    if not cleaned["field_size_ha"].dropna().gt(0).all():
        raise ValueError("Field size must be positive.")

    for column in ["cooperative_support", "high_risk"]:
        if not cleaned[column].dropna().isin([0, 1]).all():
            raise ValueError(f"{column} must contain only 0 or 1.")

    if not cleaned["yield_tons"].dropna().gt(0).all():
        raise ValueError("Harvest yield must be positive.")

    # Impute feature values. Targets are not silently invented.
    numeric_features = [
        "rainfall_mm",
        "avg_temp_c",
        "soil_moisture_pct",
        "fertilizer_kg",
        "field_size_ha",
    ]
    for column in numeric_features:
        cleaned[column] = cleaned[column].fillna(cleaned[column].median())

    if cleaned["season"].isna().any():
        cleaned["season"] = cleaned["season"].fillna(
            cleaned["season"].mode().iloc[0]
        )

    # A missing target must be investigated, not fabricated.
    if cleaned[["yield_tons", "high_risk"]].isna().any().any():
        raise ValueError("Target columns contain missing values.")

    if cleaned.isna().any().any():
        raise ValueError("Missing values remain after cleaning.")

    report = {
        "student": "Nsabimana Irene",
        "registration_number": "25/27790",
        "verification_code": "AI-A2-7790",
        "input_rows": rows_before,
        "duplicates_removed": rows_before - rows_after_duplicates,
        "rows_after_duplicate_removal": rows_after_duplicates,
        "missing_values_before": missing_before,
        "missing_values_after": {
            column: int(count)
            for column, count in cleaned.isna().sum().items()
        },
        "final_rows": len(cleaned),
        "final_columns": len(cleaned.columns),
        "validation": "passed",
        "imputation": "Median for numerical features; mode for season if needed",
    }

    return cleaned, report


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    raw = pd.read_csv(INPUT_FILE)
    cleaned, report = clean_dataset(raw)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    cleaned.to_csv(OUTPUT_FILE, index=False, float_format="%.4f")
    REPORT_FILE.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("Data cleaning completed successfully.")
    print(f"Input rows: {report['input_rows']}")
    print(f"Duplicates removed: {report['duplicates_removed']}")
    print(f"Final rows: {report['final_rows']}")
    print(f"Final columns: {report['final_columns']}")
    print(f"Remaining missing values: {int(cleaned.isna().sum().sum())}")
    print(f"Cleaned data: {OUTPUT_FILE}")
    print(f"Quality report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
