"""Generate the personalized synthetic potato-cooperative dataset."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STUDENT_ID = "25/27790"
SEED = 7790
VERIFICATION_CODE = "AI-A2-7790"

COLUMNS = [
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


def generate_dataset(seed: int = SEED) -> pd.DataFrame:
    """Generate reproducible data with the required deliberate data problems."""
    if seed != SEED:
        raise ValueError(f"Seed must be {SEED} for {VERIFICATION_CODE}.")

    rng = np.random.default_rng(seed)
    n_rows = 240 + (seed % 41)

    rainfall = rng.uniform(600, 1400, n_rows)
    temperature = rng.uniform(12, 26, n_rows)
    moisture = rng.uniform(20, 80, n_rows)
    fertilizer = rng.uniform(0, 250, n_rows)
    field_size = rng.uniform(0.2, 4.0, n_rows)
    season = rng.choice(["A", "B"], size=n_rows)
    support = rng.integers(0, 2, size=n_rows)

    season_effect = np.where(season == "A", 0.50, -0.20)
    noise = rng.normal(0, 0.80, n_rows)

    yield_tons = (
        0.004 * rainfall
        + 0.06 * moisture
        + 0.02 * fertilizer
        + 1.30 * field_size
        - 0.25 * np.abs(temperature - 18)
        + 0.70 * support
        + season_effect
        + noise
    )
    yield_tons = np.clip(yield_tons, 0.50, None)

    yield_per_hectare = yield_tons / field_size
    high_risk = (yield_per_hectare < np.median(yield_per_hectare)).astype(int)

    df = pd.DataFrame(
        {
            "rainfall_mm": rainfall,
            "avg_temp_c": temperature,
            "soil_moisture_pct": moisture,
            "fertilizer_kg": fertilizer,
            "field_size_ha": field_size,
            "season": season,
            "cooperative_support": support,
            "yield_tons": yield_tons,
            "high_risk": high_risk,
        }
    )

    # Introduce the assignment's missing values using the same RNG.
    m = round(0.05 * n_rows)
    moisture_rows = rng.choice(n_rows, size=m, replace=False)
    fertilizer_rows = rng.choice(n_rows, size=m, replace=False)

    df.loc[moisture_rows, "soil_moisture_pct"] = np.nan
    df.loc[fertilizer_rows, "fertilizer_kg"] = np.nan

    # Append three exact copies of existing rows.
    duplicate_rows = df.iloc[:3].copy()
    df = pd.concat([df, duplicate_rows], ignore_index=True)

    return df[COLUMNS]


def main() -> None:
    """Generate the CSV and record its SHA-256 fingerprint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    df = generate_dataset(args.seed)

    data_dir = ROOT / "data"
    artifacts_dir = ROOT / "artifacts"
    data_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "AI_A2_2527790.csv"
    df.to_csv(csv_path, index=False, float_format="%.4f")

    fingerprint = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    fingerprint_path = artifacts_dir / "data_fingerprint.txt"
    fingerprint_path.write_text(
        f"Student: {STUDENT_ID}\n"
        f"Verification code: {VERIFICATION_CODE}\n"
        f"Seed: {args.seed}\n"
        f"Rows before cleaning: {len(df)}\n"
        f"SHA-256: {fingerprint}\n",
        encoding="utf-8",
    )

    print(f"Student: {STUDENT_ID}")
    print(f"Verification code: {VERIFICATION_CODE}")
    print(f"Seed: {args.seed}")
    print(f"Rows generated: {len(df)}")
    print(f"Missing soil moisture: {df['soil_moisture_pct'].isna().sum()}")
    print(f"Missing fertilizer: {df['fertilizer_kg'].isna().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print(f"Dataset: {csv_path}")
    print(f"SHA-256: {fingerprint}")


if __name__ == "__main__":
    main()
