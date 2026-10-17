# Musanze Harvest Intelligence Service

**Student name:** Nsabimana Irene  
**Registration number:** 25/27790  
**Institution:** Institut d'Enseignement Supérieur de Ruhengeri (INES Ruhengeri)
**Institutional username:** Nsabimana Irene  
**Verification code:** AI-A2-7790

## Problem Statement

This project demonstrates how machine learning can be used to estimate synthetic potato harvest yield, classify high-risk harvest conditions, and explore groups of similar records through clustering.

## Important Data Notice

All data used in this assignment is synthetically generated for educational purposes. The predictions have not been validated for real farms and must not be used as the sole basis for agricultural, financial, insurance, or other high-impact decisions.

## Environment

- Python: 3.14.7
- Operating system: Windows
- Models: linear regression, random forest classification, and K-means clustering
- API: FastAPI
- Tests: pytest

## Setup

Open PowerShell in the project directory and run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the Complete Pipeline

```powershell
.\.venv\Scripts\python.exe .\run_all.py --student-id "25/27790" --seed 7790
```

The pipeline generates the synthetic data, cleans and validates it, trains and evaluates models, saves artifacts, and runs automated tests.

## Run a Prediction

```powershell
.\.venv\Scripts\python.exe .\predict.py --json .\samples\valid.json
```

Test the invalid sample:

```powershell
.\.venv\Scripts\python.exe .\predict.py --json .\samples\invalid.json
```

The invalid sample contains rainfall outside the permitted range and should be rejected with a useful validation error.

## Start the API

```powershell
.\.venv\Scripts\python.exe -m uvicorn app:app --reload
```

Open the interactive API documentation at `http://127.0.0.1:8000/docs`.

Available endpoints:

- `GET /health` — service health and verification code.
- `GET /model-info` — model and project information.
- `POST /predict` — validated prediction request.

## Evaluation Summary

The latest recorded run used 192 training records and 48 test records.

- Regression: linear regression was selected, with MAE approximately 0.798, RMSE approximately 0.963, and R² approximately 0.864.
- Classification: random forest was selected, with test precision, recall, and F1 all equal to 1.00 on this synthetic test split. This score does not imply equivalent real-world performance.
- Clustering: K=2 was selected by silhouette score (approximately 0.162), indicating weakly separated clusters.
- Tests: 6 passed in the latest recorded run.

Refer to the JSON reports and plots in `artifacts/` for detailed results.

## Generated Artifacts

- `artifacts/data_fingerprint.txt` — SHA-256 fingerprint of the generated CSV.
- `artifacts/data_quality_report.json` — data quality and cleaning summary.
- `artifacts/regression_metrics.json` — regression model comparison.
- `artifacts/classification_metrics.json` — classification metrics and confusion matrices.
- `artifacts/clustering_report.json` — clustering evaluation and profiles.
- `artifacts/test_predictions.csv` — held-out test predictions.
- `artifacts/plots/` — residual, confusion-matrix, and cluster plots.
- `models/` — saved trained models.

## Limitations and Responsible Use

The dataset is synthetic, small, and generated from assignment-specific assumptions. Results may not generalize to actual farms, seasons, locations, or farming practices. Real-world deployment would require representative field data, independent validation, expert review, and ongoing monitoring. Read `docs/RESPONSIBLE_AI.md` before interpreting results.

## Reproducibility and Submission Details

- Student ID: `25/27790`
- Seed: `7790`
- Verification code: `AI-A2-7790`
- Repository URL: https://github.com/hyrene223/AI_A2_2527790
- Submission content commit hash: To be updated after the demo video is committed.
## Declaration and AI Use

Include the required assignment declaration and AI-use disclosure in the submission evidence. Accurately disclose any AI assistance used and personally verify all submitted code, analysis, and documentation.




