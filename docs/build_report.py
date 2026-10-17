import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Table, TableStyle, Image
)

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
PLOTS = ART / "plots"
OUT = ROOT / "docs" / "AI_A2_2527790_REPORT.pdf"

def read_json(name):
    path = ART / name
    if not path.exists():
        raise FileNotFoundError(f"Required metrics file missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))

reg = read_json("regression_metrics.json")
clf = read_json("classification_metrics.json")
cluster = read_json("clustering_report.json")
quality = read_json("data_quality_report.json")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="CoverTitle", parent=styles["Title"],
    fontSize=23, leading=29, alignment=TA_CENTER, spaceAfter=18
))
styles.add(ParagraphStyle(
    name="Section", parent=styles["Heading1"],
    fontSize=16, leading=20, spaceAfter=10
))
styles.add(ParagraphStyle(
    name="SubSection", parent=styles["Heading2"],
    fontSize=12, leading=15, spaceBefore=7, spaceAfter=5
))
styles.add(ParagraphStyle(
    name="BodySmall", parent=styles["BodyText"],
    fontSize=9, leading=13, spaceAfter=6
))
styles.add(ParagraphStyle(
    name="CoverSub", parent=styles["Normal"],
    fontSize=12, leading=18, alignment=TA_CENTER
))

def para(text, style="BodySmall"):
    return Paragraph(text, styles[style])

def make_table(rows, widths=None):
    table = Table(rows, colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dcebea")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#123d3b")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("LEADING", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table

def fmt(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)

def metric_rows(data, columns):
    rows = [["Model"] + columns]
    for model, metrics in data.items():
        if isinstance(metrics, dict):
            rows.append([model] + [fmt(metrics.get(k, "N/A")) for k in columns])
    return rows

def add_image(story, filename, width=15.5*cm, height=8.5*cm):
    path = PLOTS / filename
    if path.exists():
        img = Image(str(path))
        img._restrictSize(width, height)
        story.append(img)
    else:
        story.append(para(f"Plot unavailable: {filename}"))

def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawString(2*cm, 1.1*cm, "SWE 3513 AI Individual Assignment 2 | AI-A2-7790")
    canvas.drawRightString(A4[0] - 2*cm, 1.1*cm, f"Page {doc.page}")
    canvas.restoreState()

doc = SimpleDocTemplate(
    str(OUT), pagesize=A4,
    rightMargin=2*cm, leftMargin=2*cm,
    topMargin=1.7*cm, bottomMargin=1.8*cm
)
story = []

# PAGE 1 — Cover and problem
story += [
    Spacer(1, 2.2*cm),
    para("MUSANZE HARVEST<br/>INTELLIGENCE SERVICE", "CoverTitle"),
    para("SWE 3513 — Artificial Intelligence<br/>Individual Assignment 2", "CoverSub"),
    Spacer(1, 1.5*cm),
    make_table([
        ["Submission detail", "Value"],
        ["Student name", "Nsabimana Irene"],
        ["Registration number", "25/27790"],
        ["Institution", "Institut d'Enseignement Supérieur de Ruhengeri (INES Ruhengeri)"],
        ["Verification code", "AI-A2-7790"],
        ["Personalized seed", "7790"],
        ["Python version", "3.14.7"],
    ], [5*cm, 10*cm]),
    Spacer(1, 1.2*cm),
    para("Problem statement", "Section"),
    para(
        "This project demonstrates a machine-learning workflow for a synthetic "
        "potato-cooperative dataset. It estimates harvest yield, classifies "
        "high-risk conditions, and explores groups of records through clustering."
    ),
    para(
        "<b>Data notice:</b> The dataset is synthetically generated for educational "
        "purposes. The models have not been validated for real farming decisions."
    ),
    PageBreak()
]

# PAGE 2 — Data generation and preparation
story += [
    para("1. Dataset and Data Preparation", "Section"),
    para(
        "The personalized data generator uses NumPy's seeded random generator "
        "with seed 7790. The dataset includes rainfall, average temperature, "
        "soil moisture, fertilizer quantity, field size, season, cooperative "
        "support, harvest yield, and a high-risk label."
    ),
    para("Required schema", "SubSection"),
    make_table([
        ["Feature", "Description"],
        ["rainfall_mm", "Rainfall in millimetres"],
        ["avg_temp_c", "Average temperature in degrees Celsius"],
        ["soil_moisture_pct", "Soil moisture percentage"],
        ["fertilizer_kg", "Fertilizer quantity in kilograms"],
        ["field_size_ha", "Field size in hectares"],
        ["season", "Categorical season (A or B)"],
        ["cooperative_support", "Binary cooperative support indicator"],
        ["yield_tons", "Regression target: harvest yield in tons"],
        ["high_risk", "Classification target: risk indicator"],
    ], [5*cm, 10*cm]),
    Spacer(1, 8),
    para("Cleaning and quality checks", "SubSection"),
    para(
        "The generator produced 243 rows, including three deliberate exact "
        "duplicates and missing values in soil moisture and fertilizer. Cleaning "
        "removed the three duplicate rows and imputed missing values, resulting "
        "in 240 rows and nine columns with no missing values remaining."
    ),
    para(
        "The generator records a SHA-256 fingerprint of the generated CSV in "
        "artifacts/data_fingerprint.txt. The fingerprint helps verify whether "
        "the generated file's bytes remain unchanged."
    ),
    para(
        "Data preparation is part of the reproducibility process. Imputation, "
        "scaling, and categorical encoding used in model pipelines are fitted "
        "on training data rather than the held-out test set."
    ),
    PageBreak()
]

# PAGE 3 — Methods
story += [
    para("2. Modelling Methodology", "Section"),
    para("Regression", "SubSection"),
    para(
        "The regression task predicts yield_tons. A mean-prediction baseline was "
        "compared with linear regression and random forest regression. Models "
        "were evaluated using mean absolute error (MAE), root mean squared error "
        "(RMSE), and coefficient of determination (R²). Lower MAE and RMSE are "
        "better; R² describes the proportion of test-target variation explained "
        "by the model."
    ),
    para("Classification", "SubSection"),
    para(
        "The classification task predicts high_risk. A majority-class baseline "
        "was compared with logistic regression and random forest classification. "
        "Precision, recall, F1, confusion matrices, and ROC-AUC were recorded "
        "where the metric was valid. The high_risk target was excluded from input "
        "features."
    ),
    para("Clustering", "SubSection"),
    para(
        "K-means clustering used input features after preprocessing. Candidate "
        "cluster counts from K=2 through K=6 were evaluated using silhouette "
        "scores. A higher silhouette score generally indicates better separation, "
        "although it does not prove that clusters have real-world meaning."
    ),
    para("Leakage prevention and evaluation", "SubSection"),
    para(
        "The regression target was not used as a predictor, and target-derived "
        "features were excluded. The cleaned data was divided into 192 training "
        "rows and 48 test rows using the configured random seed. The held-out "
        "test set was used for model comparison."
    ),
    PageBreak()
]

# PAGE 4 — Regression results
story += [
    para("3. Regression Results", "Section"),
    para(
        "The selected regression model was linear regression because it achieved "
        "the lowest test RMSE among the compared models."
    ),
    make_table(metric_rows(reg, ["mae", "rmse", "r2"]),
               [4.8*cm, 3.3*cm, 3.3*cm, 3.3*cm]),
    Spacer(1, 10),
    para("Interpretation", "SubSection"),
    para(
        "Linear regression achieved MAE of approximately 0.798 tons, RMSE of "
        "approximately 0.963 tons, and R² of approximately 0.864. It performed "
        "better than the mean baseline and the tested random forest regressor "
        "on this synthetic test split."
    ),
    para(
        "These metrics measure performance on the generated dataset only. "
        "They do not demonstrate accuracy on real harvests. RMSE penalizes "
        "larger errors more strongly than MAE, while R² can be negative when "
        "a model performs worse than a constant mean predictor."
    ),
    para("Residual analysis", "SubSection"),
    add_image(story, "residuals.png"),
    para(
        "Residuals are the differences between observed and predicted yields. "
        "Their pattern can reveal systematic errors or unusually large mistakes. "
        "This plot should be interpreted alongside the numerical metrics."
    ),
    PageBreak()
]

# PAGE 5 — Classification results
story += [
    para("4. Classification Results", "Section"),
    para(
        "The selected classifier was random forest. Its test-set precision, "
        "recall, F1, and ROC-AUC were each 1.000 on the recorded split."
    ),
    make_table(metric_rows(
        clf, ["precision", "recall", "f1", "roc_auc"]
    ), [4.4*cm, 2.7*cm, 2.7*cm, 2.7*cm, 2.7*cm]),
    Spacer(1, 8),
    para("Confusion matrix", "SubSection"),
    add_image(story, "confusion_matrix.png", height=7.5*cm),
    para(
        "The confusion matrix compares predicted labels with actual test labels. "
        "False positives are high-risk predictions for records whose true label "
        "is negative; false negatives are missed high-risk records."
    ),
    para(
        "<b>Important caution:</b> Perfect test scores on a small synthetic "
        "dataset can result from the way the labels were generated. They are "
        "not evidence that the classifier will achieve perfect performance on "
        "actual farms."
    ),
    PageBreak()
]

# PAGE 6 — Clustering, API, testing, responsible AI
story += [
    para("5. Clustering, API, Testing and Responsible AI", "Section"),
    para("Clustering results", "SubSection"),
]
scores = cluster.get("silhouette_scores", cluster.get("scores_by_k", {}))
if isinstance(scores, dict) and scores:
    story.append(make_table(
        [["K", "Silhouette score"]] +
        [[str(k), fmt(v)] for k, v in scores.items()],
        [5*cm, 5*cm]
    ))
else:
    story.append(para(
        "The clustering report stores candidate silhouette scores and the "
        "selected cluster count. The highest observed score was approximately "
        "0.162 at K=2."
    ))
story += [
    Spacer(1, 6),
    para(
        "K=2 was selected, with a silhouette score of approximately 0.162. "
        "This low score suggests that clusters overlap and are not strongly "
        "separated. Cluster profiles should be treated as exploratory summaries."
    ),
    add_image(story, "clusters.png", height=6.2*cm),
    para("API and automated tests", "SubSection"),
    para(
        "The FastAPI service exposes GET /health, GET /model-info, and POST "
        "/predict. Input validation is intended to reject missing fields, "
        "incorrect types, extra fields, and out-of-range values. The latest "
        "recorded automated test run passed all six tests, including inference, "
        "validation, reproducibility, and API response checks."
    ),
    para("Responsible use and limitations", "SubSection"),
    para(
        "This is an educational prototype using synthetic data, not a production "
        "agricultural service. It may not represent actual weather, soil, crop "
        "varieties, pests, local farming methods, or seasonal conditions. Do not "
        "use its output as the sole basis for loans, insurance, aid allocation, "
        "or other high-impact decisions. Real-world use would require representative "
        "field data, independent validation, agricultural expert review, privacy "
        "protections, and ongoing monitoring. See docs/RESPONSIBLE_AI.md."
    ),
    Spacer(1, 5),
    para(
        "<b>Reproducibility:</b> Run "
        "<font name='Courier'>python run_all.py --student-id &quot;25/27790&quot; "
        "--seed 7790</font> from the project root using the project's Python "
        "environment to regenerate the data, models, metrics, plots, and tests."
    ),
]

doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(f"Report created: {OUT}")
print(f"Report size: {OUT.stat().st_size:,} bytes")
