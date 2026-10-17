from fastapi.testclient import TestClient

from app import app
from predict import predict

client = TestClient(app)

VALID_INPUT = {
    "rainfall_mm": 950,
    "avg_temp_c": 18,
    "soil_moisture_pct": 50,
    "fertilizer_kg": 120,
    "field_size_ha": 1.5,
    "season": "A",
    "cooperative_support": 1,
}


def test_valid_prediction():
    response = client.post("/predict", json=VALID_INPUT)
    assert response.status_code == 200
    body = response.json()
    assert body["verification_code"] == "AI-A2-7790"
    assert isinstance(body["predicted_harvest_tons"], float)
    assert body["high_risk_prediction"] in (0, 1)
    assert 0 <= body["high_risk_probability"] <= 1
    assert isinstance(body["cluster"], int)


def test_missing_field_is_rejected():
    data = VALID_INPUT.copy()
    del data["rainfall_mm"]
    response = client.post("/predict", json=data)
    assert response.status_code == 422


def test_wrong_type_is_rejected():
    data = VALID_INPUT.copy()
    data["rainfall_mm"] = "rainy"
    response = client.post("/predict", json=data)
    assert response.status_code == 422


def test_out_of_range_value_is_rejected():
    data = VALID_INPUT.copy()
    data["rainfall_mm"] = 2000
    response = client.post("/predict", json=data)
    assert response.status_code == 422


def test_prediction_is_reproducible():
    first = predict(VALID_INPUT)
    second = predict(VALID_INPUT)
    assert first == second


def test_health_and_model_info_endpoints():
    health = client.get("/health")
    info = client.get("/model-info")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["verification_code"] == "AI-A2-7790"
    assert info.status_code == 200
    assert info.json()["verification_code"] == "AI-A2-7790"
