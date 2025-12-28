# tests/test_api.py
import os
from fastapi.testclient import TestClient

# point to your local artifacts for tests
os.environ.setdefault("MODEL_PATH", "artifacts/catboost_model.cbm")
os.environ.setdefault("META_PATH",  "artifacts/model_meta.joblib")

from api import app


def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_predict_smoke():
    payload = {
        "rows": [{
            "Age": 40,
            "Gender": "Male",
            "Tenure": 12,
            "Usage Frequency": 10,
            "Support Calls": 2,
            "Payment Delay": 3,
            "Subscription Type": "Basic",
            "Contract Length": "Monthly",
            "Total Spend": 500,
            "Last Interaction": 5
        }],
        "threshold": 0.5
    }

    with TestClient(app) as client:
        r = client.post("/predict", json=payload)
        assert r.status_code == 200, r.text
        out = r.json()
        assert 0.0 <= out["probabilities"][0] <= 1.0
        assert out["predictions"][0] in [0, 1]


def test_predict_empty_rows():
    with TestClient(app) as client:
        r = client.post("/predict", json={"rows": []})
        assert r.status_code == 400, r.text
