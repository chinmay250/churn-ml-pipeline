"""API tests — no live MLflow server required.

The model-loader dependency is overridden with a deterministic fake, and the
TestClient is NOT used as a context manager so the FastAPI lifespan (which would
try to reach MLflow) never runs.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app, get_model_loader
from src.utils.config import settings

client = TestClient(app)

VALID_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 89.10,
    "TotalCharges": 445.50,
}


class FakeLoader:
    """Deterministic stand-in: high charges -> churn."""

    is_loaded = True
    model_version = "test-1"

    def predict(self, df: pd.DataFrame):
        proba = [0.92 if df.iloc[0]["MonthlyCharges"] > 50 else 0.08]
        labels = [1 if p >= 0.5 else 0 for p in proba]
        return labels, proba


class NotLoadedLoader:
    is_loaded = False
    model_version = None

    def predict(self, df):  # pragma: no cover - should never be called
        raise AssertionError("predict called on a not-loaded model")


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


def _use(loader):
    app.dependency_overrides[get_model_loader] = lambda: loader


# --- /health ---------------------------------------------------------------

def test_health_reports_loaded():
    _use(FakeLoader())
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body == {"status": "ok", "model_loaded": True, "model_version": "test-1"}


def test_health_reports_not_loaded():
    _use(NotLoadedLoader())
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is False


# --- /predict --------------------------------------------------------------

def test_predict_valid():
    _use(FakeLoader())
    r = client.post("/predict", json=VALID_CUSTOMER)
    assert r.status_code == 200
    body = r.json()
    assert body["churn"] == 1
    assert body["churn_label"] == "Yes"
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["model_version"] == "test-1"
    assert "X-Request-ID" in r.headers


def test_predict_invalid_enum_returns_422():
    _use(FakeLoader())
    bad = {**VALID_CUSTOMER, "Contract": "Lifetime"}  # not an allowed value
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_missing_field_returns_422():
    _use(FakeLoader())
    bad = {k: v for k, v in VALID_CUSTOMER.items() if k != "tenure"}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_negative_charges_returns_422():
    _use(FakeLoader())
    bad = {**VALID_CUSTOMER, "MonthlyCharges": -5.0}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_503_when_model_not_loaded():
    _use(NotLoadedLoader())
    r = client.post("/predict", json=VALID_CUSTOMER)
    assert r.status_code == 503


# --- /drift/record ---------------------------------------------------------

def test_drift_record_appends(tmp_path, monkeypatch):
    log_file = tmp_path / "recorded.jsonl"
    monkeypatch.setattr(settings, "live_data_path", str(log_file))

    r1 = client.post("/drift/record", json=VALID_CUSTOMER)
    assert r1.status_code == 200
    assert r1.json() == {"recorded": True, "total_recorded": 1}

    r2 = client.post("/drift/record", json=VALID_CUSTOMER)
    assert r2.json()["total_recorded"] == 2
    assert log_file.exists()
    assert len(log_file.read_text().strip().splitlines()) == 2
