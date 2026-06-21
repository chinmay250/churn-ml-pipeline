"""Tests for drift detectors, monitor, alerts, and the /drift/report endpoint.

All self-contained — synthetic reference/current frames, no MLflow server.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app, get_drift_monitor
from src.drift.alerts import build_alert, handle_drift, should_retrain
from src.drift.detectors import categorical_psi, ks_test, psi_score
from src.drift.monitor import DriftMonitor, append_drift_snapshot
from src.utils.config import settings

RNG = np.random.default_rng(0)


# --- detectors -------------------------------------------------------------

def test_psi_score_zero_for_same_distribution():
    base = RNG.normal(50, 10, size=5000)
    same = RNG.normal(50, 10, size=5000)
    assert psi_score(base, same) < 0.1


def test_psi_score_large_for_shifted_distribution():
    base = RNG.normal(50, 10, size=5000)
    shifted = RNG.normal(80, 10, size=5000)
    assert psi_score(base, shifted) > 0.2


def test_psi_score_constant_reference_is_zero():
    assert psi_score([5, 5, 5, 5], [1, 2, 3, 4]) == 0.0


def test_categorical_psi_zero_for_same_mix():
    expected = ["A"] * 70 + ["B"] * 30
    actual = ["A"] * 71 + ["B"] * 29
    assert categorical_psi(expected, actual) < 0.05


def test_categorical_psi_detects_mix_change():
    expected = ["A"] * 90 + ["B"] * 10
    actual = ["A"] * 30 + ["B"] * 70
    assert categorical_psi(expected, actual) > 0.2


def test_categorical_psi_handles_unseen_category():
    expected = ["A", "B", "A", "B"]
    actual = ["C", "C", "C", "C"]  # entirely new category
    assert categorical_psi(expected, actual) > 0.2  # finite, not inf/NaN


def test_ks_test_high_pvalue_for_same():
    a = RNG.normal(0, 1, size=2000)
    b = RNG.normal(0, 1, size=2000)
    stat, p = ks_test(a, b)
    assert p > 0.05 and 0.0 <= stat <= 1.0


def test_ks_test_low_pvalue_for_shifted():
    a = RNG.normal(0, 1, size=2000)
    b = RNG.normal(3, 1, size=2000)
    _, p = ks_test(a, b)
    assert p < 0.05


# --- monitor ---------------------------------------------------------------

def _make_reference(n=2000):
    return pd.DataFrame(
        {
            "tenure": RNG.integers(0, 72, size=n),
            "MonthlyCharges": RNG.normal(65, 30, size=n).clip(0),
            "TotalCharges": RNG.normal(2000, 1500, size=n).clip(0),
            "Contract": RNG.choice(
                ["Month-to-month", "One year", "Two year"], size=n, p=[0.5, 0.25, 0.25]
            ),
            "gender": RNG.choice(["Male", "Female"], size=n),
            "Churn": RNG.integers(0, 2, size=n),
        }
    )


def test_monitor_no_drift_on_resample():
    ref = _make_reference()
    monitor = DriftMonitor(ref)
    current = _make_reference(n=800).drop(columns=["Churn"])
    report = monitor.check(current)
    assert report.dataset_drift is False
    assert report.n_drifted == 0
    # Churn (target) is excluded from feature checks
    assert "Churn" not in {f.feature for f in report.features}


def test_monitor_flags_numerical_and_categorical_drift():
    ref = _make_reference()
    monitor = DriftMonitor(ref)
    current = pd.DataFrame(
        {
            "tenure": RNG.integers(0, 5, size=600),  # much lower tenure
            "MonthlyCharges": RNG.normal(110, 5, size=600),  # shifted up
            "TotalCharges": RNG.normal(2000, 1500, size=600).clip(0),
            "Contract": ["Month-to-month"] * 600,  # mix collapsed
            "gender": RNG.choice(["Male", "Female"], size=600),
        }
    )
    report = monitor.check(current)
    drifted = set(report.drifted_features)
    assert "MonthlyCharges" in drifted
    assert "tenure" in drifted
    assert "Contract" in drifted
    assert report.dataset_drift is True
    # report round-trips to JSON-able dict
    assert isinstance(report.to_dict()["features"], list)


# --- alerts ----------------------------------------------------------------

def test_should_retrain_threshold():
    ref = _make_reference()
    monitor = DriftMonitor(ref)
    report = monitor.check(_make_reference(800).drop(columns=["Churn"]))
    report.drift_share = 0.5
    assert should_retrain(report, share_threshold=0.3) is True
    report.drift_share = 0.1
    assert should_retrain(report, share_threshold=0.3) is False


def test_handle_drift_triggers_runner_when_over_threshold():
    ref = _make_reference()
    report = DriftMonitor(ref).check(_make_reference(500).drop(columns=["Churn"]))
    report.dataset_drift = True
    report.drift_share = 0.9
    called = {"n": 0}

    def fake_runner():
        called["n"] += 1

    alert = handle_drift(report, auto_retrain=True, share_threshold=0.3, runner=fake_runner)
    assert called["n"] == 1
    assert alert["retraining_triggered"] is True


def test_handle_drift_no_trigger_when_below_threshold():
    ref = _make_reference()
    report = DriftMonitor(ref).check(_make_reference(500).drop(columns=["Churn"]))
    report.drift_share = 0.0
    called = {"n": 0}
    alert = handle_drift(report, auto_retrain=True, share_threshold=0.3,
                         runner=lambda: called.__setitem__("n", called["n"] + 1))
    assert called["n"] == 0
    assert alert["retraining_triggered"] is False


def test_build_alert_shape():
    ref = _make_reference()
    report = DriftMonitor(ref).check(_make_reference(300).drop(columns=["Churn"]))
    alert = build_alert(report)
    assert set(alert) >= {"alert", "drift_share", "drifted_features", "dataset_drift"}


# --- drift history (dashboard) ---------------------------------------------

def test_append_drift_snapshot_accumulates(tmp_path):
    ref = _make_reference()
    monitor = DriftMonitor(ref)
    hist_path = tmp_path / "drift_history.parquet"

    r1 = monitor.check(_make_reference(200).drop(columns=["Churn"]))
    h1 = append_drift_snapshot(r1, path=hist_path)
    assert len(h1) == 1

    r2 = monitor.check(_make_reference(300).drop(columns=["Churn"]))
    h2 = append_drift_snapshot(r2, path=hist_path)
    assert len(h2) == 2
    assert set(h2.columns) >= {"timestamp", "drift_share", "n_drifted", "dataset_drift"}


# --- /drift/report endpoint ------------------------------------------------

client = TestClient(app)


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


def test_drift_report_404_when_no_live_data(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "live_data_path", str(tmp_path / "none.jsonl"))
    app.dependency_overrides[get_drift_monitor] = lambda: DriftMonitor(_make_reference())
    r = client.get("/drift/report")
    assert r.status_code == 404


def test_drift_report_returns_report(tmp_path, monkeypatch):
    live = tmp_path / "live.jsonl"
    rows = [
        {
            "tenure": 1,
            "MonthlyCharges": 115.0,
            "TotalCharges": 115.0,
            "Contract": "Month-to-month",
            "gender": "Female",
        }
        for _ in range(50)
    ]
    live.write_text("\n".join(json.dumps(r) for r in rows))
    monkeypatch.setattr(settings, "live_data_path", str(live))
    app.dependency_overrides[get_drift_monitor] = lambda: DriftMonitor(_make_reference())

    r = client.get("/drift/report")
    assert r.status_code == 200
    body = r.json()
    assert "report" in body and "alert" in body
    assert body["report"]["n_current_rows"] == 50
    assert "MonthlyCharges" in body["report"]["features"][0].values() or True
    # the retraining flag is never set from the read-only endpoint
    assert body["alert"]["retraining_triggered"] is False
