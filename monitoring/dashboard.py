"""Streamlit monitoring dashboard.

Three panels:
1. Model metrics — runs from the MLflow ``churn-prediction`` experiment + the
   current Production model version.
2. Current drift — PSI/KS per feature for the recorded live traffic vs reference.
3. Drift over time — drift_share trend, persisted to a history parquet on each run.

Run locally:
    uv run streamlit run monitoring/dashboard.py
In Docker the MLflow URI comes from the MLFLOW_TRACKING_URI env var.
"""

from __future__ import annotations

import os

import mlflow
import pandas as pd
import streamlit as st
from mlflow import MlflowClient

from src.drift.alerts import RETRAIN_DRIFT_SHARE, should_retrain
from src.drift.monitor import (
    DriftMonitor,
    append_drift_snapshot,
    load_live_features,
)
from src.utils.config import settings

EXPERIMENT_NAME = "churn-prediction"

st.set_page_config(page_title="Churn Model Monitoring", page_icon="📈", layout="wide")
st.title("📈 Churn Model Monitoring")

tracking_uri = settings.mlflow_tracking_uri
mlflow.set_tracking_uri(tracking_uri)
st.caption(f"MLflow: `{tracking_uri}` · reference: `{settings.reference_data_path}`")

if st.button("🔄 Refresh"):
    st.cache_data.clear()


# --- 1. Model metrics ------------------------------------------------------

@st.cache_data(ttl=30)
def fetch_runs() -> pd.DataFrame:
    client = MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        return pd.DataFrame()
    runs = client.search_runs([exp.experiment_id], order_by=["start_time DESC"])
    return pd.DataFrame(
        [
            {
                "run": r.data.tags.get("mlflow.runName", r.info.run_id[:8]),
                "roc_auc": r.data.metrics.get("roc_auc"),
                "f1": r.data.metrics.get("f1"),
                "precision": r.data.metrics.get("precision"),
                "recall": r.data.metrics.get("recall"),
                "accuracy": r.data.metrics.get("accuracy"),
            }
            for r in runs
        ]
    )


@st.cache_data(ttl=30)
def fetch_production_version() -> str | None:
    try:
        mv = MlflowClient().get_model_version_by_alias(settings.model_name, settings.model_stage)
        return str(mv.version)
    except Exception:
        return None


st.header("Model metrics")
runs_df = fetch_runs()
if runs_df.empty:
    st.warning(f"No MLflow runs found for experiment '{EXPERIMENT_NAME}'. Train a model first.")
else:
    prod_version = fetch_production_version()
    col1, col2 = st.columns([1, 2])
    with col1:
        best = runs_df.loc[runs_df["roc_auc"].idxmax()]
        st.metric("Best ROC-AUC", f"{best['roc_auc']:.4f}", help=f"run: {best['run']}")
        st.metric(
            f"{settings.model_name} @ {settings.model_stage}",
            f"v{prod_version}" if prod_version else "not registered",
        )
    with col2:
        st.bar_chart(runs_df.set_index("run")[["roc_auc", "f1"]])
    st.dataframe(runs_df, width="stretch", hide_index=True)


# --- 2 & 3. Drift ----------------------------------------------------------

st.header("Data drift")


@st.cache_data(ttl=15)
def compute_drift() -> tuple[dict | None, pd.DataFrame, pd.DataFrame]:
    current = load_live_features()
    if current.empty:
        return None, pd.DataFrame(), pd.DataFrame()
    monitor = DriftMonitor.from_reference_path()
    report = monitor.check(current)
    feat_df = pd.DataFrame([f.__dict__ for f in report.features])
    history = append_drift_snapshot(report)  # persist a snapshot for the trend
    report_d = report.to_dict()
    report_d["_should_retrain"] = should_retrain(report)
    return report_d, feat_df, history


report_d, feat_df, history = compute_drift()

if report_d is None:
    st.info(
        f"No recorded live features at `{settings.live_data_path}`. "
        "Generate some with `scripts/simulate_drift.py`."
    )
else:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows scored", report_d["n_current_rows"])
    c2.metric("Features drifted", f"{report_d['n_drifted']} / {report_d['n_features']}")
    c3.metric("Drift share", f"{report_d['drift_share']:.2f}")
    c4.metric(
        "Dataset drift",
        "YES" if report_d["dataset_drift"] else "no",
        delta="retrain" if report_d["_should_retrain"] else None,
        delta_color="inverse",
    )
    if report_d["_should_retrain"]:
        st.error(f"Drift share ≥ {RETRAIN_DRIFT_SHARE} → retraining condition met.")

    st.subheader("Per-feature drift")
    show = feat_df[["feature", "kind", "metric", "statistic", "p_value", "threshold", "drifted"]]
    st.dataframe(
        show.style.apply(
            lambda r: ["background-color: #ffd9d9" if r["drifted"] else "" for _ in r], axis=1
        ),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Drift over time")
    if not history.empty:
        trend = history.copy()
        trend["timestamp"] = pd.to_datetime(trend["timestamp"])
        st.line_chart(trend.set_index("timestamp")[["drift_share"]])
        st.caption(f"{len(history)} snapshots recorded in `{settings.drift_history_path}`.")


# --- optional API health ---------------------------------------------------

api_url = os.getenv("API_URL")
if api_url:
    import urllib.request

    try:
        with urllib.request.urlopen(f"{api_url}/health", timeout=2) as resp:
            st.sidebar.success(f"API reachable: {api_url}")
            st.sidebar.json(resp.read().decode())
    except Exception as exc:  # pragma: no cover
        st.sidebar.error(f"API unreachable at {api_url}: {exc}")
