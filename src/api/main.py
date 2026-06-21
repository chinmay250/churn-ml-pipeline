"""FastAPI serving layer for the churn model.

Endpoints:
- GET  /health        — liveness + whether the model is loaded.
- POST /predict        — single-customer churn prediction.
- POST /drift/record   — append live features to the log for drift checks.
- GET  /drift/report   — score recorded live features against the reference distribution.

The model is loaded once at startup (lifespan). Every request is logged with
structlog (method, path, status, duration). The model is provided via a FastAPI
dependency so tests can override it without a live MLflow server.
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request

from src.api.model_loader import ModelLoader, model_loader
from src.api.schemas import (
    CustomerFeatures,
    HealthResponse,
    PredictionResponse,
    RecordResponse,
)
from src.drift.alerts import handle_drift
from src.drift.monitor import DriftMonitor, load_live_features
from src.utils.config import settings
from src.utils.logging import configure_logging, get_logger

log = get_logger("api")

# Lazily-built reference monitor, shared across requests (overridable in tests).
_drift_monitor: DriftMonitor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(dev_mode=settings.dev_mode)
    # Best-effort load: if MLflow is unreachable the app still starts so /health
    # can report not-ready (and tests that override the dependency still work).
    try:
        model_loader.load()
    except Exception as exc:
        log.warning("startup_model_load_failed", error=str(exc), uri=model_loader.model_uri)
    yield


app = FastAPI(title="Churn Prediction API", version="0.1.0", lifespan=lifespan)


def get_model_loader() -> ModelLoader:
    """Dependency — overridden in tests with a fake loader."""
    return model_loader


def get_drift_monitor() -> DriftMonitor:
    """Dependency — builds the reference monitor on first use, then caches it."""
    global _drift_monitor
    if _drift_monitor is None:
        _drift_monitor = DriftMonitor.from_reference_path()
    return _drift_monitor


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        log.exception(
            "request_failed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            duration_ms=duration_ms,
        )
        raise
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    log.info(
        "request",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=duration_ms,
    )
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/health", response_model=HealthResponse)
def health(loader: ModelLoader = Depends(get_model_loader)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=loader.is_loaded,
        model_version=loader.model_version,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(
    features: CustomerFeatures,
    loader: ModelLoader = Depends(get_model_loader),
) -> PredictionResponse:
    if not loader.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = pd.DataFrame([features.model_dump()])
    labels, probas = loader.predict(df)
    churn = int(labels[0])
    return PredictionResponse(
        churn=churn,
        churn_label="Yes" if churn == 1 else "No",
        churn_probability=round(probas[0], 6),
        model_version=loader.model_version,
    )


@app.post("/drift/record", response_model=RecordResponse)
def record(features: CustomerFeatures) -> RecordResponse:
    """Append the raw features to the live log (JSONL) for later drift analysis."""
    path = Path(settings.live_data_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(features.model_dump()) + "\n")
    total = sum(1 for _ in path.open())
    log.info("feature_recorded", path=str(path), total_recorded=total)
    return RecordResponse(recorded=True, total_recorded=total)


@app.get("/drift/report")
def drift_report(monitor: DriftMonitor = Depends(get_drift_monitor)) -> dict:
    """Score recorded live features against the reference distribution.

    Returns the DriftReport plus an alert summary (retraining is NOT auto-triggered
    from this read-only endpoint — ``auto_retrain=False``).
    """
    current = load_live_features()
    if current.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No recorded features at {settings.live_data_path}; POST to /drift/record first.",
        )
    report = monitor.check(current)
    alert = handle_drift(report, auto_retrain=False)
    return {"report": report.to_dict(), "alert": alert}
