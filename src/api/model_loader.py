"""Load and serve the registered churn model from the MLflow registry.

Loads via the **sklearn** flavor (not pyfunc) so we get ``predict_proba`` — the
logged object is the full Pipeline (preprocessing + estimator), so it consumes
the raw feature columns directly.

A module-level singleton ``model_loader`` is shared by the API; the FastAPI
lifespan calls ``.load()`` once at startup. Tests override the FastAPI
dependency with a fake, so no live MLflow server is needed for them.
"""

from __future__ import annotations

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient

from src.utils.config import settings
from src.utils.logging import get_logger

log = get_logger("model_loader")

DECISION_THRESHOLD = 0.5


class ModelLoader:
    """Lazily loads the Production-aliased model and serves predictions."""

    def __init__(self, model_name: str, alias: str):
        self.model_name = model_name
        self.alias = alias
        self._model = None
        self._version: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_version(self) -> str | None:
        return self._version

    @property
    def model_uri(self) -> str:
        return f"models:/{self.model_name}@{self.alias}"

    def load(self) -> "ModelLoader":
        """Load the model + resolve its registry version. Raises on failure."""
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        self._model = mlflow.sklearn.load_model(self.model_uri)
        try:
            mv = MlflowClient().get_model_version_by_alias(self.model_name, self.alias)
            self._version = str(mv.version)
        except Exception as exc:  # version is informational; don't fail the load
            log.warning("version_resolve_failed", error=str(exc))
            self._version = None
        log.info("model_loaded", uri=self.model_uri, version=self._version)
        return self

    def predict(self, features: pd.DataFrame) -> tuple[list[int], list[float]]:
        """Return (class_labels, churn_probabilities) for a feature DataFrame."""
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded")
        proba = self._model.predict_proba(features)[:, 1]
        labels = (proba >= DECISION_THRESHOLD).astype(int)
        return labels.tolist(), [float(p) for p in proba]


# Shared singleton — loaded by the FastAPI lifespan at startup.
model_loader = ModelLoader(settings.model_name, settings.model_stage)
