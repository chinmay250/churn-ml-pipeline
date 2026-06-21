"""Application configuration loaded from environment / .env file.

Usage:
    from src.utils.config import settings
    print(settings.mlflow_tracking_uri)
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central settings object, populated from environment variables / .env.

    Field names map to upper-case env vars (case-insensitive), e.g.
    ``mlflow_tracking_uri`` <- ``MLFLOW_TRACKING_URI``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        # model_name / model_stage would otherwise collide with pydantic's
        # protected "model_" namespace and emit warnings.
        protected_namespaces=(),
    )

    # MLflow — 5001 locally because macOS Control Center (AirPlay) squats on 5000.
    mlflow_tracking_uri: str = "http://localhost:5001"
    model_name: str = "ChurnModel"
    model_stage: str = "Production"

    # Drift thresholds
    drift_psi_threshold: float = 0.2
    drift_ks_pvalue_threshold: float = 0.05

    # Data paths
    reference_data_path: str = "data/reference/reference_data.parquet"
    raw_data_path: str = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    # Where /drift/record appends live production features (consumed in Session 4).
    live_data_path: str = "data/live/recorded_features.jsonl"
    # Time series of drift snapshots, plotted by the Streamlit dashboard (Session 5).
    drift_history_path: str = "data/live/drift_history.parquet"

    # Logging
    dev_mode: bool = True


# Import-time singleton — import this everywhere rather than re-instantiating.
settings = Settings()
