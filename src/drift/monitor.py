"""Drift monitoring orchestrator.

``DriftMonitor`` holds the reference distribution and scores a batch of current
data feature-by-feature:
- numerical features  -> KS test (drift if p-value < ks threshold) + PSI (informational)
- categorical features -> categorical PSI (drift if PSI > psi threshold)

It returns a ``DriftReport`` (JSON-serialisable) summarising per-feature results
and an overall dataset-drift verdict.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.drift.detectors import categorical_psi, ks_test, psi_score
from src.utils.config import settings
from src.utils.logging import get_logger

log = get_logger("drift.monitor")

TARGET = "Churn"
# Continuous columns get the KS test; everything else (minus target) is categorical.
NUMERICAL_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]


@dataclass
class FeatureDrift:
    feature: str
    kind: str  # "numerical" | "categorical"
    statistic: float  # KS statistic or PSI value
    metric: str  # "ks" | "psi"
    drifted: bool
    threshold: float
    p_value: float | None = None
    psi: float | None = None  # extra PSI for numerical features (informational)


@dataclass
class DriftReport:
    timestamp: str
    n_current_rows: int
    n_features: int
    n_drifted: int
    drift_share: float
    dataset_drift: bool
    features: list[FeatureDrift] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def drifted_features(self) -> list[str]:
        return [f.feature for f in self.features if f.drifted]


class DriftMonitor:
    """Scores current data against a fixed reference distribution."""

    def __init__(
        self,
        reference: pd.DataFrame,
        psi_threshold: float | None = None,
        ks_pvalue_threshold: float | None = None,
        numerical_features: list[str] | None = None,
    ):
        self.reference = reference.drop(columns=[TARGET], errors="ignore")
        self.psi_threshold = (
            psi_threshold if psi_threshold is not None else settings.drift_psi_threshold
        )
        self.ks_pvalue_threshold = (
            ks_pvalue_threshold
            if ks_pvalue_threshold is not None
            else settings.drift_ks_pvalue_threshold
        )
        self.numerical = numerical_features or [
            c for c in NUMERICAL_FEATURES if c in self.reference.columns
        ]
        self.categorical = [
            c for c in self.reference.columns if c not in self.numerical
        ]

    @classmethod
    def from_reference_path(cls, path: str | Path | None = None, **kwargs) -> "DriftMonitor":
        path = Path(path or settings.reference_data_path)
        return cls(pd.read_parquet(path), **kwargs)

    def check(self, current: pd.DataFrame) -> DriftReport:
        """Score ``current`` against the reference. Returns a DriftReport."""
        results: list[FeatureDrift] = []

        for col in self.numerical:
            if col not in current.columns:
                continue
            ref = pd.to_numeric(self.reference[col], errors="coerce").dropna()
            cur = pd.to_numeric(current[col], errors="coerce").dropna()
            stat, p_value = ks_test(ref, cur)
            psi = psi_score(ref, cur)
            results.append(
                FeatureDrift(
                    feature=col,
                    kind="numerical",
                    statistic=round(stat, 6),
                    metric="ks",
                    drifted=p_value < self.ks_pvalue_threshold,
                    threshold=self.ks_pvalue_threshold,
                    p_value=round(p_value, 6),
                    psi=round(psi, 6),
                )
            )

        for col in self.categorical:
            if col not in current.columns:
                continue
            psi = categorical_psi(self.reference[col], current[col])
            results.append(
                FeatureDrift(
                    feature=col,
                    kind="categorical",
                    statistic=round(psi, 6),
                    metric="psi",
                    drifted=psi > self.psi_threshold,
                    threshold=self.psi_threshold,
                )
            )

        n_drifted = sum(f.drifted for f in results)
        n_features = len(results)
        report = DriftReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            n_current_rows=len(current),
            n_features=n_features,
            n_drifted=n_drifted,
            drift_share=round(n_drifted / n_features, 4) if n_features else 0.0,
            dataset_drift=n_drifted > 0,
            features=results,
        )
        log.info(
            "drift_checked",
            n_current_rows=report.n_current_rows,
            n_drifted=report.n_drifted,
            drift_share=report.drift_share,
            drifted_features=report.drifted_features,
        )
        return report


def load_live_features(path: str | Path | None = None) -> pd.DataFrame:
    """Load recorded production features (JSONL) into a DataFrame.

    Returns an empty DataFrame if the file is missing or has no rows.
    """
    path = Path(path or settings.live_data_path)
    if not path.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return pd.DataFrame(rows)


def append_drift_snapshot(report: DriftReport, path: str | Path | None = None) -> pd.DataFrame:
    """Append a one-row summary of ``report`` to the drift-history parquet.

    Returns the full history DataFrame. Used by the dashboard to plot drift over time.
    """
    path = Path(path or settings.drift_history_path)
    row = {
        "timestamp": report.timestamp,
        "n_current_rows": report.n_current_rows,
        "n_drifted": report.n_drifted,
        "drift_share": report.drift_share,
        "dataset_drift": report.dataset_drift,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        history = pd.concat([pd.read_parquet(path), pd.DataFrame([row])], ignore_index=True)
    else:
        history = pd.DataFrame([row])
    history.to_parquet(path, index=False)
    return history
