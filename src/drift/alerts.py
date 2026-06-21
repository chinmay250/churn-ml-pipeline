"""Drift alerting + retraining trigger.

When a ``DriftReport`` crosses the retrain condition we log a structured alert
and (optionally) launch ``src.pipeline.train`` as a background subprocess.

The subprocess launcher is injectable (``runner``) so tests can assert the
trigger fires without actually retraining.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable

from src.drift.monitor import DriftReport
from src.utils.logging import get_logger

log = get_logger("drift.alerts")

# Retrain once the share of drifted features crosses this fraction.
RETRAIN_DRIFT_SHARE = 0.3


def build_alert(report: DriftReport) -> dict:
    """Assemble a structured alert payload from a report."""
    return {
        "alert": "data_drift",
        "timestamp": report.timestamp,
        "dataset_drift": report.dataset_drift,
        "n_drifted": report.n_drifted,
        "n_features": report.n_features,
        "drift_share": report.drift_share,
        "drifted_features": report.drifted_features,
    }


def should_retrain(report: DriftReport, share_threshold: float = RETRAIN_DRIFT_SHARE) -> bool:
    """Trigger retraining when enough of the feature set has drifted."""
    return report.drift_share >= share_threshold


def trigger_retraining() -> subprocess.Popen:
    """Launch training as a detached background subprocess (non-blocking)."""
    log.info("retraining_triggered", cmd="python -m src.pipeline.train")
    return subprocess.Popen([sys.executable, "-m", "src.pipeline.train"])


def handle_drift(
    report: DriftReport,
    auto_retrain: bool = True,
    share_threshold: float = RETRAIN_DRIFT_SHARE,
    runner: Callable[[], object] = trigger_retraining,
) -> dict:
    """Log an alert if drift is present and optionally trigger retraining.

    Returns the alert payload augmented with ``retraining_triggered``.
    """
    alert = build_alert(report)
    retrain = should_retrain(report, share_threshold)
    alert["retraining_triggered"] = False

    if report.dataset_drift:
        log.warning("drift_alert", **alert)

    if retrain and auto_retrain:
        runner()
        alert["retraining_triggered"] = True
        log.warning("drift_retrain", drift_share=report.drift_share, threshold=share_threshold)

    return alert
