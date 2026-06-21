"""Simulate production traffic — optionally drifted — against the running API.

Samples real customers from the reference parquet, optionally mutates their
features to induce distribution drift, then POSTs each to ``/predict`` and
``/drift/record``. After recording, hit ``GET /drift/report`` to see PSI/KS fire.

Examples:
    # 200 drifted customers (high charges, low tenure, all month-to-month/fiber)
    uv run python scripts/simulate_drift.py --n 200 --drift

    # 200 in-distribution customers (should NOT drift)
    uv run python scripts/simulate_drift.py --n 200 --no-drift

Requires the API running (uvicorn on --url, default http://127.0.0.1:8000).
"""

from __future__ import annotations

import argparse

import httpx
import numpy as np
import pandas as pd

from src.pipeline.preprocess import load_clean_data
from src.utils.config import settings

RNG = np.random.default_rng(7)


def _drift_row(row: dict) -> dict:
    """Push a customer toward the high-churn-risk corner of feature space."""
    row["tenure"] = int(RNG.integers(0, 4))
    row["MonthlyCharges"] = round(float(RNG.normal(110, 6)), 2)
    row["TotalCharges"] = round(row["MonthlyCharges"] * max(row["tenure"], 1), 2)
    row["Contract"] = "Month-to-month"
    row["InternetService"] = "Fiber optic"
    row["PaymentMethod"] = "Electronic check"
    row["PaperlessBilling"] = "Yes"
    return row


def _json_safe(row: dict) -> dict:
    """Coerce numpy scalar types to native Python for JSON serialisation."""
    out = {}
    for k, v in row.items():
        if isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate (drifted) traffic to the churn API.")
    parser.add_argument("--n", type=int, default=200, help="number of customers to send")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument(
        "--drift", dest="drift", action="store_true", help="mutate features to induce drift"
    )
    parser.add_argument("--no-drift", dest="drift", action="store_false")
    parser.set_defaults(drift=True)
    args = parser.parse_args()

    df = load_clean_data(settings.raw_data_path).drop(columns=["Churn"])
    sample = df.sample(n=args.n, replace=True, random_state=7).reset_index(drop=True)

    probs: list[float] = []
    with httpx.Client(base_url=args.url, timeout=10.0) as client:
        for _, raw in sample.iterrows():
            row = _json_safe(raw.to_dict())
            if args.drift:
                row = _drift_row(row)

            pred = client.post("/predict", json=row)
            pred.raise_for_status()
            probs.append(pred.json()["churn_probability"])

            rec = client.post("/drift/record", json=row)
            rec.raise_for_status()

        report = client.get("/drift/report")

    mode = "DRIFTED" if args.drift else "in-distribution"
    print(f"\nSent {args.n} {mode} customers to {args.url}")
    print(f"  mean churn probability: {np.mean(probs):.3f}")
    if report.status_code == 200:
        rep = report.json()["report"]
        print(f"  drift report: {rep['n_drifted']}/{rep['n_features']} features drifted "
              f"(share={rep['drift_share']}), dataset_drift={rep['dataset_drift']}")
        if rep["features"]:
            drifted = [f["feature"] for f in rep["features"] if f["drifted"]]
            print(f"  drifted features: {drifted}")


if __name__ == "__main__":
    main()
