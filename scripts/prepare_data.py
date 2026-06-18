"""Prepare the Telco churn dataset and build the drift reference window.

Steps:
1. Load the raw CSV.
2. Drop the ``customerID`` identifier column.
3. Coerce ``TotalCharges`` to float (blank/whitespace rows -> 0.0).
4. Encode ``Churn`` as binary int (Yes=1, No=0).
5. Persist the first 60% of rows as the drift *reference window* parquet.
6. Print class balance and null-count diagnostics.

Run:
    uv run python scripts/prepare_data.py
"""

from __future__ import annotations

from pathlib import Path

# Canonical cleaning lives in the pipeline module so training and data-prep agree.
from src.pipeline.preprocess import load_clean_data
from src.utils.config import settings
from src.utils.logging import configure_logging, get_logger

REFERENCE_FRACTION = 0.60


def main() -> None:
    configure_logging(dev_mode=settings.dev_mode)
    log = get_logger("prepare_data")

    raw_path = Path(settings.raw_data_path)
    ref_path = Path(settings.reference_data_path)

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {raw_path}. Download it first (see CLAUDE.md)."
        )

    df = load_clean_data(raw_path)
    log.info("data_loaded", rows=len(df), cols=df.shape[1], source=str(raw_path))

    # --- Diagnostics -------------------------------------------------------
    null_counts = df.isnull().sum()
    nulls = {c: int(n) for c, n in null_counts.items() if n > 0}

    churn_counts = df["Churn"].value_counts().to_dict()
    churn_rate = float(df["Churn"].mean())

    print("\n=== Null counts (non-zero only) ===")
    print(nulls if nulls else "No nulls.")

    print("\n=== Class balance (Churn) ===")
    print(f"  No  (0): {churn_counts.get(0, 0)}")
    print(f"  Yes (1): {churn_counts.get(1, 0)}")
    print(f"  Churn rate: {churn_rate:.4f}")

    print("\n=== Dtypes ===")
    print(df.dtypes.to_string())

    # --- Reference window --------------------------------------------------
    cutoff = int(len(df) * REFERENCE_FRACTION)
    reference = df.iloc[:cutoff].copy()

    ref_path.parent.mkdir(parents=True, exist_ok=True)
    reference.to_parquet(ref_path, index=False)

    log.info(
        "reference_saved",
        path=str(ref_path),
        reference_rows=len(reference),
        fraction=REFERENCE_FRACTION,
        reference_churn_rate=round(float(reference["Churn"].mean()), 4),
    )
    print(f"\nReference window ({len(reference)} rows) saved to {ref_path}")


if __name__ == "__main__":
    main()
