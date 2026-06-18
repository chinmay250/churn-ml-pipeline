"""Data loading, cleaning, and the sklearn preprocessing pipeline.

This module is the single source of truth for:
- turning the raw Telco CSV into a clean, modelling-ready DataFrame, and
- building the ``ColumnTransformer`` (StandardScaler for numerics, one-hot for
  categoricals) used by every model.

Both ``scripts/prepare_data.py`` and ``src/pipeline/train.py`` import from here
so the cleaning logic lives in exactly one place.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Churn"
ID_COLUMN = "customerID"

# Default split params — kept here so training and tests stay in sync.
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_clean_data(raw_path: str | Path) -> pd.DataFrame:
    """Load the raw Telco CSV and apply the standard cleaning transforms.

    - drop the ``customerID`` identifier
    - coerce ``TotalCharges`` to float (blank/whitespace rows -> 0.0)
    - encode ``Churn`` as binary int (Yes=1, No=0)
    """
    df = pd.read_csv(raw_path)

    if ID_COLUMN in df.columns:
        df = df.drop(columns=[ID_COLUMN])

    # TotalCharges holds ~11 single-space strings -> NaN -> 0.0.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    # Target -> binary int.
    df[TARGET] = (df[TARGET].astype(str).str.strip().str.lower() == "yes").astype(int)

    return df


def split_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numeric_columns, categorical_columns) for the feature matrix.

    Derived from dtypes so it stays correct if the schema shifts. ``SeniorCitizen``
    is already 0/1 int and lands in the numeric group (scaling a binary is harmless).
    """
    features = df.drop(columns=[TARGET])
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = features.select_dtypes(exclude="number").columns.tolist()
    return numeric, categorical


def build_preprocessor(
    numeric: list[str], categorical: list[str]
) -> ColumnTransformer:
    """Build the ColumnTransformer: scale numerics, one-hot encode categoricals."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split on the target. Returns X_train, X_test, y_train, y_test."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
