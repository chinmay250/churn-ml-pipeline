"""Tests for the preprocessing + evaluation pipeline.

These run WITHOUT an MLflow server — they exercise the pure data/model logic.
If the raw dataset isn't present the data-dependent tests skip rather than fail.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.pipeline.evaluate import compute_metrics
from src.pipeline.preprocess import (
    TARGET,
    build_preprocessor,
    load_clean_data,
    split_data,
    split_feature_columns,
)
from src.utils.config import settings

RAW_PATH = Path(settings.raw_data_path)
needs_data = pytest.mark.skipif(
    not RAW_PATH.exists(), reason=f"raw dataset missing at {RAW_PATH}"
)


@pytest.fixture(scope="module")
def clean_df():
    return load_clean_data(RAW_PATH)


@needs_data
def test_load_clean_data_schema(clean_df):
    # customerID dropped, target binarised, no nulls, TotalCharges numeric.
    assert "customerID" not in clean_df.columns
    assert clean_df[TARGET].dtype.kind == "i"
    assert set(clean_df[TARGET].unique()) <= {0, 1}
    assert clean_df["TotalCharges"].dtype.kind == "f"
    assert clean_df.isnull().sum().sum() == 0
    assert clean_df.shape == (7043, 20)


@needs_data
def test_split_feature_columns_excludes_target(clean_df):
    numeric, categorical = split_feature_columns(clean_df)
    assert TARGET not in numeric and TARGET not in categorical
    assert "tenure" in numeric and "MonthlyCharges" in numeric
    assert "Contract" in categorical and "gender" in categorical
    # every feature column is accounted for exactly once
    assert len(numeric) + len(categorical) == clean_df.shape[1] - 1


@needs_data
def test_preprocessor_transforms_to_dense_numeric(clean_df):
    numeric, categorical = split_feature_columns(clean_df)
    pre = build_preprocessor(numeric, categorical)
    X = clean_df.drop(columns=[TARGET])
    out = pre.fit_transform(X)
    assert out.shape[0] == len(clean_df)
    assert out.shape[1] > len(numeric)  # OHE expanded the categoricals
    assert not np.isnan(out).any()


@needs_data
def test_stratified_split_preserves_class_ratio(clean_df):
    X_train, X_test, y_train, y_test = split_data(clean_df)
    assert len(X_train) + len(X_test) == len(clean_df)
    # stratify keeps the churn rate close between splits
    assert abs(y_train.mean() - y_test.mean()) < 0.02


@needs_data
def test_end_to_end_pipeline_beats_baseline(clean_df):
    numeric, categorical = split_feature_columns(clean_df)
    X_train, X_test, y_train, y_test = split_data(clean_df)
    pipe = Pipeline(
        [
            ("preprocess", build_preprocessor(numeric, categorical)),
            ("model", RandomForestClassifier(n_estimators=50, random_state=42)),
        ]
    )
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test.to_numpy(), y_pred, y_proba)

    assert set(metrics) == {"roc_auc", "f1", "accuracy", "precision", "recall"}
    assert all(0.0 <= v <= 1.0 for v in metrics.values())
    # a trained model should clear random guessing comfortably
    assert metrics["roc_auc"] > 0.75


def test_compute_metrics_perfect_prediction():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_proba = np.array([0.1, 0.9, 0.2, 0.8])
    metrics = compute_metrics(y_true, y_pred, y_proba)
    assert metrics["roc_auc"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["accuracy"] == 1.0
