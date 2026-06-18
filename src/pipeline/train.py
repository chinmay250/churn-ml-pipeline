"""Training entry point: fit candidate models, track in MLflow, register the best.

Trains a RandomForest baseline and an XGBoost model, logs params/metrics/plots
for each as MLflow runs, then registers the higher ROC-AUC model to the MLflow
Model Registry as ``<model_name>`` and assigns it the ``Production`` alias.

Run (needs an MLflow tracking server up — see CLAUDE.md "Useful Commands"):
    uv run python -m src.pipeline.train
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.pipeline.evaluate import (
    compute_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
)
from src.pipeline.preprocess import (
    build_preprocessor,
    load_clean_data,
    split_data,
    split_feature_columns,
)
from src.utils.config import settings
from src.utils.logging import configure_logging, get_logger

EXPERIMENT_NAME = "churn-prediction"
PRODUCTION_ALIAS = "Production"

log = get_logger("train")


def _build_candidates(y_train) -> list[tuple[str, object, dict]]:
    """Return (name, estimator, logged_params) for each model to train."""
    # Class imbalance: ~2.77 negatives per positive. Compensate in both models.
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = round(neg / max(pos, 1), 3)

    rf_params = {"n_estimators": 300, "max_depth": 12, "class_weight": "balanced"}
    rf = RandomForestClassifier(random_state=42, n_jobs=-1, **rf_params)

    xgb_params = {
        "n_estimators": 400,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "scale_pos_weight": scale_pos_weight,
    }
    xgb = XGBClassifier(
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
        tree_method="hist",
        **xgb_params,
    )

    return [
        ("random_forest", rf, {"model_type": "random_forest", **rf_params}),
        ("xgboost", xgb, {"model_type": "xgboost", **xgb_params}),
    ]


def _train_one(
    name: str,
    estimator: object,
    params: dict,
    numeric: list[str],
    categorical: list[str],
    X_train,
    X_test,
    y_train,
    y_test,
    artifact_dir: Path,
) -> dict:
    """Train a single candidate inside an MLflow run. Returns a result summary."""
    pipe = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(numeric, categorical)),
            ("model", estimator),
        ]
    )

    with mlflow.start_run(run_name=name) as run:
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test.to_numpy(), y_pred, y_proba)

        mlflow.log_params(params)
        mlflow.log_param("n_features_in", X_train.shape[1])
        mlflow.log_metrics(metrics)

        cm = plot_confusion_matrix(
            y_test.to_numpy(), y_pred, artifact_dir / f"{name}_confusion.png", name
        )
        roc = plot_roc_curve(
            y_test.to_numpy(), y_proba, artifact_dir / f"{name}_roc.png", name
        )
        mlflow.log_artifact(str(cm), artifact_path="plots")
        mlflow.log_artifact(str(roc), artifact_path="plots")

        # cloudpickle (not the MLflow 3.x skops default) so the XGBoost-bearing
        # pipeline serializes — skops rejects xgboost types as "untrusted".
        model_info = mlflow.sklearn.log_model(
            pipe,
            name="model",
            input_example=X_train.head(3),
            serialization_format="cloudpickle",
        )

        log.info("model_trained", model=name, run_id=run.info.run_id, **metrics)
        return {
            "name": name,
            "roc_auc": metrics["roc_auc"],
            "metrics": metrics,
            "model_uri": model_info.model_uri,
            "run_id": run.info.run_id,
        }


def main() -> None:
    configure_logging(dev_mode=settings.dev_mode)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    log.info(
        "training_started",
        tracking_uri=settings.mlflow_tracking_uri,
        experiment=EXPERIMENT_NAME,
    )

    df = load_clean_data(settings.raw_data_path)
    numeric, categorical = split_feature_columns(df)
    X_train, X_test, y_train, y_test = split_data(df)
    log.info(
        "data_split",
        train_rows=len(X_train),
        test_rows=len(X_test),
        n_numeric=len(numeric),
        n_categorical=len(categorical),
    )

    results: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        artifact_dir = Path(tmp)
        for name, estimator, params in _build_candidates(y_train):
            results.append(
                _train_one(
                    name, estimator, params, numeric, categorical,
                    X_train, X_test, y_train, y_test, artifact_dir,
                )
            )

    best = max(results, key=lambda r: r["roc_auc"])
    log.info("best_model_selected", model=best["name"], roc_auc=best["roc_auc"])

    # Register best to the Model Registry and assign the Production alias.
    # (MLflow 3.x uses aliases, not the deprecated stage transitions.)
    mv = mlflow.register_model(best["model_uri"], settings.model_name)
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(
        settings.model_name, PRODUCTION_ALIAS, mv.version
    )
    log.info(
        "model_registered",
        name=settings.model_name,
        version=mv.version,
        alias=PRODUCTION_ALIAS,
        source_model=best["name"],
    )

    print(
        f"\nBest model: {best['name']} (ROC-AUC={best['roc_auc']:.4f})\n"
        f"Registered as {settings.model_name} v{mv.version} "
        f"@ {PRODUCTION_ALIAS}"
    )


if __name__ == "__main__":
    main()
