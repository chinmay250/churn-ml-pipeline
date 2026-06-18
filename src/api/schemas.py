"""Pydantic I/O models for the churn API.

``CustomerFeatures`` mirrors the 19 RAW feature columns the training pipeline
expects (post-``load_clean_data``, minus ``customerID`` and ``Churn``). The
registered model is a full sklearn Pipeline, so the API passes these raw columns
straight through — preprocessing happens inside the model.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Literal sets keep the OpenAPI docs self-describing and reject typos at the edge.
YesNo = Literal["Yes", "No"]


class CustomerFeatures(BaseModel):
    """One customer's raw features — the input to POST /predict and /drift/record."""

    gender: Literal["Male", "Female"]
    SeniorCitizen: int = Field(ge=0, le=1, description="0 or 1")
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, description="months with the company")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85,
            }
        }
    }


class PredictionResponse(BaseModel):
    """Output of POST /predict."""

    churn: int = Field(description="predicted class: 1 = will churn, 0 = will not")
    churn_label: YesNo
    churn_probability: float = Field(ge=0.0, le=1.0)
    model_version: str | None = Field(
        default=None, description="registry version of the serving model"
    )


class RecordResponse(BaseModel):
    """Output of POST /drift/record."""

    recorded: bool
    total_recorded: int = Field(description="rows in the live feature log after this write")


class HealthResponse(BaseModel):
    """Output of GET /health."""

    status: str
    model_loaded: bool
    model_version: str | None = None
