"""Pydantic schemas for API request/response models."""
from datetime import datetime as dt

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Single prediction request."""
    prediction_datetime: dt = Field(..., alias="datetime", description="Datetime for prediction (ISO format)")
    features: dict = Field(default_factory=dict, description="Additional features (wind_mw, solar_mw, etc.)")

    model_config = {"populate_by_name": True}


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    predictions: list[PredictionRequest] = Field(..., min_length=1, max_length=168)


class PredictionResponse(BaseModel):
    """Single prediction response."""
    datetime: str
    predicted_demand_mw: float
    model_name: str


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: list[PredictionResponse]
    model_name: str
    count: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_name: str | None = None
    model_version: str | None = None
    feature_count: int | None = None


class ModelInfo(BaseModel):
    """Model metadata."""
    name: str
    version: str
    trained_at: str
    metrics: dict
    features: list[str]
