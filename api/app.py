"""FastAPI application for Italian electricity demand forecasting."""
import logging
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.data.feature_engineering import build_feature_matrix, get_feature_columns

from .model_store import get_model_store, reload_models
from .schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfo,
    PredictionRequest,
    PredictionResponse,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Italian Electricity Demand Forecasting API",
    description="ML-powered demand forecasting using ENTSO-E data",
    version="1.0.0",
)

PROJECT_ROOT = Path(__file__).parent.parent


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint."""
    store = get_model_store()
    models = store.list_models()

    if models:
        first_model = models[0]
        meta = store.get_metadata(first_model)
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            model_name=first_model,
            model_version=meta.get("version", "unknown"),
            feature_count=len(meta.get("features", [])),
        )

    return HealthResponse(
        status="degraded",
        model_loaded=False,
    )


@app.get("/models", response_model=list[ModelInfo])
def list_models():
    """List all available models."""
    store = get_model_store()
    models = []

    for name in store.list_models():
        meta = store.get_metadata(name)
        models.append(ModelInfo(
            name=name,
            version=meta.get("version", "unknown"),
            trained_at=meta.get("trained_at", "unknown"),
            metrics=meta.get("metrics", {}),
            features=meta.get("features", []),
        ))

    return models


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest, model_name: str = "lightgbm_demand_forecast"):
    """Make a single demand prediction."""
    store = get_model_store()

    if model_name not in store.list_models():
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    # Create a small dataframe for feature engineering
    # We need at least 168 hours of history for lag features
    # For now, use the request datetime and fill with zeros
    # In production, you'd fetch historical data

    dt = request.prediction_datetime
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=None)

    # Create single-row dataframe
    df = pd.DataFrame({"demand_mw": [0]}, index=[dt])
    df.index.name = "datetime"

    # Add any provided features
    for key, value in request.features.items():
        df[key] = value

    # Build features
    try:
        df_features = build_feature_matrix(df, target_col="demand_mw")
        feature_cols = get_feature_columns(df_features, target_col="demand_mw")

        # Fill NaN with 0 for prediction
        X = df_features[feature_cols].fillna(0)

        prediction = store.predict(model_name, X)[0]

        return PredictionResponse(
            datetime=dt.isoformat(),
            predicted_demand_mw=round(float(prediction), 2),
            model_name=model_name,
        )
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest, model_name: str = "lightgbm_demand_forecast"):
    """Make batch demand predictions."""
    store = get_model_store()

    if model_name not in store.list_models():
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    results = []
    for req in request.predictions:
        try:
            resp = predict(req, model_name)
            results.append(resp)
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Batch prediction failed for %s: %s", req.datetime, e)
            raise HTTPException(status_code=500, detail=str(e))

    return BatchPredictionResponse(
        predictions=results,
        model_name=model_name,
        count=len(results),
    )


@app.post("/models/reload")
def reload_models_endpoint():
    """Reload all models from disk."""
    store = reload_models()
    return {"status": "ok", "models": store.list_models()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
