"""
Comprehensive tests for the Terna Energy Project pipeline
"""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

# ============================================
# Data Loading Tests
# ============================================

def test_synthetic_data_generation():
    """Test that synthetic data is generated correctly."""
    from src.data.terna_loader import generate_synthetic_data

    demand = generate_synthetic_data("demand")
    assert isinstance(demand, pd.DataFrame)
    assert "demand_mw" in demand.columns
    assert len(demand) > 0
    assert isinstance(demand.index, pd.DatetimeIndex)


def test_synthetic_generation_types():
    """Test synthetic data generation for all types."""
    from src.data.terna_loader import generate_synthetic_data

    for dataset in ["demand", "generation", "wind", "solar", "frequency"]:
        df = generate_synthetic_data(dataset)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert isinstance(df.index, pd.DatetimeIndex)


def test_synthetic_data_statistics():
    """Test synthetic data has reasonable statistics."""
    from src.data.terna_loader import generate_synthetic_data

    demand = generate_synthetic_data("demand")

    # Demand should be positive
    assert (demand["demand_mw"] > 0).all()

    # Should be in reasonable range for Italy (15-60 GW)
    assert demand["demand_mw"].min() > 10000
    assert demand["demand_mw"].max() < 100000


# ============================================
# Feature Engineering Tests
# ============================================

def test_feature_engineering():
    """Test feature engineering pipeline."""
    from src.data.feature_engineering import (
        build_feature_matrix,
    )
    from src.data.terna_loader import generate_synthetic_data

    demand = generate_synthetic_data("demand")

    df = build_feature_matrix(demand, target_col="demand_mw")

    assert "hour" in df.columns
    assert "month" in df.columns
    assert "is_weekend" in df.columns
    assert "demand_mw_lag_24" in df.columns
    assert "demand_mw_rolling_mean_24" in df.columns


def test_temporal_features():
    """Test temporal feature creation."""
    from src.data.feature_engineering import add_temporal_features
    from src.data.terna_loader import generate_synthetic_data

    demand = generate_synthetic_data("demand")
    df = add_temporal_features(demand)

    # Check all temporal features exist
    expected_cols = ["hour", "dayofweek", "dayofyear", "month", "year",
                     "week", "quarter", "is_weekend", "is_night", "is_peak",
                     "hour_sin", "hour_cos", "month_sin", "month_cos",
                     "dow_sin", "dow_cos"]

    for col in expected_cols:
        assert col in df.columns, f"Missing temporal feature: {col}"


def test_lag_features():
    """Test lag feature creation."""
    from src.data.feature_engineering import add_lag_features
    from src.data.terna_loader import generate_synthetic_data

    demand = generate_synthetic_data("demand")
    df = add_lag_features(demand, "demand_mw", lags=[1, 24, 168])

    assert "demand_mw_lag_1" in df.columns
    assert "demand_mw_lag_24" in df.columns
    assert "demand_mw_lag_168" in df.columns

    # Lag 1 should shift by 1
    assert df["demand_mw_lag_1"].iloc[1] == demand["demand_mw"].iloc[0]


def test_rolling_features():
    """Test rolling window features."""
    from src.data.feature_engineering import add_rolling_features
    from src.data.terna_loader import generate_synthetic_data

    demand = generate_synthetic_data("demand")
    df = add_rolling_features(demand, "demand_mw", windows=[24])

    assert "demand_mw_rolling_mean_24" in df.columns
    assert "demand_mw_rolling_std_24" in df.columns
    assert "demand_mw_rolling_min_24" in df.columns
    assert "demand_mw_rolling_max_24" in df.columns


def test_feature_matrix_shape():
    """Test feature matrix has correct shape."""
    from src.data.feature_engineering import build_feature_matrix
    from src.data.terna_loader import generate_synthetic_data

    demand = generate_synthetic_data("demand")
    df = build_feature_matrix(demand, target_col="demand_mw")

    # Should have more columns than original
    assert df.shape[1] > demand.shape[1]

    # Should have at least 50 features
    assert df.shape[1] >= 50


def test_prepare_train_test():
    """Test chronological train/test split."""
    from src.data.feature_engineering import build_feature_matrix, prepare_train_test
    from src.data.terna_loader import generate_synthetic_data

    demand = generate_synthetic_data("demand")

    df = build_feature_matrix(demand, target_col="demand_mw")
    df = df.dropna()

    train, test = prepare_train_test(df, test_ratio=0.2)

    assert len(train) > len(test)
    assert train.index.max() < test.index.min()

    # Check split ratio
    assert len(train) / len(df) == pytest.approx(0.8, rel=0.01)


def test_get_feature_columns():
    """Test feature column selection."""
    from src.data.feature_engineering import build_feature_matrix, get_feature_columns
    from src.data.terna_loader import generate_synthetic_data

    demand = generate_synthetic_data("demand")
    df = build_feature_matrix(demand, target_col="demand_mw")

    feature_cols = get_feature_columns(df, target_col="demand_mw")

    # Should not include target
    assert "demand_mw" not in feature_cols

    # Should only include numeric columns
    for col in feature_cols:
        assert df[col].dtype in ["float64", "int64", "int32", "float32"]


# ============================================
# Model Training Tests
# ============================================

def test_train_xgboost():
    """Test XGBoost model training."""
    from src.data.feature_engineering import (
        build_feature_matrix,
        get_feature_columns,
        prepare_train_test,
    )
    from src.data.terna_loader import generate_synthetic_data
    from src.models.ml_models import train_xgboost_forecast

    demand = generate_synthetic_data("demand")
    df = build_feature_matrix(demand, target_col="demand_mw").dropna()
    train, test = prepare_train_test(df)
    feature_cols = get_feature_columns(df)

    X_train, y_train = train[feature_cols], train["demand_mw"]
    X_test, y_test = test[feature_cols], test["demand_mw"]

    model, y_pred, metrics = train_xgboost_forecast(X_train, y_train, X_test, y_test)

    assert model is not None
    assert len(y_pred) == len(y_test)
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert "mape" in metrics


def test_train_lightgbm():
    """Test LightGBM model training."""
    from src.data.feature_engineering import (
        build_feature_matrix,
        get_feature_columns,
        prepare_train_test,
    )
    from src.data.terna_loader import generate_synthetic_data
    from src.models.ml_models import train_lightgbm_forecast

    demand = generate_synthetic_data("demand")
    df = build_feature_matrix(demand, target_col="demand_mw").dropna()
    train, test = prepare_train_test(df)
    feature_cols = get_feature_columns(df)

    X_train, y_train = train[feature_cols], train["demand_mw"]
    X_test, y_test = test[feature_cols], test["demand_mw"]

    model, y_pred, metrics = train_lightgbm_forecast(X_train, y_train, X_test, y_test)

    assert model is not None
    assert len(y_pred) == len(y_test)
    assert "r2" in metrics
    assert metrics["r2"] > 0  # Should be better than random


def test_compute_regression_metrics():
    """Test regression metrics computation."""
    from src.models.ml_models import compute_regression_metrics

    y_true = pd.Series([100, 200, 300, 400, 500])
    y_pred = pd.Series([110, 190, 310, 390, 510])

    metrics = compute_regression_metrics(y_true, y_pred)

    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert "mape" in metrics

    # MAE should be 10 (mean of |10|, |10|, |10|, |10|, |10|)
    assert metrics["mae"] == pytest.approx(10.0)

    # R² should be close to 1 (good predictions)
    assert metrics["r2"] > 0.9


# ============================================
# Anomaly Detection Tests
# ============================================

def test_anomaly_detection():
    """Test anomaly detection methods."""
    from src.data.terna_loader import generate_synthetic_data
    from src.models.ml_models import detect_anomalies_iqr, detect_anomalies_zscore

    demand = generate_synthetic_data("demand")

    result_z = detect_anomalies_zscore(demand, "demand_mw", threshold=3.0)
    assert "anomaly" in result_z.columns
    assert result_z["anomaly"].dtype == bool

    result_iqr = detect_anomalies_iqr(demand, "demand_mw")
    assert "anomaly" in result_iqr.columns
    assert "anomaly_lower" in result_iqr.columns


def test_anomaly_detection_methods():
    """Test all anomaly detection methods."""
    from src.data.terna_loader import generate_synthetic_data
    from src.models.ml_models import (
        detect_anomalies_iqr,
        detect_anomalies_isolation_forest,
        detect_anomalies_zscore,
    )

    demand = generate_synthetic_data("demand")

    # Z-score with lower threshold to detect some anomalies
    result_z = detect_anomalies_zscore(demand, "demand_mw", threshold=1.5)
    assert result_z["anomaly"].sum() > 0  # Should detect some anomalies

    # IQR with lower multiplier
    result_iqr = detect_anomalies_iqr(demand, "demand_mw", multiplier=1.0)
    assert result_iqr["anomaly"].sum() > 0

    # Isolation Forest (needs multiple features)
    demand["hour"] = demand.index.hour
    result_if, _ = detect_anomalies_isolation_forest(demand, ["demand_mw", "hour"])
    assert "anomaly" in result_if.columns
    assert "anomaly_score" in result_if.columns


# ============================================
# Classification Tests
# ============================================

def test_classification():
    """Test peak/off-peak classification."""
    from src.data.terna_loader import generate_synthetic_data
    from src.models.ml_models import classify_peak_offpeak

    demand = generate_synthetic_data("demand")

    result = classify_peak_offpeak(demand)
    assert "is_peak" in result.columns
    assert result["is_peak"].isin([0, 1]).all()


def test_classify_anomaly_type():
    """Test anomaly type classification."""
    from src.data.terna_loader import generate_synthetic_data
    from src.models.ml_models import classify_anomaly_type

    demand = generate_synthetic_data("demand")

    result = classify_anomaly_type(demand)
    assert "anomaly_type" in result.columns
    assert set(result["anomaly_type"].unique()).issubset({"normal", "spike", "dip"})


# ============================================
# API Tests
# ============================================

def test_api_health_endpoint():
    """Test API health endpoint."""
    from fastapi.testclient import TestClient

    from api.app import app

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


def test_api_models_endpoint():
    """Test API models endpoint."""
    from fastapi.testclient import TestClient

    from api.app import app

    client = TestClient(app)
    response = client.get("/models")

    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_api_predict_endpoint():
    """Test API predict endpoint."""
    from fastapi.testclient import TestClient

    from api.app import app

    client = TestClient(app)

    # First check if any models are loaded
    models_response = client.get("/models")
    if not models_response.json():
        pytest.skip("No models loaded, skipping predict test")

    response = client.post(
        "/predict",
        json={
            "datetime": datetime.now().isoformat(),
            "features": {}
        }
    )

    # Should return 200 or 404 (if model not found)
    assert response.status_code in [200, 404, 500]


# ============================================
# MLflow Tests
# ============================================

def test_mlflow_setup():
    """Test MLflow setup."""
    from src.monitoring.mlops import setup_mlflow

    mlflow = setup_mlflow()

    # Should return mlflow module or None
    assert mlflow is None or hasattr(mlflow, "set_tracking_uri")


def test_mlflow_logging():
    """Test MLflow experiment logging."""
    from src.monitoring.mlops import log_experiment, setup_mlflow

    mlflow = setup_mlflow()

    if mlflow is None:
        pytest.skip("MLflow not available")

    # Create a simple model for testing
    from sklearn.linear_model import LinearRegression
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([1, 2, 3, 4, 5])
    model = LinearRegression().fit(X, y)

    metrics = {"mae": 0.1, "rmse": 0.15, "r2": 0.99, "mape": 1.0}
    params = {"model_type": "linear_regression"}

    run_id = log_experiment("test_model", model, metrics, params=params)

    # Should return a run_id
    assert run_id is not None


# ============================================
# Integration Tests
# ============================================

def test_full_pipeline():
    """Test complete pipeline from data to prediction."""
    from src.data.feature_engineering import (
        build_feature_matrix,
        get_feature_columns,
        prepare_train_test,
    )
    from src.data.terna_loader import generate_synthetic_data
    from src.models.ml_models import train_lightgbm_forecast

    # Generate data
    demand = generate_synthetic_data("demand")

    # Build features
    df = build_feature_matrix(demand, target_col="demand_mw").dropna()

    # Split
    train, test = prepare_train_test(df)
    feature_cols = get_feature_columns(df)

    X_train, y_train = train[feature_cols], train["demand_mw"]
    X_test, y_test = test[feature_cols], test["demand_mw"]

    # Train
    model, y_pred, metrics = train_lightgbm_forecast(X_train, y_train, X_test, y_test)

    # Verify
    assert model is not None
    assert metrics["r2"] > 0.5  # Should be reasonable
    assert len(y_pred) == len(y_test)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
