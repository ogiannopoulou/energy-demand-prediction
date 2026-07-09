"""
Tests for the Terna Energy Project pipeline
"""
import sys
sys.path.insert(0, '..')

import pandas as pd
import numpy as np
import pytest


def test_synthetic_data_generation():
    """Test that synthetic data is generated correctly."""
    from src.data.terna_loader import generate_synthetic_data
    
    demand = generate_synthetic_data("demand")
    assert isinstance(demand, pd.DataFrame)
    assert "demand_mw" in demand.columns
    assert len(demand) > 0
    assert isinstance(demand.index, pd.DatetimeIndex)


def test_feature_engineering():
    """Test feature engineering pipeline."""
    from src.data.terna_loader import generate_synthetic_data
    from src.data.feature_engineering import (
        add_temporal_features, add_lag_features,
        add_rolling_features, build_feature_matrix
    )
    
    demand = generate_synthetic_data("demand")
    
    df = build_feature_matrix(demand, target_col="demand_mw")
    
    assert "hour" in df.columns
    assert "month" in df.columns
    assert "is_weekend" in df.columns
    assert "demand_mw_lag_24" in df.columns
    assert "demand_mw_rolling_mean_24" in df.columns


def test_anomaly_detection():
    """Test anomaly detection methods."""
    from src.data.terna_loader import generate_synthetic_data
    from src.models.ml_models import (
        detect_anomalies_zscore, detect_anomalies_iqr
    )
    
    demand = generate_synthetic_data("demand")
    
    result_z = detect_anomalies_zscore(demand, "demand_mw", threshold=3.0)
    assert "anomaly" in result_z.columns
    assert result_z["anomaly"].dtype == bool
    
    result_iqr = detect_anomalies_iqr(demand, "demand_mw")
    assert "anomaly" in result_iqr.columns
    assert "anomaly_lower" in result_iqr.columns


def test_classification():
    """Test peak/off-peak classification."""
    from src.data.terna_loader import generate_synthetic_data
    from src.models.ml_models import classify_peak_offpeak
    
    demand = generate_synthetic_data("demand")
    
    result = classify_peak_offpeak(demand)
    assert "is_peak" in result.columns
    assert result["is_peak"].isin([0, 1]).all()


def test_train_test_split():
    """Test chronological train/test split."""
    from src.data.terna_loader import generate_synthetic_data
    from src.data.feature_engineering import build_feature_matrix, prepare_train_test
    
    demand = generate_synthetic_data("demand")
    
    df = build_feature_matrix(demand, target_col="demand_mw")
    df = df.dropna()
    
    train, test = prepare_train_test(df, test_ratio=0.2)
    
    assert len(train) > len(test)
    assert train.index.max() < test.index.min()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
