"""
ML Models for Energy Data
Forecasting, Anomaly Detection, and Classification
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


# ──────────────────────────────────────────────
# FORECASTING
# ──────────────────────────────────────────────

def train_xgboost_forecast(X_train, y_train, X_test, y_test, params=None):
    """Train XGBoost forecasting model."""
    if params is None:
        params = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
        }
    
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    
    y_pred = model.predict(X_test)
    metrics = compute_regression_metrics(y_test, y_pred)
    
    return model, y_pred, metrics


def train_lightgbm_forecast(X_train, y_train, X_test, y_test, params=None):
    """Train LightGBM forecasting model."""
    if params is None:
        params = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
    
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
    )
    
    y_pred = model.predict(X_test)
    metrics = compute_regression_metrics(y_test, y_pred)
    
    return model, y_pred, metrics


def cross_validate_forecast(model, X, y, n_splits=5):
    """Time series cross-validation."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics_list = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        metrics = compute_regression_metrics(y_val, y_pred)
        metrics["fold"] = fold
        metrics_list.append(metrics)
    
    return pd.DataFrame(metrics_list)


def compute_regression_metrics(y_true, y_pred):
    """Compute regression metrics."""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
        "mape": np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100,
    }


# ──────────────────────────────────────────────
# ANOMALY DETECTION
# ──────────────────────────────────────────────

def detect_anomalies_isolation_forest(data: pd.DataFrame, features: list,
                                        contamination: float = 0.01):
    """Detect anomalies using Isolation Forest."""
    X = data[features].dropna()
    
    model = IsolationForest(
        contamination=contamination,
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    predictions = model.fit_predict(X_scaled)
    scores = model.decision_function(X_scaled)
    
    result = data.loc[X.index].copy()
    result["anomaly"] = predictions == -1
    result["anomaly_score"] = scores
    
    return result, model


def detect_anomalies_zscore(data: pd.DataFrame, col: str, threshold: float = 3.0):
    """Detect anomalies using Z-score method."""
    result = data.copy()
    mean = result[col].mean()
    std = result[col].std()
    
    result["zscore"] = (result[col] - mean) / std
    result["anomaly"] = np.abs(result["zscore"]) > threshold
    
    return result


def detect_anomalies_iqr(data: pd.DataFrame, col: str, multiplier: float = 1.5):
    """Detect anomalies using IQR method."""
    result = data.copy()
    Q1 = result[col].quantile(0.25)
    Q3 = result[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR
    
    result["anomaly"] = (result[col] < lower) | (result[col] > upper)
    result["anomaly_lower"] = lower
    result["anomaly_upper"] = upper
    
    return result


# ──────────────────────────────────────────────
# CLASSIFICATION
# ──────────────────────────────────────────────

def classify_peak_offpeak(data: pd.DataFrame, target_col: str = "demand_mw"):
    """Classify hours as peak or off-peak."""
    result = data.copy()
    threshold = result[target_col].quantile(0.75)
    result["is_peak"] = (result[target_col] >= threshold).astype(int)
    return result


def train_peak_classifier(X_train, y_train, X_test, y_test):
    """Train a classifier for peak/off-peak prediction."""
    model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_prob)
    
    return model, y_pred, {"classification_report": report, "auc_roc": auc}


def classify_anomaly_type(data: pd.DataFrame, demand_col: str = "demand_mw"):
    """Classify anomalies by type: spike, dip, or normal."""
    result = data.copy()
    
    mean = result[demand_col].rolling(24, center=True).mean()
    std = result[demand_col].rolling(24, center=True).std()
    
    z = (result[demand_col] - mean) / std
    
    result["anomaly_type"] = "normal"
    result.loc[z > 2, "anomaly_type"] = "spike"
    result.loc[z < -2, "anomaly_type"] = "dip"
    
    return result
