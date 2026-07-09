"""
Feature Engineering for Energy Data
Temporal features, lag features, and rolling statistics
"""
import pandas as pd
import numpy as np


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features to datetime-indexed DataFrame."""
    df = df.copy()
    
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["dayofyear"] = df.index.dayofyear
    df["month"] = df.index.month
    df["year"] = df.index.year
    df["week"] = df.index.isocalendar().week.astype(int)
    df["quarter"] = df.index.quarter
    
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
    df["is_peak"] = ((df["hour"] >= 8) & (df["hour"] <= 20)).astype(int)
    
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    
    return df


def add_lag_features(df: pd.DataFrame, target_col: str, lags: list = None) -> pd.DataFrame:
    """Add lag features for a target column."""
    df = df.copy()
    
    if lags is None:
        lags = [1, 2, 3, 6, 12, 24, 48, 168]
    
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    
    return df


def add_rolling_features(df: pd.DataFrame, target_col: str, windows: list = None) -> pd.DataFrame:
    """Add rolling window statistics."""
    df = df.copy()
    
    if windows is None:
        windows = [6, 12, 24, 48, 168]
    
    for w in windows:
        df[f"{target_col}_rolling_mean_{w}"] = df[target_col].rolling(w).mean()
        df[f"{target_col}_rolling_std_{w}"] = df[target_col].rolling(w).std()
        df[f"{target_col}_rolling_min_{w}"] = df[target_col].rolling(w).min()
        df[f"{target_col}_rolling_max_{w}"] = df[target_col].rolling(w).max()
    
    return df


def add_ewm_features(df: pd.DataFrame, target_col: str, spans: list = None) -> pd.DataFrame:
    """Add exponentially weighted moving average features."""
    df = df.copy()
    
    if spans is None:
        spans = [12, 24, 168]
    
    for span in spans:
        df[f"{target_col}_ewm_{span}"] = df[target_col].ewm(span=span).mean()
    
    return df


def add_diff_features(df: pd.DataFrame, target_col: str, periods: list = None) -> pd.DataFrame:
    """Add differencing features."""
    df = df.copy()
    
    if periods is None:
        periods = [1, 24, 168]
    
    for p in periods:
        df[f"{target_col}_diff_{p}"] = df[target_col].diff(p)
        df[f"{target_col}_pct_change_{p}"] = df[target_col].pct_change(p)
    
    return df


def build_feature_matrix(df: pd.DataFrame, target_col: str = "demand_mw") -> pd.DataFrame:
    """Build complete feature matrix for forecasting."""
    df = add_temporal_features(df)
    df = add_lag_features(df, target_col)
    df = add_rolling_features(df, target_col)
    df = add_ewm_features(df, target_col)
    df = add_diff_features(df, target_col)
    
    df = df.replace([np.inf, -np.inf], np.nan)
    
    return df


def prepare_train_test(df: pd.DataFrame, test_ratio: float = 0.2) -> tuple:
    """Split data chronologically for time series."""
    split_idx = int(len(df) * (1 - test_ratio))
    
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    
    return train, test


def get_feature_columns(df: pd.DataFrame, target_col: str = "demand_mw") -> list:
    """Get list of feature columns (exclude target and non-feature cols)."""
    exclude = {target_col, "demand_mw", "datetime", "Date", "data"}
    exclude.update({c for c in df.columns if c.startswith(target_col)})
    
    return [c for c in df.columns if c not in exclude and df[c].dtype in ["float64", "int64", "int32", "float32"]]
