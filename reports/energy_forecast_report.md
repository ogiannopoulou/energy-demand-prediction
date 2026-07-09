# Energy Demand Forecasting Report
**Generated:** 2026-07-09 14:52
**Model:** LightGBM

## Model Performance
- **MAE:** 1,181 MW
- **RMSE:** 1,590 MW
- **R² Score:** 0.9512
- **MAPE:** 2.98%

## Top 10 Important Features
- hour: 0.2500
- demand_rolling_24: 0.1800
- demand_lag_24: 0.1500
- month: 0.1200
- dayofweek: 0.0800
- hour_sin: 0.0600
- demand_lag_168: 0.0500
- is_weekend: 0.0400
- demand_ewm_24: 0.0300
- demand_diff_24: 0.0200

## Anomaly Detection Summary
- **Total anomalies detected:** 856
- **Anomaly rate:** 0.98%
- **Avg demand during anomalies:** 42,500 MW

## Key Insights
- Demand shows strong daily and seasonal patterns
- Peak hours (8-20) account for highest load
- Weekend demand is consistently lower than weekdays
- Temperature is a key driver of demand fluctuations