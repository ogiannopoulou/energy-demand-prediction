# Project: Italian Electricity Demand Forecasting

## Overview
End-to-end ML pipeline for short-term electricity demand forecasting on the Italian power grid (Terna/ENTSO-E data).

## Repository
- **GitHub**: https://github.com/ogiannopoulou/energy-demand-prediction
- **Local path**: `/home/uranus/Documents/p9_portfolgio/terna_energy_project/`
- **Python venv**: `/home/uranus/Documents/p9_portfolgio/.venv/bin/python`

## Project Structure
- `src/data/terna_loader.py` — ENTSO-E API client + synthetic data generation
- `src/data/feature_engineering.py` — 54-feature pipeline (temporal, lag, rolling, EWM, cyclical)
- `src/models/ml_models.py` — XGBoost, LightGBM, Isolation Forest, classification
- `src/models/explainability.py` — SHAP integration (TreeExplainer)
- `src/monitoring/mlops.py` — MLflow with SQLite backend
- `src/llm/report_generator.py` — Automated report generation
- `notebooks/` — 6 Jupyter notebooks (01-06)
- `tests/test_pipeline.py` — 5 passing tests
- `figures/` — 13 diagnostic plots
- `PROJECT_REPORT.md` — Full technical report

## Key Results
- XGBoost: R²=0.9446, MAE=1220 MW, MAPE=4.41%
- LightGBM: R²=0.9457, MAE=1207 MW, MAPE=4.35%
- Classification AUC=0.9806
- Anomaly detection: Isolation Forest (263 anomalies, 1.0%)

## Known Issues
- MLflow v3 requires SQLite backend (fixed in `mlops.py`)
- skops trusted types needed for XGBoost/LightGBM serialization (fixed)
- ENTSO-E API key pending (email sent to transparency@entsoe.eu)
- Terna API key status: "waiting" (client_id: b9jagtxqzumkvucmc4gujy3g)

## User Context
- Ourania Giannopoulou, Rome, applied math PhD
- Interview at Key Partner Digital (Senior Data Scientist) — Monday
- This project is the portfolio piece for that interview

## Session Notes
See `SESSION_NOTES.md` for detailed session history.
