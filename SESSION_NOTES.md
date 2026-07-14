# Session Notes — July 9, 2026

## Project: Italian Electricity Demand Forecasting
- **Repo**: https://github.com/ogiannopoulou/energy-demand-prediction
- **Local path**: `/home/uranus/Documents/p9_portfolgio/terna_energy_project/`
- **Python venv**: `/home/uranus/Documents/p9_portfolgio/.venv/bin/python`

## What was done today
1. Fixed MLflow v3 SQLite backend (was file store, v3 requires DB)
2. Fixed skops trusted types for XGBoost/LightGBM serialization
3. All 5 original notebooks (01-05) execute successfully
4. All 5 tests pass
5. Created notebook 06_diagnostics with 13 diagnostic plots:
   - Actual vs predicted scatter, residual analysis (histogram, Q-Q, ACF, residuals vs predicted)
   - Rolling MAE, error by hour of day
   - SHAP beeswarm (XGB vs LGB), dependence, waterfall
   - Anomaly comparison (3 methods overlaid), distributions, agreement
   - Classification ROC, precision-recall, calibration, confusion matrix
   - Cross-validation boxplots, learning curves
6. Wrote PROJECT_REPORT.md (detailed technical report) + PROJECT_REPORT.pdf
7. Created banner image (figures/banner.png)
8. Wrote README.md with badge shields, results table, actual vs predicted plot
9. Pushed to GitHub as `ogiannopoulou/energy-demand-prediction`

## Key results
- XGBoost: R²=0.9446, MAE=1220, MAPE=4.41%
- LightGBM: R²=0.9457, MAE=1207, MAPE=4.35%
- Classification AUC=0.9806
- Isolation Forest: 263 anomalies (1.0%)

## Pending
- ENTSO-E API key (email sent to transparency@entsoe.eu, waiting)
- Terna API key status: "waiting" (client_id: b9jagtxqzumkvucmc4gujy3g)

## Interview prep
- Read PROJECT_REPORT.md for all methods, results, physical interpretation
- Role: Senior Data Scientist at Key Partner Digital (Rome, hybrid, €35-40k)
- Monday interview
