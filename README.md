<p align="center">
  <img src="figures/banner.png" alt="Italian Electricity Demand Forecasting" width="100%">
</p>

<h3 align="center">End-to-End ML Pipeline for Short-Term Electricity Demand Forecasting</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/XGBoost-2.0-red?logo=xgboost&logoColor=white" alt="XGBoost">
  <img src="https://img.shields.io/badge/LightGBM-4.0-orange?logo=lightgbm&logoColor=white" alt="LightGBM">
  <img src="https://img.shields.io/badge/SHAP-0.45-purple" alt="SHAP">
  <img src="https://img.shields.io/badge/MLflow-2.15-blue?logo=mlflow&logoColor=white" alt="MLflow">
  <img src="https://img.shields.io/badge/Scikit--learn-1.5-orange?logo=scikit-learn&logoColor=white" alt="Scikit-learn">
</p>

<p align="center">
  Forecasting Italian electricity demand using gradient boosting, with SHAP explainability, anomaly detection, and MLOps tracking.
  <br>
  <a href="PROJECT_REPORT.md"><strong>Read the full report »</strong></a>
</p>

---

## Overview

This project demonstrates a complete ML pipeline for **short-term load forecasting** on the Italian power grid (Terna / ENTSO-E data). It covers data ingestion, feature engineering, model training, evaluation, explainability, anomaly detection, and experiment tracking.

### Key Results

| Model | MAE (MW) | RMSE (MW) | R² | MAPE |
|-------|----------|-----------|-----|------|
| **XGBoost** | 1,220 | 1,520 | 0.9446 | 4.41% |
| **LightGBM** | 1,207 | 1,505 | 0.9457 | 4.35% |

| Task | Method | Performance |
|------|--------|-------------|
| Peak Classification | LightGBM | AUC = 0.9806 |
| Anomaly Detection | Isolation Forest | 263 anomalies (1.0%) |

---

## Pipeline Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────┐
│  ENTSO-E    │────▶│ Feature          │────▶│  XGBoost /   │
│  API /      │     │ Engineering      │     │  LightGBM    │
│  Synthetic  │     │ (54 features)    │     │  Forecasting │
└─────────────┘     └──────────────────┘     └──────┬───────┘
                                                     │
                    ┌──────────────────┐     ┌───────▼───────┐
                    │  SHAP            │◀────│  Diagnostics  │
                    │  Explainability  │     │  & Evaluation │
                    └──────────────────┘     └───────┬───────┘
                                                     │
              ┌──────────────┐              ┌────────▼────────┐
              │  Anomaly     │              │  MLflow         │
              │  Detection   │              │  Tracking       │
              │  (3 methods) │              │  (SQLite)       │
              └──────────────┘              └─────────────────┘
```

---

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 01 | [Data Ingestion & EDA](notebooks/01_data_ingestion.ipynb) | Load data, statistical summary, visualizations |
| 02 | [Forecasting](notebooks/02_forecasting.ipynb) | XGBoost & LightGBM training, SHAP analysis |
| 03 | [Anomaly Detection](notebooks/03_anomaly_detection.ipynb) | Isolation Forest, Z-score, IQR methods |
| 04 | [Classification](notebooks/04_classification.ipynb) | Peak/off-peak prediction, ROC analysis |
| 05 | [LLM Report](notebooks/05_llm_report.ipynb) | Automated report generation |
| 06 | [Diagnostics](notebooks/06_diagnostics.ipynb) | Comprehensive model diagnostics (13 plots) |

---

## Diagnostic Plots

The project includes 13 diagnostic visualizations to validate model correctness:

<details>
<summary><strong>Click to expand all diagnostics</strong></summary>

### Actual vs Predicted
![Actual vs Predicted](figures/06_actual_vs_predicted.png)

### Residual Analysis
![Residual Analysis](figures/06_residual_analysis.png)

### Rolling Error
![Rolling Error](figures/06_rolling_error.png)

### Error by Hour
![Error by Hour](figures/06_error_by_hour.png)

### SHAP Beeswarm
![SHAP Beeswarm](figures/06_shap_beeswarm.png)

### SHAP Dependence
![SHAP Dependence](figures/06_shap_dependence.png)

### SHAP Waterfall
![SHAP Waterfall](figures/06_shap_waterfall.png)

### Anomaly Detection Comparison
![Anomaly Comparison](figures/06_anomaly_comparison.png)

### Anomaly Distributions
![Anomaly Distributions](figures/06_anomaly_distributions.png)

### Anomaly Agreement
![Anomaly Agreement](figures/06_anomaly_agreement.png)

### Classification Curves
![Classification Curves](figures/06_classification_curves.png)

### Classification CM & Feature Importance
![Classification](figures/06_classification_cm_fi.png)

### Cross-Validation
![CV Comparison](figures/06_cv_comparison.png)

### Learning Curves
![Learning Curves](figures/06_learning_curves.png)

</details>

---

## Project Structure

```
terna_energy_project/
├── src/
│   ├── data/
│   │   ├── terna_loader.py          # ENTSO-E API client + synthetic data
│   │   └── feature_engineering.py   # 54-feature pipeline
│   ├── models/
│   │   ├── ml_models.py             # XGBoost, LightGBM, Isolation Forest
│   │   └── explainability.py        # SHAP integration
│   ├── monitoring/
│   │   └── mlops.py                 # MLflow experiment tracking
│   └── llm/
│       └── report_generator.py      # Automated report generation
├── notebooks/                       # 6 Jupyter notebooks
├── tests/                           # Unit tests (5 passing)
├── figures/                         # 13 diagnostic plots
├── reports/                         # Generated reports
├── PROJECT_REPORT.md                # Full technical report
└── requirements.txt
```

---

## Tech Stack

- **Languages**: Python 3.12
- **ML**: XGBoost, LightGBM, Scikit-learn, Isolation Forest
- **Explainability**: SHAP (TreeExplainer)
- **MLOps**: MLflow with SQLite backend
- **Data**: ENTSO-E Transparency Platform API
- **Visualization**: Matplotlib, Seaborn
- **Testing**: Pytest
- **GenAI**: OpenAI API (optional, for enhanced reports)

---

## Quick Start

```bash
# Clone
git clone https://github.com/ogiannopoulou/terna-energy-project.git
cd terna-energy-project

# Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run notebooks (in order)
jupyter notebook notebooks/01_data_ingestion.ipynb
jupyter notebook notebooks/02_forecasting.ipynb
# ... etc

# Run tests
python -m pytest tests/ -v

# View MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow_results/mlflow.db
```

---

## Real Data

The pipeline supports real data from the [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/):

1. Register for a free API key at https://transparency.entsoe.eu/content/API_sandbox/api_key_registration.htm
2. Set environment variable: `export ENTSOE_API_KEY=your_key_here`
3. Rerun notebooks — the pipeline will automatically download real Italian demand data

Without an API key, the pipeline uses realistic synthetic data that preserves the statistical properties of real Italian electricity demand.

---

## Author

**Ourania Giannopoulou** — Applied Mathematics PhD, CFD/ML researcher

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/ogiannopoulou)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ogiannopoulou)

---

## License

This project is open source and available for portfolio demonstration purposes.
