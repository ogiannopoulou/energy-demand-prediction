<p align="center">
  <img src="figures/banner.png" alt="Italian Electricity Demand Forecasting" width="100%">
</p>

<h3 align="center">End-to-End ML Pipeline for Short-Term Electricity Demand Forecasting</h3>

<p align="center">
  <a href="https://github.com/ogiannopoulou/energy-demand-prediction/actions/workflows/ml_pipeline.yml"><img src="https://github.com/ogiannopoulou/energy-demand-prediction/actions/workflows/ml_pipeline.yml/badge.svg" alt="CI/CD"></a>
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/XGBoost-2.0-red?logo=xgboost&logoColor=white" alt="XGBoost">
  <img src="https://img.shields.io/badge/LightGBM-4.0-orange?logo=lightgbm&logoColor=white" alt="LightGBM">
  <img src="https://img.shields.io/badge/SHAP-0.45-purple" alt="SHAP">
  <img src="https://img.shields.io/badge/MLflow-2.15-blue?logo=mlflow&logoColor=white" alt="MLflow">
  <img src="https://img.shields.io/badge/Scikit--learn-1.5-orange?logo=scikit-learn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/FastAPI-0.100-green?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/GitHub%20Actions-2088FF?logo=githubactions&logoColor=white" alt="GitHub Actions">
</p>

<p align="center">
  Forecasting electricity demand (in Italy) using gradient boosting, with SHAP explainability, anomaly detection, and MLOps tracking.
  <br>
  <a href="PROJECT_REPORT.md"><strong>Read the full report »</strong></a>
</p>

---

## Overview

This project demonstrates a complete ML pipeline for **short-term load forecasting** on the Italian power grid (Terna / ENTSO-E data). It covers data ingestion, feature engineering, model training, evaluation, explainability, anomaly detection, and experiment tracking.

### Key Results

| Model | MAE (MW) | RMSE (MW) | R² | MAPE |
|-------|----------|-----------|-----|------|
| **XGBoost** | 2,639 | 3,893 | 0.7048 | 9.24% |
| **LightGBM** | 2,653 | 3,898 | 0.7039 | 9.30% |

*Results on 1 year of real ENTSO-E data (2023)*

| Task | Method | Performance |
|------|--------|-------------|
| Peak Classification | LightGBM | AUC = 0.9806 |
| Anomaly Detection | Isolation Forest | 263 anomalies (1.0%) |

<p align="center">
  <img src="figures/06_actual_vs_predicted.png" alt="Actual vs Predicted Demand" width="90%">
</p>

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

## MLOps Stack

| Component | Tool | Purpose |
|-----------|------|---------|
| **Data Versioning** | DVC | Version control for datasets |
| **Experiment Tracking** | MLflow | Log params, metrics, models |
| **API** | FastAPI | Serve predictions |
| **Containerization** | Docker | Reproducible environments |
| **CI/CD** | GitHub Actions | Automated testing, training & deployment |
| **Dashboard** | Streamlit | Monitoring & visualization |

---

## CI/CD Pipeline

Fully automated 3-stage GitHub Actions pipeline triggered on every push/PR to `main`:

```
  Push/PR ──▶ test ──▶ train ──▶ build
              (lint +    (model    (Docker
              22 tests)  training)  + GHCR)
```

| Stage | What it does | Trigger |
|-------|-------------|---------|
| **test** | Ruff lint + 22 pytest tests with coverage upload to Codecov | Every push & PR |
| **train** | Trains XGBoost & LightGBM on ENTSO-E data, logs to MLflow, saves models as artifacts | `main` only, after test passes |
| **build** | Builds multi-stage Docker image, pushes to GitHub Container Registry with SHA + `latest` tags | `main` only, after train passes |

### Testing Strategy

22 tests covering the full pipeline:

| Category | Tests | What's validated |
|----------|-------|-----------------|
| Data Loading | 3 | Synthetic generation, data types, statistics ranges |
| Feature Engineering | 7 | Temporal, lag, rolling features, train/test split, feature matrix shape |
| Model Training | 3 | XGBoost, LightGBM, regression metrics computation |
| Anomaly Detection | 2 | Z-score, IQR, Isolation Forest methods |
| Classification | 2 | Peak/off-peak classification, anomaly type detection |
| API | 3 | Health endpoint, models endpoint, predict endpoint |
| MLOps | 2 | MLflow setup, experiment logging |
| Integration | 1 | End-to-end pipeline: data → features → train → predict |

```bash
# Local CI simulation
make test          # Run all 22 tests with coverage
make lint          # Ruff lint + format check
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/models` | List available models |
| POST | `/predict` | Single prediction |
| POST | `/predict/batch` | Batch predictions |

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
├── .github/workflows/
│   └── ml_pipeline.yml          # CI/CD: test → train → build
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
├── api/
│   ├── app.py                       # FastAPI application
│   ├── schemas.py                   # Pydantic request/response models
│   └── model_store.py               # Model loading & serving
├── tests/
│   ├── conftest.py                  # Test path configuration
│   └── test_pipeline.py             # 22 tests (data, models, API, MLOps)
├── notebooks/                       # 6 Jupyter notebooks
├── figures/                         # 13 diagnostic plots
├── reports/                         # Generated reports
├── models/                          # Trained model artifacts
├── Dockerfile                       # Multi-stage Docker build
├── docker-compose.yml               # Full stack orchestration
├── Makefile                         # Dev commands: test, lint, train, api
├── pyproject.toml                   # Project config (pytest, ruff, mypy)
├── PROJECT_REPORT.md                # Full technical report
└── requirements.txt
```

---

## Tech Stack

- **Languages**: Python 3.12
- **ML**: XGBoost, LightGBM, Scikit-learn, Isolation Forest
- **Explainability**: SHAP (TreeExplainer)
- **MLOps**: MLflow with SQLite backend, DVC
- **Data**: ENTSO-E Transparency Platform API
- **API**: FastAPI + Uvicorn + Pydantic
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Testing**: Pytest + Coverage
- **Linting**: Ruff
- **Containerization**: Docker (multi-stage) + Docker Compose
- **CI/CD**: GitHub Actions (test → train → build → GHCR)
- **GenAI**: OpenAI API (optional, for enhanced reports)

---

## Quick Start

```bash
# Clone & install
git clone https://github.com/ogiannopoulou/energy-demand-prediction.git
cd energy-demand-prediction
python -m venv .venv && source .venv/bin/activate
make setup

# Run tests
make test

# Train models (requires ENTSOE_API_KEY env var, falls back to synthetic)
make train

# Start API server
make api

# Start Streamlit dashboard
make dashboard

# Run full stack with Docker
make docker-up
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

**Ourania Giannopoulou** — Applied Mathematics PhD, Comp. and Data Sciences researcher 

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/rania-giannopoulou)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ogiannopoulou)

---

## License

This project is open source and available for portfolio demonstration purposes.
