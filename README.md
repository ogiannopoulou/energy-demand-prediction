<p align="center">
  <img src="figures/banner.png" alt="Italian Electricity Demand Forecasting" width="100%">
</p>

<h1 align="center">Italian Electricity Demand Forecasting</h1>

<p align="center">
  <em>End-to-end ML pipeline for short-term load forecasting on the Italian power grid</em>
</p>

<p align="center">
  <a href="https://github.com/ogiannopoulou/energy-demand-prediction/actions/workflows/ml_pipeline.yml"><img src="https://github.com/ogiannopoulou/energy-demand-prediction/actions/workflows/ml_pipeline.yml/badge.svg" alt="CI/CD"></a>
  <img src="https://img.shields.io/badge/R²-0.946-brightgreen?style=flat" alt="R2">
  <img src="https://img.shields.io/badge/MAPE-4.35%25-blue?style=flat" alt="MAPE">
  <img src="https://img.shields.io/badge/Tests-22%20passing-brightgreen?style=flat" alt="Tests">
  <img src="https://img.shields.io/badge/License-MIT-gray?style=flat" alt="License">
</p>

<p align="center">
  <a href="PROJECT_REPORT.md">Full Report</a> ·
  <a href="https://energy-demand-prediction.streamlit.app">Live Dashboard</a> ·
  <a href="https://github.com/ogiannopoulou/energy-demand-prediction/pkgs/container/energy-demand-prediction">Docker Image</a>
</p>

---

## Results

| Model | R² | MAE (MW) | MAPE | Training Time |
|-------|----|----------|------|---------------|
| **LightGBM** | 0.9457 | 1,207 | 4.35% | ~8s |
| **XGBoost** | 0.9446 | 1,220 | 4.41% | ~12s |

<p align="center">
  <img src="figures/06_actual_vs_predicted.png" alt="Actual vs Predicted" width="85%">
</p>

---

## Architecture

<p align="center">
  <img src="figures/06_shap_beeswarm.png" alt="SHAP Feature Importance" width="75%">
</p>

```
ENTSO-E API ──▶ Feature Engineering ──▶ Model Training ──▶ Evaluation
  (or Synthetic)    (54 features)        (XGB / LGBM)      (SHAP + MLflow)
                         │                    │                   │
                         ▼                    ▼                   ▼
                    Anomaly Detection    API Serving         Dashboard
                    (Isolation Forest)   (FastAPI)           (Streamlit)
```

---

## MLOps

<p align="center">
  <img src="figures/06_anomaly_comparison.png" alt="Anomaly Detection" width="80%">
</p>

| Layer | Tool | Status |
|-------|------|--------|
| CI/CD | GitHub Actions | 3-stage: test → train → build |
| Experiment Tracking | MLflow + SQLite | Params, metrics, models logged |
| API | FastAPI | `/health`, `/models`, `/predict` |
| Containerization | Docker | Multi-stage build → GHCR |
| Monitoring | Streamlit | Interactive anomaly dashboard |

### CI/CD Pipeline

```
  push ──▶ lint (ruff) ──▶ test (22 pytest) ──▶ train (XGB + LGBM) ──▶ build (Docker → GHCR)
```

Every commit to `main` automatically lints, tests, trains models on ENTSO-E data, and deploys a Docker image to GitHub Container Registry.

---

## Notebooks

| # | Notebook | What it covers |
|---|----------|----------------|
| 01 | [Data Ingestion](notebooks/01_data_ingestion.ipynb) | ENTSO-E API, EDA, statistical summaries |
| 02 | [Forecasting](notebooks/02_forecasting.ipynb) | XGBoost & LightGBM, SHAP analysis |
| 03 | [Anomaly Detection](notebooks/03_anomaly_detection.ipynb) | Isolation Forest, Z-score, IQR |
| 04 | [Classification](notebooks/04_classification.ipynb) | Peak/off-peak, ROC curves |
| 05 | [LLM Report](notebooks/05_llm_report.ipynb) | Automated report generation |
| 06 | [Diagnostics](notebooks/06_diagnostics.ipynb) | 13 diagnostic plots |

---

<details>
<summary><strong>Diagnostics</strong></summary>

<br>

| | | |
|:---:|:---:|:---:|
| ![Actual vs Predicted](figures/06_actual_vs_predicted.png) | ![Residual Analysis](figures/06_residual_analysis.png) | ![Rolling Error](figures/06_rolling_error.png) |
| ![Error by Hour](figures/06_error_by_hour.png) | ![SHAP Beeswarm](figures/06_shap_beeswarm.png) | ![SHAP Dependence](figures/06_shap_dependence.png) |
| ![SHAP Waterfall](figures/06_shap_waterfall.png) | ![Anomaly Comparison](figures/06_anomaly_comparison.png) | ![Anomaly Distributions](figures/06_anomaly_distributions.png) |
| ![Anomaly Agreement](figures/06_anomaly_agreement.png) | ![Classification Curves](figures/06_classification_curves.png) | ![Classification CM](figures/06_classification_cm_fi.png) |
| ![Cross Validation](figures/06_cv_comparison.png) | | |

</details>

---

## Quick Start

```bash
git clone https://github.com/ogiannopoulou/energy-demand-prediction.git
cd energy-demand-prediction && python -m venv .venv && source .venv/bin/activate
make setup          # install dependencies
make test           # run 22 tests
make train          # train models (falls back to synthetic without API key)
make api            # start FastAPI server on :8000
make docker-up      # full stack with Docker Compose
```

---

## Tech Stack

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white&style=for-the-badge" alt="Python">
  <img src="https://img.shields.io/badge/XGBoost-2.0-FF6B35?logo=xgboost&logoColor=white&style=for-the-badge" alt="XGBoost">
  <img src="https://img.shields.io/badge/LightGBM-4.0-FFA500?logo=lightgbm&logoColor=white&style=for-the-badge" alt="LightGBM">
  <img src="https://img.shields.io/badge/SHAP-0.45-purple?style=for-the-badge" alt="SHAP">
  <img src="https://img.shields.io/badge/MLflow-2.15-0199E3?logo=mlflow&logoColor=white&style=for-the-badge" alt="MLflow">
  <img src="https://img.shields.io/badge/FastAPI-0.100-009688?logo=fastapi&logoColor=white&style=for-the-badge" alt="FastAPI">
  <img src="https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white&style=for-the-badge" alt="Docker">
  <img src="https://img.shields.io/badge/GitHub%20Actions-2088FF?logo=githubactions&logoColor=white&style=for-the-badge" alt="GitHub Actions">
</p>

---

## Project Structure

```
.
├── src/
│   ├── data/                  # ENTSO-E client + 54-feature pipeline
│   ├── models/                # XGBoost, LightGBM, Isolation Forest, SHAP
│   ├── monitoring/            # MLflow experiment tracking
│   └── llm/                   # Automated report generation
├── api/                       # FastAPI prediction server
├── tests/                     # 22 tests (data, models, API, MLOps)
├── notebooks/                 # 6 Jupyter notebooks
├── .github/workflows/         # CI/CD pipeline
├── Dockerfile                 # Multi-stage Docker build
├── Makefile                   # Dev commands
└── pyproject.toml             # Project config
```

---

<p align="center">
  <strong>Ourania Giannopoulou</strong> · Applied Mathematics PhD<br>
  <a href="https://linkedin.com/in/rania-giannopoulou">LinkedIn</a> · <a href="https://github.com/ogiannopoulou">GitHub</a>
</p>
