.PHONY: help setup test lint train api dashboard docker docker-up docker-down clean

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Install dependencies
	pip install -r requirements-dev.txt
	pip install -e .

test: ## Run tests with coverage
	pytest tests/ -v --cov=src --cov-report=term-missing

lint: ## Run linter
	ruff check src/ api/ tests/
	ruff format --check src/ api/ tests/

format: ## Format code
	ruff format src/ api/ tests/

train: ## Train models with real ENTSO-E data
	PYTHONPATH=. python -c "\
	from src.data.terna_loader import EntsoeClient;\
	from src.data.feature_engineering import build_feature_matrix, prepare_train_test, get_feature_columns;\
	from src.models.ml_models import train_xgboost_forecast, train_lightgbm_forecast;\
	from src.monitoring.mlops import setup_mlflow, log_experiment;\
	import skops.io as sio;\
	from datetime import datetime;\
	import json;\
	from pathlib import Path;\
	client = EntsoeClient();\
	demand = client.get_actual_load('202301010000', '202401010000');\
	df_features = build_feature_matrix(demand, target_col='demand_mw');\
	df_features = df_features.dropna();\
	train, test = prepare_train_test(df_features, test_ratio=0.2);\
	feature_cols = get_feature_columns(df_features, target_col='demand_mw');\
	X_train, y_train = train[feature_cols], train['demand_mw'];\
	X_test, y_test = test[feature_cols], test['demand_mw'];\
	setup_mlflow();\
	xgb_model, _, xgb_metrics = train_xgboost_forecast(X_train, y_train, X_test, y_test);\
	lgb_model, _, lgb_metrics = train_lightgbm_forecast(X_train, y_train, X_test, y_test);\
	log_experiment('XGBoost_Demand_Forecast', xgb_model, xgb_metrics);\
	log_experiment('LightGBM_Demand_Forecast', lgb_model, lgb_metrics);\
	models_dir = Path('models');\
	models_dir.mkdir(exist_ok=True);\
	for name, model, metrics in [('xgboost', xgb_model, xgb_metrics), ('lightgbm', lgb_model, lgb_metrics)]:\
	    sio.dump(model, models_dir / f'{name}_demand_forecast.skops');\
	    meta = {'name': f'{name}_demand_forecast', 'version': datetime.now().strftime('%Y%m%d%H%M%S'), 'trained_at': datetime.now().isoformat(), 'metrics': metrics, 'features': feature_cols};\
	    json.dump(meta, open(models_dir / f'{name}_demand_forecast.json', 'w'), indent=2);\
	print('Training complete!')"

api: ## Start FastAPI server
	UVICORN_LOG_LEVEL=info uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

dashboard: ## Start Streamlit dashboard
	streamlit run dashboard/app.py --server.port 8501

docker: ## Build Docker image
	docker build -t terna-energy-api .

docker-up: ## Start all services with docker-compose
	docker-compose up -d

docker-down: ## Stop all services
	docker-compose down

mlflow: ## Start MLflow UI
	mlflow ui --backend-store-uri sqlite:///mlflow_results/mlflow.db

clean: ## Clean generated files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage coverage.xml
	rm -rf *.egg-info dist build
