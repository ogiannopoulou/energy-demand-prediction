"""
MLOps - MLflow Experiment Tracking and Model Registry
"""
import logging
import os
from pathlib import Path
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("mlflow not installed, tracking features disabled")


MLFLOW_DIR = Path(__file__).parent.parent.parent / "mlflow_results"
MLFLOW_DB = MLFLOW_DIR / "mlflow.db"
MODEL_REGISTRY = "terna_energy_models"


def setup_mlflow(tracking_dir: str = None):
    """Configure MLflow tracking with SQLite backend."""
    if not MLFLOW_AVAILABLE:
        logger.warning("mlflow not installed, skipping tracking setup")
        return None
    
    if tracking_dir is None:
        tracking_dir = str(MLFLOW_DIR)
    
    os.makedirs(tracking_dir, exist_ok=True)
    db_path = Path(tracking_dir) / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    
    # Create or get experiment
    experiment_name = "italian_energy_forecast"
    mlflow.set_experiment(experiment_name)
    
    logger.info("MLflow tracking URI: sqlite:///%s", db_path)
    return mlflow


def log_experiment(model_name: str, model, metrics: dict, params: dict = None, 
                   feature_importance: pd.DataFrame = None, tags: dict = None,
                   register_model: bool = False):
    """Log an experiment run to MLflow."""
    if not MLFLOW_AVAILABLE:
        logger.warning("mlflow not available, skipping logging")
        return None
    
    with mlflow.start_run(run_name=model_name) as run:
        # Log parameters
        if params:
            mlflow.log_params(params)
            logger.info("Logged %d parameters", len(params))
        
        # Log metrics
        mlflow.log_metrics(metrics)
        logger.info("Logged metrics: %s", metrics)
        
        # Log tags
        if tags:
            mlflow.set_tags(tags)
        
        # Add default tags
        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("trained_at", datetime.now().isoformat())
        
        # Log model
        mlflow.sklearn.log_model(
            model, f"{model_name}_model",
            skops_trusted_types=[
                "xgboost.core.Booster", "xgboost.sklearn.XGBRegressor",
                "xgboost.sklearn.XGBClassifier",
                "lightgbm.sklearn.LGBMRegressor", "lightgbm.sklearn.LGBMClassifier",
                "lightgbm.basic.Booster",
                "sklearn.ensemble._forest.RandomForestRegressor",
                "sklearn.ensemble._forest.RandomForestClassifier",
                "sklearn.linear_model._base.LinearRegression",
                "sklearn.pipeline.Pipeline",
                "sklearn.preprocessing._standard.StandardScaler",
                "collections.OrderedDict",
            ]
        )
        
        # Log feature importance if provided
        if feature_importance is not None:
            importance_path = MLFLOW_DIR / f"{model_name}_feature_importance.csv"
            feature_importance.to_csv(importance_path, index=False)
            mlflow.log_artifact(str(importance_path))
        
        # Register model if requested
        if register_model:
            model_uri = f"runs:/{run.info.run_id}/{model_name}_model"
            try:
                result = mlflow.register_model(model_uri, MODEL_REGISTRY)
                logger.info("Registered model %s version %s", MODEL_REGISTRY, result.version)
            except Exception as e:
                logger.error("Failed to register model: %s", e)
        
        logger.info("Logged experiment: %s (run_id: %s)", model_name, run.info.run_id)
        return run.info.run_id


def get_best_model(metric: str = "rmse", ascending: bool = True):
    """Get the best model from MLflow experiments."""
    if not MLFLOW_AVAILABLE:
        logger.warning("mlflow not available")
        return None
    
    client = MlflowClient()
    
    # Get all runs
    experiment = mlflow.get_experiment_by_name("italian_energy_forecast")
    if experiment is None:
        logger.warning("No experiment found")
        return None
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
        max_results=1,
    )
    
    if not runs:
        logger.warning("No runs found")
        return None
    
    best_run = runs[0]
    logger.info("Best run: %s (metric: %.4f)", best_run.info.run_id, best_run.data.metrics.get(metric, 0))
    
    return best_run


def compare_experiments(results: list[dict] = None) -> pd.DataFrame:
    """Compare multiple experiment results."""
    if not MLFLOW_AVAILABLE:
        logger.warning("mlflow not available")
        return pd.DataFrame()
    
    if results is None:
        # Fetch from MLflow
        experiment = mlflow.get_experiment_by_name("italian_energy_forecast")
        if experiment is None:
            return pd.DataFrame()
        
        client = MlflowClient()
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        
        results = []
        for run in runs:
            result = {"run_id": run.info.run_id, "run_name": run.info.run_name}
            result.update(run.data.metrics)
            result.update(run.data.params)
            results.append(result)
    
    comparison = pd.DataFrame(results)
    if "rmse" in comparison.columns:
        comparison = comparison.sort_values("rmse")
    
    return comparison
