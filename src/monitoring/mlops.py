"""
MLOps - MLflow Experiment Tracking
"""
import os
from pathlib import Path
import pandas as pd

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


MLFLOW_DIR = Path(__file__).parent.parent.parent / "mlflow_results"
MLFLOW_DB = MLFLOW_DIR / "mlflow.db"


def setup_mlflow(tracking_dir: str = None):
    """Configure MLflow tracking with SQLite backend."""
    if not MLFLOW_AVAILABLE:
        print("mlflow not installed, skipping tracking setup")
        return
    
    if tracking_dir is None:
        tracking_dir = str(MLFLOW_DIR)
    
    os.makedirs(tracking_dir, exist_ok=True)
    db_path = Path(tracking_dir) / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    
    return mlflow


def log_experiment(model_name: str, model, metrics: dict, params: dict = None, 
                   feature_importance: pd.DataFrame = None, tags: dict = None):
    """Log an experiment run to MLflow."""
    if not MLFLOW_AVAILABLE:
        print("mlflow not available, skipping logging")
        return
    
    with mlflow.start_run(run_name=model_name):
        if params:
            mlflow.log_params(params)
        
        mlflow.log_metrics(metrics)
        
        if tags:
            mlflow.set_tags(tags)
        
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
        
        if feature_importance is not None:
            importance_path = MLFLOW_DIR / f"{model_name}_feature_importance.csv"
            feature_importance.to_csv(importance_path, index=False)
            mlflow.log_artifact(str(importance_path))
        
        print(f"Logged experiment: {model_name}")
        print(f"  Metrics: {metrics}")


def compare_experiments(results: list[dict]) -> pd.DataFrame:
    """Compare multiple experiment results."""
    comparison = pd.DataFrame(results)
    comparison = comparison.sort_values("rmse")
    
    return comparison
