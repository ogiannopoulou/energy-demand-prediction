"""
Model Explainability with SHAP
"""
import pandas as pd
import numpy as np

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def compute_shap_values(model, X: pd.DataFrame, model_type: str = "tree"):
    """Compute SHAP values for a tree-based model."""
    if not SHAP_AVAILABLE:
        raise ImportError("shap is required: pip install shap")
    
    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.KernelExplainer(model.predict, shap.sample(X, 100))
    
    shap_values = explainer.shap_values(X)
    
    return explainer, shap_values


def get_feature_importance(shap_values, feature_names: list) -> pd.DataFrame:
    """Extract feature importance from SHAP values."""
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": np.abs(shap_values).mean(axis=0),
    }).sort_values("importance", ascending=False)
    
    return importance


def explain_prediction(explainer, X_single: pd.DataFrame, feature_names: list):
    """Explain a single prediction."""
    shap_values = explainer.shap_values(X_single)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    explanation = pd.DataFrame({
        "feature": feature_names,
        "value": X_single.values[0],
        "shap_value": shap_values[0],
    }).sort_values("shap_value", key=abs, ascending=False)
    
    return explanation
