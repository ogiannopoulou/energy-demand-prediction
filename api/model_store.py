"""Model store for loading and managing trained models."""
import json
import logging
from pathlib import Path

import pandas as pd
import skops.io as sio

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "models"


class ModelStore:
    """Manages trained models for serving."""

    def __init__(self):
        self.models: dict = {}
        self.metadata: dict = {}
        self._load_models()

    def _load_models(self):
        """Load all available models from models/ directory."""
        if not MODELS_DIR.exists():
            logger.warning("Models directory not found: %s", MODELS_DIR)
            return

        for model_path in MODELS_DIR.glob("*.skops"):
            model_name = model_path.stem
            try:
                model = sio.load(model_path, trusted=sio.get_untrusted_types(file=model_path))
                self.models[model_name] = model
                logger.info("Loaded model: %s", model_name)

                # Load metadata if exists
                meta_path = model_path.with_suffix(".json")
                if meta_path.exists():
                    with open(meta_path) as f:
                        self.metadata[model_name] = json.load(f)
                    logger.info("Loaded metadata for: %s", model_name)

            except Exception as e:
                logger.error("Failed to load model %s: %s", model_name, e)

    def get_model(self, name: str):
        """Get a model by name."""
        return self.models.get(name)

    def get_metadata(self, name: str) -> dict:
        """Get model metadata."""
        return self.metadata.get(name, {})

    def list_models(self) -> list[str]:
        """List all available model names."""
        return list(self.models.keys())

    def predict(self, model_name: str, features: pd.DataFrame) -> pd.Series:
        """Make prediction using specified model."""
        model = self.get_model(model_name)
        if model is None:
            raise ValueError(f"Model not found: {model_name}")
        return model.predict(features)


# Global instance
_store: ModelStore | None = None


def get_model_store() -> ModelStore:
    """Get or create the global model store."""
    global _store
    if _store is None:
        _store = ModelStore()
    return _store


def reload_models():
    """Force reload all models."""
    global _store
    _store = None
    return get_model_store()
