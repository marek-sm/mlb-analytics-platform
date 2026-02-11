"""Model artifact save/load registry.

This module handles serialization and deserialization of trained models to disk.
Models are versioned by timestamp.
"""

import os
import pickle
from pathlib import Path

from mlb.config.settings import get_config


def _get_artifacts_dir() -> Path:
    """Get the models/artifacts directory path."""
    # Assume models/artifacts is at src/mlb/models/artifacts/
    base_dir = Path(__file__).parent / "artifacts"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def save_model(model, name: str) -> None:
    """Save a trained model to disk.

    Args:
        model: Trained LightGBM model
        name: Model name (e.g., "home_mu_20260210_120000")
    """
    artifacts_dir = _get_artifacts_dir()
    model_path = artifacts_dir / f"{name}.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved: {model_path}")


def load_model(name: str):
    """Load a trained model from disk.

    Args:
        name: Model name (e.g., "home_mu_20260210_120000")

    Returns:
        Loaded LightGBM model

    Raises:
        FileNotFoundError: If model artifact not found
    """
    artifacts_dir = _get_artifacts_dir()
    model_path = artifacts_dir / f"{name}.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model
