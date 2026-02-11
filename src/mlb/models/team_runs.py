"""Team run-scoring models: training and inference.

This module owns model training, serialization, and inference for team run-scoring
predictions. Two models per game-half: μ (expected runs) and r (dispersion).
"""

import asyncpg
import lightgbm as lgb
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime

from mlb.config.settings import get_config
from mlb.db.models import Table
from mlb.models.features import GameFeatures, build_game_features
from mlb.models.registry import save_model, load_model


@dataclass
class TeamRunParams:
    """Output of inference, input to Unit 6 simulation engine.

    Park factor is applied exactly once in this unit (D-010).
    """

    game_id: str
    home_mu: float  # expected runs, park-adjusted
    away_mu: float
    home_disp: float  # Negative Binomial dispersion parameter (r)
    away_disp: float
    model_version: str  # artifact identifier


class TeamRunModel:
    """Wrapper for home/away μ and r models."""

    def __init__(self):
        self.home_mu_model = None
        self.away_mu_model = None
        self.home_disp_model = None
        self.away_disp_model = None
        self.model_version = None

    def is_trained(self) -> bool:
        """Check if all four models are trained."""
        return all(
            [
                self.home_mu_model,
                self.away_mu_model,
                self.home_disp_model,
                self.away_disp_model,
            ]
        )


def _features_to_array(features: GameFeatures, is_home: bool) -> pd.DataFrame:
    """Convert GameFeatures to DataFrame with named columns for model input (D-026).

    Args:
        features: GameFeatures dataclass
        is_home: If True, use home team features; otherwise away team features

    Returns:
        Single-row DataFrame with feature values and named columns
    """
    # Handle None values for weather (dome/retractable parks)
    temp_f = features.temp_f if features.temp_f is not None else 72
    wind_speed = features.wind_speed_mph if features.wind_speed_mph is not None else 5
    precip_pct = features.precip_pct if features.precip_pct is not None else 0

    # Wind direction as categorical (simplified to N/S/E/W/NE/NW/SE/SW)
    wind_dir_map = {
        "N": 0,
        "NE": 1,
        "E": 2,
        "SE": 3,
        "S": 4,
        "SW": 5,
        "W": 6,
        "NW": 7,
    }
    wind_dir = wind_dir_map.get(features.wind_dir, 0) if features.wind_dir else 0

    if is_home:
        feature_values = [
            features.park_factor,
            1 if features.is_outdoor else 0,
            temp_f,
            wind_speed,
            wind_dir,
            precip_pct,
            features.home_starter_rest,
            features.home_starter_pitch_ct_avg,
            features.home_starter_era_recent,
            features.away_starter_era_recent,  # opponent pitcher
            features.home_lineup_ops,
            features.home_bullpen_usage,
            features.home_run_env,
            features.away_run_env,  # opponent run environment
        ]
    else:
        feature_values = [
            features.park_factor,
            1 if features.is_outdoor else 0,
            temp_f,
            wind_speed,
            wind_dir,
            precip_pct,
            features.away_starter_rest,
            features.away_starter_pitch_ct_avg,
            features.away_starter_era_recent,
            features.home_starter_era_recent,  # opponent pitcher
            features.away_lineup_ops,
            features.away_bullpen_usage,
            features.away_run_env,
            features.home_run_env,  # opponent run environment
        ]

    # Get feature names and create DataFrame
    feature_names = GameFeatures.feature_names(is_home)
    return pd.DataFrame([feature_values], columns=feature_names, dtype=np.float32)


async def train(conn: asyncpg.Connection) -> str:
    """Train all four models on historical data and serialize to disk.

    Args:
        conn: Database connection

    Returns:
        Model version string (timestamp-based artifact identifier)

    Raises:
        ValueError: If insufficient training data (< 30 games)
    """
    config = get_config()

    # Fetch historical games with final scores
    games = await conn.fetch(
        f"""
        SELECT game_id, home_score, away_score
        FROM {Table.GAMES}
        WHERE status = 'final'
          AND home_score IS NOT NULL
          AND away_score IS NOT NULL
        ORDER BY game_date DESC
        LIMIT 1000
        """
    )

    if len(games) < 30:
        raise ValueError(f"Insufficient training data: {len(games)} games (need ≥30)")

    # Build feature vectors
    home_features = []
    away_features = []
    home_targets_mu = []
    away_targets_mu = []

    for game in games:
        try:
            features = await build_game_features(conn, game["game_id"])
        except ValueError:
            # Skip games without confirmed lineups or starters
            continue

        home_features.append(_features_to_array(features, is_home=True))
        away_features.append(_features_to_array(features, is_home=False))
        home_targets_mu.append(game["home_score"])
        away_targets_mu.append(game["away_score"])

    if len(home_features) < 30:
        raise ValueError(
            f"Only {len(home_features)} games with complete features (need ≥30)"
        )

    X_home = pd.concat(home_features, ignore_index=True)
    X_away = pd.concat(away_features, ignore_index=True)
    y_home_mu = np.array(home_targets_mu, dtype=np.float32)
    y_away_mu = np.array(away_targets_mu, dtype=np.float32)

    # Train μ models (expected runs)
    home_mu_model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
    )
    away_mu_model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
    )

    home_mu_model.fit(X_home, y_home_mu)
    away_mu_model.fit(X_away, y_away_mu)

    # Train dispersion models (D-024: separate from μ, same features)
    # Use residuals as proxy for dispersion
    home_mu_pred = home_mu_model.predict(X_home)
    away_mu_pred = away_mu_model.predict(X_away)
    home_residuals = np.abs(y_home_mu - home_mu_pred)
    away_residuals = np.abs(y_away_mu - away_mu_pred)

    home_disp_model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=50,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )
    away_disp_model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=50,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )

    home_disp_model.fit(X_home, home_residuals)
    away_disp_model.fit(X_away, away_residuals)

    # Calculate training metrics (RMSE)
    home_rmse = np.sqrt(np.mean((y_home_mu - home_mu_pred) ** 2))
    away_rmse = np.sqrt(np.mean((y_away_mu - away_mu_pred) ** 2))

    print(f"Training complete: {len(home_features)} games")
    print(f"Home RMSE: {home_rmse:.3f}")
    print(f"Away RMSE: {away_rmse:.3f}")

    # Generate version identifier
    model_version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # Save models to disk
    save_model(home_mu_model, f"home_mu_{model_version}")
    save_model(away_mu_model, f"away_mu_{model_version}")
    save_model(home_disp_model, f"home_disp_{model_version}")
    save_model(away_disp_model, f"away_disp_{model_version}")

    return model_version


async def predict(
    conn: asyncpg.Connection, game_id: str, model_version: str
) -> TeamRunParams:
    """Run inference on a game and return TeamRunParams.

    Args:
        conn: Database connection
        game_id: Game identifier
        model_version: Model artifact version to load

    Returns:
        TeamRunParams with park-adjusted μ and dispersion parameters

    Raises:
        ValueError: If game not found or models not trained
    """
    # Build features
    features = await build_game_features(conn, game_id)

    # Load models from disk
    home_mu_model = load_model(f"home_mu_{model_version}")
    away_mu_model = load_model(f"away_mu_{model_version}")
    home_disp_model = load_model(f"home_disp_{model_version}")
    away_disp_model = load_model(f"away_disp_{model_version}")

    # Prepare feature DataFrames
    X_home = _features_to_array(features, is_home=True)
    X_away = _features_to_array(features, is_home=False)

    # Predict μ (expected runs) - models output base μ
    home_mu_base = float(home_mu_model.predict(X_home)[0])
    away_mu_base = float(away_mu_model.predict(X_away)[0])

    # Apply park factor exactly once (D-010)
    home_mu_adjusted = home_mu_base * features.park_factor
    away_mu_adjusted = away_mu_base * features.park_factor

    # Clamp to plausible bounds after park adjustment
    home_mu = np.clip(home_mu_adjusted, 0.5, 15.0)
    away_mu = np.clip(away_mu_adjusted, 0.5, 15.0)

    # Predict dispersion
    home_disp_raw = float(home_disp_model.predict(X_home)[0])
    away_disp_raw = float(away_disp_model.predict(X_away)[0])

    # Ensure dispersion > 0
    home_disp = max(home_disp_raw, 0.1)
    away_disp = max(away_disp_raw, 0.1)

    return TeamRunParams(
        game_id=game_id,
        home_mu=home_mu,
        away_mu=away_mu,
        home_disp=home_disp,
        away_disp=away_disp,
        model_version=model_version,
    )
