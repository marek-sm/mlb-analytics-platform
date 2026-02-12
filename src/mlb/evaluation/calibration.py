"""Market-specific calibration models.

Fits lightweight calibration functions (isotonic regression or Platt scaling)
to improve probability calibration. Calibration models are saved and loaded
via the models registry for use in Unit 7 edge computation.
"""

import logging
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone

import asyncpg
import numpy as np
from numpy.typing import NDArray
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from mlb.db.models import Table
from mlb.models.registry import load_model, save_model

logger = logging.getLogger(__name__)


@dataclass
class CalibrationModel:
    """Calibration model for a specific market."""

    market: str
    method: str  # 'isotonic' | 'platt'
    fitted_at: datetime
    params: bytes | dict  # serialized calibrator (pickle or coefficients)


async def fit_calibration(
    conn: asyncpg.Connection,
    market: str,
    method: str = "isotonic",
    min_samples: int = 50,
) -> CalibrationModel | None:
    """Fit a calibration model for a specific market using historical data.

    Args:
        conn: Database connection
        market: Market type to calibrate (e.g., 'ml', 'rl', 'total')
        method: Calibration method ('isotonic' or 'platt')
        min_samples: Minimum number of samples required to fit (default 50)

    Returns:
        CalibrationModel if successful, None if insufficient data

    Raises:
        ValueError: If method is not 'isotonic' or 'platt'

    Notes:
        - Uses historical (p_model, outcome) pairs from final games only
        - Isotonic regression is nonparametric and handles non-monotonic miscalibration
        - Platt scaling is parametric (logistic regression) and smoother
        - Requires at least min_samples observations to fit
        - Saves model to registry using naming convention: calibration_{market}_{method}
    """
    if method not in ("isotonic", "platt"):
        raise ValueError(f"method must be 'isotonic' or 'platt', got {method}")

    # Fetch historical (p_model, outcome) pairs for this market
    # Join sim_market_probs -> projections -> games (final only)
    rows = await conn.fetch(
        f"""
        SELECT
            smp.prob AS p_model,
            CASE
                WHEN smp.side = 'home' THEN
                    CASE WHEN g.home_score > g.away_score THEN 1.0 ELSE 0.0 END
                WHEN smp.side = 'away' THEN
                    CASE WHEN g.away_score > g.home_score THEN 1.0 ELSE 0.0 END
                WHEN smp.market = 'total' AND smp.side = 'over' THEN
                    CASE WHEN (g.home_score + g.away_score) > smp.line THEN 1.0 ELSE 0.0 END
                WHEN smp.market = 'total' AND smp.side = 'under' THEN
                    CASE WHEN (g.home_score + g.away_score) < smp.line THEN 1.0 ELSE 0.0 END
                WHEN smp.market = 'team_total' AND smp.side = 'home' THEN
                    CASE WHEN g.home_score > smp.line THEN 1.0 ELSE 0.0 END
                WHEN smp.market = 'team_total' AND smp.side = 'away' THEN
                    CASE WHEN g.away_score > smp.line THEN 1.0 ELSE 0.0 END
                WHEN smp.market = 'rl' AND smp.side = 'home' THEN
                    CASE WHEN (g.home_score - g.away_score) > smp.line THEN 1.0 ELSE 0.0 END
                WHEN smp.market = 'rl' AND smp.side = 'away' THEN
                    CASE WHEN (g.away_score - g.home_score) > smp.line THEN 1.0 ELSE 0.0 END
                ELSE NULL
            END AS outcome
        FROM {Table.SIM_MARKET_PROBS} smp
        JOIN {Table.PROJECTIONS} p ON smp.projection_id = p.projection_id
        JOIN {Table.GAMES} g ON p.game_id = g.game_id
        WHERE smp.market = $1
          AND g.status = 'final'
          AND g.home_score IS NOT NULL
          AND g.away_score IS NOT NULL
        """,
        market,
    )

    if len(rows) < min_samples:
        logger.warning(
            f"Insufficient data for calibration: {len(rows)} samples < {min_samples} for market {market}"
        )
        return None

    # Extract arrays
    p_model = np.array([float(row["p_model"]) for row in rows])
    outcomes = np.array([float(row["outcome"]) for row in rows if row["outcome"] is not None])

    # Filter out None outcomes (shouldn't happen with the query, but defensive)
    valid_mask = ~np.isnan(outcomes)
    p_model = p_model[valid_mask]
    outcomes = outcomes[valid_mask]

    if len(p_model) < min_samples:
        logger.warning(
            f"Insufficient valid data for calibration: {len(p_model)} samples < {min_samples} for market {market}"
        )
        return None

    # Fit calibration model
    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        calibrator.fit(p_model, outcomes)
        params = pickle.dumps(calibrator)
    else:  # platt
        # Platt scaling: fit logistic regression on raw probabilities
        calibrator = LogisticRegression(solver="lbfgs", max_iter=1000)
        calibrator.fit(p_model.reshape(-1, 1), outcomes)
        params = {
            "coef": calibrator.coef_.tolist(),
            "intercept": calibrator.intercept_.tolist(),
        }

    fitted_at = datetime.now(timezone.utc)

    model = CalibrationModel(
        market=market,
        method=method,
        fitted_at=fitted_at,
        params=params,
    )

    # Save to registry
    model_name = f"calibration_{market}_{method}"
    save_model(model, model_name)
    logger.info(f"Calibration model saved: {model_name} ({len(p_model)} samples)")

    return model


def apply_calibration(
    p_raw: float | NDArray[np.float64],
    calibration_model: CalibrationModel,
) -> float | NDArray[np.float64]:
    """Apply calibration to raw probabilities.

    Args:
        p_raw: Raw probability or array of probabilities
        calibration_model: Fitted calibration model

    Returns:
        Calibrated probability or array of probabilities in [0, 1]

    Raises:
        ValueError: If calibration_model.method is not recognized

    Notes:
        - Isotonic: uses sklearn IsotonicRegression.predict
        - Platt: applies logistic transform with fitted coefficients
        - Output is clipped to [0, 1]
    """
    is_scalar = isinstance(p_raw, (int, float))
    p_array = np.atleast_1d(p_raw).astype(np.float64)

    if calibration_model.method == "isotonic":
        calibrator = pickle.loads(calibration_model.params)
        p_calibrated = calibrator.predict(p_array)
    elif calibration_model.method == "platt":
        params = calibration_model.params
        coef = np.array(params["coef"]).flatten()
        intercept = np.array(params["intercept"]).flatten()
        # Logistic transform: p = 1 / (1 + exp(-(coef * p_raw + intercept)))
        logit = coef[0] * p_array + intercept[0]
        p_calibrated = 1.0 / (1.0 + np.exp(-logit))
    else:
        raise ValueError(f"Unknown calibration method: {calibration_model.method}")

    # Clip to [0, 1]
    p_calibrated = np.clip(p_calibrated, 0.0, 1.0)

    if is_scalar:
        return float(p_calibrated[0])
    return p_calibrated


def load_calibration(market: str, method: str = "isotonic") -> CalibrationModel | None:
    """Load a calibration model from the registry.

    Args:
        market: Market type
        method: Calibration method

    Returns:
        CalibrationModel if found, None otherwise
    """
    model_name = f"calibration_{market}_{method}"
    try:
        model = load_model(model_name)
        return model
    except FileNotFoundError:
        logger.info(f"No calibration model found: {model_name}")
        return None
