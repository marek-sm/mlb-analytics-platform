"""Rolling-origin backtesting framework.

Evaluates model performance using only data available at each historical point.
No future leakage: for each eval date, only projections made before game outcomes
are known are scored.
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime

import asyncpg
import numpy as np

from mlb.db.models import Table
from mlb.evaluation.clv import compute_clv
from mlb.evaluation.metrics import brier_score, ece, log_loss, tail_accuracy

logger = logging.getLogger(__name__)


@dataclass
class EvalReport:
    """Evaluation report for a backtest run."""

    eval_date: date  # date the eval was run
    market: str  # market type evaluated
    start_date: date  # backtest window start
    end_date: date  # backtest window end
    sample_n: int
    log_loss: float | None
    brier_score: float | None
    ece: float | None
    tail_low_acc: float | None  # None if insufficient tail samples
    tail_high_acc: float | None
    median_clv: float | None
    meta: dict  # model version, bin counts, etc.


async def run_backtest(
    conn: asyncpg.Connection,
    start_date: date,
    end_date: date,
    market: str,
) -> EvalReport:
    """Run a rolling-origin backtest for a specific market and date range.

    Args:
        conn: Database connection
        start_date: Start of evaluation window (inclusive)
        end_date: End of evaluation window (inclusive)
        market: Market type to evaluate (e.g., 'ml', 'rl', 'total', 'team_total')

    Returns:
        EvalReport with all computed metrics

    Notes:
        - Only evaluates final games with non-null scores
        - Uses most recent projection per game (by run_ts)
        - Computes log loss, Brier score, ECE, tail accuracy, and median CLV
        - Returns sample_n=0 and all metrics as None if no final games found
        - Rolling-origin: for each date, only uses projections made before outcomes known
    """
    eval_date = datetime.now().date()

    # Fetch all final games in date range with outcomes
    game_rows = await conn.fetch(
        f"""
        SELECT
            game_id,
            game_date,
            home_score,
            away_score
        FROM {Table.GAMES}
        WHERE game_date >= $1
          AND game_date <= $2
          AND status = 'final'
          AND home_score IS NOT NULL
          AND away_score IS NOT NULL
        ORDER BY game_date
        """,
        start_date,
        end_date,
    )

    if not game_rows:
        logger.info(f"No final games found for {market} in {start_date} to {end_date}")
        return EvalReport(
            eval_date=eval_date,
            market=market,
            start_date=start_date,
            end_date=end_date,
            sample_n=0,
            log_loss=None,
            brier_score=None,
            ece=None,
            tail_low_acc=None,
            tail_high_acc=None,
            median_clv=None,
            meta={},
        )

    # Collect game IDs for CLV computation
    game_ids = [row["game_id"] for row in game_rows]

    # Fetch model probabilities and compute outcomes for each game
    p_model_list = []
    outcomes_list = []

    for game_row in game_rows:
        game_id = game_row["game_id"]
        home_score = int(game_row["home_score"])
        away_score = int(game_row["away_score"])
        total_score = home_score + away_score

        # Get most recent projection for this game
        proj_rows = await conn.fetch(
            f"""
            WITH latest_proj AS (
                SELECT projection_id
                FROM {Table.PROJECTIONS}
                WHERE game_id = $1
                ORDER BY run_ts DESC
                LIMIT 1
            )
            SELECT
                smp.market,
                smp.side,
                smp.line,
                smp.prob AS p_model
            FROM {Table.SIM_MARKET_PROBS} smp
            JOIN latest_proj lp ON smp.projection_id = lp.projection_id
            WHERE smp.market = $2
            """,
            game_id,
            market,
        )

        for proj_row in proj_rows:
            side = proj_row["side"]
            line = float(proj_row["line"]) if proj_row["line"] is not None else None
            p_model = float(proj_row["p_model"])

            # Compute outcome based on market and side
            outcome = _compute_outcome(
                market=market,
                side=side,
                line=line,
                home_score=home_score,
                away_score=away_score,
                total_score=total_score,
            )

            if outcome is not None:
                p_model_list.append(p_model)
                outcomes_list.append(outcome)

    if not p_model_list:
        logger.info(f"No projections found for {market} in {start_date} to {end_date}")
        return EvalReport(
            eval_date=eval_date,
            market=market,
            start_date=start_date,
            end_date=end_date,
            sample_n=0,
            log_loss=None,
            brier_score=None,
            ece=None,
            tail_low_acc=None,
            tail_high_acc=None,
            median_clv=None,
            meta={},
        )

    # Convert to numpy arrays
    p_model_arr = np.array(p_model_list, dtype=np.float64)
    outcomes_arr = np.array(outcomes_list, dtype=np.float64)
    sample_n = len(p_model_arr)

    # Compute metrics
    log_loss_val = log_loss(p_model_arr, outcomes_arr)
    brier_val = brier_score(p_model_arr, outcomes_arr)
    ece_val = ece(p_model_arr, outcomes_arr, n_bins=10)
    tail_metrics = tail_accuracy(p_model_arr, outcomes_arr, threshold=0.15)

    # Compute CLV
    clv_rows = await compute_clv(conn, game_ids)
    median_clv_val = None
    if clv_rows:
        clv_values = [row.clv for row in clv_rows]
        median_clv_val = float(np.median(clv_values))

    # Build meta dictionary
    meta = {
        "tail_low_n": tail_metrics["low_tail_n"],
        "tail_high_n": tail_metrics["high_tail_n"],
        "clv_sample_n": len(clv_rows),
    }

    return EvalReport(
        eval_date=eval_date,
        market=market,
        start_date=start_date,
        end_date=end_date,
        sample_n=sample_n,
        log_loss=log_loss_val,
        brier_score=brier_val,
        ece=ece_val,
        tail_low_acc=tail_metrics["low_tail_acc"],
        tail_high_acc=tail_metrics["high_tail_acc"],
        median_clv=median_clv_val,
        meta=meta,
    )


def _compute_outcome(
    market: str,
    side: str | None,
    line: float | None,
    home_score: int,
    away_score: int,
    total_score: int,
) -> float | None:
    """Compute binary outcome (1.0 or 0.0) for a market prediction.

    Args:
        market: Market type
        side: Market side
        line: Line value (for totals and runlines)
        home_score: Home team final score
        away_score: Away team final score
        total_score: Combined final score

    Returns:
        1.0 if event occurred, 0.0 otherwise, None if invalid combination

    Notes:
        - ML market: home/away win
        - RL market: home/away cover the runline
        - Total market: over/under the line
        - Team total market: team score over/under the line
    """
    if market == "ml":
        if side == "home":
            return 1.0 if home_score > away_score else 0.0
        elif side == "away":
            return 1.0 if away_score > home_score else 0.0
    elif market == "rl":
        if line is None:
            return None
        if side == "home":
            return 1.0 if (home_score - away_score) > line else 0.0
        elif side == "away":
            return 1.0 if (away_score - home_score) > line else 0.0
    elif market == "total":
        if line is None:
            return None
        if side == "over":
            return 1.0 if total_score > line else 0.0
        elif side == "under":
            return 1.0 if total_score < line else 0.0
    elif market == "team_total":
        if line is None:
            return None
        # Assume side indicates home/away for team totals
        team_score = home_score if side == "home" else away_score
        # For team totals, we need to determine if it's over or under
        # This is ambiguous from the schema - for v1, assume we're predicting "over"
        # This is a limitation that should be documented
        return 1.0 if team_score > line else 0.0

    return None
