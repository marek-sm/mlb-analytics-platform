"""Closing Line Value (CLV) computation.

CLV measures how model probabilities compare to the closing market probability.
Positive CLV indicates the model beat the closing line (favorable).
"""

import logging
from dataclasses import dataclass
from datetime import timedelta

import asyncpg

from mlb.db.models import Table
from mlb.odds.devig import proportional_devig

logger = logging.getLogger(__name__)


@dataclass
class CLVRow:
    """CLV result for a single projection."""

    prob_id: int
    game_id: str
    market: str
    p_model: float
    p_close_fair: float  # fair prob from closing odds (T-5 min)
    clv: float  # p_model - p_close_fair (positive = model beat the close)


async def compute_clv(
    conn: asyncpg.Connection,
    game_ids: list[str],
) -> list[CLVRow]:
    """Compute CLV for all projections associated with the given games.

    Args:
        conn: Database connection
        game_ids: List of game IDs to evaluate

    Returns:
        List of CLVRow objects, one per (projection, market) pair with closing odds

    Notes:
        - Closing line is defined as latest odds snapshot at T-5 minutes before first_pitch
        - If no snapshot exists within 30 minutes before first pitch, game is excluded
        - Devig uses proportional method (D-036)
        - Only markets with both sides available from same book are included (D-039)
        - Games without final status are excluded from CLV computation
    """
    if not game_ids:
        return []

    # Get games with first_pitch and final status
    game_rows = await conn.fetch(
        f"""
        SELECT game_id, first_pitch
        FROM {Table.GAMES}
        WHERE game_id = ANY($1)
          AND status = 'final'
          AND first_pitch IS NOT NULL
        """,
        game_ids,
    )

    if not game_rows:
        logger.info("No final games with first_pitch found for CLV computation")
        return []

    clv_rows: list[CLVRow] = []

    for game_row in game_rows:
        game_id = game_row["game_id"]
        first_pitch = game_row["first_pitch"]

        # Define closing window: T-30 to T-5 minutes before first pitch
        close_cutoff = first_pitch - timedelta(minutes=5)
        close_window_start = first_pitch - timedelta(minutes=30)

        # Get latest odds snapshot within closing window for each (book, market, side)
        # We need to find the latest snapshot per book/market/side, then devig
        odds_rows = await conn.fetch(
            f"""
            WITH ranked_odds AS (
                SELECT
                    snapshot_id,
                    book,
                    market,
                    side,
                    line,
                    price,
                    snapshot_ts,
                    ROW_NUMBER() OVER (
                        PARTITION BY book, market, side, line
                        ORDER BY snapshot_ts DESC
                    ) AS rn
                FROM {Table.ODDS_SNAPSHOTS}
                WHERE game_id = $1
                  AND snapshot_ts >= $2
                  AND snapshot_ts <= $3
            )
            SELECT
                book,
                market,
                side,
                line,
                price,
                snapshot_ts
            FROM ranked_odds
            WHERE rn = 1
            ORDER BY book, market, line, side
            """,
            game_id,
            close_window_start,
            close_cutoff,
        )

        if not odds_rows:
            logger.debug(
                f"No closing odds found for {game_id} in window {close_window_start} to {close_cutoff}"
            )
            continue

        # Group odds by (book, market, line) and apply devig
        closing_fair_probs = _devig_closing_odds(odds_rows)

        # Get model probabilities from most recent projection for this game
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
                smp.prob_id,
                smp.market,
                smp.side,
                smp.line,
                smp.prob AS p_model
            FROM {Table.SIM_MARKET_PROBS} smp
            JOIN latest_proj lp ON smp.projection_id = lp.projection_id
            """,
            game_id,
        )

        # Match model probs to closing fair probs
        for proj_row in proj_rows:
            market = proj_row["market"]
            side = proj_row["side"]
            line = float(proj_row["line"]) if proj_row["line"] is not None else None
            p_model = float(proj_row["p_model"])

            # Find matching closing fair prob
            key = (market, side, line)
            if key in closing_fair_probs:
                p_close_fair = closing_fair_probs[key]
                clv = p_model - p_close_fair

                clv_rows.append(
                    CLVRow(
                        prob_id=proj_row["prob_id"],
                        game_id=game_id,
                        market=market,
                        p_model=p_model,
                        p_close_fair=p_close_fair,
                        clv=clv,
                    )
                )

    return clv_rows


def _devig_closing_odds(
    odds_rows: list[asyncpg.Record],
) -> dict[tuple[str, str | None, float | None], float]:
    """Apply devig to closing odds and return fair probabilities.

    Args:
        odds_rows: List of odds records with columns: book, market, side, line, price

    Returns:
        Dictionary mapping (market, side, line) to fair probability

    Notes:
        - Groups odds by (book, market, line) and applies proportional devig
        - Only includes markets where a book provides both sides (D-039)
        - Returns empty dict if no valid two-sided markets found
    """
    # Group by (book, market, line)
    groups: dict[tuple[str, str, float | None], list[asyncpg.Record]] = {}
    for row in odds_rows:
        book = row["book"]
        market = row["market"]
        line = float(row["line"]) if row["line"] is not None else None
        key = (book, market, line)
        if key not in groups:
            groups[key] = []
        groups[key].append(row)

    fair_probs: dict[tuple[str, str | None, float | None], float] = {}

    # Apply devig to each group that has both sides
    for (book, market, line), rows in groups.items():
        if len(rows) < 2:
            # Not enough sides from this book - skip
            continue

        # Extract prices and sides
        prices = [float(row["price"]) for row in rows]
        sides = [row["side"] for row in rows]

        try:
            devigged = proportional_devig(prices)
            for side, fair_prob in zip(sides, devigged):
                key = (market, side, line)
                fair_probs[key] = fair_prob
        except ValueError as e:
            logger.warning(f"Devig failed for {book} {market} {line}: {e}")
            continue

    return fair_probs
