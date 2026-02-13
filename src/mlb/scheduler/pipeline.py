"""Pipeline orchestration — end-to-end execution.

Implements:
- run_global() for three daily runs (night-before, morning, midday)
- run_game() for per-game and event-driven reruns
- run_daily_eval() for nightly evaluation trigger
"""

import asyncio
import logging
from datetime import date, datetime, timezone
from typing import Literal

import asyncpg

from mlb.config.settings import get_config
from mlb.db.models import Table
from mlb.db.pool import get_pool
from mlb.evaluation.backtest import run_backtest
from mlb.ingestion.games import V1GameProvider
from mlb.ingestion.lineups import V1LineupProvider
from mlb.ingestion.odds import V1OddsProvider
from mlb.ingestion.weather import V1WeatherProvider
from mlb.models.player_props import predict_hitters, predict_pitcher
from mlb.models.team_runs import predict as predict_team_runs
from mlb.odds.edge import compute_edges
from mlb.simulation.engine import simulate_game
from mlb.simulation.persistence import persist_simulation_results

logger = logging.getLogger(__name__)


async def run_global(run_type: Literal["night_before", "morning", "midday"]) -> None:
    """Execute a global pipeline run for all games on a given date.

    Steps:
        1. Fetch today's schedule
        2. Ingest odds for all games
        3. Ingest weather for all games
        4. For each game:
           a. Ingest latest lineups
           b. Build features + predict
           c. Simulate
           d. Compute edges

    Args:
        run_type: Type of global run ('night_before' | 'morning' | 'midday')

    Notes:
        - Uses current UTC date converted to ET for determining "today"
        - Skips postponed games
        - Conservative: if any ingestion fails after retries, log and skip
    """
    logger.info(f"Starting global run: {run_type}")

    # Determine target date (convert UTC to ET and get date)
    # Simplified: use UTC date for v1 (production would convert to ET)
    target_date = datetime.now(timezone.utc).date()

    pool = await get_pool()

    async with pool.acquire() as conn:
        # Step 1: Fetch today's schedule
        game_provider = V1GameProvider()
        game_rows = await _retry_ingestion(
            lambda: game_provider.fetch_schedule(target_date),
            "schedule",
        )

        if not game_rows:
            logger.info(f"No games scheduled for {target_date}")
            return

        # Write games to database
        await game_provider.write_games(game_rows)

        game_ids = [row.game_id for row in game_rows]
        logger.info(f"Found {len(game_ids)} games for {target_date}")

        # Step 2: Ingest odds for all games
        odds_provider = V1OddsProvider()
        for game_id in game_ids:
            odds_rows = await _retry_ingestion(
                lambda: odds_provider.fetch_odds(game_id),
                f"odds-{game_id}",
            )
            if odds_rows:
                await odds_provider.write_odds(odds_rows)

        # Step 3: Ingest weather for all games
        weather_provider = V1WeatherProvider()
        for game_id in game_ids:
            weather_row = await _retry_ingestion(
                lambda: weather_provider.fetch_weather(game_id),
                f"weather-{game_id}",
            )
            if weather_row:
                await weather_provider.write_weather([weather_row])

        # Step 4: Process each game (lineups, models, simulation, edge)
        for game_id in game_ids:
            # Check if game is postponed before processing
            game_status = await conn.fetchval(
                f"SELECT status FROM {Table.GAMES} WHERE game_id = $1",
                game_id,
            )

            if game_status == "postponed":
                logger.info(f"Skipping postponed game: {game_id}")
                continue

            await _process_game(conn, game_id)

    logger.info(f"Completed global run: {run_type}")


async def run_game(game_id: str) -> None:
    """Execute pipeline for a single game.

    Same as steps 2-4d of run_global, but for a single game.
    Called by per-game scheduler and event-driven reruns.

    Args:
        game_id: Game identifier

    Notes:
        - Checks game status before processing (skips postponed)
        - Used for T-90, T-30 runs and event-driven reruns
    """
    logger.info(f"Starting per-game run: {game_id}")

    pool = await get_pool()

    async with pool.acquire() as conn:
        # Check game status
        game_row = await conn.fetchrow(
            f"SELECT status, first_pitch FROM {Table.GAMES} WHERE game_id = $1",
            game_id,
        )

        if not game_row:
            logger.warning(f"Game not found: {game_id}")
            return

        if game_row["status"] == "postponed":
            logger.info(f"Skipping postponed game: {game_id}")
            return

        # Ingest odds
        odds_provider = V1OddsProvider()
        odds_rows = await _retry_ingestion(
            lambda: odds_provider.fetch_odds(game_id),
            f"odds-{game_id}",
        )
        if odds_rows:
            await odds_provider.write_odds(odds_rows)

        # Ingest weather
        weather_provider = V1WeatherProvider()
        weather_row = await _retry_ingestion(
            lambda: weather_provider.fetch_weather(game_id),
            f"weather-{game_id}",
        )
        if weather_row:
            await weather_provider.write_weather([weather_row])

        # Process game (lineups, models, simulation, edge)
        await _process_game(conn, game_id)

    logger.info(f"Completed per-game run: {game_id}")


async def run_daily_eval() -> None:
    """Trigger nightly evaluation after all games are final.

    Runs backtest for each major market type for today's date.

    Notes:
        - Should be triggered after the last game of the day is final
        - For v1, runs evaluation for the current date only
        - Writes results to eval_results table
    """
    logger.info("Starting daily evaluation")

    target_date = datetime.now(timezone.utc).date()
    pool = await get_pool()

    async with pool.acquire() as conn:
        # Check if there are final games for today
        final_count = await conn.fetchval(
            f"""
            SELECT COUNT(*)
            FROM {Table.GAMES}
            WHERE game_date = $1 AND status = 'final'
            """,
            target_date,
        )

        if not final_count or final_count == 0:
            logger.info(f"No final games for {target_date}, skipping evaluation")
            return

        # Run backtest for each major market
        markets = ["ml", "rl", "total", "team_total"]

        for market in markets:
            try:
                logger.info(f"Running backtest for market: {market}")
                eval_report = await run_backtest(
                    conn,
                    start_date=target_date,
                    end_date=target_date,
                    market=market,
                )

                # Persist evaluation results
                await _save_eval_report(conn, eval_report)

            except Exception as e:
                logger.error(f"Backtest failed for {market}: {e}", exc_info=True)

    logger.info("Completed daily evaluation")


async def _process_game(conn: asyncpg.Connection, game_id: str) -> None:
    """Process a single game: lineups → models → simulation → edge.

    Args:
        conn: Database connection
        game_id: Game identifier

    Notes:
        - Conservative: if any step fails, log error and skip remaining steps
        - Sets edge_computed_at timestamp for publishing gate
    """
    try:
        # Step 4a: Ingest lineups
        lineup_provider = V1LineupProvider()
        lineup_rows = await _retry_ingestion(
            lambda: lineup_provider.fetch_lineups(game_id),
            f"lineups-{game_id}",
        )
        if lineup_rows:
            await lineup_provider.write_lineups(lineup_rows)

        # Get game metadata for model inference
        game = await conn.fetchrow(
            f"""
            SELECT game_date, home_team_id, away_team_id, first_pitch
            FROM {Table.GAMES}
            WHERE game_id = $1
            """,
            game_id,
        )

        if not game:
            logger.error(f"Game metadata not found: {game_id}")
            return

        # Step 4b: Build features + predict
        # For v1, use a fixed model version (in v2, retrieve from registry)
        model_version = "latest"

        # Predict team runs (Unit 4)
        try:
            team_params = await predict_team_runs(conn, game_id, model_version)
        except ValueError as e:
            logger.warning(f"Team prediction failed for {game_id}: {e}")
            return

        # Step 4c: Simulate (Unit 6)
        try:
            sim_result = await simulate_game(
                conn,
                game_id,
                model_version,
            )
        except ValueError as e:
            logger.warning(f"Simulation failed for {game_id}: {e}")
            return

        # Save simulation results
        projection_id = await persist_simulation_results(conn, sim_result)

        # Step 4d: Compute edges (Unit 7)
        try:
            edge_result = await compute_edges(conn, projection_id)

            # Update projection with edge_computed_at timestamp
            await conn.execute(
                f"""
                UPDATE {Table.PROJECTIONS}
                SET edge_computed_at = $1
                WHERE projection_id = $2
                """,
                edge_result.computed_at,
                projection_id,
            )

            logger.info(
                f"Pipeline complete for {game_id}: "
                f"projection_id={projection_id}, "
                f"market_edges={len(edge_result.market_edges)}, "
                f"player_edges={len(edge_result.player_edges)}"
            )

        except Exception as e:
            logger.error(f"Edge computation failed for {game_id}: {e}", exc_info=True)

    except Exception as e:
        logger.error(f"Pipeline failed for {game_id}: {e}", exc_info=True)


async def _retry_ingestion(coro_func, label: str):
    """Retry an ingestion operation with exponential backoff.

    Args:
        coro_func: Async callable that returns ingestion result
        label: Label for logging

    Returns:
        Ingestion result or empty list/None on failure

    Notes:
        - Retries up to max_retry_attempts (from config)
        - Exponential backoff: 1s, 2s, 4s, ...
        - Conservative: returns empty on final failure
    """
    config = get_config()
    max_attempts = config.max_retry_attempts + 1  # +1 for initial attempt

    for attempt in range(max_attempts):
        try:
            result = await coro_func()

            # Check if result is empty (ingestion failure per D-019)
            if not result:
                if attempt < max_attempts - 1:
                    logger.warning(
                        f"Ingestion returned empty for {label}, "
                        f"retrying ({attempt + 1}/{max_attempts})"
                    )
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    logger.warning(
                        f"Ingestion failed for {label} after {max_attempts} attempts"
                    )
                    return [] if isinstance(result, list) else None

            return result

        except Exception as e:
            if attempt < max_attempts - 1:
                logger.warning(
                    f"Ingestion error for {label}: {e}, "
                    f"retrying ({attempt + 1}/{max_attempts})"
                )
                await asyncio.sleep(2 ** attempt)
            else:
                logger.error(
                    f"Ingestion failed for {label} after {max_attempts} attempts: {e}",
                    exc_info=True,
                )
                return []


async def _save_eval_report(
    conn: asyncpg.Connection,
    report,
) -> None:
    """Save evaluation report to eval_results table.

    Args:
        conn: Database connection
        report: EvalReport from backtest
    """
    await conn.execute(
        f"""
        INSERT INTO {Table.EVAL_RESULTS}
        (eval_date, market, start_date, end_date, sample_n,
         log_loss, brier_score, ece, tail_low_acc, tail_high_acc,
         median_clv, meta)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        """,
        report.eval_date,
        report.market,
        report.start_date,
        report.end_date,
        report.sample_n,
        report.log_loss,
        report.brier_score,
        report.ece,
        report.tail_low_acc,
        report.tail_high_acc,
        report.median_clv,
        report.meta,
    )

    logger.info(
        f"Saved eval report: {report.market} "
        f"{report.start_date} to {report.end_date}, "
        f"n={report.sample_n}"
    )
