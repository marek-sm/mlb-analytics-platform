"""Historical backfill orchestration script.

Populates the database with ≥30 completed games and all associated data
(lineups, game logs, odds, weather), then triggers initial model training
(Units 4 + 5).

This module orchestrates existing ingestion adapters in FK-safe order.
It does not change schema.

Usage:
    python -m mlb.operations.backfill --start 2025-06-01 --end 2025-06-21
"""

import argparse
import asyncio
import logging
import os
from datetime import date, timedelta
from pathlib import Path

import mlb.models.player_props as player_props_module
import mlb.models.team_runs as team_runs_module
from mlb.db.models import Table
from mlb.db.pool import get_pool
from mlb.ingestion.games import V1GameProvider
from mlb.ingestion.lineups import V1LineupProvider
from mlb.ingestion.odds import V1OddsProvider
from mlb.ingestion.stats import V1StatsProvider
from mlb.ingestion.weather import V1WeatherProvider

logger = logging.getLogger(__name__)


def _daterange(start: date, end: date):
    """Yield each date from start to end inclusive."""
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


async def run_backfill(start_date: date, end_date: date) -> None:
    """Populate the database with ≥30 completed games and all associated data.

    Executes seven phases in FK-safe order:
      1. Schedule ingestion for the full date range.
      2. Lineup ingestion — two calls per final game (home + away team).
      3. Game-log (stats) ingestion, one call per date with final games.
      4. Odds ingestion per final game via historical endpoint (D-067).
      5. Weather ingestion per final game (best-effort, outdoor parks only).
      6. Verification queries — fail if games < 30 or game_logs = 0.
      7. Model training for team_runs (Unit 4) and player_props (Unit 5).

    All adapter failures ([] or None returns, exceptions) are logged as
    WARNING and skipped; the backfill never aborts due to a single failure.

    Args:
        start_date: First date to ingest (inclusive).
        end_date:   Last date to ingest (inclusive).

    Raises:
        RuntimeError: If fewer than 30 final games are found after Phase 1,
                      or if verification queries in Phase 6 fail.
    """
    game_provider = V1GameProvider()
    lineup_provider = V1LineupProvider()
    stats_provider = V1StatsProvider()
    odds_provider = V1OddsProvider()
    weather_provider = V1WeatherProvider()

    # -------------------------------------------------------------------------
    # Phase 1: Schedule
    # -------------------------------------------------------------------------
    logger.info("Phase 1: Ingesting schedule for %s → %s", start_date, end_date)
    all_game_rows = []

    for current_date in _daterange(start_date, end_date):
        try:
            rows = await game_provider.fetch_schedule(current_date)
            await game_provider.write_games(rows)
            all_game_rows.extend(rows)
            logger.info("  %s: %d game(s) ingested.", current_date, len(rows))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Schedule ingest failed for %s: %s", current_date, exc)
        await asyncio.sleep(1)

    # Verify ≥30 final games landed in the database.
    pool = await get_pool()
    async with pool.acquire() as conn:
        final_count = await conn.fetchval(
            f"SELECT COUNT(*) FROM {Table.GAMES} "
            f"WHERE status = 'final' AND game_date BETWEEN $1 AND $2",
            start_date,
            end_date,
        )

    if final_count < 30:
        raise RuntimeError(
            f"Insufficient final games: {final_count} found in "
            f"{start_date}–{end_date} (need ≥30). Widen the date range and retry."
        )

    logger.info("Phase 1 complete: %d final game(s) confirmed in DB.", final_count)

    # Collect final games from memory for subsequent phases (avoids extra DB round-trip).
    final_games = [g for g in all_game_rows if g.status == "final"]

    # -------------------------------------------------------------------------
    # Phase 2: Lineups
    # Two calls per game — once for home team, once for away team.
    # The boxscore cache (D-060) makes the second call cheap.
    # fetch_lineup() handles DB persistence internally (_persist_lineups).
    # -------------------------------------------------------------------------
    logger.info("Phase 2: Ingesting lineups for %d final game(s).", len(final_games))

    for game in final_games:
        for team_id in (game.home_team_id, game.away_team_id):
            try:
                await lineup_provider.fetch_lineup(game.game_id, team_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Lineup ingest failed for game %s, team %s: %s",
                    game.game_id,
                    team_id,
                    exc,
                )
        await asyncio.sleep(0.5)

    logger.info("Phase 2 complete.")

    # -------------------------------------------------------------------------
    # Phase 3: Stats (game logs)
    # One fetch_game_logs call per unique date that has final games.
    # D-059 roster discovery captures starters, relievers, pinch hitters, and
    # substitutions via the boxscore endpoint.  D-020 auto-creates any unknown
    # players encountered during log ingestion.
    # -------------------------------------------------------------------------
    final_dates = sorted({g.game_date for g in final_games})
    logger.info("Phase 3: Ingesting stats for %d date(s).", len(final_dates))

    for game_date in final_dates:
        try:
            logs = await stats_provider.fetch_game_logs(game_date)
            await stats_provider.write_game_logs(logs)
            logger.info("  %s: %d game-log row(s).", game_date, len(logs))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Stats ingest failed for %s: %s", game_date, exc)
        await asyncio.sleep(1)

    logger.info("Phase 3 complete.")

    # -------------------------------------------------------------------------
    # Phase 4: Odds
    # Pass game.first_pitch as event_date so the adapter routes to the
    # historical endpoint (/v4/historical/sports/baseball_mlb/odds?date=...)
    # rather than the live feed (D-067).  ~10 credits per call on the $30/month
    # plan; 315 games × 10 = 3,150 credits — well within the 20k monthly limit.
    # -------------------------------------------------------------------------
    logger.info("Phase 4: Ingesting odds for %d final game(s).", len(final_games))

    for game in final_games:
        try:
            odds = await odds_provider.fetch_odds(
                game.game_id, event_date=game.first_pitch
            )
            if odds:
                await odds_provider._write_odds(odds)  # noqa: SLF001
                logger.info(
                    "  game %s: %d odds row(s) ingested.", game.game_id, len(odds)
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Odds ingest failed for game %s: %s", game.game_id, exc)
        await asyncio.sleep(0.5)

    logger.info("Phase 4 complete.")

    # -------------------------------------------------------------------------
    # Phase 5: Weather  (best-effort)
    # Indoor and retractable-roof parks return None (D-018).  No retry on None
    # per D-066.  Open-Meteo archive has a 1-5 day lag (FC-43); very recent
    # games may return None — this is acceptable.
    # -------------------------------------------------------------------------
    logger.info("Phase 5: Ingesting weather for %d final game(s).", len(final_games))

    for game in final_games:
        try:
            weather = await weather_provider.fetch_weather(game.game_id, game.park_id)
            if weather is not None:
                await weather_provider.write_weather(weather)
                logger.info("  game %s: weather ingested.", game.game_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Weather ingest failed for game %s: %s", game.game_id, exc
            )
        await asyncio.sleep(0.5)

    logger.info("Phase 5 complete.")

    # -------------------------------------------------------------------------
    # Phase 6: Verify
    # Fail only if final games < 30 or player_game_logs is empty.
    # -------------------------------------------------------------------------
    logger.info("Phase 6: Running verification queries.")

    async with pool.acquire() as conn:
        games_count = await conn.fetchval(
            f"SELECT COUNT(*) FROM {Table.GAMES} WHERE status = 'final'"
        )
        logs_count = await conn.fetchval(
            f"SELECT COUNT(*) FROM {Table.PLAYER_GAME_LOGS}"
        )
        players_count = await conn.fetchval(
            f"SELECT COUNT(DISTINCT player_id) FROM {Table.PLAYERS}"
        )
        lineups_count = await conn.fetchval(
            f"SELECT COUNT(*) FROM {Table.LINEUPS}"
        )
        weather_count = await conn.fetchval(
            f"SELECT COUNT(*) FROM {Table.WEATHER}"
        )

    logger.info(
        "Verification counts — games=%s, game_logs=%s, players=%s, "
        "lineups=%s, weather=%s",
        games_count,
        logs_count,
        players_count,
        lineups_count,
        weather_count,
    )

    if games_count < 30:
        raise RuntimeError(
            f"Verification failed: only {games_count} final games in DB (need ≥30)."
        )
    if logs_count == 0:
        raise RuntimeError(
            "Verification failed: player_game_logs table is empty. "
            "Stats ingestion (Phase 3) may have failed for all dates."
        )

    logger.info("Phase 6 complete.")

    # -------------------------------------------------------------------------
    # Phase 7: Train models
    # Both train() functions accept an asyncpg Connection and return a version
    # string used to identify saved artifacts.
    # -------------------------------------------------------------------------
    logger.info("Phase 7: Training models.")

    async with pool.acquire() as conn:
        team_version = await team_runs_module.train(conn)
        logger.info("  team_runs model trained: version %s", team_version)

        props_version = await player_props_module.train(conn)
        logger.info("  player_props model trained: version %s", props_version)

    # Verify artifact files were serialized to disk.
    artifacts_dir = Path("models") / "artifacts"
    artifacts = [f for f in os.listdir(artifacts_dir) if f.endswith(".pkl")]
    if not artifacts:
        raise RuntimeError(
            f"No model artifacts (.pkl) found in {artifacts_dir}/. "
            "Training may have completed but failed to serialize."
        )

    logger.info(
        "Phase 7 complete: %d artifact(s) in %s/", len(artifacts), artifacts_dir
    )
    logger.info("Backfill complete.")


def main() -> None:
    """CLI entry point for the historical backfill."""
    parser = argparse.ArgumentParser(
        description="MLB Analytics — Historical Backfill (≥30 final games)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python -m mlb.operations.backfill --start 2025-06-01 --end 2025-06-21"
        ),
    )
    parser.add_argument(
        "--start",
        required=True,
        metavar="YYYY-MM-DD",
        help="First date of the backfill range (inclusive).",
    )
    parser.add_argument(
        "--end",
        required=True,
        metavar="YYYY-MM-DD",
        help="Last date of the backfill range (inclusive).",
    )
    args = parser.parse_args()

    try:
        start_date = date.fromisoformat(args.start)
        end_date = date.fromisoformat(args.end)
    except ValueError as exc:
        parser.error(f"Invalid date format: {exc}")
        return  # unreachable — parser.error() exits

    if start_date > end_date:
        parser.error("--start must be ≤ --end")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    asyncio.run(run_backfill(start_date, end_date))


if __name__ == "__main__":
    main()
