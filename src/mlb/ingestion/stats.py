"""Concrete stats provider with game log upsert logic."""

import logging
from datetime import date

import asyncpg

from mlb.db.models import Table
from mlb.db.pool import get_pool
from mlb.ingestion.base import GameLogRow, StatsProvider, ensure_player_exists

logger = logging.getLogger(__name__)


class V1StatsProvider(StatsProvider):
    """
    V1 concrete stats provider.

    Upserts game logs to player_game_logs table on (player_id, game_id).
    """

    async def fetch_game_logs(self, game_date: date) -> list[GameLogRow]:
        """
        Fetch game logs for all players on a date.

        Conservative fallback: on error, log warning and return empty list.

        Args:
            game_date: Date to fetch logs for

        Returns:
            List of GameLogRow objects
        """
        try:
            # Stub: In production, this would call an external API
            logger.info(f"Fetching game logs for date {game_date}")

            # Simulate API call that returns no data in v1
            rows = []

            return rows

        except Exception as e:
            # Conservative fallback: log and return empty
            logger.warning(
                f"Failed to fetch game logs for date {game_date}: {e}",
                exc_info=True,
            )
            return []

    async def write_game_logs(self, rows: list[GameLogRow]) -> None:
        """
        Write game log rows to database with upsert logic.

        Uses ON CONFLICT to update existing rows on (player_id, game_id).
        Ensures all players exist before writing (per D-020).

        Args:
            rows: List of GameLogRow objects to persist
        """
        if not rows:
            return

        # Ensure all players exist before writing game logs (D-020)
        unique_player_ids = {row.player_id for row in rows}

        pool = await get_pool()

        async with pool.acquire() as conn:
            # Ensure all players exist before writing game logs
            for player_id in unique_player_ids:
                await ensure_player_exists(conn, player_id)

            # Upsert statement
            upsert_sql = f"""
                INSERT INTO {Table.PLAYER_GAME_LOGS}
                (player_id, game_id, pa, ab, h, tb, hr, rbi, r, bb, k, ip_outs, er, pitch_count, is_starter)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                ON CONFLICT (player_id, game_id)
                DO UPDATE SET
                    pa = EXCLUDED.pa,
                    ab = EXCLUDED.ab,
                    h = EXCLUDED.h,
                    tb = EXCLUDED.tb,
                    hr = EXCLUDED.hr,
                    rbi = EXCLUDED.rbi,
                    r = EXCLUDED.r,
                    bb = EXCLUDED.bb,
                    k = EXCLUDED.k,
                    ip_outs = EXCLUDED.ip_outs,
                    er = EXCLUDED.er,
                    pitch_count = EXCLUDED.pitch_count,
                    is_starter = EXCLUDED.is_starter
            """

            await conn.executemany(
                upsert_sql,
                [
                    (
                        row.player_id,
                        row.game_id,
                        row.pa,
                        row.ab,
                        row.h,
                        row.tb,
                        row.hr,
                        row.rbi,
                        row.r,
                        row.bb,
                        row.k,
                        row.ip_outs,
                        row.er,
                        row.pitch_count,
                        row.is_starter,
                    )
                    for row in rows
                ],
            )

        logger.info(f"Upserted {len(rows)} game log rows to database")

    async def ensure_player_exists(
        self, player_id: int, name: str | None = None, team_id: int | None = None
    ) -> None:
        """
        Ensure player exists in players table (per D-020).

        Wrapper that delegates to shared helper function.

        Args:
            player_id: Player identifier
            name: Player name (optional)
            team_id: Team identifier (optional)
        """
        pool = await get_pool()
        async with pool.acquire() as conn:
            await ensure_player_exists(conn, player_id, name, team_id)
