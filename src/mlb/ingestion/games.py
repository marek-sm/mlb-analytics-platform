"""Concrete game/schedule provider."""

import logging
from datetime import date

import asyncpg

from mlb.db.models import Table
from mlb.db.pool import get_pool
from mlb.ingestion.base import GameProvider, GameRow

logger = logging.getLogger(__name__)


class V1GameProvider(GameProvider):
    """
    V1 concrete game/schedule provider.

    Upserts games to games table on game_id.
    Handles status changes (scheduled → postponed, etc.).
    """

    async def fetch_schedule(self, game_date: date) -> list[GameRow]:
        """
        Fetch game schedule for a date.

        Conservative fallback: on error, log warning and return empty list.

        Args:
            game_date: Date to fetch schedule for

        Returns:
            List of GameRow objects
        """
        try:
            # Stub: In production, this would call an external API
            logger.info(f"Fetching game schedule for date {game_date}")

            # Simulate API call that returns no data in v1
            rows = []

            return rows

        except Exception as e:
            # Conservative fallback: log and return empty
            logger.warning(
                f"Failed to fetch schedule for date {game_date}: {e}",
                exc_info=True,
            )
            return []

    async def write_games(self, rows: list[GameRow]) -> None:
        """
        Write game rows to database with upsert logic.

        Uses ON CONFLICT to update existing games on game_id.
        Handles status changes (e.g., scheduled → postponed).

        Args:
            rows: List of GameRow objects to persist
        """
        if not rows:
            return

        pool = await get_pool()

        async with pool.acquire() as conn:
            # Upsert statement
            upsert_sql = f"""
                INSERT INTO {Table.GAMES}
                (game_id, game_date, home_team_id, away_team_id, park_id, first_pitch, status, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, now())
                ON CONFLICT (game_id)
                DO UPDATE SET
                    game_date = EXCLUDED.game_date,
                    home_team_id = EXCLUDED.home_team_id,
                    away_team_id = EXCLUDED.away_team_id,
                    park_id = EXCLUDED.park_id,
                    first_pitch = EXCLUDED.first_pitch,
                    status = EXCLUDED.status,
                    updated_at = now()
            """

            await conn.executemany(
                upsert_sql,
                [
                    (
                        row.game_id,
                        row.game_date,
                        row.home_team_id,
                        row.away_team_id,
                        row.park_id,
                        row.first_pitch,
                        row.status,
                    )
                    for row in rows
                ],
            )

        logger.info(f"Upserted {len(rows)} games to database")
