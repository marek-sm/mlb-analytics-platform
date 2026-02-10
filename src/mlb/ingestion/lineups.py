"""Concrete lineup provider with confirmation flip logic."""

import logging
from datetime import datetime, timezone

import asyncpg

from mlb.db.models import Table
from mlb.db.pool import get_pool
from mlb.ingestion.base import LineupProvider, LineupRow, ensure_player_exists

logger = logging.getLogger(__name__)


class V1LineupProvider(LineupProvider):
    """
    V1 concrete lineup provider.

    Implements the lineup confirmation contract (D-011):
    When inserting a new confirmed lineup, set prior confirmed rows
    to is_confirmed = FALSE first.
    """

    async def fetch_lineup(self, game_id: str, team_id: int) -> list[LineupRow]:
        """
        Fetch lineup for a team in a game.

        Conservative fallback: on error, log warning and return empty list.

        Args:
            game_id: Game identifier
            team_id: Team identifier

        Returns:
            List of LineupRow objects
        """
        try:
            # Stub: In production, this would call an external API
            logger.info(f"Fetching lineup for game {game_id}, team {team_id}")

            # Simulate API call that returns no data in v1
            rows = []

            return rows

        except Exception as e:
            # Conservative fallback: log and return empty
            logger.warning(
                f"Failed to fetch lineup for game {game_id}, team {team_id}: {e}",
                exc_info=True,
            )
            return []

    async def is_confirmed(self, game_id: str, team_id: int) -> bool:
        """
        Check if lineup is confirmed by team.

        Args:
            game_id: Game identifier
            team_id: Team identifier

        Returns:
            True if lineup is officially confirmed
        """
        try:
            # Stub: In production, this would check provider metadata
            logger.info(
                f"Checking lineup confirmation for game {game_id}, team {team_id}"
            )

            # Default to False in stub implementation
            return False

        except Exception as e:
            logger.warning(
                f"Failed to check lineup confirmation for game {game_id}, team {team_id}: {e}",
                exc_info=True,
            )
            return False

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

    async def write_lineup(
        self, rows: list[LineupRow], is_confirmed: bool
    ) -> None:
        """
        Write lineup rows to database with confirmation flip logic.

        Per D-011: If is_confirmed=True, first set all prior confirmed rows
        for this (game_id, team_id) to is_confirmed=FALSE.

        Ensures all players exist before writing (per D-020).

        Args:
            rows: List of LineupRow objects to persist
            is_confirmed: Whether this lineup is officially confirmed
        """
        if not rows:
            return

        # All rows should be for same game_id and team_id
        game_id = rows[0].game_id
        team_id = rows[0].team_id

        # Ensure all players exist before writing lineups (D-020)
        unique_player_ids = {row.player_id for row in rows}

        pool = await get_pool()

        async with pool.acquire() as conn:
            async with conn.transaction():
                # Ensure all players exist before writing lineups
                for player_id in unique_player_ids:
                    await ensure_player_exists(conn, player_id, team_id=team_id)

                # If this is a confirmed lineup, flip prior confirmed rows to FALSE
                if is_confirmed:
                    flip_sql = f"""
                        UPDATE {Table.LINEUPS}
                        SET is_confirmed = FALSE
                        WHERE game_id = $1
                          AND team_id = $2
                          AND is_confirmed = TRUE
                    """
                    result = await conn.execute(flip_sql, game_id, team_id)
                    logger.info(
                        f"Flipped prior confirmed lineups for game {game_id}, team {team_id}: {result}"
                    )

                # Insert new lineup rows
                insert_sql = f"""
                    INSERT INTO {Table.LINEUPS}
                    (game_id, team_id, player_id, batting_order, is_confirmed, source_ts)
                    VALUES ($1, $2, $3, $4, $5, $6)
                """

                await conn.executemany(
                    insert_sql,
                    [
                        (
                            row.game_id,
                            row.team_id,
                            row.player_id,
                            row.batting_order,
                            row.is_confirmed,
                            row.source_ts,
                        )
                        for row in rows
                    ],
                )

        logger.info(
            f"Wrote {len(rows)} lineup rows for game {game_id}, team {team_id} "
            f"(confirmed={is_confirmed})"
        )
