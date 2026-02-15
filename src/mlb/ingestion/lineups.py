"""Concrete lineup provider with confirmation flip logic."""

import asyncio
import logging
from datetime import datetime, timezone

import aiohttp
import asyncpg

from mlb.config.settings import get_config
from mlb.db.models import Table
from mlb.db.pool import get_pool
from mlb.ingestion.base import LineupProvider, LineupRow, ensure_player_exists

logger = logging.getLogger(__name__)


def _is_confirmed(game_status: str, player_count: int) -> bool:
    """
    Determine if lineup should be marked as confirmed.

    Rules:
    - Games started/finished (L, F) → always confirmed
    - Pre-game (P, S) with exactly 9 players → confirmed
    - Otherwise → unconfirmed (partial or postponed)

    Args:
        game_status: Abstract game code from MLB API (F, L, P, S, D)
        player_count: Number of players in lineup

    Returns:
        True if lineup should be confirmed
    """
    if game_status in ("L", "F"):  # Live or Final
        return True
    if game_status in ("P", "S") and player_count == 9:  # Preview/Scheduled with full lineup
        return True
    return False


def _parse_batting_order(player_data: dict, batting_order_array: list[int], player_id: int) -> int | None:
    """
    Parse batting order from player data.

    Primary: battingOrder field ("100"–"900" → 1–9)
    Fallback: battingOrder array index (0-based → 1-based)

    Args:
        player_data: Player data dict from API
        batting_order_array: Fallback batting order array
        player_id: Player ID to look up in array

    Returns:
        Batting order (1–9) or None if invalid/bench
    """
    # Primary: parse battingOrder field
    batting_order_str = player_data.get("battingOrder", "0")
    if batting_order_str and batting_order_str != "0":
        try:
            batting_order = int(batting_order_str) // 100  # "300" → 3
            if 1 <= batting_order <= 9:
                return batting_order
        except (ValueError, TypeError):
            pass

    # Fallback: use battingOrder array index
    try:
        array_idx = batting_order_array.index(player_id)
        batting_order = array_idx + 1  # 0-based → 1-based
        if 1 <= batting_order <= 9:
            return batting_order
    except (ValueError, IndexError):
        pass

    return None


class V1LineupProvider(LineupProvider):
    """
    V1 concrete lineup provider.

    Implements the lineup confirmation contract (D-011):
    When inserting a new confirmed lineup, set prior confirmed rows
    to is_confirmed = FALSE first.

    Uses MLB Stats API boxscore endpoint (D-057).
    """

    async def fetch_lineup(self, game_id: str, team_id: int | None = None) -> list[LineupRow]:
        """
        Fetch lineup for a team in a game from MLB Stats API boxscore.

        If team_id=None, fetch both home+away; if specified, fetch only that team.

        Conservative fallback: on error, log and return empty list.

        Args:
            game_id: Game identifier (gamePk)
            team_id: Team identifier (optional, fetch both if None)

        Returns:
            List of LineupRow objects
        """
        try:
            config = get_config()
            api_url = f"{config.mlb_stats_api_base_url}/game/{game_id}/boxscore"

            # Fetch boxscore data with 10s timeout
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(api_url) as response:
                    response.raise_for_status()
                    data = await response.json()

            # Extract teams data
            teams_data = data.get("teams", {})
            if not teams_data:
                logger.warning(f"No teams data in boxscore for game {game_id}")
                return []

            # Extract game status
            status_data = data.get("status", {})
            game_status = status_data.get("abstractGameCode", "P")

            # Parse lineups
            all_rows = []
            source_ts = datetime.now(timezone.utc)

            for side in ["home", "away"]:
                side_data = teams_data.get(side, {})
                if not side_data:
                    continue

                # Get team_id for this side
                side_team_id = side_data.get("team", {}).get("id")
                if not side_team_id:
                    logger.warning(f"Missing team.id for {side} in game {game_id}")
                    continue

                # Skip if caller requested specific team and this isn't it
                if team_id is not None and side_team_id != team_id:
                    continue

                # Extract players and batting order array
                players_data = side_data.get("players", {})
                batting_order_array = side_data.get("battingOrder", [])

                # Parse lineup rows
                lineup_rows = []
                for player_key, player_data in players_data.items():
                    person = player_data.get("person", {})
                    player_id = person.get("id")
                    if not player_id:
                        continue

                    # Parse batting order
                    batting_order = _parse_batting_order(player_data, batting_order_array, player_id)
                    if batting_order is None:
                        # Skip bench players or invalid batting orders
                        continue

                    # Extract player metadata for D-020 upsert
                    player_name = person.get("fullName")
                    position = player_data.get("position", {}).get("abbreviation")

                    lineup_rows.append({
                        "player_id": player_id,
                        "player_name": player_name,
                        "position": position,
                        "batting_order": batting_order,
                    })

                # Determine confirmation status
                is_confirmed = _is_confirmed(game_status, len(lineup_rows))

                # Log lineup info
                logger.info(
                    f"Parsed {len(lineup_rows)} players for game {game_id}, "
                    f"team {side_team_id} ({side}), status={game_status}, confirmed={is_confirmed}"
                )

                # Convert to LineupRow objects
                for row_data in lineup_rows:
                    all_rows.append(LineupRow(
                        game_id=game_id,
                        team_id=side_team_id,
                        player_id=row_data["player_id"],
                        batting_order=row_data["batting_order"],
                        is_confirmed=is_confirmed,
                        source_ts=source_ts,
                    ))

            # Persist to database with D-011 flip logic and D-020 player upsert
            if all_rows:
                await self._persist_lineups(all_rows)

            return all_rows

        except asyncio.TimeoutError:
            logger.warning(f"Timeout fetching lineup for game {game_id}, team {team_id}")
            return []
        except aiohttp.ClientError as e:
            logger.error(f"HTTP error fetching lineup for game {game_id}, team {team_id}: {e}")
            return []
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Parse error for game {game_id}, team {team_id}: {e}")
            return []
        except Exception as e:
            # Conservative fallback: log and return empty
            logger.error(
                f"Failed to fetch lineup for game {game_id}, team {team_id}: {e}",
                exc_info=True,
            )
            return []

    async def _persist_lineups(self, rows: list[LineupRow]) -> None:
        """
        Persist lineup rows to database with D-011 flip logic and D-020 player upsert.

        Groups rows by (game_id, team_id) and processes each group in a transaction.

        Args:
            rows: List of LineupRow objects to persist
        """
        # Group rows by (game_id, team_id)
        groups: dict[tuple[str, int], list[LineupRow]] = {}
        for row in rows:
            key = (row.game_id, row.team_id)
            if key not in groups:
                groups[key] = []
            groups[key].append(row)

        # Process each group in a transaction
        pool = await get_pool()

        for (game_id, team_id), group_rows in groups.items():
            try:
                async with pool.acquire() as conn:
                    async with conn.transaction():
                        # 1. Upsert missing players (D-020) - BEFORE lineup insert
                        # Extract unique players with metadata
                        players_to_upsert = {}
                        for row in group_rows:
                            if row.player_id not in players_to_upsert:
                                players_to_upsert[row.player_id] = team_id

                        for player_id, pid_team_id in players_to_upsert.items():
                            await ensure_player_exists(conn, player_id, team_id=pid_team_id)

                        # 2. Flip prior confirmed rows (D-011) - only if new lineup is confirmed
                        is_confirmed = group_rows[0].is_confirmed  # All rows in group have same confirmation
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

                        # 3. Insert new lineup rows (append-only, no ON CONFLICT)
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
                                for row in group_rows
                            ],
                        )

                        logger.info(
                            f"Inserted {len(group_rows)} lineup rows for game {game_id}, team {team_id} "
                            f"(confirmed={is_confirmed})"
                        )

            except Exception as e:
                # Database exceptions must be caught, logged, and not bubble up
                logger.error(
                    f"Database error persisting lineup for game {game_id}, team {team_id}: {e}",
                    exc_info=True,
                )
                # Transaction automatically rolled back on exception

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
