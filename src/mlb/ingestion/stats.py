"""Concrete stats provider with game log upsert logic."""

import asyncio
import json
import logging
from datetime import date

import aiohttp
import asyncpg

from mlb.config.settings import get_config
from mlb.db.models import Table
from mlb.db.pool import get_pool
from mlb.ingestion.base import GameLogRow, StatsProvider, ensure_player_exists
from mlb.ingestion.cache import get_cache

logger = logging.getLogger(__name__)

# Cache TTL for season-wide game logs: 1 hour
GAMELOG_CACHE_TTL_SECONDS = 3600


def parse_ip_to_outs(ip_str: str | None) -> int | None:
    """
    Convert MLB innings pitched format to total outs recorded.

    MLB format: "full_innings.partial_outs"

    Examples:
    - "5.0" → 15 (5 × 3)
    - "5.2" → 17 (5 × 3 + 2)
    - "0.1" → 1
    - None/"" → None

    Args:
        ip_str: Innings pitched string from API

    Returns:
        Total outs recorded, or None if invalid
    """
    if not ip_str:
        return None

    full, _, partial = ip_str.partition(".")

    try:
        full_innings = int(full)
        partial_outs = int(partial) if partial else 0
    except ValueError:
        logger.warning(f"Invalid innings format (non-numeric): {ip_str}")
        return None

    if partial_outs not in {0, 1, 2}:
        logger.warning(f"Invalid innings format: {ip_str}")
        return None

    return full_innings * 3 + partial_outs


def detect_starter(ip_outs: int | None) -> bool | None:
    """
    Detect if pitcher is a starter using v1 heuristic.

    V1 heuristic: is_starter = True if ip_outs >= 9 (3+ innings)

    Known limitations (acceptable for v1):
    - Misclassifies 2-inning openers as non-starters
    - Misclassifies long relievers (4+ IP) as starters
    - Misclassifies injury/rain-shortened starts (< 3 IP) as relievers

    Args:
        ip_outs: Total outs recorded (ip × 3)

    Returns:
        True if starter, False if reliever, None if not a pitcher
    """
    if ip_outs is None:
        return None
    return ip_outs >= 9


class V1StatsProvider(StatsProvider):
    """
    V1 concrete stats provider.

    Upserts game logs to player_game_logs table on (player_id, game_id).
    """

    async def _get_all_players_in_game(
        self, game_id: str, session: aiohttp.ClientSession
    ) -> set[int]:
        """
        Fetch all player IDs who appeared in a game from boxscore API.

        This includes starters, bench players, relief pitchers, pinch hitters, etc.
        Falls back to empty set on error (caller will use lineups-only).

        Checks cache first (shared with Step 1C) to avoid redundant HTTP calls.

        Args:
            game_id: Game identifier
            session: aiohttp ClientSession for making HTTP requests

        Returns:
            Set of player IDs who appeared in the game
        """
        config = get_config()
        base_url = config.mlb_stats_api_base_url
        boxscore_url = f"{base_url}/game/{game_id}/boxscore"
        cache = get_cache()
        cache_key = f"boxscore:{game_id}"

        try:
            # Check cache first (shared with Step 1C)
            cached_bytes = cache.get(cache_key)
            if cached_bytes is not None:
                try:
                    boxscore_data = json.loads(cached_bytes.decode("utf-8"))
                    logger.info(f"Boxscore cache hit for game {game_id} (Step 1D)")
                except Exception as e:
                    logger.warning(
                        f"Failed to deserialize cached boxscore for game {game_id}: {e}"
                    )
                    cached_bytes = None

            # If cache miss, fetch from API
            if cached_bytes is None:
                async with session.get(
                    boxscore_url, timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        response_bytes = await resp.read()
                        # Cache the raw response bytes (TTL: 2 hours)
                        cache.set(cache_key, response_bytes, 7200)
                        boxscore_data = json.loads(response_bytes.decode("utf-8"))
                        logger.info(
                            f"Boxscore fetched and cached for game {game_id} (Step 1D)"
                        )
                    else:
                        logger.warning(
                            f"HTTP {resp.status} fetching boxscore for game {game_id}"
                        )
                        return set()

            # Extract player IDs from both teams
            player_ids = set()

            # Extract player IDs from home team
            home_players = (
                boxscore_data.get("teams", {})
                .get("home", {})
                .get("players", {})
            )
            for player_key, player_data in home_players.items():
                player_id = player_data.get("person", {}).get("id")
                if player_id:
                    player_ids.add(player_id)

            # Extract player IDs from away team
            away_players = (
                boxscore_data.get("teams", {})
                .get("away", {})
                .get("players", {})
            )
            for player_key, player_data in away_players.items():
                player_id = player_data.get("person", {}).get("id")
                if player_id:
                    player_ids.add(player_id)

            logger.info(
                f"Extracted {len(player_ids)} player IDs from boxscore for game {game_id}"
            )
            return player_ids

        except asyncio.TimeoutError:
            logger.warning(f"Timeout fetching boxscore for game {game_id}")
            return set()
        except Exception as e:
            logger.warning(f"Error fetching boxscore for game {game_id}: {e}")
            return set()

    async def fetch_game_logs(self, game_date: date) -> list[GameLogRow]:
        """
        Fetch game logs for all players on a date.

        Orchestration:
        1. Query games table for completed games on date
        2. Query lineups table for player IDs in those games
        3. For each player, fetch hitting + pitching logs from MLB Stats API
        4. Merge two-way player stats into single GameLogRow
        5. Return list of GameLogRow objects

        Conservative fallback: on error, log warning and return empty list.

        Args:
            game_date: Date to fetch logs for

        Returns:
            List of GameLogRow objects
        """
        try:
            logger.info(f"Fetching game logs for date {game_date}")

            pool = await get_pool()

            # Step 1: Get completed games for date
            async with pool.acquire() as conn:
                games = await conn.fetch(
                    f"""
                    SELECT game_id
                    FROM {Table.GAMES}
                    WHERE game_date = $1 AND status = 'final'
                    """,
                    game_date,
                )

            if not games:
                logger.info(f"No completed games found for date {game_date}")
                return []

            game_ids = [row["game_id"] for row in games]

            # Step 2: Get player IDs from lineups (starting 9 per team)
            async with pool.acquire() as conn:
                lineups = await conn.fetch(
                    f"""
                    SELECT DISTINCT player_id
                    FROM {Table.LINEUPS}
                    WHERE game_id = ANY($1::text[])
                    """,
                    game_ids,
                )

            lineup_player_ids = {row["player_id"] for row in lineups}

            # Step 2b: Get ALL players who appeared in each game from boxscore
            # This includes relief pitchers, pinch hitters, substitutions
            all_player_ids = set(lineup_player_ids)

            async with aiohttp.ClientSession() as session:
                for game_id in game_ids:
                    boxscore_player_ids = await self._get_all_players_in_game(
                        game_id, session
                    )
                    all_player_ids.update(boxscore_player_ids)

            # If no players found at all (both lineups and boxscore failed), return empty
            if not all_player_ids:
                logger.info(
                    f"No players found for completed games on date {game_date}"
                )
                return []

            player_ids = list(all_player_ids)
            logger.info(
                f"Fetching logs for {len(player_ids)} players ({len(lineup_player_ids)} from lineups, "
                f"{len(all_player_ids) - len(lineup_player_ids)} additional from boxscore) "
                f"across {len(game_ids)} games"
            )

            # Step 3: Fetch hitting + pitching logs for each player
            config = get_config()
            base_url = config.mlb_stats_api_base_url
            season = game_date.year
            cache = get_cache()

            logs_by_player_game: dict[tuple[int, str], GameLogRow] = {}

            async with aiohttp.ClientSession() as session:
                for player_id in player_ids:
                    # Fetch hitting logs (with caching)
                    hitting_cache_key = f"gamelog:{player_id}:{season}:hitting"
                    hitting_data = None

                    # Check cache first
                    cached_bytes = cache.get(hitting_cache_key)
                    if cached_bytes is not None:
                        try:
                            hitting_data = json.loads(cached_bytes.decode("utf-8"))
                        except Exception as e:
                            logger.warning(
                                f"Failed to deserialize cached hitting logs for player {player_id}: {e}"
                            )

                    # If cache miss, fetch from API
                    if hitting_data is None:
                        hitting_url = f"{base_url}/people/{player_id}/stats?stats=gameLog&group=hitting&season={season}"
                        try:
                            async with session.get(
                                hitting_url, timeout=aiohttp.ClientTimeout(total=10)
                            ) as resp:
                                if resp.status == 200:
                                    response_bytes = await resp.read()
                                    # Cache the raw response
                                    cache.set(
                                        hitting_cache_key,
                                        response_bytes,
                                        GAMELOG_CACHE_TTL_SECONDS,
                                    )
                                    hitting_data = json.loads(
                                        response_bytes.decode("utf-8")
                                    )
                                elif resp.status == 404:
                                    logger.warning(
                                        f"Player {player_id} not found in MLB Stats API (hitting)"
                                    )
                                else:
                                    logger.warning(
                                        f"HTTP {resp.status} fetching hitting logs for player {player_id}"
                                    )
                        except asyncio.TimeoutError:
                            logger.warning(
                                f"Timeout fetching hitting logs for player {player_id}"
                            )
                        except Exception as e:
                            logger.warning(
                                f"Error fetching hitting logs for player {player_id}: {e}"
                            )

                    # Process hitting data if we got it (from cache or API)
                    if hitting_data is not None:
                        await self._process_hitting_splits(
                            hitting_data,
                            player_id,
                            game_ids,
                            logs_by_player_game,
                        )

                    # Fetch pitching logs (with caching)
                    pitching_cache_key = f"gamelog:{player_id}:{season}:pitching"
                    pitching_data = None

                    # Check cache first
                    cached_bytes = cache.get(pitching_cache_key)
                    if cached_bytes is not None:
                        try:
                            pitching_data = json.loads(cached_bytes.decode("utf-8"))
                        except Exception as e:
                            logger.warning(
                                f"Failed to deserialize cached pitching logs for player {player_id}: {e}"
                            )

                    # If cache miss, fetch from API
                    if pitching_data is None:
                        pitching_url = f"{base_url}/people/{player_id}/stats?stats=gameLog&group=pitching&season={season}"
                        try:
                            async with session.get(
                                pitching_url, timeout=aiohttp.ClientTimeout(total=10)
                            ) as resp:
                                if resp.status == 200:
                                    response_bytes = await resp.read()
                                    # Cache the raw response
                                    cache.set(
                                        pitching_cache_key,
                                        response_bytes,
                                        GAMELOG_CACHE_TTL_SECONDS,
                                    )
                                    pitching_data = json.loads(
                                        response_bytes.decode("utf-8")
                                    )
                                elif resp.status == 404:
                                    # Not all hitters pitch - this is OK, skip silently
                                    pass
                                else:
                                    logger.warning(
                                        f"HTTP {resp.status} fetching pitching logs for player {player_id}"
                                    )
                        except asyncio.TimeoutError:
                            logger.warning(
                                f"Timeout fetching pitching logs for player {player_id}"
                            )
                        except Exception as e:
                            logger.warning(
                                f"Error fetching pitching logs for player {player_id}: {e}"
                            )

                    # Process pitching data if we got it (from cache or API)
                    if pitching_data is not None:
                        await self._process_pitching_splits(
                            pitching_data,
                            player_id,
                            game_ids,
                            logs_by_player_game,
                        )

            rows = list(logs_by_player_game.values())
            logger.info(f"Fetched {len(rows)} game log rows for date {game_date}")
            return rows

        except Exception as e:
            # Conservative fallback: log and return empty
            logger.warning(
                f"Failed to fetch game logs for date {game_date}: {e}",
                exc_info=True,
            )
            return []

    async def _process_hitting_splits(
        self,
        data: dict,
        player_id: int,
        target_game_ids: list[str],
        logs: dict[tuple[int, str], GameLogRow],
    ) -> None:
        """Process hitting splits from API response."""
        if "stats" not in data or not data["stats"]:
            return

        for stat_group in data["stats"]:
            if "splits" not in stat_group:
                continue

            for split in stat_group["splits"]:
                game_pk = split.get("game", {}).get("gamePk")
                if not game_pk:
                    continue

                game_id = str(game_pk)
                if game_id not in target_game_ids:
                    continue

                stat = split.get("stat", {})
                key = (player_id, game_id)

                # Get or create GameLogRow
                if key not in logs:
                    logs[key] = GameLogRow(player_id=player_id, game_id=game_id)

                # Populate hitting fields
                logs[key].pa = stat.get("plateAppearances")
                logs[key].ab = stat.get("atBats")
                logs[key].h = stat.get("hits")
                logs[key].tb = stat.get("totalBases")
                logs[key].hr = stat.get("homeRuns")
                logs[key].rbi = stat.get("rbi")
                logs[key].r = stat.get("runs")
                logs[key].bb = stat.get("baseOnBalls")
                logs[key].k = stat.get("strikeOuts")

    async def _process_pitching_splits(
        self,
        data: dict,
        player_id: int,
        target_game_ids: list[str],
        logs: dict[tuple[int, str], GameLogRow],
    ) -> None:
        """Process pitching splits from API response."""
        if "stats" not in data or not data["stats"]:
            return

        for stat_group in data["stats"]:
            if "splits" not in stat_group:
                continue

            for split in stat_group["splits"]:
                game_pk = split.get("game", {}).get("gamePk")
                if not game_pk:
                    continue

                game_id = str(game_pk)
                if game_id not in target_game_ids:
                    continue

                stat = split.get("stat", {})
                key = (player_id, game_id)

                # Get or create GameLogRow
                if key not in logs:
                    logs[key] = GameLogRow(player_id=player_id, game_id=game_id)

                # Populate pitching fields
                ip_str = stat.get("inningsPitched")
                ip_outs = parse_ip_to_outs(ip_str)

                logs[key].ip_outs = ip_outs
                logs[key].er = stat.get("earnedRuns")
                logs[key].pitch_count = stat.get("pitchesThrown")
                logs[key].is_starter = detect_starter(ip_outs)

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
