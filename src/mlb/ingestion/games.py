"""Concrete game/schedule provider."""

import logging
from datetime import date, datetime, timezone

import aiohttp
import asyncpg

from mlb.config.settings import get_config
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
            config = get_config()
            url = f"{config.mlb_stats_api_base_url}/schedule"
            params = {
                "sportId": "1",
                "date": game_date.strftime("%Y-%m-%d"),
                "hydrate": "team,venue",
            }

            logger.info(f"Fetching game schedule for date {game_date}")

            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()

            # Parse response
            rows = []
            if not data.get("dates"):
                return rows

            games = data["dates"][0].get("games", [])
            if not games:
                return rows

            # Load valid park IDs and home team fallback map (FC-32)
            pool = await get_pool()
            async with pool.acquire() as conn:
                # Get set of valid park_ids
                park_rows = await conn.fetch(f"SELECT park_id FROM {Table.PARKS}")
                valid_parks: set[int] = {row["park_id"] for row in park_rows}

                # Get home team fallback map: {team_id: park_id}
                fallback_rows = await conn.fetch(
                    f"SELECT team_id, park_id FROM {Table.PARKS}"
                )
                fallback_map: dict[int, int] = {
                    row["team_id"]: row["park_id"] for row in fallback_rows
                }

            for game in games:
                game_id = str(game["gamePk"])
                official_date = datetime.strptime(
                    game["officialDate"], "%Y-%m-%d"
                ).date()
                home_team_id = game["teams"]["home"]["team"]["id"]
                away_team_id = game["teams"]["away"]["team"]["id"]

                # Parse first_pitch as UTC datetime
                game_date_str = game.get("gameDate")
                first_pitch = None
                if game_date_str:
                    first_pitch = datetime.strptime(
                        game_date_str, "%Y-%m-%dT%H:%M:%SZ"
                    ).replace(tzinfo=timezone.utc)

                # Parse venue.id with validation (FC-32)
                venue_id = game.get("venue", {}).get("id")
                park_id = None

                # If venue_id exists and is valid, use it
                if venue_id is not None and venue_id in valid_parks:
                    park_id = venue_id
                # Otherwise, fallback to home team's park
                elif home_team_id in fallback_map:
                    park_id = fallback_map[home_team_id]
                    if venue_id is not None and venue_id not in valid_parks:
                        logger.warning(
                            f"Venue ID {venue_id} not in parks table for game {game_id}, using home team's park {park_id}"
                        )
                # If no valid venue and no home team fallback, skip game
                else:
                    logger.warning(
                        f"No valid venue.id and no park found for home_team_id={home_team_id}, skipping game {game_id}"
                    )
                    continue

                # Map status
                abstract_game_code = game.get("status", {}).get("abstractGameCode", "")
                if abstract_game_code == "F":
                    status = "final"
                elif abstract_game_code == "D":
                    status = "postponed"
                else:
                    status = "scheduled"

                # Parse scores for final games
                home_score = None
                away_score = None
                if status == "final":
                    home_score = game["teams"]["home"].get("score")
                    away_score = game["teams"]["away"].get("score")

                rows.append(
                    GameRow(
                        game_id=game_id,
                        game_date=official_date,
                        home_team_id=home_team_id,
                        away_team_id=away_team_id,
                        park_id=park_id,
                        first_pitch=first_pitch,
                        status=status,
                        home_score=home_score,
                        away_score=away_score,
                    )
                )

            logger.info(f"Fetched {len(rows)} games for date {game_date}")
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
                (game_id, game_date, home_team_id, away_team_id, park_id, first_pitch, status, home_score, away_score, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, now())
                ON CONFLICT (game_id)
                DO UPDATE SET
                    game_date = EXCLUDED.game_date,
                    home_team_id = EXCLUDED.home_team_id,
                    away_team_id = EXCLUDED.away_team_id,
                    park_id = EXCLUDED.park_id,
                    first_pitch = EXCLUDED.first_pitch,
                    status = EXCLUDED.status,
                    home_score = EXCLUDED.home_score,
                    away_score = EXCLUDED.away_score,
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
                        row.home_score,
                        row.away_score,
                    )
                    for row in rows
                ],
            )

        logger.info(f"Upserted {len(rows)} games to database")
