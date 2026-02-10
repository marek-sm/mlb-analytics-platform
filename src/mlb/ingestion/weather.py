"""Concrete weather provider with park type checking."""

import logging
from datetime import datetime, timezone

import asyncpg

from mlb.db.models import Table
from mlb.db.pool import get_pool
from mlb.ingestion.base import WeatherProvider, WeatherRow

logger = logging.getLogger(__name__)


class V1WeatherProvider(WeatherProvider):
    """
    V1 concrete weather provider.

    Returns None for indoor or retractable-roof parks (per D-018).
    Fetches weather only for outdoor parks.
    """

    async def fetch_weather(self, game_id: str, park_id: int) -> WeatherRow | None:
        """
        Fetch weather for a game at a park.

        Returns None for indoor or retractable-roof parks.

        Conservative fallback: on error, log warning and return None.

        Args:
            game_id: Game identifier
            park_id: Park identifier

        Returns:
            WeatherRow if outdoor park, None otherwise
        """
        try:
            # Check if park is outdoor and not retractable
            pool = await get_pool()

            async with pool.acquire() as conn:
                park_info = await conn.fetchrow(
                    f"""
                    SELECT is_outdoor, is_retractable
                    FROM {Table.PARKS}
                    WHERE park_id = $1
                    """,
                    park_id,
                )

                if park_info is None:
                    logger.warning(f"Park {park_id} not found in database")
                    return None

                is_outdoor = park_info["is_outdoor"]
                is_retractable = park_info["is_retractable"]

                # Per D-018: no weather for indoor or retractable parks
                if not is_outdoor or is_retractable:
                    logger.debug(
                        f"Skipping weather for game {game_id} at park {park_id} "
                        f"(outdoor={is_outdoor}, retractable={is_retractable})"
                    )
                    return None

                # Stub: In production, this would call an external weather API
                logger.info(
                    f"Fetching weather for game {game_id} at park {park_id}"
                )

                # Simulate API call that returns no data in v1
                return None

        except Exception as e:
            # Conservative fallback: log and return None
            logger.warning(
                f"Failed to fetch weather for game {game_id} at park {park_id}: {e}",
                exc_info=True,
            )
            return None

    async def write_weather(self, row: WeatherRow) -> None:
        """
        Write weather row to database.

        Weather table is append-only per D-015.

        Args:
            row: WeatherRow object to persist
        """
        pool = await get_pool()

        async with pool.acquire() as conn:
            insert_sql = f"""
                INSERT INTO {Table.WEATHER}
                (game_id, temp_f, wind_speed_mph, wind_dir, precip_pct, fetched_at)
                VALUES ($1, $2, $3, $4, $5, $6)
            """

            await conn.execute(
                insert_sql,
                row.game_id,
                row.temp_f,
                row.wind_speed_mph,
                row.wind_dir,
                row.precip_pct,
                row.fetched_at,
            )

        logger.info(f"Wrote weather row for game {row.game_id}")
