"""Concrete weather provider with Open-Meteo integration."""

import asyncio
import json
import logging
from datetime import date, datetime, timezone

import aiohttp
import asyncpg

from mlb.config.settings import get_config
from mlb.db.models import Table
from mlb.db.pool import get_pool
from mlb.ingestion.base import WeatherProvider, WeatherRow

logger = logging.getLogger(__name__)


def degrees_to_cardinal(degrees: float | None) -> str | None:
    """
    Convert wind direction from degrees (0-360) to 8-point cardinal direction.

    Uses 22.5° bins centered on each cardinal direction (D-062).

    Args:
        degrees: Wind direction in degrees (0-360)

    Returns:
        Cardinal direction ("N", "NE", "E", "SE", "S", "SW", "W", "NW") or None
    """
    if degrees is None:
        return None

    # Normalize to 0-360 range
    degrees = degrees % 360

    # 8-point compass with 45° bins (22.5° on each side of cardinal)
    if degrees >= 337.5 or degrees < 22.5:
        return "N"
    elif degrees < 67.5:
        return "NE"
    elif degrees < 112.5:
        return "E"
    elif degrees < 157.5:
        return "SE"
    elif degrees < 202.5:
        return "S"
    elif degrees < 247.5:
        return "SW"
    elif degrees < 292.5:
        return "W"
    else:  # 292.5 <= degrees < 337.5
        return "NW"


def select_closest_hour(
    hourly_times: list[str], first_pitch: datetime
) -> int | None:
    """
    Select the closest hourly index to first_pitch time.

    Primary: floor first_pitch to hour and look for exact match.
    Fallback: find entry with smallest absolute time difference (D-063).

    Args:
        hourly_times: List of ISO timestamps from API (e.g., ["2026-06-15T19:00", ...])
        first_pitch: Game first pitch time (UTC)

    Returns:
        Index into hourly_times list, or None if list is empty
    """
    if not hourly_times:
        return None

    # Floor first_pitch to hour
    target_hour = first_pitch.replace(minute=0, second=0, microsecond=0)

    # Try exact match first
    for i, time_str in enumerate(hourly_times):
        try:
            # Parse ISO timestamp - Open-Meteo returns UTC times without explicit 'Z'
            api_time = datetime.fromisoformat(time_str)
            # Make timezone-aware if needed (assume UTC)
            if api_time.tzinfo is None:
                api_time = api_time.replace(tzinfo=timezone.utc)
            if api_time == target_hour:
                return i
        except (ValueError, AttributeError):
            continue

    # Fallback: find closest hour
    min_diff = None
    min_idx = None

    for i, time_str in enumerate(hourly_times):
        try:
            api_time = datetime.fromisoformat(time_str)
            if api_time.tzinfo is None:
                api_time = api_time.replace(tzinfo=timezone.utc)
            diff = abs((api_time - target_hour).total_seconds())
            if min_diff is None or diff < min_diff:
                min_diff = diff
                min_idx = i
        except (ValueError, AttributeError):
            continue

    return min_idx


class V1WeatherProvider(WeatherProvider):
    """
    V1 concrete weather provider with Open-Meteo integration.

    Returns None for indoor or retractable-roof parks (per D-018).
    Fetches weather only for outdoor parks.
    """

    async def fetch_weather(self, game_id: str, park_id: int) -> WeatherRow | None:
        """
        Fetch weather for a game at a park from Open-Meteo API.

        Returns None for indoor or retractable-roof parks (D-018).

        Conservative fallback: on error, log warning and return None (D-019).

        Args:
            game_id: Game identifier
            park_id: Park identifier

        Returns:
            WeatherRow if outdoor park with valid data, None otherwise
        """
        try:
            config = get_config()
            pool = await get_pool()

            async with pool.acquire() as conn:
                # Step 1: Check park type and coordinates
                park_info = await conn.fetchrow(
                    f"""
                    SELECT is_outdoor, is_retractable, latitude, longitude
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
                latitude = park_info["latitude"]
                longitude = park_info["longitude"]

                # Per D-018: no weather for indoor or retractable parks
                if not is_outdoor or is_retractable:
                    logger.debug(
                        f"Skipping weather for game {game_id} at park {park_id} "
                        f"(outdoor={is_outdoor}, retractable={is_retractable})"
                    )
                    return None

                # Check for NULL coordinates
                if latitude is None or longitude is None:
                    logger.warning(
                        f"Park {park_id} has NULL coordinates (lat={latitude}, lon={longitude})"
                    )
                    return None

                # Step 2: Get first_pitch time and game_date
                game_info = await conn.fetchrow(
                    f"""
                    SELECT first_pitch, game_date
                    FROM {Table.GAMES}
                    WHERE game_id = $1
                    """,
                    game_id,
                )

                if game_info is None:
                    logger.warning(f"Game {game_id} not found in database")
                    return None

                first_pitch = game_info["first_pitch"]
                game_date = game_info["game_date"]

                if first_pitch is None:
                    logger.warning(f"Game {game_id} has NULL first_pitch")
                    return None

            # Step 3: Choose API endpoint (forecast vs archive)
            today = date.today()
            if game_date < today:
                api_url = config.open_meteo_archive_url
            else:
                api_url = config.open_meteo_forecast_url

            # Step 4: Build query parameters
            params = {
                "latitude": float(latitude),
                "longitude": float(longitude),
                "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m,precipitation_probability",
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "timezone": "UTC",
                "start_date": game_date.isoformat(),
                "end_date": game_date.isoformat(),
            }

            # Step 5: Fetch from Open-Meteo
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(api_url, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()

            # Step 6: Extract hourly data
            hourly = data.get("hourly", {})
            hourly_times = hourly.get("time", [])
            temperatures = hourly.get("temperature_2m", [])
            wind_speeds = hourly.get("wind_speed_10m", [])
            wind_directions = hourly.get("wind_direction_10m", [])
            precip_probs = hourly.get("precipitation_probability", [])

            if not hourly_times:
                logger.warning(
                    f"No hourly data returned from Open-Meteo for game {game_id}"
                )
                return None

            # Step 7: Select hour closest to first_pitch
            hour_idx = select_closest_hour(hourly_times, first_pitch)
            if hour_idx is None:
                logger.warning(
                    f"Could not select hour for game {game_id} (first_pitch={first_pitch})"
                )
                return None

            # Step 8: Extract values for selected hour
            try:
                temp_raw = temperatures[hour_idx]
                wind_speed_raw = wind_speeds[hour_idx]
                wind_dir_raw = wind_directions[hour_idx]
                precip_raw = precip_probs[hour_idx]
            except IndexError:
                logger.warning(
                    f"Hourly data arrays have mismatched lengths for game {game_id}"
                )
                return None

            # Step 9: Validate and convert
            # If any required field is None/null, return None (skip incomplete data)
            if temp_raw is None:
                logger.warning(
                    f"temperature_2m is null for game {game_id} at hour {hourly_times[hour_idx]}"
                )
                return None

            if wind_speed_raw is None:
                logger.warning(
                    f"wind_speed_10m is null for game {game_id} at hour {hourly_times[hour_idx]}"
                )
                return None

            if wind_dir_raw is None:
                logger.warning(
                    f"wind_direction_10m is null for game {game_id} at hour {hourly_times[hour_idx]}"
                )
                return None

            if precip_raw is None:
                logger.warning(
                    f"precipitation_probability is null for game {game_id} at hour {hourly_times[hour_idx]}"
                )
                return None

            # Convert to integers
            temp_f = round(temp_raw)
            wind_speed_mph = round(wind_speed_raw)
            wind_dir = degrees_to_cardinal(wind_dir_raw)
            precip_pct = round(precip_raw)

            # Wind direction conversion failure should never happen given non-null input
            if wind_dir is None:
                logger.warning(
                    f"Failed to convert wind_direction {wind_dir_raw} for game {game_id}"
                )
                return None

            # Step 10: Build WeatherRow
            weather_row = WeatherRow(
                game_id=game_id,
                temp_f=temp_f,
                wind_speed_mph=wind_speed_mph,
                wind_dir=wind_dir,
                precip_pct=precip_pct,
                fetched_at=datetime.now(timezone.utc),
            )

            logger.info(
                f"Fetched weather for game {game_id}: {temp_f}°F, {wind_speed_mph}mph {wind_dir}, {precip_pct}% precip"
            )

            return weather_row

        except asyncio.TimeoutError:
            logger.warning(f"Timeout fetching weather for game {game_id}, park {park_id}")
            return None
        except aiohttp.ClientError as e:
            logger.warning(
                f"HTTP error fetching weather for game {game_id}, park {park_id}: {e}"
            )
            return None
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(
                f"Parse error for game {game_id}, park {park_id}: {e}"
            )
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
        Write weather row to database (append-only per D-015).

        On UNIQUE(game_id, fetched_at) violation, log WARNING but do not raise.

        Args:
            row: WeatherRow object to persist
        """
        pool = await get_pool()

        try:
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

        except asyncpg.UniqueViolationError:
            # Duplicate (game_id, fetched_at) - data already persisted
            logger.warning(
                f"Duplicate weather row for game {row.game_id} at {row.fetched_at} (already persisted)"
            )
        except Exception as e:
            # Log error but do not raise (conservative fallback)
            logger.error(
                f"Failed to write weather row for game {row.game_id}: {e}",
                exc_info=True,
            )
