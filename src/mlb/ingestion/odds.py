"""Concrete odds provider with American → European decimal conversion."""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Optional

import aiohttp
import asyncpg

from mlb.config import get_config
from mlb.db.models import Table
from mlb.db.pool import get_pool
from mlb.ingestion.base import OddsProvider, OddsRow
from mlb.ingestion.cache import get_cache

logger = logging.getLogger(__name__)


def american_to_decimal(american: float) -> float:
    """
    Convert American odds to European decimal format.

    Args:
        american: American odds (e.g., +150, -110)

    Returns:
        European decimal odds (≥ 1.0)

    Raises:
        ValueError: If american odds is 0 or invalid
    """
    if american == 0:
        raise ValueError("American odds cannot be 0")

    if american > 0:
        # Positive: decimal = (american / 100) + 1
        return (american / 100) + 1
    else:
        # Negative: decimal = (100 / abs(american)) + 1
        return (100 / abs(american)) + 1


def detect_and_convert_odds(value: float, format_hint: str | None = None) -> float:
    """
    Detect odds format and convert to European decimal.

    Args:
        value: Odds value
        format_hint: Optional format hint ('american' or 'decimal')

    Returns:
        European decimal odds (≥ 1.0)

    Raises:
        ValueError: If format is ambiguous or invalid
    """
    # If format is explicitly specified, trust it
    if format_hint == "decimal":
        if value < 1.0:
            raise ValueError(f"Decimal odds must be ≥ 1.0, got {value}")
        return value
    elif format_hint == "american":
        return american_to_decimal(value)

    # Auto-detect based on value range (per D-017)
    # American: [-99999, -100] ∪ [100, 99999]
    # Decimal: [1.0, 50.0]
    #
    # Known limitation (FC-14): European decimal odds ≥ 100.0 would be
    # misclassified as American. This is not reachable for v1 main-line
    # MLB markets (moneyline, run line, totals). Extreme long shots with
    # decimal odds ≥ 100.0 (~99/1 or worse) do not occur in practice for
    # these markets.
    if 1.0 <= value <= 50.0:
        # Likely decimal
        return value
    elif (100 <= value <= 99999) or (-99999 <= value <= -100):
        # Likely American
        return american_to_decimal(value)
    else:
        # Ambiguous range
        raise ValueError(
            f"Ambiguous odds value {value} - cannot determine format. "
            f"Expected American (≥100 or ≤-100) or decimal (1.0-50.0)"
        )


class V1OddsProvider(OddsProvider):
    """
    V1 concrete odds provider.

    This is a stub implementation that demonstrates the interface.
    In production, this would fetch from a real odds API.
    """

    async def fetch_odds(
        self,
        game_id: str,
        *,
        event_date: Optional[datetime] = None,
    ) -> list[OddsRow]:
        """
        Fetch odds snapshots for a game.

        When event_date is provided the historical endpoint is used
        (/v4/historical/sports/baseball_mlb/odds?date=...) instead of the live
        endpoint, enabling historical backfill.  The response schema is
        identical between the two endpoints (per D-067).

        Conservative fallback: on error, log warning and return empty list.

        Args:
            game_id:    Game identifier.
            event_date: UTC datetime of the snapshot to retrieve.  When None
                        (default) the live endpoint is called.  When provided
                        the historical endpoint is called and costs 10 API
                        credits vs 1 for the live endpoint (per D-067).

        Returns:
            List of OddsRow objects with price in European decimal (≥ 1.0).
        """
        try:
            config = get_config()
            cache = get_cache()

            if event_date is not None:
                # ----------------------------------------------------------
                # Historical endpoint (D-067): no cache — historical snapshots
                # are fixed and not re-requested during normal operation.
                # ----------------------------------------------------------
                if not config.odds_api_key:
                    logger.warning("odds_api_key not configured, returning empty odds")
                    return []

                url = f"{config.odds_api_base_url}/historical/sports/baseball_mlb/odds"
                date_param = event_date.strftime("%Y-%m-%dT%H:%M:%SZ")
                params = {
                    "apiKey": config.odds_api_key,
                    "regions": "us",
                    "markets": "h2h,spreads,totals",
                    "oddsFormat": "american",
                    "dateFormat": "iso",
                    "date": date_param,
                }

                try:
                    timeout = aiohttp.ClientTimeout(total=10)
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.get(url, params=params) as resp:
                            if resp.status != 200:
                                logger.warning(
                                    f"Odds API historical returned status {resp.status}, "
                                    f"returning empty"
                                )
                                return []

                            response_text = await resp.text()
                            raw = json.loads(response_text)
                            # Historical endpoint wraps events: {"data": [...], ...}
                            # Live endpoint returns a bare array.
                            response_data = raw.get("data", raw) if isinstance(raw, dict) else raw
                            logger.info(
                                f"Fetched historical odds for date {date_param}"
                            )

                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    logger.warning(
                        f"Historical odds API request failed: {e}", exc_info=True
                    )
                    return []

            else:
                # ----------------------------------------------------------
                # Live endpoint: cached for 5 minutes (existing behaviour).
                # ----------------------------------------------------------
                cache_key = "odds_api_mlb"
                cached_response = cache.get(cache_key)

                if cached_response is not None:
                    logger.info("Using cached odds API response")
                    response_data = json.loads(cached_response.decode("utf-8"))
                else:
                    if not config.odds_api_key:
                        logger.warning(
                            "odds_api_key not configured, returning empty odds"
                        )
                        return []

                    url = f"{config.odds_api_base_url}/sports/baseball_mlb/odds"
                    params = {
                        "apiKey": config.odds_api_key,
                        "regions": "us",
                        "markets": "h2h,spreads,totals",
                        "oddsFormat": "american",
                        "dateFormat": "iso",
                    }

                    try:
                        timeout = aiohttp.ClientTimeout(total=10)
                        async with aiohttp.ClientSession(timeout=timeout) as session:
                            async with session.get(url, params=params) as resp:
                                if resp.status != 200:
                                    logger.warning(
                                        f"Odds API returned status {resp.status}, returning empty"
                                    )
                                    return []

                                response_text = await resp.text()
                                response_data = json.loads(response_text)

                                # Cache the response for 5 minutes
                                cache.set(
                                    cache_key,
                                    response_text.encode("utf-8"),
                                    ttl_seconds=300,
                                )
                                logger.info("Fetched and cached odds from API")

                    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                        logger.warning(f"Odds API request failed: {e}", exc_info=True)
                        return []

            # Build team name → team_id lookup and query games
            pool = await get_pool()
            async with pool.acquire() as conn:
                team_rows = await conn.fetch(
                    f"SELECT team_id, name FROM {Table.TEAMS}"
                )

                team_name_to_id = {row["name"]: row["team_id"] for row in team_rows}

                # Parse events and build OddsRow objects
                all_odds_rows = []

                for event in response_data:
                    home_team_name = event.get("home_team")
                    away_team_name = event.get("away_team")
                    commence_time_str = event.get("commence_time")

                    if not home_team_name or not commence_time_str:
                        continue

                    # Find home_team_id
                    home_team_id = team_name_to_id.get(home_team_name)
                    if home_team_id is None:
                        logger.debug(f"Unknown home team: {home_team_name}")
                        continue

                    # Parse commence_time to get game_date
                    commence_time = datetime.fromisoformat(commence_time_str.replace("Z", "+00:00"))
                    game_date = commence_time.date()

                    # Find matching game_id
                    game_rows = await conn.fetch(
                        f"""
                        SELECT game_id, first_pitch
                        FROM {Table.GAMES}
                        WHERE home_team_id = $1 AND game_date = $2
                        """,
                        home_team_id,
                        game_date,
                    )

                    if not game_rows:
                        logger.debug(
                            f"No game found for home_team_id={home_team_id}, game_date={game_date}"
                        )
                        continue

                    # For doubleheaders, pick closest first_pitch to commence_time
                    if len(game_rows) == 1:
                        event_game_id = game_rows[0]["game_id"]
                    else:
                        # Multiple games (doubleheader) - pick closest by first_pitch
                        closest_game = min(
                            game_rows,
                            key=lambda g: abs(
                                (g["first_pitch"] - commence_time).total_seconds()
                            )
                            if g["first_pitch"]
                            else float("inf"),
                        )
                        event_game_id = closest_game["game_id"]

                    # Parse bookmakers
                    bookmakers = event.get("bookmakers", [])
                    for bookmaker in bookmakers:
                        book_key = bookmaker.get("key")
                        markets = bookmaker.get("markets", [])

                        for market in markets:
                            market_key = market.get("key")
                            last_update_str = market.get("last_update")

                            # Map market key to canonical format
                            market_mapping = {
                                "h2h": "ml",
                                "spreads": "rl",
                                "totals": "total",
                            }
                            canonical_market = market_mapping.get(market_key)
                            if canonical_market is None:
                                # Skip unknown market
                                continue

                            # Parse snapshot_ts
                            if last_update_str:
                                snapshot_ts = datetime.fromisoformat(
                                    last_update_str.replace("Z", "+00:00")
                                )
                            else:
                                snapshot_ts = datetime.now(timezone.utc)

                            # Parse outcomes
                            outcomes = market.get("outcomes", [])
                            for outcome in outcomes:
                                outcome_name = outcome.get("name")
                                price_american = outcome.get("price")
                                point = outcome.get("point")

                                if price_american is None:
                                    continue

                                # Determine side
                                if canonical_market == "ml":
                                    if outcome_name == home_team_name:
                                        side = "home"
                                    elif outcome_name == away_team_name:
                                        side = "away"
                                    else:
                                        continue
                                    line = None
                                elif canonical_market == "rl":
                                    if outcome_name == home_team_name:
                                        side = "home"
                                    elif outcome_name == away_team_name:
                                        side = "away"
                                    else:
                                        continue
                                    line = float(point) if point is not None else None
                                elif canonical_market == "total":
                                    if outcome_name == "Over":
                                        side = "over"
                                    elif outcome_name == "Under":
                                        side = "under"
                                    else:
                                        continue
                                    line = float(point) if point is not None else None
                                else:
                                    continue

                                # Convert price to decimal
                                try:
                                    price_decimal = american_to_decimal(float(price_american))
                                    if price_decimal < 1.0:
                                        logger.warning(
                                            f"Invalid decimal price {price_decimal} < 1.0, skipping"
                                        )
                                        continue
                                except (ValueError, ZeroDivisionError) as e:
                                    logger.warning(f"Failed to convert odds {price_american}: {e}")
                                    continue

                                # Create OddsRow
                                odds_row = OddsRow(
                                    game_id=event_game_id,
                                    book=book_key,
                                    market=canonical_market,
                                    side=side,
                                    line=line,
                                    price=price_decimal,
                                    snapshot_ts=snapshot_ts,
                                )
                                all_odds_rows.append(odds_row)

            # Filter to only the requested game_id
            filtered_rows = [row for row in all_odds_rows if row.game_id == game_id]

            logger.info(f"Fetched {len(filtered_rows)} odds rows for game {game_id}")
            return filtered_rows

        except Exception as e:
            # Conservative fallback: log and return empty
            logger.warning(
                f"Failed to fetch odds for game {game_id}: {e}",
                exc_info=True,
            )
            return []

    async def _write_odds(self, rows: list[OddsRow]) -> None:
        """
        Write odds rows to database.

        Args:
            rows: List of OddsRow objects to persist
        """
        if not rows:
            return

        pool = await get_pool()

        async with pool.acquire() as conn:
            # Prepare insert statement
            insert_sql = f"""
                INSERT INTO {Table.ODDS_SNAPSHOTS}
                (game_id, book, market, side, line, price, snapshot_ts)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            """

            # Batch insert all rows
            await conn.executemany(
                insert_sql,
                [
                    (
                        row.game_id,
                        row.book,
                        row.market,
                        row.side,
                        row.line,
                        row.price,
                        row.snapshot_ts,
                    )
                    for row in rows
                ],
            )

        logger.info(f"Wrote {len(rows)} odds snapshots to database")
