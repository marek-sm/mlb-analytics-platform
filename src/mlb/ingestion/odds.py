"""Concrete odds provider with American → European decimal conversion."""

import logging
from datetime import datetime, timezone

import asyncpg

from mlb.config import get_config
from mlb.db.models import Table
from mlb.db.pool import get_pool
from mlb.ingestion.base import OddsProvider, OddsRow

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

    async def fetch_odds(self, game_id: str) -> list[OddsRow]:
        """
        Fetch odds snapshots for a game.

        Conservative fallback: on error, log warning and return empty list.

        Args:
            game_id: Game identifier

        Returns:
            List of OddsRow objects with price in European decimal (≥ 1.0)
        """
        try:
            # Stub: In production, this would call an external API
            # For now, return empty list (no odds available)
            logger.info(f"Fetching odds for game {game_id}")

            # Simulate API call that returns no data in v1
            rows = []

            # If we had real data, we would:
            # 1. Parse provider response
            # 2. Detect/convert odds format using detect_and_convert_odds()
            # 3. Build OddsRow objects
            # 4. Write to database using _write_odds()

            return rows

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
