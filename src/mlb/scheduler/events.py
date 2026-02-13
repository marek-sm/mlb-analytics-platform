"""Event detection and rerun throttle logic.

Implements:
- check_for_changes() to detect lineup confirmations, pitcher changes, odds movements
- Rerun throttle to prevent pipeline thrashing
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import asyncpg

from mlb.config.settings import get_config
from mlb.db.models import Table
from mlb.db.pool import get_pool
from mlb.scheduler.pipeline import run_game

logger = logging.getLogger(__name__)


@dataclass
class ChangeEvent:
    """Detected change event that may trigger a rerun."""

    game_id: str
    event_type: str  # 'lineup_confirmed' | 'pitcher_change' | 'odds_movement'
    detected_at: datetime


# In-memory throttle state (game_id -> last_rerun_ts)
# For production, consider persisting to database or Redis
_last_rerun: dict[str, datetime] = {}


async def check_for_changes(game_id: str) -> list[ChangeEvent]:
    """Check for changes that warrant a rerun (lineup, pitcher, odds).

    Args:
        game_id: Game identifier

    Returns:
        List of ChangeEvent objects (may be empty)

    Notes:
        - Lineup confirmed: lineup.is_confirmed changed from False to True
        - Pitcher change: starting pitcher changed in lineup
        - Odds movement: significant price change (>5% implied probability shift)
        - For v1, simplified detection (actual implementation would track state)
    """
    events = []
    pool = await get_pool()

    async with pool.acquire() as conn:
        # Check for lineup confirmation
        # Simplified: check if lineups are now confirmed but weren't last run
        confirmed_lineups = await conn.fetch(
            f"""
            SELECT team_id, is_confirmed
            FROM {Table.LINEUPS}
            WHERE game_id = $1
              AND batting_order = 1
              AND is_confirmed = TRUE
            """,
            game_id,
        )

        if confirmed_lineups and len(confirmed_lineups) == 2:
            # Both lineups confirmed - this could be a new event
            # (In production, track previous state to detect change)
            events.append(
                ChangeEvent(
                    game_id=game_id,
                    event_type="lineup_confirmed",
                    detected_at=datetime.now(timezone.utc),
                )
            )

        # Check for pitcher change
        # Simplified: not implemented in v1 (would require state tracking)

        # Check for odds movement
        # Simplified: not implemented in v1 (would require tracking previous snapshot)

    return events


async def trigger_rerun_if_needed(game_id: str) -> bool:
    """Check for changes and trigger rerun if throttle allows.

    Args:
        game_id: Game identifier

    Returns:
        True if rerun was triggered, False if throttled or no changes

    Notes:
        - Applies rerun_throttle_minutes from config
        - At most 1 rerun per game per throttle window
    """
    config = get_config()

    # Check if throttled
    now = datetime.now(timezone.utc)
    last_run = _last_rerun.get(game_id)

    if last_run:
        time_since_last = (now - last_run).total_seconds() / 60  # minutes
        if time_since_last < config.rerun_throttle_minutes:
            logger.info(
                f"Rerun throttled for {game_id}: "
                f"{time_since_last:.1f}min < {config.rerun_throttle_minutes}min"
            )
            return False

    # Check for changes
    events = await check_for_changes(game_id)

    if not events:
        logger.debug(f"No changes detected for {game_id}")
        return False

    # Trigger rerun
    logger.info(
        f"Triggering rerun for {game_id}: {len(events)} events detected "
        f"({', '.join(e.event_type for e in events)})"
    )

    await run_game(game_id)

    # Update throttle state
    _last_rerun[game_id] = now

    return True
