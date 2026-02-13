"""Publishing gate logic for lineup uncertainty policy.

Implements D-046: Team markets always publishable when edge computed.
Player props require confirmed lineup OR p_start >= threshold.
"""

import logging

import asyncpg

from mlb.config.settings import get_config
from mlb.db.models import Table
from mlb.db.pool import get_pool

logger = logging.getLogger(__name__)


async def is_publishable(
    game_id: str,
    market: str,
    player_id: int | None = None,
) -> bool:
    """Check if a projection is eligible for publication.

    Args:
        game_id: Game identifier
        market: Market type ('ml', 'rl', 'total', 'team_total', or player stat)
        player_id: Player ID (required for player props, None for team markets)

    Returns:
        True if publishable, False otherwise

    Rules:
        Team markets (player_id is None):
            - Publishable if edge_computed_at IS NOT NULL

        Player props (player_id is not None):
            - Publishable if edge_computed_at IS NOT NULL AND
              (lineup confirmed for player's team OR p_start >= threshold)

    Notes:
        - Enforces lineup uncertainty policy (D-046)
        - Team markets exempt from lineup requirement
        - p_start_threshold configurable (default 0.85)
    """
    config = get_config()
    pool = await get_pool()

    async with pool.acquire() as conn:
        # Team markets: check if edge computed
        if player_id is None:
            # Get latest projection for this game
            edge_computed = await conn.fetchval(
                f"""
                SELECT edge_computed_at
                FROM {Table.PROJECTIONS}
                WHERE game_id = $1
                ORDER BY run_ts DESC
                LIMIT 1
                """,
                game_id,
            )

            # Team markets are publishable if edge was computed
            return edge_computed is not None

        # Player props: check edge AND (lineup confirmed OR p_start >= threshold)
        # Get latest projection
        projection_row = await conn.fetchrow(
            f"""
            SELECT projection_id, edge_computed_at
            FROM {Table.PROJECTIONS}
            WHERE game_id = $1
            ORDER BY run_ts DESC
            LIMIT 1
            """,
            game_id,
        )

        if not projection_row or not projection_row["edge_computed_at"]:
            return False

        projection_id = projection_row["projection_id"]

        # Get player's team
        player_team = await conn.fetchval(
            f"""
            SELECT team_id
            FROM {Table.LINEUPS}
            WHERE game_id = $1 AND player_id = $2
            LIMIT 1
            """,
            game_id,
            player_id,
        )

        if not player_team:
            logger.warning(
                f"Player {player_id} not found in lineup for game {game_id}"
            )
            return False

        # Check if lineup is confirmed for this team
        lineup_confirmed = await conn.fetchval(
            f"""
            SELECT EXISTS(
                SELECT 1
                FROM {Table.LINEUPS}
                WHERE game_id = $1
                  AND team_id = $2
                  AND is_confirmed = TRUE
                LIMIT 1
            )
            """,
            game_id,
            player_team,
        )

        if lineup_confirmed:
            return True

        # Lineup not confirmed - check p_start threshold
        p_start = await conn.fetchval(
            f"""
            SELECT p_start
            FROM {Table.PLAYER_PROJECTIONS}
            WHERE projection_id = $1
              AND player_id = $2
            LIMIT 1
            """,
            projection_id,
            player_id,
        )

        if p_start is None:
            logger.warning(
                f"No p_start found for player {player_id} in projection {projection_id}"
            )
            return False

        # Check if p_start meets threshold
        return p_start >= config.p_start_threshold
