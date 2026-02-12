"""Persistence of edge calculation results to database.

Updates sim_market_probs and player_projections with edge, kelly_fraction,
and edge_computed_at (D-012).
"""

import asyncpg

from mlb.db.models import Table
from mlb.odds.edge import EdgeResult


async def persist_edges(
    conn: asyncpg.Connection,
    edge_result: EdgeResult,
) -> None:
    """Persist edge calculation results to database.

    Updates existing rows in sim_market_probs and player_projections with
    edge, kelly_fraction, and edge_computed_at values.

    Args:
        conn: Database connection
        edge_result: EdgeResult from compute_edges()

    Notes:
        - Updates are idempotent (acceptance criterion #9)
        - Only updates rows that have matching odds (others remain NULL)
        - Sets edge_computed_at on ALL sim_market_probs rows for the projection,
          even those with no odds (D-012)

    Raises:
        asyncpg.PostgresError: If database update fails
    """
    # Update sim_market_probs with edges
    if edge_result.market_edges:
        await _update_market_edges(conn, edge_result)

    # Update player_projections with edges (may be all NULL for v1)
    if edge_result.player_edges:
        await _update_player_edges(conn, edge_result)

    # Set edge_computed_at on ALL sim_market_probs for this projection (D-012)
    # This marks the edge pass as complete, even for rows with no odds
    await conn.execute(
        f"""
        UPDATE {Table.SIM_MARKET_PROBS}
        SET edge_computed_at = $1
        WHERE projection_id = $2
        """,
        edge_result.computed_at,
        edge_result.projection_id,
    )


async def _update_market_edges(
    conn: asyncpg.Connection,
    edge_result: EdgeResult,
) -> None:
    """Update sim_market_probs with market edges.

    Args:
        conn: Database connection
        edge_result: EdgeResult containing market edges
    """
    # Batch update using executemany for efficiency
    update_data = [
        (
            edge.edge,
            edge.kelly_fraction,
            edge.prob_id,
        )
        for edge in edge_result.market_edges
    ]

    if update_data:
        await conn.executemany(
            f"""
            UPDATE {Table.SIM_MARKET_PROBS}
            SET edge = $1, kelly_fraction = $2
            WHERE prob_id = $3
            """,
            update_data,
        )


async def _update_player_edges(
    conn: asyncpg.Connection,
    edge_result: EdgeResult,
) -> None:
    """Update player_projections with player prop edges.

    Args:
        conn: Database connection
        edge_result: EdgeResult containing player edges

    Notes:
        - Only updates rows that have non-NULL edge values
        - Rows with no matching odds (edge = None) are skipped
    """
    # Filter out player edges with no odds (edge is None)
    update_data = [
        (
            edge.edge,
            edge.kelly_fraction,
            edge.pp_id,
        )
        for edge in edge_result.player_edges
        if edge.edge is not None
    ]

    if update_data:
        await conn.executemany(
            f"""
            UPDATE {Table.PLAYER_PROJECTIONS}
            SET edge = $1, kelly_fraction = $2
            WHERE pp_id = $3
            """,
            update_data,
        )
