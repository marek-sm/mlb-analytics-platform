"""Persistence of simulation results to database.

Writes SimResult → projections + sim_market_probs + player_projections.
Leaves edge, kelly_fraction, and edge_computed_at as NULL (populated by Unit 7).
"""

import asyncpg
import json
from typing import Any

from mlb.db.models import Table
from mlb.simulation.engine import SimResult
from mlb.simulation.markets import MarketProb, PlayerPropProb


async def persist_simulation_results(
    conn: asyncpg.Connection,
    sim_result: SimResult,
    team_markets: list[MarketProb],
    player_props: list[PlayerPropProb],
) -> int:
    """Persist simulation results to database.

    Args:
        conn: Database connection
        sim_result: SimResult from simulation kernel
        team_markets: Team market probabilities from derive_team_markets()
        player_props: Player prop probabilities from derive_player_props()

    Returns:
        projection_id (int) for the newly created projection

    Raises:
        asyncpg.PostgresError: If persistence fails
    """
    # Build meta JSONB (model_version + feature snapshot)
    meta: dict[str, Any] = {
        "model_version": sim_result.model_version,
        "correlation": 0.15,  # Default correlation used (D-032)
        "n_hitters": len(sim_result.hitter_sims),
        "n_pitchers": len(sim_result.pitcher_sims),
    }

    # Insert into projections table
    projection_id = await conn.fetchval(
        f"""
        INSERT INTO {Table.PROJECTIONS} (
            game_id,
            run_ts,
            home_mu,
            away_mu,
            home_disp,
            away_disp,
            sim_n,
            meta
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        RETURNING projection_id
        """,
        sim_result.game_id,
        sim_result.run_ts,
        sim_result.home_mu,
        sim_result.away_mu,
        sim_result.home_disp,
        sim_result.away_disp,
        sim_result.sim_n,
        json.dumps(meta),
    )

    # Insert team markets into sim_market_probs (batch)
    if team_markets:
        market_rows = [
            (
                projection_id,
                market.market,
                market.side,
                market.line,
                market.prob,
                None,  # edge (NULL, populated by Unit 7)
                None,  # kelly_fraction (NULL, populated by Unit 7)
            )
            for market in team_markets
        ]

        await conn.executemany(
            f"""
            INSERT INTO {Table.SIM_MARKET_PROBS} (
                projection_id,
                market,
                side,
                line,
                prob,
                edge,
                kelly_fraction
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            market_rows,
        )

    # Insert player props into player_projections (batch)
    if player_props:
        prop_rows = [
            (
                projection_id,
                prop.player_id,
                sim_result.game_id,
                prop.p_start,
                prop.stat,
                prop.line,
                prop.prob_over,
                None,  # edge (NULL, populated by Unit 7)
                None,  # kelly_fraction (NULL, populated by Unit 7)
            )
            for prop in player_props
        ]

        await conn.executemany(
            f"""
            INSERT INTO {Table.PLAYER_PROJECTIONS} (
                projection_id,
                player_id,
                game_id,
                p_start,
                stat,
                line,
                prob_over,
                edge,
                kelly_fraction
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
            prop_rows,
        )

    return projection_id
