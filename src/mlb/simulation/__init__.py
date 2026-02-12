"""Monte Carlo simulation engine for MLB games.

This module provides simulation kernel and market probability derivation.
Consumes TeamRunParams (Unit 4) and player prop params (Unit 5).
Produces SimResult, which is persisted to projections + sim_market_probs + player_projections.
"""

from mlb.simulation.engine import (
    simulate_game,
    SimResult,
    HitterSimResult,
    PitcherSimResult,
)
from mlb.simulation.markets import (
    derive_team_markets,
    derive_player_props,
    MarketProb,
    PlayerPropProb,
)
from mlb.simulation.persistence import persist_simulation_results

__all__ = [
    "simulate_game",
    "SimResult",
    "HitterSimResult",
    "PitcherSimResult",
    "derive_team_markets",
    "derive_player_props",
    "MarketProb",
    "PlayerPropProb",
    "persist_simulation_results",
]
