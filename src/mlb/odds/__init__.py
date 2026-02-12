"""Odds processing, edge calculation, and bankroll sizing (Unit 7)."""

from mlb.odds.best_line import BestLine, get_best_lines
from mlb.odds.devig import proportional_devig
from mlb.odds.edge import EdgeResult, MarketEdge, PlayerEdge, compute_edges

__all__ = [
    "BestLine",
    "get_best_lines",
    "proportional_devig",
    "EdgeResult",
    "MarketEdge",
    "PlayerEdge",
    "compute_edges",
]
