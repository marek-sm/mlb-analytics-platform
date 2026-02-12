"""Market probability derivation from simulation results.

Pure functions that convert SimResult arrays into market probabilities.
Implements D-034 hardcoded lines for v1.
"""

import numpy as np
from dataclasses import dataclass

from mlb.simulation.engine import SimResult, HitterSimResult, PitcherSimResult


@dataclass
class MarketProb:
    """Team market probability row for persistence."""

    market: str  # 'ml' | 'rl' | 'total' | 'team_total'
    side: str | None
    line: float | None
    prob: float


@dataclass
class PlayerPropProb:
    """Player prop probability row for persistence."""

    player_id: int
    p_start: float
    stat: str  # 'H' | 'TB' | 'HR' | 'RBI' | 'R' | 'BB' | 'K' | 'OUTS' | 'ER'
    line: float  # main line only (D-034)
    prob_over: float


# Hardcoded main lines for v1 (D-034)
MAIN_LINES = {
    "H": 0.5,
    "TB": 1.5,
    "HR": 0.5,
    "RBI": 0.5,
    "R": 0.5,
    "BB": 0.5,
    "K": 4.5,
    "OUTS": 16.5,
    "ER": 2.5,
}


def derive_team_markets(sim_result: SimResult) -> list[MarketProb]:
    """Derive all team market probabilities from simulation results.

    Derives:
    - Moneyline (home, away)
    - Run Line ±1.5 (home, away)
    - Game Total 8.5 (over, under)
    - Team Totals 4.5 (home over/under, away over/under)

    Args:
        sim_result: SimResult from simulation kernel

    Returns:
        List of MarketProb objects
    """
    home_scores = sim_result.home_scores
    away_scores = sim_result.away_scores
    total_scores = home_scores + away_scores

    n = len(home_scores)

    markets = []

    # Moneyline
    p_home_win = np.sum(home_scores > away_scores) / n
    p_away_win = np.sum(away_scores > home_scores) / n

    markets.append(MarketProb(market="ml", side="home", line=None, prob=p_home_win))
    markets.append(MarketProb(market="ml", side="away", line=None, prob=p_away_win))

    # Run Line ±1.5
    p_home_covers = np.sum(home_scores - away_scores > 1.5) / n
    p_away_covers = np.sum(away_scores - home_scores > 1.5) / n

    markets.append(MarketProb(market="rl", side="home", line=-1.5, prob=p_home_covers))
    markets.append(MarketProb(market="rl", side="away", line=1.5, prob=p_away_covers))

    # Game Total 8.5
    p_over = np.sum(total_scores > 8.5) / n
    p_under = np.sum(total_scores < 8.5) / n

    markets.append(MarketProb(market="total", side="over", line=8.5, prob=p_over))
    markets.append(MarketProb(market="total", side="under", line=8.5, prob=p_under))

    # Team Totals 4.5
    p_home_over = np.sum(home_scores > 4.5) / n
    p_home_under = np.sum(home_scores < 4.5) / n
    p_away_over = np.sum(away_scores > 4.5) / n
    p_away_under = np.sum(away_scores < 4.5) / n

    markets.append(
        MarketProb(market="team_total", side="home_over", line=4.5, prob=p_home_over)
    )
    markets.append(
        MarketProb(market="team_total", side="home_under", line=4.5, prob=p_home_under)
    )
    markets.append(
        MarketProb(market="team_total", side="away_over", line=4.5, prob=p_away_over)
    )
    markets.append(
        MarketProb(market="team_total", side="away_under", line=4.5, prob=p_away_under)
    )

    return markets


def derive_player_props(
    sim_result: SimResult,
) -> list[PlayerPropProb]:
    """Derive all player prop probabilities from simulation results.

    Uses hardcoded main lines from D-034.

    Args:
        sim_result: SimResult from simulation kernel

    Returns:
        List of PlayerPropProb objects
    """
    props = []

    # Hitter props
    for player_id, hitter_sim in sim_result.hitter_sims.items():
        n = len(hitter_sim.pa)

        # H
        p_over_h = np.sum(hitter_sim.h > MAIN_LINES["H"]) / n
        props.append(
            PlayerPropProb(
                player_id=player_id,
                p_start=hitter_sim.p_start,
                stat="H",
                line=MAIN_LINES["H"],
                prob_over=p_over_h,
            )
        )

        # TB
        p_over_tb = np.sum(hitter_sim.tb > MAIN_LINES["TB"]) / n
        props.append(
            PlayerPropProb(
                player_id=player_id,
                p_start=hitter_sim.p_start,
                stat="TB",
                line=MAIN_LINES["TB"],
                prob_over=p_over_tb,
            )
        )

        # HR
        p_over_hr = np.sum(hitter_sim.hr > MAIN_LINES["HR"]) / n
        props.append(
            PlayerPropProb(
                player_id=player_id,
                p_start=hitter_sim.p_start,
                stat="HR",
                line=MAIN_LINES["HR"],
                prob_over=p_over_hr,
            )
        )

        # RBI
        p_over_rbi = np.sum(hitter_sim.rbi > MAIN_LINES["RBI"]) / n
        props.append(
            PlayerPropProb(
                player_id=player_id,
                p_start=hitter_sim.p_start,
                stat="RBI",
                line=MAIN_LINES["RBI"],
                prob_over=p_over_rbi,
            )
        )

        # R
        p_over_r = np.sum(hitter_sim.r > MAIN_LINES["R"]) / n
        props.append(
            PlayerPropProb(
                player_id=player_id,
                p_start=hitter_sim.p_start,
                stat="R",
                line=MAIN_LINES["R"],
                prob_over=p_over_r,
            )
        )

        # BB (optional)
        if hitter_sim.bb is not None:
            p_over_bb = np.sum(hitter_sim.bb > MAIN_LINES["BB"]) / n
            props.append(
                PlayerPropProb(
                    player_id=player_id,
                    p_start=hitter_sim.p_start,
                    stat="BB",
                    line=MAIN_LINES["BB"],
                    prob_over=p_over_bb,
                )
            )

    # Pitcher props
    for player_id, pitcher_sim in sim_result.pitcher_sims.items():
        n = len(pitcher_sim.outs)

        # K
        p_over_k = np.sum(pitcher_sim.k > MAIN_LINES["K"]) / n
        props.append(
            PlayerPropProb(
                player_id=player_id,
                p_start=1.0,  # Pitchers always "start" (are listed starters)
                stat="K",
                line=MAIN_LINES["K"],
                prob_over=p_over_k,
            )
        )

        # OUTS
        p_over_outs = np.sum(pitcher_sim.outs > MAIN_LINES["OUTS"]) / n
        props.append(
            PlayerPropProb(
                player_id=player_id,
                p_start=1.0,
                stat="OUTS",
                line=MAIN_LINES["OUTS"],
                prob_over=p_over_outs,
            )
        )

        # ER
        p_over_er = np.sum(pitcher_sim.er > MAIN_LINES["ER"]) / n
        props.append(
            PlayerPropProb(
                player_id=player_id,
                p_start=1.0,
                stat="ER",
                line=MAIN_LINES["ER"],
                prob_over=p_over_er,
            )
        )

    return props
