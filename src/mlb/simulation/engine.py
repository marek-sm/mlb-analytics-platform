"""Monte Carlo simulation kernel for team runs and player props.

Implements:
- Negative Binomial sampling with correlated noise (D-032)
- Extra-innings tie-break (D-033)
- Player prop sampling for hitters and pitchers (D-034, D-035)
"""

import asyncpg
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from scipy import stats

from mlb.config.settings import get_config
from mlb.models.team_runs import TeamRunParams, predict as predict_team_runs
from mlb.models.player_props import (
    HitterPropParams,
    PitcherPropParams,
    predict_hitters,
    predict_pitcher,
)
from mlb.models.features import build_game_features
from mlb.db.models import Table


@dataclass
class HitterSimResult:
    """Per-hitter simulation results across all trials."""

    player_id: int
    p_start: float
    pa: np.ndarray  # shape (sim_n,)
    h: np.ndarray  # shape (sim_n,)
    tb: np.ndarray  # shape (sim_n,)
    hr: np.ndarray  # shape (sim_n,)
    rbi: np.ndarray  # shape (sim_n,)
    r: np.ndarray  # shape (sim_n,)
    bb: np.ndarray | None  # shape (sim_n,) or None if BB model not available


@dataclass
class PitcherSimResult:
    """Per-pitcher simulation results across all trials."""

    player_id: int
    outs: np.ndarray  # shape (sim_n,)
    k: np.ndarray  # shape (sim_n,)
    er: np.ndarray  # shape (sim_n,)


@dataclass
class SimResult:
    """Output of simulate_game() — contains team scores and player prop samples."""

    game_id: str
    run_ts: datetime  # pipeline run timestamp (UTC)
    home_mu: float  # logged for audit, not re-used
    away_mu: float
    home_disp: float
    away_disp: float
    sim_n: int  # actual number of trials run
    home_scores: np.ndarray  # shape (sim_n,) — final scores incl. tie-break
    away_scores: np.ndarray  # shape (sim_n,)
    model_version: str
    hitter_sims: dict[int, HitterSimResult]  # player_id → HitterSimResult
    pitcher_sims: dict[int, PitcherSimResult]  # player_id → PitcherSimResult


def _sample_nb_scores(
    mu: float, r: float, n: int, correlation: float = 0.0, shared_noise: np.ndarray | None = None
) -> np.ndarray:
    """Sample from Negative Binomial distribution with optional correlated noise.

    Args:
        mu: Expected value (mean)
        r: Dispersion parameter
        n: Number of samples
        correlation: Correlation coefficient for shared noise (0.0 = independent)
        shared_noise: Pre-generated shared standard normal noise (for correlated sampling)

    Returns:
        Array of shape (n,) with NB-distributed scores
    """
    if correlation > 0.0 and shared_noise is not None:
        # Apply correlated noise using copula (D-032)
        # Generate independent standard normal noise
        independent_noise = np.random.standard_normal(n)

        # Combine shared and independent noise with correlation structure
        # correlated_noise = rho * shared_noise + sqrt(1 - rho^2) * independent_noise
        correlated_noise = (
            correlation * shared_noise +
            np.sqrt(1 - correlation ** 2) * independent_noise
        )

        # Convert correlated normal to uniform via CDF
        uniform_samples = stats.norm.cdf(correlated_noise)

        # Convert uniform to NB via inverse CDF
        p = mu / (mu + r)  # NB probability parameter
        return stats.nbinom.ppf(uniform_samples, r, 1 - p).astype(int)
    else:
        # Standard independent sampling
        p = mu / (mu + r)
        return np.random.negative_binomial(r, 1 - p, size=n)


def _resolve_tie(
    home_mu: float,
    away_mu: float,
    home_bullpen_usage: float,
    away_bullpen_usage: float,
) -> bool:
    """Resolve a tie probabilistically (D-033).

    Args:
        home_mu: Home team expected runs
        away_mu: Away team expected runs
        home_bullpen_usage: Home bullpen innings used in recent days (fatigue proxy)
        away_bullpen_usage: Away bullpen innings used in recent days (fatigue proxy)

    Returns:
        True if home team wins, False if away team wins
    """
    # Base probability from relative strength
    p_home_win = home_mu / (home_mu + away_mu)

    # Adjust by ±0.02 for bullpen fatigue differential (D-033)
    usage_diff = home_bullpen_usage - away_bullpen_usage
    usage_threshold = 0.1 * max(home_bullpen_usage, away_bullpen_usage)  # 10% threshold

    if abs(usage_diff) > usage_threshold:
        if usage_diff > 0:
            # Home bullpen more fatigued → reduce home win probability
            p_home_win -= 0.02
        else:
            # Away bullpen more fatigued → increase home win probability
            p_home_win += 0.02

    return np.random.random() < p_home_win


def _sample_hitter_stats(
    params: HitterPropParams, sim_n: int
) -> HitterSimResult:
    """Sample hitter stats for all trials (D-035).

    Args:
        params: HitterPropParams from Unit 5
        sim_n: Number of simulation trials

    Returns:
        HitterSimResult with sampled stats
    """
    # Sample PA from distribution
    pa_probs = np.array(params.pa_dist)
    pa_values = np.arange(len(pa_probs))  # 0, 1, 2, 3, 4, 5, 6
    pa_samples = np.random.choice(pa_values, size=sim_n, p=pa_probs)

    # For each trial, determine if hitter starts
    starts = np.random.random(sim_n) < params.p_start
    pa_samples = pa_samples * starts  # Zero out PA if didn't start

    # Sample events (D-035: independent Bernoulli per PA)
    # H: Binomial(PA, h_rate)
    h_samples = np.array([
        np.random.binomial(pa, params.h_rate) if pa > 0 else 0
        for pa in pa_samples
    ])

    # HR: Binomial(PA, hr_rate)
    hr_samples = np.array([
        np.random.binomial(pa, params.hr_rate) if pa > 0 else 0
        for pa in pa_samples
    ])

    # TB: derived from H + extra bases (not independently sampled)
    # Simplified: TB ≈ H × (1 + extra_base_rate)
    # Use tb_rate directly as TB per PA
    tb_samples = np.array([
        max(h_samples[i], np.random.poisson(pa * params.tb_rate))
        for i, pa in enumerate(pa_samples)
    ])

    # RBI: Poisson(PA × rbi_rate)
    rbi_samples = np.array([
        np.random.poisson(pa * params.rbi_rate) if pa > 0 else 0
        for pa in pa_samples
    ])

    # R: Poisson(PA × r_rate)
    r_samples = np.array([
        np.random.poisson(pa * params.r_rate) if pa > 0 else 0
        for pa in pa_samples
    ])

    # BB: Binomial(PA, bb_rate) if bb_rate is available
    bb_samples = None
    if params.bb_rate is not None:
        bb_samples = np.array([
            np.random.binomial(pa, params.bb_rate) if pa > 0 else 0
            for pa in pa_samples
        ])

    return HitterSimResult(
        player_id=params.player_id,
        p_start=params.p_start,
        pa=pa_samples,
        h=h_samples,
        tb=tb_samples,
        hr=hr_samples,
        rbi=rbi_samples,
        r=r_samples,
        bb=bb_samples,
    )


def _sample_pitcher_stats(
    params: PitcherPropParams, sim_n: int
) -> PitcherSimResult:
    """Sample pitcher stats for all trials.

    Args:
        params: PitcherPropParams from Unit 5
        sim_n: Number of simulation trials

    Returns:
        PitcherSimResult with sampled stats
    """
    # Sample outs from distribution (buckets: 0-3, 4-6, 7-9, ..., 25-27+)
    outs_probs = np.array(params.outs_dist)
    outs_buckets = len(outs_probs)

    # Sample bucket indices
    bucket_samples = np.random.choice(outs_buckets, size=sim_n, p=outs_probs)

    # Convert buckets to actual outs (sample within bucket)
    outs_samples = np.array([
        np.random.randint(bucket * 3, min(bucket * 3 + 3, 28))
        if bucket < 9 else np.random.randint(27, 30)
        for bucket in bucket_samples
    ])

    # K: based on k_rate × BF
    # BF approximated as outs × bf_per_out_ratio (D-031)
    config = get_config()
    bf_samples = (outs_samples * config.bf_per_out_ratio).astype(int)
    k_samples = np.array([
        np.random.binomial(bf, params.k_rate) if bf > 0 else 0
        for bf in bf_samples
    ])

    # ER: based on er_rate × outs
    er_samples = np.array([
        max(0, np.random.poisson(outs * params.er_rate)) if outs > 0 else 0
        for outs in outs_samples
    ])

    return PitcherSimResult(
        player_id=params.player_id,
        outs=outs_samples,
        k=k_samples,
        er=er_samples,
    )


async def simulate_game(
    conn: asyncpg.Connection,
    game_id: str,
    model_version: str,
    sim_n: int | None = None,
    correlation: float | None = None,
) -> SimResult:
    """Run Monte Carlo simulation for a single game.

    Args:
        conn: Database connection
        game_id: Game identifier
        model_version: Model version to use for inference
        sim_n: Number of trials (None = use config default)
        correlation: Score correlation coefficient (None = use config default, 0.15)

    Returns:
        SimResult with team scores and player prop samples

    Raises:
        ValueError: If game not found or models not available
    """
    config = get_config()

    # Determine simulation parameters
    if sim_n is None:
        sim_n = config.default_sim_n

    # Clamp to [2000, 10000] per D-004
    sim_n = max(2000, min(10000, sim_n))

    if correlation is None:
        correlation = 0.15  # Default per D-032

    run_ts = datetime.utcnow()

    # Get team run parameters (Unit 4)
    team_params = await predict_team_runs(conn, game_id, model_version)

    # Get game features for bullpen usage (FC-19)
    game_features = await build_game_features(conn, game_id)

    # Generate shared noise for correlated sampling (D-032)
    shared_noise = np.random.standard_normal(sim_n) if correlation > 0 else None

    # Sample home and away scores from Negative Binomial
    home_scores = _sample_nb_scores(
        team_params.home_mu,
        team_params.home_disp,
        sim_n,
        correlation,
        shared_noise,
    )
    away_scores = _sample_nb_scores(
        team_params.away_mu,
        team_params.away_disp,
        sim_n,
        correlation,
        shared_noise,
    )

    # Resolve ties (D-033)
    ties = home_scores == away_scores
    n_ties = np.sum(ties)
    if n_ties > 0:
        # For each tied game, resolve probabilistically
        for i in np.where(ties)[0]:
            if _resolve_tie(
                team_params.home_mu,
                team_params.away_mu,
                game_features.home_bullpen_usage,
                game_features.away_bullpen_usage,
            ):
                home_scores[i] += 1  # Home wins
            else:
                away_scores[i] += 1  # Away wins

    # Get game metadata for player predictions
    game = await conn.fetchrow(
        f"""
        SELECT game_date, home_team_id, away_team_id
        FROM {Table.GAMES}
        WHERE game_id = $1
        """,
        game_id,
    )

    if not game:
        raise ValueError(f"Game not found: {game_id}")

    game_date = game["game_date"]
    home_team_id = game["home_team_id"]
    away_team_id = game["away_team_id"]

    # Get starting pitchers
    starters = await conn.fetch(
        f"""
        SELECT l.player_id, l.team_id, p.throws
        FROM {Table.LINEUPS} l
        JOIN {Table.PLAYERS} p ON l.player_id = p.player_id
        WHERE l.game_id = $1
          AND l.is_confirmed = TRUE
          AND l.batting_order = 1
        """,
        game_id,
    )

    starter_map = {s["team_id"]: s for s in starters}

    # Get opposing starter throws for hitter predictions
    home_opp_throws = (
        starter_map[away_team_id]["throws"]
        if away_team_id in starter_map
        else "R"
    )
    away_opp_throws = (
        starter_map[home_team_id]["throws"]
        if home_team_id in starter_map
        else "R"
    )

    # Predict and simulate hitters
    hitter_sims = {}

    # Home hitters
    home_hitters = await predict_hitters(
        conn,
        game_id,
        game_date,
        home_team_id,
        home_opp_throws,
        team_params.home_mu,
        model_version,
    )
    for hitter_params in home_hitters:
        hitter_sims[hitter_params.player_id] = _sample_hitter_stats(
            hitter_params, sim_n
        )

    # Away hitters
    away_hitters = await predict_hitters(
        conn,
        game_id,
        game_date,
        away_team_id,
        away_opp_throws,
        team_params.away_mu,
        model_version,
    )
    for hitter_params in away_hitters:
        hitter_sims[hitter_params.player_id] = _sample_hitter_stats(
            hitter_params, sim_n
        )

    # Predict and simulate pitchers
    pitcher_sims = {}

    # Home pitcher (if available)
    if home_team_id in starter_map:
        home_pitcher_id = starter_map[home_team_id]["player_id"]

        # Get away lineup OPS (simplified: use game_mu as proxy)
        # In v2, calculate actual lineup OPS from player stats
        away_lineup_ops = 0.700  # League average placeholder

        home_pitcher_params = await predict_pitcher(
            conn,
            game_id,
            game_date,
            home_pitcher_id,
            away_lineup_ops,
            team_params.away_mu,
            model_version,
        )
        pitcher_sims[home_pitcher_id] = _sample_pitcher_stats(
            home_pitcher_params, sim_n
        )

    # Away pitcher (if available)
    if away_team_id in starter_map:
        away_pitcher_id = starter_map[away_team_id]["player_id"]
        home_lineup_ops = 0.700  # League average placeholder

        away_pitcher_params = await predict_pitcher(
            conn,
            game_id,
            game_date,
            away_pitcher_id,
            home_lineup_ops,
            team_params.home_mu,
            model_version,
        )
        pitcher_sims[away_pitcher_id] = _sample_pitcher_stats(
            away_pitcher_params, sim_n
        )

    return SimResult(
        game_id=game_id,
        run_ts=run_ts,
        home_mu=team_params.home_mu,
        away_mu=team_params.away_mu,
        home_disp=team_params.home_disp,
        away_disp=team_params.away_disp,
        sim_n=sim_n,
        home_scores=home_scores,
        away_scores=away_scores,
        model_version=model_version,
        hitter_sims=hitter_sims,
        pitcher_sims=pitcher_sims,
    )
