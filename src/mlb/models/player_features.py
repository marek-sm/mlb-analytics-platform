"""Player-level feature engineering for hitter and pitcher prop models.

This module builds player-specific features from ingested data for Unit 5.
Reuses game-level covariates from models/features.py (Unit 4).
"""

import asyncpg
from dataclasses import dataclass
from datetime import date, timedelta

from mlb.config.settings import get_config
from mlb.db.models import Table


@dataclass
class HitterFeatures:
    """Per-hitter per-game feature vector for player prop models.

    Top-7 lineup hitters only (batting_order 1-7).
    """

    player_id: int
    game_id: str
    batting_order: int  # 1-7 (top 7 only)
    bats: str  # 'L' | 'R' | 'S'
    opp_starter_throws: str  # 'L' | 'R'
    platoon_adv: bool  # True if bats opposite of starter throws (D-030: switch always True)
    days_rest: int  # days since last game started
    starts_last_7: int  # games started in last 7 days
    starts_last_14: int
    rolling_pa_per_game: float  # shrunk PA/G over rolling window
    rolling_h_rate: float  # shrunk H/PA
    rolling_tb_rate: float  # shrunk TB/PA
    rolling_hr_rate: float  # shrunk HR/PA
    rolling_rbi_rate: float  # shrunk RBI/PA
    rolling_r_rate: float  # shrunk R/PA
    rolling_bb_rate: float  # shrunk BB/PA
    game_mu: float  # team expected runs from Unit 4 (context)


@dataclass
class PitcherFeatures:
    """Per-starter per-game feature vector for pitcher prop models.

    Starting pitchers only.
    """

    player_id: int
    game_id: str
    throws: str  # 'L' | 'R'
    days_rest: int  # days since last start
    rolling_pitch_count: float  # shrunk avg pitch count over window
    rolling_ip_outs: float  # shrunk avg outs per start
    rolling_k_rate: float  # shrunk K per batter faced (BF approximated as ip_outs × bf_per_out_ratio)
    rolling_er_rate: float  # shrunk ER per out
    opp_lineup_ops: float  # opposing lineup aggregate OPS (from Unit 4)
    game_mu: float  # opposing team expected runs (context)


async def build_hitter_features(
    conn: asyncpg.Connection,
    game_id: str,
    game_date: date,
    team_id: int,
    opp_starter_throws: str,
    game_mu: float
) -> list[HitterFeatures]:
    """Build hitter features for top-7 lineup positions only.

    Args:
        conn: Database connection
        game_id: Game identifier
        game_date: Date of the game (for rolling windows)
        team_id: Team identifier (to filter lineup to one team)
        opp_starter_throws: Opposing starting pitcher's throwing hand
        game_mu: Expected runs for this team (from Unit 4 model)

    Returns:
        List of HitterFeatures for batting_order 1-7 only (single team)
    """
    config = get_config()

    # Get confirmed lineup for this game and team (top 7 only)
    lineup = await conn.fetch(
        f"""
        SELECT l.player_id, l.batting_order, p.bats
        FROM {Table.LINEUPS} l
        JOIN {Table.PLAYERS} p ON l.player_id = p.player_id
        WHERE l.game_id = $1
          AND l.team_id = $2
          AND l.is_confirmed = TRUE
          AND l.batting_order >= 1
          AND l.batting_order <= 7
        ORDER BY l.batting_order
        """,
        game_id,
        team_id,
    )

    if not lineup:
        return []

    features_list = []

    for row in lineup:
        player_id = row["player_id"]
        batting_order = row["batting_order"]
        bats = row["bats"] or "R"  # Default to R if NULL

        # Calculate platoon advantage (D-030: switch hitters always True)
        if bats == "S":
            platoon_adv = True
        else:
            platoon_adv = (bats == "L" and opp_starter_throws == "R") or \
                          (bats == "R" and opp_starter_throws == "L")

        # Calculate days_rest and start counts
        start_stats = await _get_hitter_start_stats(conn, player_id, game_date)

        # Calculate rolling stats (shrunk)
        rolling_stats = await _get_hitter_rolling_stats(conn, player_id, game_date)

        features = HitterFeatures(
            player_id=player_id,
            game_id=game_id,
            batting_order=batting_order,
            bats=bats,
            opp_starter_throws=opp_starter_throws,
            platoon_adv=platoon_adv,
            days_rest=start_stats["days_rest"],
            starts_last_7=start_stats["starts_last_7"],
            starts_last_14=start_stats["starts_last_14"],
            rolling_pa_per_game=rolling_stats["pa_per_game"],
            rolling_h_rate=rolling_stats["h_rate"],
            rolling_tb_rate=rolling_stats["tb_rate"],
            rolling_hr_rate=rolling_stats["hr_rate"],
            rolling_rbi_rate=rolling_stats["rbi_rate"],
            rolling_r_rate=rolling_stats["r_rate"],
            rolling_bb_rate=rolling_stats["bb_rate"],
            game_mu=game_mu,
        )

        features_list.append(features)

    return features_list


async def build_pitcher_features(
    conn: asyncpg.Connection,
    game_id: str,
    game_date: date,
    pitcher_id: int,
    opp_lineup_ops: float,
    opp_game_mu: float
) -> PitcherFeatures:
    """Build pitcher features for a starting pitcher.

    Args:
        conn: Database connection
        game_id: Game identifier
        game_date: Date of the game
        pitcher_id: Pitcher player_id
        opp_lineup_ops: Opposing lineup aggregate OPS (from Unit 4)
        opp_game_mu: Opposing team expected runs (from Unit 4)

    Returns:
        PitcherFeatures dataclass
    """
    config = get_config()

    # Get pitcher handedness
    pitcher = await conn.fetchrow(
        f"""
        SELECT throws
        FROM {Table.PLAYERS}
        WHERE player_id = $1
        """,
        pitcher_id,
    )

    if not pitcher:
        raise ValueError(f"Pitcher {pitcher_id} not found")

    throws = pitcher["throws"] or "R"  # Default to R if NULL

    # Calculate rest and rolling stats
    rest_stats = await _get_pitcher_rest_stats(conn, pitcher_id, game_date)
    rolling_stats = await _get_pitcher_rolling_stats(conn, pitcher_id, game_date)

    return PitcherFeatures(
        player_id=pitcher_id,
        game_id=game_id,
        throws=throws,
        days_rest=rest_stats["days_rest"],
        rolling_pitch_count=rolling_stats["pitch_count"],
        rolling_ip_outs=rolling_stats["ip_outs"],
        rolling_k_rate=rolling_stats["k_rate"],
        rolling_er_rate=rolling_stats["er_rate"],
        opp_lineup_ops=opp_lineup_ops,
        game_mu=opp_game_mu,
    )


async def _get_hitter_start_stats(
    conn: asyncpg.Connection, player_id: int, game_date: date
) -> dict:
    """Calculate hitter start statistics (days_rest, starts_last_7, starts_last_14)."""
    # Get recent games where player was in lineup (is_starter or appeared)
    recent_games = await conn.fetch(
        f"""
        SELECT g.game_date
        FROM {Table.LINEUPS} l
        JOIN {Table.GAMES} g ON l.game_id = g.game_id
        WHERE l.player_id = $1
          AND g.game_date < $2
          AND g.game_date >= $3
          AND l.is_confirmed = TRUE
        ORDER BY g.game_date DESC
        """,
        player_id,
        game_date,
        game_date - timedelta(days=14),
    )

    if not recent_games:
        return {
            "days_rest": 7,  # Default assumption
            "starts_last_7": 0,
            "starts_last_14": 0,
        }

    # Days since last game
    last_game_date = recent_games[0]["game_date"]
    days_rest = (game_date - last_game_date).days

    # Count starts in last 7 and 14 days
    cutoff_7 = game_date - timedelta(days=7)
    cutoff_14 = game_date - timedelta(days=14)

    starts_last_7 = sum(1 for g in recent_games if g["game_date"] >= cutoff_7)
    starts_last_14 = sum(1 for g in recent_games if g["game_date"] >= cutoff_14)

    return {
        "days_rest": days_rest,
        "starts_last_7": starts_last_7,
        "starts_last_14": starts_last_14,
    }


async def _get_hitter_rolling_stats(
    conn: asyncpg.Connection, player_id: int, game_date: date
) -> dict:
    """Calculate shrunk rolling batting stats."""
    config = get_config()
    window_start = game_date - timedelta(days=config.rolling_window_batting_days)
    k_batter = config.shrinkage_k_batter

    # League average priors (D-029: shrinkage toward league means)
    league_pa_per_game = 3.5
    league_h_rate = 0.250
    league_tb_rate = 0.400
    league_hr_rate = 0.030
    league_rbi_rate = 0.130
    league_r_rate = 0.130
    league_bb_rate = 0.090

    # Get batting stats in window
    stats = await conn.fetchrow(
        f"""
        SELECT
            COUNT(DISTINCT pgl.game_id) as games,
            COALESCE(SUM(pgl.pa), 0) as total_pa,
            COALESCE(SUM(pgl.h), 0) as total_h,
            COALESCE(SUM(pgl.tb), 0) as total_tb,
            COALESCE(SUM(pgl.hr), 0) as total_hr,
            COALESCE(SUM(pgl.rbi), 0) as total_rbi,
            COALESCE(SUM(pgl.r), 0) as total_r,
            COALESCE(SUM(pgl.bb), 0) as total_bb
        FROM {Table.PLAYER_GAME_LOGS} pgl
        JOIN {Table.GAMES} g ON pgl.game_id = g.game_id
        WHERE pgl.player_id = $1
          AND g.game_date >= $2
          AND g.game_date < $3
          AND pgl.pa IS NOT NULL
        """,
        player_id,
        window_start,
        game_date,
    )

    if not stats or stats["games"] == 0 or stats["total_pa"] == 0:
        # Insufficient history: return league averages
        return {
            "pa_per_game": league_pa_per_game,
            "h_rate": league_h_rate,
            "tb_rate": league_tb_rate,
            "hr_rate": league_hr_rate,
            "rbi_rate": league_rbi_rate,
            "r_rate": league_r_rate,
            "bb_rate": league_bb_rate,
        }

    total_pa = stats["total_pa"]
    games = stats["games"]

    # Empirical Bayes shrinkage (D-021, D-029)
    shrinkage_weight = total_pa / (total_pa + k_batter)

    # PA per game (shrunk toward league mean)
    raw_pa_per_game = total_pa / games
    pa_per_game = shrinkage_weight * raw_pa_per_game + \
                  (1 - shrinkage_weight) * league_pa_per_game

    # Per-PA rates (shrunk toward league means)
    raw_h_rate = stats["total_h"] / total_pa
    h_rate = shrinkage_weight * raw_h_rate + (1 - shrinkage_weight) * league_h_rate

    raw_tb_rate = stats["total_tb"] / total_pa
    tb_rate = shrinkage_weight * raw_tb_rate + (1 - shrinkage_weight) * league_tb_rate

    raw_hr_rate = stats["total_hr"] / total_pa
    hr_rate = shrinkage_weight * raw_hr_rate + (1 - shrinkage_weight) * league_hr_rate

    raw_rbi_rate = stats["total_rbi"] / total_pa
    rbi_rate = shrinkage_weight * raw_rbi_rate + (1 - shrinkage_weight) * league_rbi_rate

    raw_r_rate = stats["total_r"] / total_pa
    r_rate = shrinkage_weight * raw_r_rate + (1 - shrinkage_weight) * league_r_rate

    raw_bb_rate = stats["total_bb"] / total_pa
    bb_rate = shrinkage_weight * raw_bb_rate + (1 - shrinkage_weight) * league_bb_rate

    return {
        "pa_per_game": pa_per_game,
        "h_rate": h_rate,
        "tb_rate": tb_rate,
        "hr_rate": hr_rate,
        "rbi_rate": rbi_rate,
        "r_rate": r_rate,
        "bb_rate": bb_rate,
    }


async def _get_pitcher_rest_stats(
    conn: asyncpg.Connection, pitcher_id: int, game_date: date
) -> dict:
    """Calculate pitcher rest days."""
    # Get last start date
    last_start = await conn.fetchrow(
        f"""
        SELECT g.game_date
        FROM {Table.PLAYER_GAME_LOGS} pgl
        JOIN {Table.GAMES} g ON pgl.game_id = g.game_id
        WHERE pgl.player_id = $1
          AND g.game_date < $2
          AND pgl.is_starter = TRUE
        ORDER BY g.game_date DESC
        LIMIT 1
        """,
        pitcher_id,
        game_date,
    )

    if not last_start:
        return {"days_rest": 4}  # Default typical starter rest

    days_rest = (game_date - last_start["game_date"]).days

    return {"days_rest": days_rest}


async def _get_pitcher_rolling_stats(
    conn: asyncpg.Connection, pitcher_id: int, game_date: date
) -> dict:
    """Calculate shrunk rolling pitching stats."""
    config = get_config()
    window_start = game_date - timedelta(days=config.rolling_window_pitching_days)
    k_pitcher = config.shrinkage_k_pitcher

    # League average priors (D-029)
    league_pitch_count = 90.0
    league_ip_outs = 18.0  # 6 IP average
    league_k_rate = 0.220
    league_er_rate = 0.150  # ER per out

    # Get pitching stats in window
    stats = await conn.fetchrow(
        f"""
        SELECT
            COUNT(*) as starts,
            COALESCE(SUM(pgl.pitch_count), 0) as total_pitches,
            COALESCE(SUM(pgl.ip_outs), 0) as total_outs,
            COALESCE(SUM(pgl.k), 0) as total_k,
            COALESCE(SUM(pgl.er), 0) as total_er
        FROM {Table.PLAYER_GAME_LOGS} pgl
        JOIN {Table.GAMES} g ON pgl.game_id = g.game_id
        WHERE pgl.player_id = $1
          AND g.game_date >= $2
          AND g.game_date < $3
          AND pgl.is_starter = TRUE
        """,
        pitcher_id,
        window_start,
        game_date,
    )

    if not stats or stats["starts"] == 0:
        # Insufficient history: return league averages
        return {
            "pitch_count": league_pitch_count,
            "ip_outs": league_ip_outs,
            "k_rate": league_k_rate,
            "er_rate": league_er_rate,
        }

    # Calculate IP for shrinkage weight
    total_outs = stats["total_outs"] or 0
    total_ip = total_outs / 3.0

    # Empirical Bayes shrinkage
    shrinkage_weight = total_ip / (total_ip + k_pitcher)

    # Average pitch count per start (shrunk)
    raw_pitch_count = (stats["total_pitches"] or 0) / stats["starts"]
    pitch_count = shrinkage_weight * raw_pitch_count + \
                  (1 - shrinkage_weight) * league_pitch_count

    # Average outs per start (shrunk)
    raw_ip_outs = total_outs / stats["starts"]
    ip_outs = shrinkage_weight * raw_ip_outs + \
              (1 - shrinkage_weight) * league_ip_outs

    # K rate (per batter faced, shrunk)
    # BF approximated as ip_outs × bf_per_out_ratio (D-031)
    bf_per_out = config.bf_per_out_ratio
    total_bf_approx = total_outs * bf_per_out

    if total_bf_approx > 0:
        raw_k_rate = (stats["total_k"] or 0) / total_bf_approx
        k_rate = shrinkage_weight * raw_k_rate + (1 - shrinkage_weight) * league_k_rate
    else:
        k_rate = league_k_rate

    # ER rate (per out, shrunk)
    if total_outs > 0:
        raw_er_rate = (stats["total_er"] or 0) / total_outs
        er_rate = shrinkage_weight * raw_er_rate + (1 - shrinkage_weight) * league_er_rate
    else:
        er_rate = league_er_rate

    return {
        "pitch_count": pitch_count,
        "ip_outs": ip_outs,
        "k_rate": k_rate,
        "er_rate": er_rate,
    }
