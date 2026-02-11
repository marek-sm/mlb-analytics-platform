"""Player prop models for hitters and pitchers.

This module trains and runs inference for:
- P(start) model for hitters (D-027)
- PA distribution model for hitters (D-028)
- Hitter event-rate models (D-029)
- Pitcher outs distribution model
- Pitcher event-rate models (D-029)

All models consume features from player_features.py and apply shrinkage/pooling.
"""

import asyncpg
import lightgbm as lgb
import numpy as np
from dataclasses import dataclass
from datetime import date

from mlb.db.models import Table
from mlb.models.player_features import (
    build_hitter_features,
    build_pitcher_features,
    HitterFeatures,
    PitcherFeatures,
)
from mlb.models.registry import save_model, load_model


@dataclass
class HitterPropParams:
    """Output of hitter prop inference, input to Unit 6 simulation."""

    player_id: int
    game_id: str
    p_start: float  # P(start) — publishing gate input (Unit 9)
    pa_dist: list[float]  # probability mass for PA = 0, 1, 2, 3, 4, 5, 6+
    h_rate: float  # P(hit | PA) — shrunk
    tb_rate: float  # expected TB / PA — shrunk
    hr_rate: float  # P(HR | PA) — shrunk
    rbi_rate: float  # expected RBI / PA — shrunk
    r_rate: float  # expected R / PA — shrunk
    bb_rate: float | None  # P(BB | PA) — shrunk (optional, may be None)


@dataclass
class PitcherPropParams:
    """Output of pitcher prop inference, input to Unit 6 simulation."""

    player_id: int
    game_id: str
    outs_dist: list[float]  # probability mass for outs = 0–3, 4–6, ..., 25–27+
    k_rate: float  # K per batter faced (BF approximated as ip_outs × bf_per_out_ratio) — shrunk
    er_rate: float  # expected ER / out — shrunk


async def train(conn: asyncpg.Connection) -> str:
    """Train all player prop models on historical data.

    Trains:
    - P(start) model (LightGBM binary classifier, D-027)
    - PA distribution model (LightGBM multiclass, D-028)
    - Pitcher outs distribution model (LightGBM multiclass)

    Event-rate models use shrunk rolling means (D-029), no ML training.

    Args:
        conn: Database connection

    Returns:
        Model version string (timestamp)

    Raises:
        ValueError: If insufficient training data (< 30 games)
    """
    from datetime import datetime

    # Generate model version timestamp
    model_version = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Get training data: final games with confirmed lineups (≥30 games)
    training_games = await conn.fetch(
        f"""
        SELECT DISTINCT g.game_id, g.game_date, g.home_team_id, g.away_team_id,
                        g.home_score, g.away_score
        FROM {Table.GAMES} g
        WHERE g.status = 'final'
          AND g.home_score IS NOT NULL
          AND g.away_score IS NOT NULL
        ORDER BY g.game_date DESC
        LIMIT 200
        """
    )

    if len(training_games) < 30:
        raise ValueError(
            f"Insufficient training data: {len(training_games)} games (need ≥30)"
        )

    print(f"Training on {len(training_games)} games...")

    # Build training datasets for P(start) and PA models
    p_start_X = []
    p_start_y = []
    pa_X = []
    pa_y = []
    outs_X = []
    outs_y = []

    for game in training_games:
        game_id = game["game_id"]
        game_date = game["game_date"]

        # Get lineups (all positions, not just top 7)
        lineups = await conn.fetch(
            f"""
            SELECT l.player_id, l.team_id, l.batting_order, p.bats
            FROM {Table.LINEUPS} l
            JOIN {Table.PLAYERS} p ON l.player_id = p.player_id
            WHERE l.game_id = $1
              AND l.is_confirmed = TRUE
            ORDER BY l.team_id, l.batting_order
            """,
            game_id,
        )

        # Get starters for this game
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

        if len(starters) < 2:
            continue  # Skip games without starter info

        starter_map = {s["team_id"]: s for s in starters}

        # Build P(start) features for each lineup player
        for lineup_row in lineups:
            player_id = lineup_row["player_id"]
            team_id = lineup_row["team_id"]
            batting_order = lineup_row["batting_order"]
            bats = lineup_row["bats"] or "R"

            # Get opposing starter
            opp_team_id = (
                game["away_team_id"]
                if team_id == game["home_team_id"]
                else game["home_team_id"]
            )
            if opp_team_id not in starter_map:
                continue

            opp_starter_throws = starter_map[opp_team_id]["throws"] or "R"

            # Calculate platoon advantage (D-030)
            if bats == "S":
                platoon_adv = 1
            else:
                platoon_adv = int(
                    (bats == "L" and opp_starter_throws == "R") or
                    (bats == "R" and opp_starter_throws == "L")
                )

            # Get start history
            start_stats = await _get_hitter_start_stats_for_training(
                conn, player_id, game_date
            )

            # P(start) label: did this player actually start?
            # Check if player has PA in this game
            did_start = await conn.fetchval(
                f"""
                SELECT EXISTS(
                    SELECT 1
                    FROM {Table.PLAYER_GAME_LOGS}
                    WHERE player_id = $1 AND game_id = $2 AND pa > 0
                )
                """,
                player_id,
                game_id,
            )

            # P(start) features
            p_start_X.append([
                platoon_adv,
                start_stats["days_rest"],
                start_stats["starts_last_7"],
                start_stats["starts_last_14"],
                batting_order,
            ])
            p_start_y.append(1 if did_start else 0)

            # PA distribution: only for players who started and in top 7
            if did_start and batting_order <= 7:
                # Get actual PA count
                actual_pa = await conn.fetchval(
                    f"""
                    SELECT COALESCE(pa, 0)
                    FROM {Table.PLAYER_GAME_LOGS}
                    WHERE player_id = $1 AND game_id = $2
                    """,
                    player_id,
                    game_id,
                )

                # PA features (same as P(start) for simplicity)
                pa_X.append([
                    platoon_adv,
                    start_stats["days_rest"],
                    start_stats["starts_last_7"],
                    start_stats["starts_last_14"],
                    batting_order,
                ])

                # PA label: bucket into 0, 1, 2, 3, 4, 5, 6+ (7 classes)
                pa_label = min(actual_pa or 0, 6)
                pa_y.append(pa_label)

        # Build pitcher outs distribution training data
        for starter in starters:
            pitcher_id = starter["player_id"]
            team_id = starter["team_id"]

            # Get actual outs recorded
            actual_outs = await conn.fetchval(
                f"""
                SELECT COALESCE(ip_outs, 0)
                FROM {Table.PLAYER_GAME_LOGS}
                WHERE player_id = $1 AND game_id = $2 AND is_starter = TRUE
                """,
                pitcher_id,
                game_id,
            )

            if actual_outs is None:
                continue

            # Get pitcher features
            rest_stats = await _get_pitcher_rest_stats_for_training(
                conn, pitcher_id, game_date
            )
            rolling_stats = await _get_pitcher_rolling_stats_for_training(
                conn, pitcher_id, game_date
            )

            # Outs features
            outs_X.append([
                rest_stats["days_rest"],
                rolling_stats["pitch_count"],
                rolling_stats["ip_outs"],
                rolling_stats["k_rate"],
                rolling_stats["er_rate"],
            ])

            # Outs label: bucket into 0–3, 4–6, 7–9, ..., 25–27+ (10 classes)
            outs_label = min(actual_outs // 3, 9)  # 0-9 (9 = 27+ outs)
            outs_y.append(outs_label)

    if len(p_start_X) < 100:
        raise ValueError(
            f"Insufficient P(start) training samples: {len(p_start_X)} (need ≥100)"
        )

    if len(pa_X) < 100:
        raise ValueError(
            f"Insufficient PA training samples: {len(pa_X)} (need ≥100)"
        )

    if len(outs_X) < 30:
        raise ValueError(
            f"Insufficient outs training samples: {len(outs_X)} (need ≥30)"
        )

    # Train P(start) model (binary classifier, D-027)
    print(f"Training P(start) model on {len(p_start_X)} samples...")
    p_start_model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        verbose=-1,
    )
    p_start_model.fit(np.array(p_start_X), np.array(p_start_y))
    save_model(p_start_model, f"p_start_{model_version}")

    # Train PA distribution model (multiclass, D-028)
    print(f"Training PA model on {len(pa_X)} samples...")
    pa_model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=7,  # 0, 1, 2, 3, 4, 5, 6+
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        verbose=-1,
    )
    pa_model.fit(np.array(pa_X), np.array(pa_y))
    save_model(pa_model, f"pa_dist_{model_version}")

    # Train pitcher outs distribution model (multiclass)
    print(f"Training outs model on {len(outs_X)} samples...")
    outs_model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=10,  # 0–3, 4–6, ..., 27+
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        verbose=-1,
    )
    outs_model.fit(np.array(outs_X), np.array(outs_y))
    save_model(outs_model, f"outs_dist_{model_version}")

    print(f"Training complete. Models saved with version: {model_version}")

    return model_version


async def predict_hitters(
    conn: asyncpg.Connection,
    game_id: str,
    game_date: date,
    team_id: int,
    opp_starter_throws: str,
    game_mu: float,
    model_version: str,
) -> list[HitterPropParams]:
    """Run inference for all top-7 hitters in a game.

    Args:
        conn: Database connection
        game_id: Game identifier
        game_date: Date of the game
        team_id: Team identifier
        opp_starter_throws: Opposing starting pitcher's throwing hand
        game_mu: Expected runs for this team (from Unit 4)
        model_version: Model version to load

    Returns:
        List of HitterPropParams for top-7 hitters
    """
    # Load models
    p_start_model = load_model(f"p_start_{model_version}")
    pa_model = load_model(f"pa_dist_{model_version}")

    # Build features
    hitter_features_list = await build_hitter_features(
        conn, game_id, game_date, team_id, opp_starter_throws, game_mu
    )

    results = []

    for features in hitter_features_list:
        # P(start) prediction (D-027)
        p_start_X = np.array([
            [
                1 if features.platoon_adv else 0,
                features.days_rest,
                features.starts_last_7,
                features.starts_last_14,
                features.batting_order,
            ]
        ])
        p_start = float(p_start_model.predict_proba(p_start_X)[0, 1])

        # PA distribution prediction (D-028)
        pa_dist_proba = pa_model.predict_proba(p_start_X)[0]
        pa_dist = [float(p) for p in pa_dist_proba]

        # Normalize to sum to 1.0
        pa_dist_sum = sum(pa_dist)
        if pa_dist_sum > 0:
            pa_dist = [p / pa_dist_sum for p in pa_dist]
        else:
            pa_dist = [1.0] + [0.0] * 6  # Default to 0 PA

        # Event rates: use shrunk rolling means (D-029)
        # These come directly from features (already shrunk)
        result = HitterPropParams(
            player_id=features.player_id,
            game_id=game_id,
            p_start=p_start,
            pa_dist=pa_dist,
            h_rate=features.rolling_h_rate,
            tb_rate=features.rolling_tb_rate,
            hr_rate=features.rolling_hr_rate,
            rbi_rate=features.rolling_rbi_rate,
            r_rate=features.rolling_r_rate,
            bb_rate=features.rolling_bb_rate,  # May be None if BB model not trained
        )

        results.append(result)

    return results


async def predict_pitcher(
    conn: asyncpg.Connection,
    game_id: str,
    game_date: date,
    pitcher_id: int,
    opp_lineup_ops: float,
    opp_game_mu: float,
    model_version: str,
) -> PitcherPropParams:
    """Run inference for a starting pitcher.

    Args:
        conn: Database connection
        game_id: Game identifier
        game_date: Date of the game
        pitcher_id: Pitcher player_id
        opp_lineup_ops: Opposing lineup aggregate OPS
        opp_game_mu: Opposing team expected runs
        model_version: Model version to load

    Returns:
        PitcherPropParams
    """
    # Load model
    outs_model = load_model(f"outs_dist_{model_version}")

    # Build features
    features = await build_pitcher_features(
        conn, game_id, game_date, pitcher_id, opp_lineup_ops, opp_game_mu
    )

    # Outs distribution prediction
    outs_X = np.array([
        [
            features.days_rest,
            features.rolling_pitch_count,
            features.rolling_ip_outs,
            features.rolling_k_rate,
            features.rolling_er_rate,
        ]
    ])
    outs_dist_proba = outs_model.predict_proba(outs_X)[0]
    outs_dist = [float(p) for p in outs_dist_proba]

    # Normalize to sum to 1.0
    outs_dist_sum = sum(outs_dist)
    if outs_dist_sum > 0:
        outs_dist = [p / outs_dist_sum for p in outs_dist]
    else:
        outs_dist = [0.0] * 9 + [1.0]  # Default to 27+ outs (unlikely)

    # Event rates: use shrunk rolling means (D-029)
    result = PitcherPropParams(
        player_id=pitcher_id,
        game_id=game_id,
        outs_dist=outs_dist,
        k_rate=features.rolling_k_rate,
        er_rate=features.rolling_er_rate,
    )

    return result


# Helper functions for training (duplicated from player_features.py for training context)


async def _get_hitter_start_stats_for_training(
    conn: asyncpg.Connection, player_id: int, game_date: date
) -> dict:
    """Get hitter start stats for training (same as player_features.py)."""
    from datetime import timedelta

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
            "days_rest": 7,
            "starts_last_7": 0,
            "starts_last_14": 0,
        }

    last_game_date = recent_games[0]["game_date"]
    days_rest = (game_date - last_game_date).days

    cutoff_7 = game_date - timedelta(days=7)
    cutoff_14 = game_date - timedelta(days=14)

    starts_last_7 = sum(1 for g in recent_games if g["game_date"] >= cutoff_7)
    starts_last_14 = sum(1 for g in recent_games if g["game_date"] >= cutoff_14)

    return {
        "days_rest": days_rest,
        "starts_last_7": starts_last_7,
        "starts_last_14": starts_last_14,
    }


async def _get_pitcher_rest_stats_for_training(
    conn: asyncpg.Connection, pitcher_id: int, game_date: date
) -> dict:
    """Get pitcher rest stats for training."""
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
        return {"days_rest": 4}

    days_rest = (game_date - last_start["game_date"]).days

    return {"days_rest": days_rest}


async def _get_pitcher_rolling_stats_for_training(
    conn: asyncpg.Connection, pitcher_id: int, game_date: date
) -> dict:
    """Get pitcher rolling stats for training."""
    from datetime import timedelta
    from mlb.config.settings import get_config

    config = get_config()
    window_start = game_date - timedelta(days=config.rolling_window_pitching_days)
    k_pitcher = config.shrinkage_k_pitcher

    league_pitch_count = 90.0
    league_ip_outs = 18.0
    league_k_rate = 0.220
    league_er_rate = 0.150

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
        return {
            "pitch_count": league_pitch_count,
            "ip_outs": league_ip_outs,
            "k_rate": league_k_rate,
            "er_rate": league_er_rate,
        }

    total_outs = stats["total_outs"] or 0
    total_ip = total_outs / 3.0

    shrinkage_weight = total_ip / (total_ip + k_pitcher)

    raw_pitch_count = (stats["total_pitches"] or 0) / stats["starts"]
    pitch_count = shrinkage_weight * raw_pitch_count + \
                  (1 - shrinkage_weight) * league_pitch_count

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
