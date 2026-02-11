"""Feature engineering for team run-scoring models.

This module builds game-level features from ingested data. It is the single source
of feature vectors consumed by both home and away team models (Unit 4).
"""

import asyncpg
from dataclasses import dataclass
from datetime import date, timedelta

from mlb.config.settings import get_config
from mlb.db.models import Table


@dataclass
class GameFeatures:
    """Game-level feature vector for team run-scoring models.

    Shared by home and away models via game-level covariates.
    Weather fields are None when park is indoor/retractable (D-018).
    """

    game_id: str
    game_date: date
    park_factor: float  # 1.000 = neutral, from parks table
    is_outdoor: bool
    temp_f: int | None  # None for dome/retractable
    wind_speed_mph: int | None
    wind_dir: str | None
    precip_pct: int | None
    home_starter_id: int
    away_starter_id: int
    home_starter_rest: int  # days since last start
    away_starter_rest: int
    home_starter_pitch_ct_avg: float  # rolling avg pitch count
    away_starter_pitch_ct_avg: float
    home_starter_era_recent: float  # rolling ERA (shrunk toward league mean)
    away_starter_era_recent: float
    home_lineup_ops: float  # rolling aggregate OPS for confirmed lineup
    away_lineup_ops: float
    home_bullpen_usage: float  # innings used in last N days (fatigue proxy)
    away_bullpen_usage: float
    home_run_env: float  # team rolling R/G
    away_run_env: float

    @staticmethod
    def feature_names(is_home: bool) -> list[str]:
        """Return ordered list of feature names for model input (D-026).

        Args:
            is_home: If True, return home team feature names; otherwise away team

        Returns:
            List of feature names in the order they appear in the feature vector
        """
        if is_home:
            return [
                "park_factor",
                "is_outdoor",
                "temp_f",
                "wind_speed_mph",
                "wind_dir",
                "precip_pct",
                "home_starter_rest",
                "home_starter_pitch_ct_avg",
                "home_starter_era_recent",
                "away_starter_era_recent",
                "home_lineup_ops",
                "home_bullpen_usage",
                "home_run_env",
                "away_run_env",
            ]
        else:
            return [
                "park_factor",
                "is_outdoor",
                "temp_f",
                "wind_speed_mph",
                "wind_dir",
                "precip_pct",
                "away_starter_rest",
                "away_starter_pitch_ct_avg",
                "away_starter_era_recent",
                "home_starter_era_recent",
                "away_lineup_ops",
                "away_bullpen_usage",
                "away_run_env",
                "home_run_env",
            ]


async def build_game_features(conn: asyncpg.Connection, game_id: str) -> GameFeatures:
    """Build feature vector for a given game_id.

    Args:
        conn: Database connection
        game_id: Game identifier

    Returns:
        GameFeatures dataclass with all fields populated

    Raises:
        ValueError: If game not found or missing required data
    """
    config = get_config()

    # Fetch game metadata and park info
    game_row = await conn.fetchrow(
        f"""
        SELECT g.game_date, g.home_team_id, g.away_team_id, g.park_id,
               p.park_factor, p.is_outdoor
        FROM {Table.GAMES} g
        JOIN {Table.PARKS} p ON g.park_id = p.park_id
        WHERE g.game_id = $1
        """,
        game_id,
    )
    if not game_row:
        raise ValueError(f"Game {game_id} not found")

    game_date = game_row["game_date"]
    home_team_id = game_row["home_team_id"]
    away_team_id = game_row["away_team_id"]
    park_factor = float(game_row["park_factor"])
    is_outdoor = game_row["is_outdoor"]

    # Fetch weather (most recent before game_date)
    weather = await _get_weather(conn, game_id, is_outdoor)

    # Fetch starters from lineups
    starters = await _get_starters(conn, game_id, home_team_id, away_team_id)

    # Calculate pitcher features (rest, pitch count, ERA)
    home_starter_features = await _get_pitcher_features(
        conn, starters["home_starter_id"], game_date
    )
    away_starter_features = await _get_pitcher_features(
        conn, starters["away_starter_id"], game_date
    )

    # Calculate lineup features
    home_lineup_ops = await _get_lineup_ops(conn, game_id, home_team_id, game_date)
    away_lineup_ops = await _get_lineup_ops(conn, game_id, away_team_id, game_date)

    # Calculate bullpen usage
    home_bullpen_usage = await _get_bullpen_usage(conn, home_team_id, game_date)
    away_bullpen_usage = await _get_bullpen_usage(conn, away_team_id, game_date)

    # Calculate team run environment
    home_run_env = await _get_team_run_env(conn, home_team_id, game_date)
    away_run_env = await _get_team_run_env(conn, away_team_id, game_date)

    return GameFeatures(
        game_id=game_id,
        game_date=game_date,
        park_factor=park_factor,
        is_outdoor=is_outdoor,
        temp_f=weather["temp_f"],
        wind_speed_mph=weather["wind_speed_mph"],
        wind_dir=weather["wind_dir"],
        precip_pct=weather["precip_pct"],
        home_starter_id=starters["home_starter_id"],
        away_starter_id=starters["away_starter_id"],
        home_starter_rest=home_starter_features["rest"],
        away_starter_rest=away_starter_features["rest"],
        home_starter_pitch_ct_avg=home_starter_features["pitch_ct_avg"],
        away_starter_pitch_ct_avg=away_starter_features["pitch_ct_avg"],
        home_starter_era_recent=home_starter_features["era_recent"],
        away_starter_era_recent=away_starter_features["era_recent"],
        home_lineup_ops=home_lineup_ops,
        away_lineup_ops=away_lineup_ops,
        home_bullpen_usage=home_bullpen_usage,
        away_bullpen_usage=away_bullpen_usage,
        home_run_env=home_run_env,
        away_run_env=away_run_env,
    )


async def _get_weather(
    conn: asyncpg.Connection, game_id: str, is_outdoor: bool
) -> dict:
    """Get weather data for a game. Returns None fields for indoor/retractable parks."""
    if not is_outdoor:
        return {
            "temp_f": None,
            "wind_speed_mph": None,
            "wind_dir": None,
            "precip_pct": None,
        }

    # Get most recent weather snapshot for this game (D-015)
    weather_row = await conn.fetchrow(
        f"""
        SELECT temp_f, wind_speed_mph, wind_dir, precip_pct
        FROM {Table.WEATHER}
        WHERE game_id = $1
        ORDER BY fetched_at DESC
        LIMIT 1
        """,
        game_id,
    )

    if not weather_row:
        # Conservative fallback for outdoor parks (D-023)
        return {
            "temp_f": 72,
            "wind_speed_mph": 5,
            "wind_dir": "N",
            "precip_pct": 0,
        }

    return {
        "temp_f": weather_row["temp_f"],
        "wind_speed_mph": weather_row["wind_speed_mph"],
        "wind_dir": weather_row["wind_dir"],
        "precip_pct": weather_row["precip_pct"],
    }


async def _get_starters(
    conn: asyncpg.Connection, game_id: str, home_team_id: int, away_team_id: int
) -> dict:
    """Get starting pitcher IDs from confirmed lineups."""
    # Home starter (batting_order = 0 indicates pitcher in NL parks, otherwise lookup)
    home_starter = await conn.fetchrow(
        f"""
        SELECT player_id
        FROM {Table.LINEUPS}
        WHERE game_id = $1 AND team_id = $2 AND is_confirmed = TRUE
        ORDER BY source_ts DESC
        LIMIT 1
        """,
        game_id,
        home_team_id,
    )

    away_starter = await conn.fetchrow(
        f"""
        SELECT player_id
        FROM {Table.LINEUPS}
        WHERE game_id = $1 AND team_id = $2 AND is_confirmed = TRUE
        ORDER BY source_ts DESC
        LIMIT 1
        """,
        game_id,
        away_team_id,
    )

    if not home_starter or not away_starter:
        raise ValueError(
            f"Confirmed lineup not found for game {game_id}. "
            "Cannot build features without starters."
        )

    return {
        "home_starter_id": home_starter["player_id"],
        "away_starter_id": away_starter["player_id"],
    }


async def _get_pitcher_features(
    conn: asyncpg.Connection, pitcher_id: int, game_date: date
) -> dict:
    """Calculate pitcher features: rest days, pitch count avg, ERA (shrunk)."""
    config = get_config()

    # Get pitcher's recent starts (30-day window per D-022)
    window_start = game_date - timedelta(days=30)
    recent_starts = await conn.fetch(
        f"""
        SELECT g.game_date, pgl.ip_outs, pgl.er, pgl.pitch_count
        FROM {Table.PLAYER_GAME_LOGS} pgl
        JOIN {Table.GAMES} g ON pgl.game_id = g.game_id
        WHERE pgl.player_id = $1
          AND g.game_date >= $2
          AND g.game_date < $3
          AND pgl.is_starter = TRUE
        ORDER BY g.game_date DESC
        """,
        pitcher_id,
        window_start,
        game_date,
    )

    if not recent_starts:
        # Insufficient history: fall back to league-average starter profile
        return {
            "rest": 4,  # typical starter rest
            "pitch_ct_avg": 90.0,  # league average
            "era_recent": 4.50,  # league average ERA
        }

    # Calculate rest (days since last start)
    last_start_date = recent_starts[0]["game_date"]
    rest = (game_date - last_start_date).days

    # Calculate pitch count average
    pitch_counts = [s["pitch_count"] for s in recent_starts if s["pitch_count"]]
    pitch_ct_avg = sum(pitch_counts) / len(pitch_counts) if pitch_counts else 90.0

    # Calculate ERA with shrinkage (D-021)
    total_ip_outs = sum(s["ip_outs"] or 0 for s in recent_starts)
    total_er = sum(s["er"] or 0 for s in recent_starts)

    ip = total_ip_outs / 3.0  # Convert outs to innings
    k_pitcher = 80  # Shrinkage constant for pitchers (D-021)
    league_era = 4.50

    if ip > 0:
        raw_era = (total_er / ip) * 9.0
        shrinkage_weight = ip / (ip + k_pitcher)
        era_recent = shrinkage_weight * raw_era + (1 - shrinkage_weight) * league_era
    else:
        era_recent = league_era

    return {
        "rest": rest,
        "pitch_ct_avg": pitch_ct_avg,
        "era_recent": era_recent,
    }


async def _get_lineup_ops(
    conn: asyncpg.Connection, game_id: str, team_id: int, game_date: date
) -> float:
    """Calculate rolling OPS for confirmed lineup (60-day window per D-022)."""
    config = get_config()

    # Get confirmed lineup player IDs (D-011: most recent confirmed)
    lineup_players = await conn.fetch(
        f"""
        SELECT DISTINCT player_id
        FROM {Table.LINEUPS}
        WHERE game_id = $1 AND team_id = $2 AND is_confirmed = TRUE
        """,
        game_id,
        team_id,
    )

    if not lineup_players:
        # No confirmed lineup: use league average
        return 0.750

    player_ids = [row["player_id"] for row in lineup_players]

    # Get batting stats for lineup players (60-day window)
    window_start = game_date - timedelta(days=60)
    batting_stats = await conn.fetch(
        f"""
        SELECT pgl.player_id, SUM(pgl.h) as h, SUM(pgl.ab) as ab,
               SUM(pgl.bb) as bb, SUM(pgl.tb) as tb, SUM(pgl.pa) as pa
        FROM {Table.PLAYER_GAME_LOGS} pgl
        JOIN {Table.GAMES} g ON pgl.game_id = g.game_id
        WHERE pgl.player_id = ANY($1)
          AND g.game_date >= $2
          AND g.game_date < $3
        GROUP BY pgl.player_id
        """,
        player_ids,
        window_start,
        game_date,
    )

    # Calculate shrunk OPS for each player
    k_batter = 200  # Shrinkage constant for batters (D-021)
    league_ops = 0.750
    ops_values = []

    for stats in batting_stats:
        pa = stats["pa"] or 0
        ab = stats["ab"] or 0
        h = stats["h"] or 0
        bb = stats["bb"] or 0
        tb = stats["tb"] or 0

        if ab > 0 and pa > 0:
            obp = (h + bb) / pa
            slg = tb / ab
            raw_ops = obp + slg
            shrinkage_weight = pa / (pa + k_batter)
            shrunk_ops = shrinkage_weight * raw_ops + (1 - shrinkage_weight) * league_ops
            ops_values.append(shrunk_ops)

    if ops_values:
        return sum(ops_values) / len(ops_values)
    else:
        return league_ops


async def _get_bullpen_usage(
    conn: asyncpg.Connection, team_id: int, game_date: date
) -> float:
    """Calculate bullpen innings used in last 7 days (fatigue proxy)."""
    window_start = game_date - timedelta(days=7)

    # Get total relief innings (non-starters) for the team
    bullpen_ip = await conn.fetchval(
        f"""
        SELECT COALESCE(SUM(pgl.ip_outs), 0) / 3.0 as innings
        FROM {Table.PLAYER_GAME_LOGS} pgl
        JOIN {Table.GAMES} g ON pgl.game_id = g.game_id
        JOIN {Table.LINEUPS} l ON pgl.player_id = l.player_id AND pgl.game_id = l.game_id
        WHERE l.team_id = $1
          AND g.game_date >= $2
          AND g.game_date < $3
          AND pgl.is_starter = FALSE
          AND pgl.ip_outs IS NOT NULL
        """,
        team_id,
        window_start,
        game_date,
    )

    return float(bullpen_ip or 0.0)


async def _get_team_run_env(
    conn: asyncpg.Connection, team_id: int, game_date: date
) -> float:
    """Calculate team rolling runs per game (30-day window)."""
    window_start = game_date - timedelta(days=30)

    # Get runs scored by this team in recent games
    runs = await conn.fetch(
        f"""
        SELECT
            CASE
                WHEN g.home_team_id = $1 THEN g.home_score
                WHEN g.away_team_id = $1 THEN g.away_score
            END as runs
        FROM {Table.GAMES} g
        WHERE (g.home_team_id = $1 OR g.away_team_id = $1)
          AND g.game_date >= $2
          AND g.game_date < $3
          AND g.status = 'final'
        """,
        team_id,
        window_start,
        game_date,
    )

    if not runs:
        return 4.5  # League average runs per game

    run_values = [r["runs"] for r in runs if r["runs"] is not None]
    if run_values:
        return sum(run_values) / len(run_values)
    else:
        return 4.5
