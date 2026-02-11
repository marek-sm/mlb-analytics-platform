"""Abstract base classes and canonical row schemas for data ingestion."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime


# Canonical row schemas
@dataclass
class OddsRow:
    """Canonical odds snapshot row."""

    game_id: str
    book: str  # e.g. 'draftkings', 'fanduel'
    market: str  # 'ml' | 'rl' | 'total' | 'team_total'
    side: str | None  # 'home' | 'away' | 'over' | 'under'
    line: float | None  # e.g. -1.5, 8.5
    price: float  # European decimal, ≥ 1.0 (converted on ingestion)
    snapshot_ts: datetime  # UTC


@dataclass
class LineupRow:
    """Canonical lineup row."""

    game_id: str
    team_id: int
    player_id: int
    batting_order: int  # 1–9
    is_confirmed: bool
    source_ts: datetime  # UTC


@dataclass
class GameLogRow:
    """Canonical game log row."""

    player_id: int
    game_id: str
    pa: int | None = None
    ab: int | None = None
    h: int | None = None
    tb: int | None = None
    hr: int | None = None
    rbi: int | None = None
    r: int | None = None
    bb: int | None = None
    k: int | None = None
    ip_outs: int | None = None  # outs recorded = IP × 3
    er: int | None = None
    pitch_count: int | None = None
    is_starter: bool | None = None


@dataclass
class GameRow:
    """Canonical game row."""

    game_id: str
    game_date: date
    home_team_id: int
    away_team_id: int
    park_id: int
    first_pitch: datetime | None = None  # UTC
    status: str = "scheduled"  # 'scheduled' | 'confirmed' | 'final' | 'postponed'
    home_score: int | None = None  # Populated when status='final'
    away_score: int | None = None  # Populated when status='final'


@dataclass
class WeatherRow:
    """Canonical weather row."""

    game_id: str
    temp_f: int
    wind_speed_mph: int
    wind_dir: str
    precip_pct: int
    fetched_at: datetime  # UTC


# Abstract base classes
class OddsProvider(ABC):
    """Abstract odds provider interface."""

    @abstractmethod
    async def fetch_odds(self, game_id: str) -> list[OddsRow]:
        """
        Fetch odds snapshots for a game.

        Args:
            game_id: Game identifier

        Returns:
            List of OddsRow objects with price in European decimal (≥ 1.0)
        """
        pass


class LineupProvider(ABC):
    """Abstract lineup provider interface."""

    @abstractmethod
    async def fetch_lineup(self, game_id: str, team_id: int) -> list[LineupRow]:
        """
        Fetch lineup for a team in a game.

        Args:
            game_id: Game identifier
            team_id: Team identifier

        Returns:
            List of LineupRow objects
        """
        pass

    @abstractmethod
    async def is_confirmed(self, game_id: str, team_id: int) -> bool:
        """
        Check if lineup is confirmed by team.

        Args:
            game_id: Game identifier
            team_id: Team identifier

        Returns:
            True if lineup is officially confirmed
        """
        pass


class StatsProvider(ABC):
    """Abstract stats provider interface."""

    @abstractmethod
    async def fetch_game_logs(self, game_date: date) -> list[GameLogRow]:
        """
        Fetch game logs for all players on a date.

        Args:
            game_date: Date to fetch logs for

        Returns:
            List of GameLogRow objects
        """
        pass


class GameProvider(ABC):
    """Abstract game/schedule provider interface."""

    @abstractmethod
    async def fetch_schedule(self, game_date: date) -> list[GameRow]:
        """
        Fetch game schedule for a date.

        Args:
            game_date: Date to fetch schedule for

        Returns:
            List of GameRow objects
        """
        pass


class WeatherProvider(ABC):
    """Abstract weather provider interface."""

    @abstractmethod
    async def fetch_weather(self, game_id: str, park_id: int) -> WeatherRow | None:
        """
        Fetch weather for a game at a park.

        Returns None for indoor or retractable-roof parks.

        Args:
            game_id: Game identifier
            park_id: Park identifier

        Returns:
            WeatherRow if outdoor park, None otherwise
        """
        pass


# Shared helper functions
async def ensure_player_exists(
    conn,
    player_id: int,
    name: str | None = None,
    team_id: int | None = None,
) -> None:
    """
    Ensure player exists in players table (per D-020).

    Upserts player with available metadata. Missing fields are NULL.

    Args:
        conn: Database connection
        player_id: Player identifier
        name: Player name (optional)
        team_id: Team identifier (optional)
    """
    from mlb.db.models import Table

    upsert_sql = f"""
        INSERT INTO {Table.PLAYERS}
        (player_id, name, team_id, position, bats, throws)
        VALUES ($1, $2, $3, NULL, NULL, NULL)
        ON CONFLICT (player_id)
        DO UPDATE SET
            name = COALESCE(EXCLUDED.name, {Table.PLAYERS}.name),
            team_id = COALESCE(EXCLUDED.team_id, {Table.PLAYERS}.team_id)
    """

    await conn.execute(
        upsert_sql,
        player_id,
        name or f"Player {player_id}",  # Default name if not provided
        team_id,
    )
