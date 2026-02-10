"""Provider-agnostic data ingestion interfaces and adapters."""

from mlb.ingestion.base import (
    GameLogRow,
    GameProvider,
    GameRow,
    LineupProvider,
    LineupRow,
    OddsProvider,
    OddsRow,
    StatsProvider,
    WeatherProvider,
    WeatherRow,
)
from mlb.ingestion.cache import CacheEntry, get_cache
from mlb.ingestion.games import V1GameProvider
from mlb.ingestion.lineups import V1LineupProvider
from mlb.ingestion.odds import V1OddsProvider
from mlb.ingestion.stats import V1StatsProvider
from mlb.ingestion.weather import V1WeatherProvider

__all__ = [
    # ABCs
    "OddsProvider",
    "LineupProvider",
    "StatsProvider",
    "GameProvider",
    "WeatherProvider",
    # Row schemas
    "OddsRow",
    "LineupRow",
    "GameLogRow",
    "GameRow",
    "WeatherRow",
    # Cache
    "CacheEntry",
    "get_cache",
    # Concrete implementations
    "V1OddsProvider",
    "V1LineupProvider",
    "V1StatsProvider",
    "V1GameProvider",
    "V1WeatherProvider",
]
