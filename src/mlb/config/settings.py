"""Application configuration schema and validation."""

from typing import Literal

from pydantic import Field, PostgresDsn, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    env: Literal["dev", "staging", "prod"] = Field(
        ...,
        description="Application environment",
    )
    db_dsn: PostgresDsn = Field(
        ...,
        description="PostgreSQL database connection string",
    )
    db_pool_min: int = Field(
        default=2,
        ge=1,
        description="Minimum database connection pool size",
    )
    db_pool_max: int = Field(
        default=10,
        ge=1,
        description="Maximum database connection pool size",
    )
    api_key_odds: SecretStr = Field(
        default=SecretStr(""),
        description="API key for odds provider (placeholder)",
    )
    api_key_weather: SecretStr = Field(
        default=SecretStr(""),
        description="API key for weather provider (placeholder)",
    )
    discord_token: SecretStr = Field(
        default=SecretStr(""),
        description="Discord bot token (placeholder)",
    )
    stripe_secret: SecretStr = Field(
        default=SecretStr(""),
        description="Stripe secret key (placeholder)",
    )
    default_sim_n: int = Field(
        default=5000,
        ge=2000,
        le=10000,
        description="Default number of Monte Carlo simulations",
    )
    shrinkage_k_batter: int = Field(
        default=200,
        ge=50,
        le=500,
        description="Shrinkage constant for batters (PA)",
    )
    shrinkage_k_pitcher: int = Field(
        default=80,
        ge=20,
        le=200,
        description="Shrinkage constant for pitchers (IP)",
    )
    rolling_window_batting_days: int = Field(
        default=60,
        ge=14,
        le=120,
        description="Rolling window for batting stats (days)",
    )
    rolling_window_pitching_days: int = Field(
        default=30,
        ge=7,
        le=60,
        description="Rolling window for pitching stats (days)",
    )
    bf_per_out_ratio: float = Field(
        default=1.35,
        ge=1.0,
        le=2.0,
        description="League-average batters faced per out (for K/BF approximation)",
    )
    min_edge_threshold: float = Field(
        default=0.02,
        ge=0.0,
        le=0.10,
        description="Minimum edge threshold (2%) for kelly_fraction > 0 (D-037)",
    )
    kelly_fraction_multiplier: float = Field(
        default=0.25,
        ge=0.05,
        le=1.0,
        description="Fractional Kelly multiplier (0.25 = quarter-Kelly) (D-038)",
    )
    schedule_night_before_et: str = Field(
        default="22:00",
        description="ET time for night-before global run (D-043)",
    )
    schedule_morning_et: str = Field(
        default="08:00",
        description="ET time for morning global run (D-043)",
    )
    schedule_midday_et: str = Field(
        default="12:00",
        description="ET time for midday global run (D-043)",
    )
    game_run_t_minus_minutes: list[int] = Field(
        default=[90, 30],
        description="Minutes before first_pitch for per-game runs (D-044)",
    )
    rerun_throttle_minutes: int = Field(
        default=10,
        ge=1,
        le=60,
        description="Minimum gap between reruns per game (D-045)",
    )
    p_start_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Publishing gate threshold for p_start (D-046)",
    )
    max_retry_attempts: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Maximum ingestion retry attempts",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level",
    )
    discord_guild_id: str = Field(
        default="",
        description="Discord guild (server) ID for bot operations",
    )
    free_pick_channel: str = Field(
        default="free-picks",
        description="Channel name for daily free picks",
    )
    paid_channels: list[str] = Field(
        default=["team-moneyline", "team-runline", "team-totals", "player-props-h", "player-props-p"],
        description="Channel names for paid-tier picks",
    )
    announcements_channel: str = Field(
        default="announcements",
        description="Channel name for bot announcements",
    )
    free_pick_window_min: int = Field(
        default=60,
        ge=30,
        le=120,
        description="Earliest time before first_pitch to post free pick (minutes)",
    )
    free_pick_window_max: int = Field(
        default=90,
        ge=30,
        le=120,
        description="Latest time before first_pitch to post free pick (minutes)",
    )

    @field_validator("db_pool_max")
    @classmethod
    def validate_pool_max(cls, v: int, info) -> int:
        """Ensure pool_max >= pool_min."""
        if "db_pool_min" in info.data and v < info.data["db_pool_min"]:
            raise ValueError("db_pool_max must be >= db_pool_min")
        return v


_config: AppConfig | None = None


def get_config() -> AppConfig:
    """Get or create the singleton AppConfig instance."""
    global _config
    if _config is None:
        _config = AppConfig()
    return _config
