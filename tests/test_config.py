"""Unit tests for configuration loading and validation."""

import os
from typing import Any, Dict

import pytest
from pydantic import ValidationError

from mlb.config.settings import AppConfig


@pytest.fixture
def base_env() -> Dict[str, str]:
    """Provide minimal valid environment variables."""
    return {
        "ENV": "dev",
        "DB_DSN": "postgresql://user:pass@localhost:5432/testdb",
    }


@pytest.fixture
def clean_env(monkeypatch) -> None:
    """Clean environment variables before each test."""
    env_vars = [
        "ENV",
        "DB_DSN",
        "DB_POOL_MIN",
        "DB_POOL_MAX",
        "API_KEY_ODDS",
        "API_KEY_WEATHER",
        "DISCORD_TOKEN",
        "STRIPE_SECRET",
        "DEFAULT_SIM_N",
        "LOG_LEVEL",
    ]
    for var in env_vars:
        monkeypatch.delenv(var, raising=False)


def test_valid_config_minimal(monkeypatch, clean_env, base_env):
    """Test valid configuration with minimal required fields."""
    for key, value in base_env.items():
        monkeypatch.setenv(key, value)

    config = AppConfig()

    assert config.env == "dev"
    assert str(config.db_dsn) == "postgresql://user:pass@localhost:5432/testdb"
    assert config.db_pool_min == 2
    assert config.db_pool_max == 10
    assert config.default_sim_n == 5000
    assert config.log_level == "INFO"


def test_valid_config_all_fields(monkeypatch, clean_env, base_env):
    """Test valid configuration with all fields specified."""
    env = {
        **base_env,
        "DB_POOL_MIN": "5",
        "DB_POOL_MAX": "20",
        "API_KEY_ODDS": "test-odds-key",
        "API_KEY_WEATHER": "test-weather-key",
        "DISCORD_TOKEN": "test-discord-token",
        "STRIPE_SECRET": "test-stripe-secret",
        "DEFAULT_SIM_N": "7500",
        "LOG_LEVEL": "DEBUG",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    config = AppConfig()

    assert config.env == "dev"
    assert config.db_pool_min == 5
    assert config.db_pool_max == 20
    assert config.api_key_odds.get_secret_value() == "test-odds-key"
    assert config.api_key_weather.get_secret_value() == "test-weather-key"
    assert config.discord_token.get_secret_value() == "test-discord-token"
    assert config.stripe_secret.get_secret_value() == "test-stripe-secret"
    assert config.default_sim_n == 7500
    assert config.log_level == "DEBUG"


def test_missing_db_dsn(monkeypatch, clean_env):
    """Test that missing DB_DSN raises validation error."""
    monkeypatch.setenv("ENV", "dev")

    with pytest.raises(ValidationError) as exc_info:
        AppConfig()

    errors = exc_info.value.errors()
    assert any(error["loc"] == ("db_dsn",) for error in errors)


def test_missing_env(monkeypatch, clean_env):
    """Test that missing ENV raises validation error."""
    monkeypatch.setenv("DB_DSN", "postgresql://user:pass@localhost:5432/testdb")

    with pytest.raises(ValidationError) as exc_info:
        AppConfig()

    errors = exc_info.value.errors()
    assert any(error["loc"] == ("env",) for error in errors)


def test_invalid_env_value(monkeypatch, clean_env, base_env):
    """Test that invalid ENV value raises validation error."""
    base_env["ENV"] = "production"  # Not in allowed set
    for key, value in base_env.items():
        monkeypatch.setenv(key, value)

    with pytest.raises(ValidationError) as exc_info:
        AppConfig()

    errors = exc_info.value.errors()
    assert any(
        error["loc"] == ("env",) and "literal_error" in error["type"]
        for error in errors
    )


def test_valid_env_values(monkeypatch, clean_env, base_env):
    """Test all valid ENV values are accepted."""
    for env_value in ["dev", "staging", "prod"]:
        base_env["ENV"] = env_value
        for key, value in base_env.items():
            monkeypatch.setenv(key, value)

        config = AppConfig()
        assert config.env == env_value


def test_pool_max_less_than_min(monkeypatch, clean_env, base_env):
    """Test that pool_max < pool_min raises validation error."""
    base_env["DB_POOL_MIN"] = "10"
    base_env["DB_POOL_MAX"] = "5"
    for key, value in base_env.items():
        monkeypatch.setenv(key, value)

    with pytest.raises(ValidationError) as exc_info:
        AppConfig()

    errors = exc_info.value.errors()
    assert any(
        error["loc"] == ("db_pool_max",) and "db_pool_min" in str(error["ctx"])
        for error in errors
    )


def test_pool_max_equal_to_min(monkeypatch, clean_env, base_env):
    """Test that pool_max == pool_min is valid."""
    base_env["DB_POOL_MIN"] = "5"
    base_env["DB_POOL_MAX"] = "5"
    for key, value in base_env.items():
        monkeypatch.setenv(key, value)

    config = AppConfig()
    assert config.db_pool_min == 5
    assert config.db_pool_max == 5


def test_sim_count_below_minimum(monkeypatch, clean_env, base_env):
    """Test that default_sim_n < 2000 raises validation error."""
    base_env["DEFAULT_SIM_N"] = "1999"
    for key, value in base_env.items():
        monkeypatch.setenv(key, value)

    with pytest.raises(ValidationError) as exc_info:
        AppConfig()

    errors = exc_info.value.errors()
    assert any(
        error["loc"] == ("default_sim_n",) and "greater_than_equal" in error["type"]
        for error in errors
    )


def test_sim_count_above_maximum(monkeypatch, clean_env, base_env):
    """Test that default_sim_n > 10000 raises validation error."""
    base_env["DEFAULT_SIM_N"] = "10001"
    for key, value in base_env.items():
        monkeypatch.setenv(key, value)

    with pytest.raises(ValidationError) as exc_info:
        AppConfig()

    errors = exc_info.value.errors()
    assert any(
        error["loc"] == ("default_sim_n",) and "less_than_equal" in error["type"]
        for error in errors
    )


def test_sim_count_at_boundaries(monkeypatch, clean_env, base_env):
    """Test that default_sim_n at boundaries [2000, 10000] is valid."""
    for boundary in ["2000", "10000"]:
        base_env["DEFAULT_SIM_N"] = boundary
        for key, value in base_env.items():
            monkeypatch.setenv(key, value)

        config = AppConfig()
        assert config.default_sim_n == int(boundary)


def test_empty_secret_strings_allowed(monkeypatch, clean_env, base_env):
    """Test that empty secret strings are allowed."""
    base_env["API_KEY_ODDS"] = ""
    base_env["API_KEY_WEATHER"] = ""
    base_env["DISCORD_TOKEN"] = ""
    base_env["STRIPE_SECRET"] = ""
    for key, value in base_env.items():
        monkeypatch.setenv(key, value)

    config = AppConfig()
    assert config.api_key_odds.get_secret_value() == ""
    assert config.api_key_weather.get_secret_value() == ""
    assert config.discord_token.get_secret_value() == ""
    assert config.stripe_secret.get_secret_value() == ""


def test_extra_env_vars_ignored(monkeypatch, clean_env, base_env):
    """Test that extra/unknown environment variables are ignored."""
    base_env["UNKNOWN_VAR"] = "should-be-ignored"
    base_env["ANOTHER_UNKNOWN"] = "also-ignored"
    for key, value in base_env.items():
        monkeypatch.setenv(key, value)

    config = AppConfig()
    assert config.env == "dev"
    assert not hasattr(config, "unknown_var")
    assert not hasattr(config, "another_unknown")


def test_case_insensitive_env_vars(monkeypatch, clean_env):
    """Test that environment variable names are case-insensitive."""
    monkeypatch.setenv("env", "staging")  # lowercase
    monkeypatch.setenv("db_dsn", "postgresql://user:pass@localhost:5432/testdb")

    config = AppConfig()
    assert config.env == "staging"
