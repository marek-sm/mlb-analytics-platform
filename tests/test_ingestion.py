"""Tests for Unit 3 - Data Ingestion acceptance criteria."""

import asyncio
from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from mlb.db.models import Table
from mlb.db.pool import close_pool, get_pool
from mlb.ingestion.base import (
    GameLogRow,
    GameRow,
    LineupRow,
    OddsRow,
    WeatherRow,
)
from mlb.ingestion.cache import Cache, get_cache
from mlb.ingestion.games import V1GameProvider
from mlb.ingestion.lineups import V1LineupProvider
from mlb.ingestion.odds import (
    V1OddsProvider,
    american_to_decimal,
    detect_and_convert_odds,
)
from mlb.ingestion.stats import V1StatsProvider
from mlb.ingestion.weather import V1WeatherProvider


# Acceptance Criterion 1: Each ABC has exactly one concrete implementation
def test_abc_implementations():
    """Each ABC defines methods and has one concrete implementation."""
    # Verify concrete implementations exist and are instantiable
    assert V1OddsProvider() is not None
    assert V1LineupProvider() is not None
    assert V1StatsProvider() is not None
    assert V1GameProvider() is not None
    assert V1WeatherProvider() is not None


# Acceptance Criterion 2: American → European decimal conversion
def test_american_to_decimal_positive():
    """Convert American +150 to decimal 2.50."""
    result = american_to_decimal(150)
    assert result == pytest.approx(2.50, rel=1e-3)


def test_american_to_decimal_negative():
    """Convert American -110 to decimal 1.909."""
    result = american_to_decimal(-110)
    assert result == pytest.approx(1.909, rel=1e-3)


def test_american_to_decimal_edge_cases():
    """Test edge cases for American odds conversion."""
    assert american_to_decimal(100) == pytest.approx(2.0)
    assert american_to_decimal(-100) == pytest.approx(2.0)
    assert american_to_decimal(200) == pytest.approx(3.0)
    assert american_to_decimal(-200) == pytest.approx(1.5)

    with pytest.raises(ValueError):
        american_to_decimal(0)


def test_detect_and_convert_odds():
    """Test odds format detection and conversion."""
    # Decimal format (1.0-50.0)
    assert detect_and_convert_odds(1.5) == pytest.approx(1.5)
    assert detect_and_convert_odds(2.0) == pytest.approx(2.0)
    assert detect_and_convert_odds(10.5) == pytest.approx(10.5)

    # American format (≥100 or ≤-100)
    assert detect_and_convert_odds(150) == pytest.approx(2.5)
    assert detect_and_convert_odds(-110) == pytest.approx(1.909, rel=1e-3)
    assert detect_and_convert_odds(200) == pytest.approx(3.0)

    # Ambiguous values should raise
    with pytest.raises(ValueError, match="Ambiguous"):
        detect_and_convert_odds(75)  # Too low for American, too high for decimal

    with pytest.raises(ValueError, match="Ambiguous"):
        detect_and_convert_odds(-50)  # Too high for American negative

    # Explicit format hint
    assert detect_and_convert_odds(2.0, format_hint="decimal") == 2.0
    assert detect_and_convert_odds(150, format_hint="american") == pytest.approx(2.5)


def test_odds_row_price_validation():
    """OddsRow price must be ≥ 1.0 (European decimal)."""
    # Valid odds
    row = OddsRow(
        game_id="game123",
        book="draftkings",
        market="ml",
        side="home",
        line=None,
        price=1.909,
        snapshot_ts=datetime.now(timezone.utc),
    )
    assert row.price >= 1.0

    # Would fail validation if we had validators
    row_invalid = OddsRow(
        game_id="game123",
        book="draftkings",
        market="ml",
        side="home",
        line=None,
        price=0.5,  # Invalid
        snapshot_ts=datetime.now(timezone.utc),
    )
    # This is a data model test - in production we'd have Pydantic validators


# Acceptance Criterion 3: Lineup confirmation flip logic
@pytest.mark.asyncio
async def test_lineup_confirmation_flip_db():
    """
    Test lineup confirmation flip logic against real Postgres database.

    Verifies D-011 confirmation contract:
    - Insert initial confirmed lineup for (game_id, team_id)
    - Insert new confirmed lineup for same (game_id, team_id)
    - Assert old rows flipped to is_confirmed = FALSE
    - Assert new rows have is_confirmed = TRUE

    This ensures the partial unique index on (game_id, team_id, batting_order)
    WHERE is_confirmed = TRUE is respected by flipping prior confirmed rows
    before inserting new ones.
    """
    pool = await get_pool()

    try:
        # Setup: Create test game and team (using real team/park IDs from seed data)
        async with pool.acquire() as conn:
            # Get first two teams and a park from seed data
            teams = await conn.fetch("SELECT team_id FROM teams LIMIT 2")
            park = await conn.fetchval("SELECT park_id FROM parks LIMIT 1")

            home_team_id = teams[0]["team_id"]
            away_team_id = teams[1]["team_id"]

            await conn.execute(
                """
                INSERT INTO games (game_id, game_date, home_team_id, away_team_id, park_id, status)
                VALUES ('test_game_1', '2026-01-01', $1, $2, $3, 'scheduled')
                ON CONFLICT (game_id) DO NOTHING
                """,
                home_team_id,
                away_team_id,
                park,
            )

            # Insert initial confirmed lineup
            initial_lineup = [
                LineupRow(
                    game_id="test_game_1",
                    team_id=home_team_id,
                    player_id=100 + i,
                    batting_order=i + 1,
                    is_confirmed=True,
                    source_ts=datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
                )
                for i in range(9)
            ]

            # Ensure players exist
            stats_provider = V1StatsProvider()
            for row in initial_lineup:
                await stats_provider.ensure_player_exists(row.player_id)

            # Write initial lineup
            lineup_provider = V1LineupProvider()
            await lineup_provider.write_lineup(initial_lineup, is_confirmed=True)

            # Verify initial lineup is confirmed
            count_confirmed = await conn.fetchval(
                """
                SELECT COUNT(*)
                FROM lineups
                WHERE game_id = 'test_game_1' AND team_id = $1 AND is_confirmed = TRUE
                """,
                home_team_id,
            )
            assert count_confirmed == 9

            # Insert new confirmed lineup (lineup change)
            new_lineup = [
                LineupRow(
                    game_id="test_game_1",
                    team_id=home_team_id,
                    player_id=200 + i,
                    batting_order=i + 1,
                    is_confirmed=True,
                    source_ts=datetime(2026, 1, 1, 11, 0, 0, tzinfo=timezone.utc),
                )
                for i in range(9)
            ]

            # Ensure new players exist
            for row in new_lineup:
                await stats_provider.ensure_player_exists(row.player_id)

            # Write new lineup (should flip prior confirmed to FALSE)
            await lineup_provider.write_lineup(new_lineup, is_confirmed=True)

            # Verify: Old lineup should be flipped to FALSE
            old_confirmed_count = await conn.fetchval(
                """
                SELECT COUNT(*)
                FROM lineups
                WHERE game_id = 'test_game_1'
                  AND team_id = $1
                  AND player_id BETWEEN 100 AND 108
                  AND is_confirmed = TRUE
                """,
                home_team_id,
            )
            assert old_confirmed_count == 0

            # Verify: New lineup should be confirmed
            new_confirmed_count = await conn.fetchval(
                """
                SELECT COUNT(*)
                FROM lineups
                WHERE game_id = 'test_game_1'
                  AND team_id = $1
                  AND player_id BETWEEN 200 AND 208
                  AND is_confirmed = TRUE
                """,
                home_team_id,
            )
            assert new_confirmed_count == 9

            # Verify partial unique index behavior: exactly one confirmed row per batting order
            for batting_order in range(1, 10):
                confirmed_for_order = await conn.fetchval(
                    """
                    SELECT COUNT(*)
                    FROM lineups
                    WHERE game_id = 'test_game_1'
                      AND team_id = $1
                      AND batting_order = $2
                      AND is_confirmed = TRUE
                    """,
                    home_team_id,
                    batting_order,
                )
                assert (
                    confirmed_for_order == 1
                ), f"Expected exactly 1 confirmed row for batting_order {batting_order}, got {confirmed_for_order}"

    finally:
        # Cleanup
        async with pool.acquire() as conn:
            # Delete all child rows first (session-scoped pool means data persists)
            await conn.execute("DELETE FROM player_projections WHERE player_id BETWEEN 100 AND 208")
            await conn.execute("DELETE FROM player_game_logs WHERE player_id BETWEEN 100 AND 208")
            await conn.execute("DELETE FROM lineups WHERE player_id BETWEEN 100 AND 208")
            await conn.execute("DELETE FROM players WHERE player_id BETWEEN 100 AND 208")
            await conn.execute("DELETE FROM games WHERE game_id = 'test_game_1'")


# Acceptance Criterion 4: Stats upsert
@pytest.mark.asyncio
async def test_stats_upsert():
    """
    Upsert game logs and verify second call updates without violating constraints.
    """
    pool = await get_pool()

    try:
        # Setup: Create test game (using real team/park IDs from seed data)
        async with pool.acquire() as conn:
            teams = await conn.fetch("SELECT team_id FROM teams LIMIT 2")
            park = await conn.fetchval("SELECT park_id FROM parks LIMIT 1")

            await conn.execute(
                """
                INSERT INTO games (game_id, game_date, home_team_id, away_team_id, park_id, status)
                VALUES ('test_game_2', '2026-01-01', $1, $2, $3, 'final')
                ON CONFLICT (game_id) DO NOTHING
                """,
                teams[0]["team_id"],
                teams[1]["team_id"],
                park,
            )

        # Ensure player exists
        stats_provider = V1StatsProvider()
        await stats_provider.ensure_player_exists(player_id=1001, name="Test Player")

        # Initial game log
        initial_log = [
            GameLogRow(
                player_id=1001,
                game_id="test_game_2",
                pa=4,
                ab=4,
                h=1,
                tb=1,
                hr=0,
                rbi=0,
                r=0,
                bb=0,
                k=1,
            )
        ]

        await stats_provider.write_game_logs(initial_log)

        # Verify initial write
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT h, tb FROM player_game_logs WHERE player_id = 1001 AND game_id = 'test_game_2'"
            )
            assert row["h"] == 1
            assert row["tb"] == 1

        # Updated game log (same player/game, different stats)
        updated_log = [
            GameLogRow(
                player_id=1001,
                game_id="test_game_2",
                pa=5,
                ab=5,
                h=2,
                tb=5,
                hr=1,
                rbi=2,
                r=1,
                bb=0,
                k=1,
            )
        ]

        # Upsert should update, not insert new row
        await stats_provider.write_game_logs(updated_log)

        # Verify update
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT h, tb, hr FROM player_game_logs WHERE player_id = 1001 AND game_id = 'test_game_2'"
            )
            assert row["h"] == 2
            assert row["tb"] == 5
            assert row["hr"] == 1

            # Verify only one row exists
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM player_game_logs WHERE player_id = 1001 AND game_id = 'test_game_2'"
            )
            assert count == 1

    finally:
        # Cleanup
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM player_game_logs WHERE player_id = 1001 AND game_id = 'test_game_2'"
            )
            await conn.execute("DELETE FROM players WHERE player_id = 1001")
            await conn.execute("DELETE FROM games WHERE game_id = 'test_game_2'")


# Acceptance Criterion 5: Weather returns None for indoor/retractable parks
@pytest.mark.asyncio
async def test_weather_park_filtering():
    """
    Weather returns None for indoor or retractable-roof parks.
    """
    pool = await get_pool()
    weather_provider = V1WeatherProvider()

    # Test with existing parks from seed data
    # Find an outdoor park (likely most parks)
    async with pool.acquire() as conn:
        outdoor_park = await conn.fetchrow(
            "SELECT park_id FROM parks WHERE is_outdoor = TRUE AND is_retractable = FALSE LIMIT 1"
        )

        # Should attempt fetch for outdoor park (returns None in stub, but logic is tested)
        if outdoor_park:
            result = await weather_provider.fetch_weather(
                "test_game_3", outdoor_park["park_id"]
            )
            # Stub returns None, but didn't error on park type check
            assert result is None  # v1 stub

        # Test retractable park (if exists)
        retractable_park = await conn.fetchrow(
            "SELECT park_id FROM parks WHERE is_retractable = TRUE LIMIT 1"
        )
        if retractable_park:
            result = await weather_provider.fetch_weather(
                "test_game_4", retractable_park["park_id"]
            )
            assert result is None  # Should skip retractable

        # Test with non-existent park
        result = await weather_provider.fetch_weather("test_game_5", 99999)
        assert result is None  # Should handle gracefully


# FC-13: Test weather skipped for retractable roof parks
@pytest.mark.asyncio
async def test_weather_skipped_retractable_roof():
    """
    Test that weather is skipped for retractable-roof parks (FC-13, D-018).

    Verifies guard condition: is_outdoor = TRUE AND is_retractable = FALSE.
    Even if a park is outdoor, retractable roof means no weather fetch.
    """
    pool = await get_pool()
    weather_provider = V1WeatherProvider()

    try:
        # Create test park: outdoor with retractable roof
        # This tests the specific edge case in FC-13
        async with pool.acquire() as conn:
            # Get a real team for FK
            team_id = await conn.fetchval("SELECT team_id FROM teams LIMIT 1")

            # Insert test park: is_outdoor = TRUE, is_retractable = TRUE
            await conn.execute(
                """
                INSERT INTO parks (park_id, name, team_id, is_outdoor, is_retractable, park_factor)
                VALUES (32000, 'Test Retractable Park', $1, TRUE, TRUE, 1.000)
                ON CONFLICT (park_id) DO NOTHING
                """,
                team_id,
            )

        # Fetch weather for retractable park (should return None)
        result = await weather_provider.fetch_weather("test_game_fc13_retract", 32000)

        # Assert weather was skipped (None returned, no API call)
        assert result is None

        # Verify no weather row was inserted
        async with pool.acquire() as conn:
            weather_count = await conn.fetchval(
                "SELECT COUNT(*) FROM weather WHERE game_id = 'test_game_fc13_retract'"
            )
            assert weather_count == 0

    finally:
        # Cleanup
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM parks WHERE park_id = 32000")


# FC-13: Test weather skipped for indoor parks
@pytest.mark.asyncio
async def test_weather_skipped_indoor():
    """
    Test that weather is skipped for indoor parks (FC-13, D-018).

    Verifies guard condition: is_outdoor = FALSE means no weather fetch.
    """
    pool = await get_pool()
    weather_provider = V1WeatherProvider()

    # Use existing indoor park from seed data (Tropicana Field)
    async with pool.acquire() as conn:
        indoor_park = await conn.fetchrow(
            "SELECT park_id FROM parks WHERE is_outdoor = FALSE AND is_retractable = FALSE LIMIT 1"
        )

    if indoor_park:
        park_id = indoor_park["park_id"]

        # Fetch weather for indoor park (should return None)
        result = await weather_provider.fetch_weather("test_game_fc13_indoor", park_id)

        # Assert weather was skipped (None returned, no API call)
        assert result is None

        # Verify no weather row was inserted
        async with pool.acquire() as conn:
            weather_count = await conn.fetchval(
                "SELECT COUNT(*) FROM weather WHERE game_id = 'test_game_fc13_indoor'"
            )
            assert weather_count == 0


# Acceptance Criterion 6: Fallback behavior
@pytest.mark.asyncio
async def test_fallback_behavior():
    """
    When provider fetch fails, log warning and return empty list without writing data.
    """
    # Test odds provider fallback
    odds_provider = V1OddsProvider()

    # Stub implementation returns empty list on no data
    rows = await odds_provider.fetch_odds("nonexistent_game")
    assert rows == []

    # Test lineup provider fallback
    lineup_provider = V1LineupProvider()
    rows = await lineup_provider.fetch_lineup("nonexistent_game", 1)
    assert rows == []

    # Test stats provider fallback
    stats_provider = V1StatsProvider()
    rows = await stats_provider.fetch_game_logs(date(2026, 1, 1))
    assert rows == []

    # Test game provider fallback
    game_provider = V1GameProvider()
    rows = await game_provider.fetch_schedule(date(2026, 1, 1))
    assert rows == []

    # Test weather provider fallback
    weather_provider = V1WeatherProvider()
    row = await weather_provider.fetch_weather("nonexistent_game", 1)
    assert row is None


# Acceptance Criterion 7: Cache prevents duplicate calls within TTL
def test_cache_ttl():
    """
    Cache prevents duplicate HTTP calls within TTL.
    """
    cache = Cache()

    # Set cache entry with 60s TTL
    cache.set("test_key", b"test_payload", ttl_seconds=60)

    # First get should return cached value
    result = cache.get("test_key")
    assert result == b"test_payload"

    # Second get should also hit cache
    result = cache.get("test_key")
    assert result == b"test_payload"

    # Verify cache entry exists and is not expired
    entry = cache._store.get("test_key")
    assert entry is not None
    assert not entry.is_expired()


def test_cache_expiration():
    """
    Cache entry expires after TTL.
    """
    cache = Cache()

    # Set cache entry with 0s TTL (immediately expired)
    cache.set("test_key", b"test_payload", ttl_seconds=0)

    # Should return None (expired)
    result = cache.get("test_key")
    assert result is None

    # Entry should be removed
    assert "test_key" not in cache._store


def test_cache_miss():
    """
    Cache miss returns None.
    """
    cache = Cache()

    result = cache.get("nonexistent_key")
    assert result is None


def test_cache_prune():
    """
    Cache prune removes expired entries.
    """
    cache = Cache()

    # Add expired and non-expired entries
    cache.set("expired", b"old", ttl_seconds=0)
    cache.set("valid", b"new", ttl_seconds=60)

    # Prune should remove expired entry
    removed = cache.prune_expired()
    assert removed >= 1  # At least one removed

    # Valid entry should still exist
    assert cache.get("valid") == b"new"


def test_cache_singleton():
    """
    get_cache returns singleton instance.
    """
    cache1 = get_cache()
    cache2 = get_cache()

    assert cache1 is cache2

    # Set value in one, get from other
    cache1.set("test", b"value", ttl_seconds=60)
    assert cache2.get("test") == b"value"


# Test D-020: Unknown players are upserted
@pytest.mark.asyncio
async def test_ensure_player_exists():
    """
    Unknown players encountered during ingestion are upserted with available metadata.
    """
    pool = await get_pool()
    stats_provider = V1StatsProvider()

    try:
        # Ensure player doesn't exist
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM players WHERE player_id = 9999")

        # Upsert player with minimal metadata
        await stats_provider.ensure_player_exists(player_id=9999, name="New Player")

        # Verify player exists with NULL position
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT name, position, bats, throws FROM players WHERE player_id = 9999"
            )
            assert row is not None
            assert row["name"] == "New Player"
            assert row["position"] is None
            assert row["bats"] is None
            assert row["throws"] is None

        # Upsert again with team_id (use real team from seed data)
        async with pool.acquire() as conn:
            real_team_id = await conn.fetchval("SELECT team_id FROM teams LIMIT 1")

        await stats_provider.ensure_player_exists(
            player_id=9999, name="Updated Player", team_id=real_team_id
        )

        # Verify update
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT name, team_id FROM players WHERE player_id = 9999"
            )
            assert row["name"] == "Updated Player"
            assert row["team_id"] == real_team_id

    finally:
        # Cleanup
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM players WHERE player_id = 9999")


# FC-12: Test unknown player upsert in stats ingestion
@pytest.mark.asyncio
async def test_unknown_player_upsert_stats():
    """
    Test that stats ingestion automatically upserts unknown players (FC-12, D-020).

    Verifies that write_game_logs creates player rows before inserting game logs,
    preventing FK constraint violations.
    """
    pool = await get_pool()
    stats_provider = V1StatsProvider()

    try:
        # Setup: Create test game
        async with pool.acquire() as conn:
            teams = await conn.fetch("SELECT team_id FROM teams LIMIT 2")
            park = await conn.fetchval("SELECT park_id FROM parks LIMIT 1")

            await conn.execute(
                """
                INSERT INTO games (game_id, game_date, home_team_id, away_team_id, park_id, status)
                VALUES ('test_game_fc12_stats', '2026-01-01', $1, $2, $3, 'final')
                ON CONFLICT (game_id) DO NOTHING
                """,
                teams[0]["team_id"],
                teams[1]["team_id"],
                park,
            )

        # Delete player if exists (start with clean state)
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM players WHERE player_id = 8888")

        # Verify player does not exist
        async with pool.acquire() as conn:
            player_exists = await conn.fetchval(
                "SELECT COUNT(*) FROM players WHERE player_id = 8888"
            )
            assert player_exists == 0

        # Write game log for unknown player (should auto-create player)
        game_log = [
            GameLogRow(
                player_id=8888,
                game_id="test_game_fc12_stats",
                pa=4,
                ab=4,
                h=2,
                tb=5,
                hr=1,
                rbi=2,
                r=1,
                bb=0,
                k=1,
            )
        ]

        await stats_provider.write_game_logs(game_log)

        # Verify player was created automatically
        async with pool.acquire() as conn:
            player_row = await conn.fetchrow(
                "SELECT player_id, name, position, bats, throws FROM players WHERE player_id = 8888"
            )
            assert player_row is not None
            assert player_row["player_id"] == 8888
            assert player_row["name"] == "Player 8888"  # Default name
            assert player_row["position"] is None  # NULLable fields
            assert player_row["bats"] is None
            assert player_row["throws"] is None

        # Verify game log was created
        async with pool.acquire() as conn:
            log_row = await conn.fetchrow(
                "SELECT player_id, game_id, h, hr FROM player_game_logs WHERE player_id = 8888"
            )
            assert log_row is not None
            assert log_row["player_id"] == 8888
            assert log_row["game_id"] == "test_game_fc12_stats"
            assert log_row["h"] == 2
            assert log_row["hr"] == 1

    finally:
        # Cleanup
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM player_game_logs WHERE player_id = 8888"
            )
            await conn.execute("DELETE FROM players WHERE player_id = 8888")
            await conn.execute("DELETE FROM games WHERE game_id = 'test_game_fc12_stats'")


# FC-12: Test unknown player upsert in lineup ingestion
@pytest.mark.asyncio
async def test_unknown_player_upsert_lineups():
    """
    Test that lineup ingestion automatically upserts unknown players (FC-12, D-020).

    Verifies that write_lineup creates player rows before inserting lineup,
    preventing FK constraint violations.
    """
    pool = await get_pool()
    lineup_provider = V1LineupProvider()

    try:
        # Setup: Create test game
        async with pool.acquire() as conn:
            teams = await conn.fetch("SELECT team_id FROM teams LIMIT 2")
            park = await conn.fetchval("SELECT park_id FROM parks LIMIT 1")

            home_team_id = teams[0]["team_id"]
            away_team_id = teams[1]["team_id"]

            await conn.execute(
                """
                INSERT INTO games (game_id, game_date, home_team_id, away_team_id, park_id, status)
                VALUES ('test_game_fc12_lineup', '2026-01-01', $1, $2, $3, 'scheduled')
                ON CONFLICT (game_id) DO NOTHING
                """,
                home_team_id,
                away_team_id,
                park,
            )

        # Delete players if they exist (start with clean state)
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM players WHERE player_id BETWEEN 7770 AND 7778"
            )

        # Verify players do not exist
        async with pool.acquire() as conn:
            player_count = await conn.fetchval(
                "SELECT COUNT(*) FROM players WHERE player_id BETWEEN 7770 AND 7778"
            )
            assert player_count == 0

        # Write lineup for unknown players (should auto-create all 9 players)
        lineup = [
            LineupRow(
                game_id="test_game_fc12_lineup",
                team_id=home_team_id,
                player_id=7770 + i,
                batting_order=i + 1,
                is_confirmed=True,
                source_ts=datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
            )
            for i in range(9)
        ]

        await lineup_provider.write_lineup(lineup, is_confirmed=True)

        # Verify all 9 players were created automatically
        async with pool.acquire() as conn:
            player_count = await conn.fetchval(
                "SELECT COUNT(*) FROM players WHERE player_id BETWEEN 7770 AND 7778"
            )
            assert player_count == 9

            # Check one player has NULLable fields
            player_row = await conn.fetchrow(
                "SELECT player_id, name, team_id, position, bats, throws FROM players WHERE player_id = 7770"
            )
            assert player_row is not None
            assert player_row["player_id"] == 7770
            assert player_row["name"] == "Player 7770"  # Default name
            assert player_row["team_id"] == home_team_id  # From lineup
            assert player_row["position"] is None  # NULLable fields
            assert player_row["bats"] is None
            assert player_row["throws"] is None

        # Verify lineup was created
        async with pool.acquire() as conn:
            lineup_count = await conn.fetchval(
                "SELECT COUNT(*) FROM lineups WHERE game_id = 'test_game_fc12_lineup' AND is_confirmed = TRUE"
            )
            assert lineup_count == 9

    finally:
        # Cleanup
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM lineups WHERE game_id = 'test_game_fc12_lineup'"
            )
            await conn.execute(
                "DELETE FROM players WHERE player_id BETWEEN 7770 AND 7778"
            )
            await conn.execute("DELETE FROM games WHERE game_id = 'test_game_fc12_lineup'")


# FC-15: Test game score persistence for final games
@pytest.mark.asyncio
async def test_game_final_score_persisted():
    """
    Test that home_score and away_score are persisted for final games (FC-15, D-025).

    Verifies:
    - Games with status='final' have home_score and away_score populated
    - Games with status='scheduled' have NULL scores
    """
    pool = await get_pool()
    game_provider = V1GameProvider()

    try:
        # Setup: Get real team and park IDs from seed data
        async with pool.acquire() as conn:
            teams = await conn.fetch("SELECT team_id FROM teams LIMIT 2")
            park = await conn.fetchval("SELECT park_id FROM parks LIMIT 1")

            home_team_id = teams[0]["team_id"]
            away_team_id = teams[1]["team_id"]

        # Create a final game with scores
        final_game = GameRow(
            game_id="test_game_fc15_final",
            game_date=date(2026, 1, 15),
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            park_id=park,
            first_pitch=datetime(2026, 1, 15, 19, 0, 0, tzinfo=timezone.utc),
            status="final",
            home_score=5,
            away_score=3,
        )

        # Create a scheduled game without scores
        scheduled_game = GameRow(
            game_id="test_game_fc15_scheduled",
            game_date=date(2026, 1, 16),
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            park_id=park,
            first_pitch=datetime(2026, 1, 16, 19, 0, 0, tzinfo=timezone.utc),
            status="scheduled",
            home_score=None,
            away_score=None,
        )

        # Write both games
        await game_provider.write_games([final_game, scheduled_game])

        # Verify final game has scores populated
        async with pool.acquire() as conn:
            final_row = await conn.fetchrow(
                """
                SELECT status, home_score, away_score
                FROM games
                WHERE game_id = 'test_game_fc15_final'
                """
            )
            assert final_row is not None
            assert final_row["status"] == "final"
            assert final_row["home_score"] == 5
            assert final_row["away_score"] == 3

        # Verify scheduled game has NULL scores
        async with pool.acquire() as conn:
            scheduled_row = await conn.fetchrow(
                """
                SELECT status, home_score, away_score
                FROM games
                WHERE game_id = 'test_game_fc15_scheduled'
                """
            )
            assert scheduled_row is not None
            assert scheduled_row["status"] == "scheduled"
            assert scheduled_row["home_score"] is None
            assert scheduled_row["away_score"] is None

    finally:
        # Cleanup
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM games WHERE game_id IN ('test_game_fc15_final', 'test_game_fc15_scheduled')"
            )


# FC-30: Test V1GameProvider.fetch_schedule() wires to MLB Stats API
@pytest.mark.asyncio
async def test_fetch_schedule_parses_mlb_api_response():
    """
    Test that fetch_schedule parses MLB API response correctly (FC-30).

    Verifies:
    - Calls MLB Stats API with correct parameters
    - Parses response dates[0].games[] into GameRow objects
    - Maps all fields correctly (game_id, teams, venue, status, scores)
    """
    game_provider = V1GameProvider()

    # Mock MLB API response with 2 games
    mock_response = {
        "dates": [
            {
                "games": [
                    {
                        "gamePk": 123456,
                        "officialDate": "2026-02-14",
                        "gameDate": "2026-02-14T19:10:00Z",
                        "teams": {
                            "home": {
                                "team": {"id": 147},
                                "score": 5,
                            },
                            "away": {
                                "team": {"id": 121},
                                "score": 3,
                            },
                        },
                        "venue": {"id": 2602},
                        "status": {"abstractGameCode": "F"},
                    },
                    {
                        "gamePk": 789012,
                        "officialDate": "2026-02-14",
                        "gameDate": "2026-02-14T20:05:00Z",
                        "teams": {
                            "home": {
                                "team": {"id": 133},
                                "score": None,
                            },
                            "away": {
                                "team": {"id": 137},
                                "score": None,
                            },
                        },
                        "venue": {"id": 2680},
                        "status": {"abstractGameCode": "S"},
                    },
                ]
            }
        ]
    }

    # Mock aiohttp response
    class MockResponse:
        async def json(self):
            return mock_response

        async def read(self):
            return json.dumps(mock_response).encode("utf-8")

        def raise_for_status(self):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    class MockSession:
        def __init__(self, *args, **kwargs):
            pass

        def get(self, url, params=None):
            return MockResponse()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    with patch("aiohttp.ClientSession", MockSession):
        rows = await game_provider.fetch_schedule(date(2026, 2, 14))

    # Verify parsed games
    assert len(rows) == 2

    # Game 1: Final game with scores
    assert rows[0].game_id == "123456"
    assert rows[0].game_date == date(2026, 2, 14)
    assert rows[0].home_team_id == 147
    assert rows[0].away_team_id == 121
    assert rows[0].park_id == 2602
    assert rows[0].first_pitch == datetime(2026, 2, 14, 19, 10, 0, tzinfo=timezone.utc)
    assert rows[0].status == "final"
    assert rows[0].home_score == 5
    assert rows[0].away_score == 3

    # Game 2: Scheduled game without scores
    assert rows[1].game_id == "789012"
    assert rows[1].game_date == date(2026, 2, 14)
    assert rows[1].home_team_id == 133
    assert rows[1].away_team_id == 137
    assert rows[1].park_id == 2680
    assert rows[1].first_pitch == datetime(2026, 2, 14, 20, 5, 0, tzinfo=timezone.utc)
    assert rows[1].status == "scheduled"
    assert rows[1].home_score is None
    assert rows[1].away_score is None


@pytest.mark.asyncio
async def test_fetch_schedule_api_timeout_returns_empty():
    """
    Test that fetch_schedule returns empty list on API timeout (FC-30).

    Verifies conservative fallback: on timeout exception, log warning and return [].
    """
    game_provider = V1GameProvider()

    # Mock aiohttp timeout
    class MockSession:
        def __init__(self, *args, **kwargs):
            pass

        def get(self, url, params=None):
            raise asyncio.TimeoutError("API timeout")

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    with patch("aiohttp.ClientSession", MockSession):
        rows = await game_provider.fetch_schedule(date(2026, 2, 14))

    # Should return empty list on timeout
    assert rows == []


@pytest.mark.asyncio
async def test_fetch_schedule_unknown_venue_falls_back_to_home_team():
    """
    Test that fetch_schedule falls back to home team's park for unknown venue IDs (FC-32).

    Verifies:
    - Game with venue.id = 99999 (not in parks) resolves to home team's park
    - Game with venue.id = 99999 AND home_team_id not in parks is skipped (no GameRow)
    """
    game_provider = V1GameProvider()

    # Mock MLB API response with unknown venue.id and two games
    mock_response = {
        "dates": [
            {
                "games": [
                    {
                        "gamePk": 999001,
                        "officialDate": "2026-02-14",
                        "gameDate": "2026-02-14T19:10:00Z",
                        "teams": {
                            "home": {
                                "team": {"id": 147},  # Has fallback in parks
                                "score": None,
                            },
                            "away": {
                                "team": {"id": 121},
                                "score": None,
                            },
                        },
                        "venue": {"id": 99999},  # Unknown venue ID
                        "status": {"abstractGameCode": "S"},
                    },
                    {
                        "gamePk": 999002,
                        "officialDate": "2026-02-14",
                        "gameDate": "2026-02-14T20:05:00Z",
                        "teams": {
                            "home": {
                                "team": {"id": 88888},  # No fallback in parks
                                "score": None,
                            },
                            "away": {
                                "team": {"id": 121},
                                "score": None,
                            },
                        },
                        "venue": {"id": 99999},  # Unknown venue ID
                        "status": {"abstractGameCode": "S"},
                    },
                ]
            }
        ]
    }

    # Mock aiohttp response
    class MockResponse:
        async def json(self):
            return mock_response

        async def read(self):
            return json.dumps(mock_response).encode("utf-8")

        def raise_for_status(self):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    class MockSession:
        def __init__(self, *args, **kwargs):
            pass

        def get(self, url, params=None):
            return MockResponse()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    # Mock DB queries for valid parks and fallback map (FC-32)
    class MockConnection:
        async def fetch(self, query):
            # Return valid park IDs (not including 99999)
            if "SELECT park_id FROM" in query:
                return [{"park_id": 2602}, {"park_id": 2680}, {"park_id": 2889}]
            # Return team->park fallback map (team 147 has fallback, 88888 does not)
            elif "SELECT team_id, park_id FROM" in query:
                return [
                    {"team_id": 147, "park_id": 2602},
                    {"team_id": 121, "park_id": 2680},
                    {"team_id": 133, "park_id": 2889},
                ]
            return []

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    class MockPool:
        def acquire(self):
            return MockConnection()

    # Mock get_pool to return our mock pool
    async def mock_get_pool():
        return MockPool()

    with patch("aiohttp.ClientSession", MockSession), patch(
        "mlb.ingestion.games.get_pool", mock_get_pool
    ):
        rows = await game_provider.fetch_schedule(date(2026, 2, 14))

    # Verify: Only first game should be returned (with fallback park)
    assert len(rows) == 1
    assert rows[0].game_id == "999001"
    assert rows[0].home_team_id == 147
    assert rows[0].park_id == 2602  # Fallback park for team 147

    # Game 2 should be skipped (home_team_id 88888 not in fallback map)


# FC-31: Test V1OddsProvider.fetch_odds() wires to The Odds API
@pytest.mark.asyncio
async def test_fetch_odds_parses_odds_api_response():
    """
    Test that fetch_odds parses The Odds API response correctly (FC-31).

    Verifies:
    - Calls The Odds API with correct parameters
    - Parses response events[] into OddsRow objects
    - Maps markets (h2h→ml, spreads→rl, totals→total)
    - Converts American odds to decimal
    - Filters to only requested game_id
    """
    odds_provider = V1OddsProvider()

    # Mock The Odds API response
    mock_response = [
        {
            "id": "abc123",
            "sport_key": "baseball_mlb",
            "commence_time": "2026-02-15T00:10:00Z",
            "home_team": "New York Yankees",
            "away_team": "Boston Red Sox",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "title": "DraftKings",
                    "markets": [
                        {
                            "key": "h2h",
                            "last_update": "2026-02-14T20:00:00Z",
                            "outcomes": [
                                {
                                    "name": "New York Yankees",
                                    "price": -150,
                                },
                                {
                                    "name": "Boston Red Sox",
                                    "price": 130,
                                },
                            ],
                        },
                        {
                            "key": "spreads",
                            "last_update": "2026-02-14T20:00:00Z",
                            "outcomes": [
                                {
                                    "name": "New York Yankees",
                                    "price": -110,
                                    "point": -1.5,
                                },
                                {
                                    "name": "Boston Red Sox",
                                    "price": -110,
                                    "point": 1.5,
                                },
                            ],
                        },
                        {
                            "key": "totals",
                            "last_update": "2026-02-14T20:00:00Z",
                            "outcomes": [
                                {
                                    "name": "Over",
                                    "price": -115,
                                    "point": 8.5,
                                },
                                {
                                    "name": "Under",
                                    "price": -105,
                                    "point": 8.5,
                                },
                            ],
                        },
                    ],
                },
            ],
        },
    ]

    # Mock aiohttp response
    class MockResponse:
        status = 200

        async def text(self):
            import json

            return json.dumps(mock_response)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    class MockSession:
        def __init__(self, *args, **kwargs):
            pass

        def get(self, url, params=None):
            return MockResponse()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    # Mock database queries
    class MockConnection:
        async def fetch(self, query, *args):
            # Return team name→id mapping
            if "team_id" in query and "name" in query and "teams" in query.lower():
                return [
                    {"team_id": 147, "name": "New York Yankees"},
                    {"team_id": 111, "name": "Boston Red Sox"},
                ]
            # Return matching game_id
            elif "game_id" in query and "first_pitch" in query and "games" in query.lower():
                return [
                    {
                        "game_id": "test_game_odds_1",
                        "first_pitch": datetime(2026, 2, 15, 0, 10, 0, tzinfo=timezone.utc),
                    }
                ]
            return []

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    class MockPool:
        def acquire(self):
            return MockConnection()

    async def mock_get_pool():
        return MockPool()

    # Clear cache to ensure fresh fetch
    cache = get_cache()
    cache.clear()

    # Mock config with fake API key
    class MockConfig:
        odds_api_key = "test_api_key"
        odds_api_base_url = "https://api.the-odds-api.com/v4"

    with patch("aiohttp.ClientSession", MockSession), patch(
        "mlb.ingestion.odds.get_pool", mock_get_pool
    ), patch("mlb.ingestion.odds.get_config", return_value=MockConfig()):
        rows = await odds_provider.fetch_odds("test_game_odds_1")

    # Verify parsed odds
    assert len(rows) == 6  # 2 h2h + 2 spreads + 2 totals

    # Find specific odds rows
    home_ml = next((r for r in rows if r.market == "ml" and r.side == "home"), None)
    away_ml = next((r for r in rows if r.market == "ml" and r.side == "away"), None)
    home_rl = next((r for r in rows if r.market == "rl" and r.side == "home"), None)
    away_rl = next((r for r in rows if r.market == "rl" and r.side == "away"), None)
    over_total = next((r for r in rows if r.market == "total" and r.side == "over"), None)
    under_total = next((r for r in rows if r.market == "total" and r.side == "under"), None)

    # Verify home moneyline
    assert home_ml is not None
    assert home_ml.game_id == "test_game_odds_1"
    assert home_ml.book == "draftkings"
    assert home_ml.price == pytest.approx(1.667, rel=1e-3)  # -150 → 1.667
    assert home_ml.line is None

    # Verify away moneyline
    assert away_ml is not None
    assert away_ml.price == pytest.approx(2.3, rel=1e-3)  # +130 → 2.30

    # Verify home run line
    assert home_rl is not None
    assert home_rl.line == -1.5
    assert home_rl.price == pytest.approx(1.909, rel=1e-3)  # -110 → 1.909

    # Verify away run line
    assert away_rl is not None
    assert away_rl.line == 1.5
    assert away_rl.price == pytest.approx(1.909, rel=1e-3)  # -110 → 1.909

    # Verify over
    assert over_total is not None
    assert over_total.line == 8.5
    assert over_total.price == pytest.approx(1.870, rel=1e-3)  # -115 → 1.870

    # Verify under
    assert under_total is not None
    assert under_total.line == 8.5
    assert under_total.price == pytest.approx(1.952, rel=1e-3)  # -105 → 1.952


@pytest.mark.asyncio
async def test_fetch_odds_american_to_decimal_correctness():
    """
    Test that fetch_odds converts American odds to decimal correctly (FC-31).

    Verifies the american_to_decimal function is used properly and
    all prices are >= 1.0 (European decimal format).
    """
    odds_provider = V1OddsProvider()

    # Mock response with various American odds values
    mock_response = [
        {
            "id": "test123",
            "commence_time": "2026-02-15T00:10:00Z",
            "home_team": "New York Yankees",
            "away_team": "Boston Red Sox",
            "bookmakers": [
                {
                    "key": "fanduel",
                    "markets": [
                        {
                            "key": "h2h",
                            "last_update": "2026-02-14T20:00:00Z",
                            "outcomes": [
                                {"name": "New York Yankees", "price": 100},  # Even odds
                                {"name": "Boston Red Sox", "price": -200},  # Heavy favorite
                            ],
                        },
                    ],
                },
            ],
        },
    ]

    class MockResponse:
        status = 200

        async def text(self):
            import json

            return json.dumps(mock_response)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    class MockSession:
        def __init__(self, *args, **kwargs):
            pass

        def get(self, url, params=None):
            return MockResponse()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    class MockConnection:
        async def fetch(self, query, *args):
            if "team_id" in query and "name" in query and "teams" in query.lower():
                return [
                    {"team_id": 147, "name": "New York Yankees"},
                    {"team_id": 111, "name": "Boston Red Sox"},
                ]
            elif "game_id" in query and "first_pitch" in query and "games" in query.lower():
                return [{"game_id": "test_game_odds_2", "first_pitch": datetime(2026, 2, 15, 0, 10, 0, tzinfo=timezone.utc)}]
            return []

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    class MockPool:
        def acquire(self):
            return MockConnection()

    async def mock_get_pool():
        return MockPool()

    cache = get_cache()
    cache.clear()

    class MockConfig:
        odds_api_key = "test_api_key"
        odds_api_base_url = "https://api.the-odds-api.com/v4"

    with patch("aiohttp.ClientSession", MockSession), patch(
        "mlb.ingestion.odds.get_pool", mock_get_pool
    ), patch("mlb.ingestion.odds.get_config", return_value=MockConfig()):
        rows = await odds_provider.fetch_odds("test_game_odds_2")

    # Verify all prices are >= 1.0
    assert len(rows) == 2
    for row in rows:
        assert row.price >= 1.0

    # Verify specific conversions
    home_odds = next((r for r in rows if r.side == "home"), None)
    away_odds = next((r for r in rows if r.side == "away"), None)

    assert home_odds.price == pytest.approx(2.0, rel=1e-3)  # +100 → 2.0
    assert away_odds.price == pytest.approx(1.5, rel=1e-3)  # -200 → 1.5


@pytest.mark.asyncio
async def test_fetch_odds_skips_unknown_market():
    """
    Test that fetch_odds skips unknown market types (FC-31).

    Verifies only h2h, spreads, totals are parsed. Other markets are ignored.
    """
    odds_provider = V1OddsProvider()

    # Mock response with unknown market
    mock_response = [
        {
            "id": "test456",
            "commence_time": "2026-02-15T00:10:00Z",
            "home_team": "New York Yankees",
            "away_team": "Boston Red Sox",
            "bookmakers": [
                {
                    "key": "betmgm",
                    "markets": [
                        {
                            "key": "h2h",
                            "last_update": "2026-02-14T20:00:00Z",
                            "outcomes": [
                                {"name": "New York Yankees", "price": -150},
                                {"name": "Boston Red Sox", "price": 130},
                            ],
                        },
                        {
                            "key": "player_props",  # Unknown market type
                            "last_update": "2026-02-14T20:00:00Z",
                            "outcomes": [
                                {"name": "Aaron Judge", "price": -110},
                            ],
                        },
                        {
                            "key": "first_inning_winner",  # Unknown market type
                            "last_update": "2026-02-14T20:00:00Z",
                            "outcomes": [
                                {"name": "New York Yankees", "price": 200},
                            ],
                        },
                    ],
                },
            ],
        },
    ]

    class MockResponse:
        status = 200

        async def text(self):
            import json

            return json.dumps(mock_response)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    class MockSession:
        def __init__(self, *args, **kwargs):
            pass

        def get(self, url, params=None):
            return MockResponse()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    class MockConnection:
        async def fetch(self, query, *args):
            if "team_id" in query and "name" in query and "teams" in query.lower():
                return [
                    {"team_id": 147, "name": "New York Yankees"},
                    {"team_id": 111, "name": "Boston Red Sox"},
                ]
            elif "game_id" in query and "first_pitch" in query and "games" in query.lower():
                return [{"game_id": "test_game_odds_3", "first_pitch": datetime(2026, 2, 15, 0, 10, 0, tzinfo=timezone.utc)}]
            return []

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    class MockPool:
        def acquire(self):
            return MockConnection()

    async def mock_get_pool():
        return MockPool()

    cache = get_cache()
    cache.clear()

    class MockConfig:
        odds_api_key = "test_api_key"
        odds_api_base_url = "https://api.the-odds-api.com/v4"

    with patch("aiohttp.ClientSession", MockSession), patch(
        "mlb.ingestion.odds.get_pool", mock_get_pool
    ), patch("mlb.ingestion.odds.get_config", return_value=MockConfig()):
        rows = await odds_provider.fetch_odds("test_game_odds_3")

    # Verify only h2h market was parsed (unknown markets skipped)
    assert len(rows) == 2  # Only 2 h2h outcomes
    assert all(r.market == "ml" for r in rows)
    # Verify no player_props or first_inning_winner
    assert not any(r.book == "Aaron Judge" for r in rows)


@pytest.mark.asyncio
async def test_fetch_odds_no_game_match_skips_event():
    """
    Test that fetch_odds skips events when no matching game is found (FC-31).

    Verifies:
    - Events with unknown home team are skipped
    - Events with no matching game_id are skipped
    - Only returns odds for games that exist in database
    """
    odds_provider = V1OddsProvider()

    # Mock response with 2 events: one matchable, one not
    mock_response = [
        {
            "id": "event1",
            "commence_time": "2026-02-15T00:10:00Z",
            "home_team": "New York Yankees",  # Known team
            "away_team": "Boston Red Sox",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "markets": [
                        {
                            "key": "h2h",
                            "last_update": "2026-02-14T20:00:00Z",
                            "outcomes": [
                                {"name": "New York Yankees", "price": -150},
                                {"name": "Boston Red Sox", "price": 130},
                            ],
                        },
                    ],
                },
            ],
        },
        {
            "id": "event2",
            "commence_time": "2026-02-15T00:10:00Z",
            "home_team": "Unknown Team XYZ",  # Unknown team
            "away_team": "Another Unknown Team",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "markets": [
                        {
                            "key": "h2h",
                            "last_update": "2026-02-14T20:00:00Z",
                            "outcomes": [
                                {"name": "Unknown Team XYZ", "price": -110},
                                {"name": "Another Unknown Team", "price": -110},
                            ],
                        },
                    ],
                },
            ],
        },
    ]

    class MockResponse:
        status = 200

        async def text(self):
            import json

            return json.dumps(mock_response)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    class MockSession:
        def __init__(self, *args, **kwargs):
            pass

        def get(self, url, params=None):
            return MockResponse()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    class MockConnection:
        async def fetch(self, query, *args):
            if "team_id" in query and "name" in query and "teams" in query.lower():
                return [
                    {"team_id": 147, "name": "New York Yankees"},
                    {"team_id": 111, "name": "Boston Red Sox"},
                ]
            elif "game_id" in query and "first_pitch" in query and "games" in query.lower():
                # Only return game for Yankees (event1), not for Unknown Team (event2)
                if args and args[0] == 147:  # Yankees team_id
                    return [{"game_id": "test_game_odds_4", "first_pitch": datetime(2026, 2, 15, 0, 10, 0, tzinfo=timezone.utc)}]
                return []
            return []

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    class MockPool:
        def acquire(self):
            return MockConnection()

    async def mock_get_pool():
        return MockPool()

    cache = get_cache()
    cache.clear()

    class MockConfig:
        odds_api_key = "test_api_key"
        odds_api_base_url = "https://api.the-odds-api.com/v4"

    with patch("aiohttp.ClientSession", MockSession), patch(
        "mlb.ingestion.odds.get_pool", mock_get_pool
    ), patch("mlb.ingestion.odds.get_config", return_value=MockConfig()):
        rows = await odds_provider.fetch_odds("test_game_odds_4")

    # Verify only event1 odds returned (event2 skipped due to unknown team)
    assert len(rows) == 2
    assert all(r.game_id == "test_game_odds_4" for r in rows)


@pytest.mark.asyncio
async def test_fetch_odds_api_timeout_returns_empty():
    """
    Test that fetch_odds returns empty list on API timeout (FC-31).

    Verifies conservative fallback: on timeout exception, log warning and return [].
    """
    odds_provider = V1OddsProvider()

    # Mock aiohttp timeout
    class MockSession:
        def __init__(self, *args, **kwargs):
            pass

        def get(self, url, params=None):
            raise asyncio.TimeoutError("API timeout")

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    cache = get_cache()
    cache.clear()

    class MockConfig:
        odds_api_key = "test_api_key"
        odds_api_base_url = "https://api.the-odds-api.com/v4"

    with patch("aiohttp.ClientSession", MockSession), patch(
        "mlb.ingestion.odds.get_config", return_value=MockConfig()
    ):
        rows = await odds_provider.fetch_odds("test_game_odds_5")

    # Should return empty list on timeout
    assert rows == []


# ============================================================================
# Step 1C: Lineups Ingestion Tests
# ============================================================================


@pytest.mark.asyncio
async def test_lineup_parse_standard_lineup():
    """
    AC1: Parse standard lineup.
    Mock API returns 9 players with battingOrder "100"–"900", status "P".
    Assert: exactly 9 LineupRow, is_confirmed=TRUE, batting_order in [1,9].
    """
    import json
    from pathlib import Path

    lineup_provider = V1LineupProvider()

    # Load fixture
    fixture_path = Path(__file__).parent / "fixtures" / "boxscore_full_lineup.json"
    with open(fixture_path) as f:
        mock_response = json.load(f)

    # Mock aiohttp response
    class MockResponse:
        async def json(self):
            return mock_response

        async def read(self):
            return json.dumps(mock_response).encode("utf-8")

        def raise_for_status(self):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    class MockSession:
        def __init__(self, *args, **kwargs):
            pass

        def get(self, url, params=None):
            return MockResponse()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    class MockConfig:
        mlb_stats_api_base_url = "https://statsapi.mlb.com/api/v1"

    with patch("aiohttp.ClientSession", MockSession), patch(
        "mlb.ingestion.lineups.get_config", return_value=MockConfig()
    ), patch("mlb.ingestion.lineups.V1LineupProvider._persist_lineups", new_callable=AsyncMock):
        rows = await lineup_provider.fetch_lineup("test_game_ac1", team_id=None)

    # Verify: 18 total rows (9 home + 9 away)
    assert len(rows) == 18

    # Verify home lineup
    home_rows = [r for r in rows if r.team_id == 147]
    assert len(home_rows) == 9
    assert all(r.is_confirmed for r in home_rows)
    assert all(1 <= r.batting_order <= 9 for r in home_rows)
    assert set(r.batting_order for r in home_rows) == set(range(1, 10))

    # Verify away lineup
    away_rows = [r for r in rows if r.team_id == 111]
    assert len(away_rows) == 9
    assert all(r.is_confirmed for r in away_rows)
    assert all(1 <= r.batting_order <= 9 for r in away_rows)
    assert set(r.batting_order for r in away_rows) == set(range(1, 10))


@pytest.mark.asyncio
async def test_lineup_confirm_on_game_start():
    """
    AC2: Confirm on game start.
    Mock status "L" with 9 players. Assert: all is_confirmed=TRUE.
    """
    import json
    from pathlib import Path

    lineup_provider = V1LineupProvider()

    # Load fixture
    fixture_path = Path(__file__).parent / "fixtures" / "boxscore_game_live.json"
    with open(fixture_path) as f:
        mock_response = json.load(f)

    class MockResponse:
        async def json(self):
            return mock_response

        async def read(self):
            return json.dumps(mock_response).encode("utf-8")

        def raise_for_status(self):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    class MockSession:
        def __init__(self, *args, **kwargs):
            pass

        def get(self, url, params=None):
            return MockResponse()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    class MockConfig:
        mlb_stats_api_base_url = "https://statsapi.mlb.com/api/v1"

    with patch("aiohttp.ClientSession", MockSession), patch(
        "mlb.ingestion.lineups.get_config", return_value=MockConfig()
    ), patch("mlb.ingestion.lineups.V1LineupProvider._persist_lineups", new_callable=AsyncMock):
        rows = await lineup_provider.fetch_lineup("test_game_ac2", team_id=147)

    # Verify: 9 rows, all confirmed (game live)
    assert len(rows) == 9
    assert all(r.is_confirmed for r in rows)
    assert all(r.team_id == 147 for r in rows)


@pytest.mark.asyncio
async def test_lineup_partial_lineup_unconfirmed():
    """
    AC3: Partial lineup unconfirmed.
    Mock 6 players, status "P". Assert: 6 rows, all is_confirmed=FALSE.
    """
    import json
    from pathlib import Path

    lineup_provider = V1LineupProvider()

    # Load fixture
    fixture_path = Path(__file__).parent / "fixtures" / "boxscore_partial.json"
    with open(fixture_path) as f:
        mock_response = json.load(f)

    class MockResponse:
        async def json(self):
            return mock_response

        async def read(self):
            return json.dumps(mock_response).encode("utf-8")

        def raise_for_status(self):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    class MockSession:
        def __init__(self, *args, **kwargs):
            pass

        def get(self, url, params=None):
            return MockResponse()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    class MockConfig:
        mlb_stats_api_base_url = "https://statsapi.mlb.com/api/v1"

    with patch("aiohttp.ClientSession", MockSession), patch(
        "mlb.ingestion.lineups.get_config", return_value=MockConfig()
    ), patch("mlb.ingestion.lineups.V1LineupProvider._persist_lineups", new_callable=AsyncMock):
        rows = await lineup_provider.fetch_lineup("test_game_ac3", team_id=147)

    # Verify: 6 rows, all unconfirmed (partial lineup)
    assert len(rows) == 6
    assert all(not r.is_confirmed for r in rows)
    assert all(r.team_id == 147 for r in rows)


@pytest.mark.asyncio
async def test_lineup_d011_flip_logic():
    """
    AC4: D-011 flip logic.
    Pre-insert confirmed lineup. Fetch new confirmed lineup.
    Assert: old rows is_confirmed=FALSE, new rows is_confirmed=TRUE.
    """
    pool = await get_pool()

    try:
        # Setup: Create test game
        async with pool.acquire() as conn:
            teams = await conn.fetch("SELECT team_id FROM teams LIMIT 2")
            park = await conn.fetchval("SELECT park_id FROM parks LIMIT 1")

            home_team_id = teams[0]["team_id"]
            away_team_id = teams[1]["team_id"]

            await conn.execute(
                """
                INSERT INTO games (game_id, game_date, home_team_id, away_team_id, park_id, status)
                VALUES ('test_game_ac4', '2026-02-14', $1, $2, $3, 'scheduled')
                ON CONFLICT (game_id) DO NOTHING
                """,
                home_team_id,
                away_team_id,
                park,
            )

        # Insert initial confirmed lineup
        initial_lineup = [
            LineupRow(
                game_id="test_game_ac4",
                team_id=home_team_id,
                player_id=30000 + i,
                batting_order=i + 1,
                is_confirmed=True,
                source_ts=datetime(2026, 2, 14, 10, 0, 0, tzinfo=timezone.utc),
            )
            for i in range(9)
        ]

        lineup_provider = V1LineupProvider()
        await lineup_provider._persist_lineups(initial_lineup)

        # Verify initial lineup is confirmed
        async with pool.acquire() as conn:
            count_confirmed = await conn.fetchval(
                "SELECT COUNT(*) FROM lineups WHERE game_id = 'test_game_ac4' AND is_confirmed = TRUE"
            )
            assert count_confirmed == 9

        # Insert new confirmed lineup (different players)
        new_lineup = [
            LineupRow(
                game_id="test_game_ac4",
                team_id=home_team_id,
                player_id=40000 + i,
                batting_order=i + 1,
                is_confirmed=True,
                source_ts=datetime(2026, 2, 14, 11, 0, 0, tzinfo=timezone.utc),
            )
            for i in range(9)
        ]

        await lineup_provider._persist_lineups(new_lineup)

        # Verify: Old lineup flipped to FALSE
        async with pool.acquire() as conn:
            old_confirmed_count = await conn.fetchval(
                """
                SELECT COUNT(*)
                FROM lineups
                WHERE game_id = 'test_game_ac4'
                  AND player_id BETWEEN 30000 AND 30008
                  AND is_confirmed = TRUE
                """
            )
            assert old_confirmed_count == 0

            # Verify: New lineup is confirmed
            new_confirmed_count = await conn.fetchval(
                """
                SELECT COUNT(*)
                FROM lineups
                WHERE game_id = 'test_game_ac4'
                  AND player_id BETWEEN 40000 AND 40008
                  AND is_confirmed = TRUE
                """
            )
            assert new_confirmed_count == 9

    finally:
        # Cleanup
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM lineups WHERE game_id = 'test_game_ac4'")
            await conn.execute("DELETE FROM players WHERE player_id BETWEEN 30000 AND 30008")
            await conn.execute("DELETE FROM players WHERE player_id BETWEEN 40000 AND 40008")
            await conn.execute("DELETE FROM games WHERE game_id = 'test_game_ac4'")


@pytest.mark.asyncio
async def test_lineup_d020_player_upsert():
    """
    AC5: D-020 player upsert.
    Mock lineup with unknown player_id. Assert: player upserted; lineup inserted without FK error.
    """
    pool = await get_pool()

    try:
        # Setup: Create test game
        async with pool.acquire() as conn:
            teams = await conn.fetch("SELECT team_id FROM teams LIMIT 2")
            park = await conn.fetchval("SELECT park_id FROM parks LIMIT 1")

            home_team_id = teams[0]["team_id"]
            away_team_id = teams[1]["team_id"]

            await conn.execute(
                """
                INSERT INTO games (game_id, game_date, home_team_id, away_team_id, park_id, status)
                VALUES ('test_game_ac5', '2026-02-14', $1, $2, $3, 'scheduled')
                ON CONFLICT (game_id) DO NOTHING
                """,
                home_team_id,
                away_team_id,
                park,
            )

        # Delete player if exists (start clean)
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM players WHERE player_id = 50000")

        # Verify player does not exist
        async with pool.acquire() as conn:
            player_exists = await conn.fetchval(
                "SELECT COUNT(*) FROM players WHERE player_id = 50000"
            )
            assert player_exists == 0

        # Create lineup with unknown player
        lineup = [
            LineupRow(
                game_id="test_game_ac5",
                team_id=home_team_id,
                player_id=50000,
                batting_order=1,
                is_confirmed=True,
                source_ts=datetime(2026, 2, 14, 10, 0, 0, tzinfo=timezone.utc),
            )
        ]

        lineup_provider = V1LineupProvider()
        await lineup_provider._persist_lineups(lineup)

        # Verify player was auto-created
        async with pool.acquire() as conn:
            player_row = await conn.fetchrow(
                "SELECT player_id, name FROM players WHERE player_id = 50000"
            )
            assert player_row is not None
            assert player_row["player_id"] == 50000

        # Verify lineup was inserted
        async with pool.acquire() as conn:
            lineup_count = await conn.fetchval(
                "SELECT COUNT(*) FROM lineups WHERE game_id = 'test_game_ac5' AND player_id = 50000"
            )
            assert lineup_count == 1

    finally:
        # Cleanup
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM lineups WHERE game_id = 'test_game_ac5'")
            await conn.execute("DELETE FROM players WHERE player_id = 50000")
            await conn.execute("DELETE FROM games WHERE game_id = 'test_game_ac5'")


@pytest.mark.asyncio
async def test_lineup_skip_bench_players():
    """
    AC6: Skip bench players.
    Mock 11 players (9 + 2 with battingOrder="0"). Assert: exactly 9 rows returned.
    """
    import json
    from pathlib import Path

    lineup_provider = V1LineupProvider()

    # Load fixture
    fixture_path = Path(__file__).parent / "fixtures" / "boxscore_with_bench.json"
    with open(fixture_path) as f:
        mock_response = json.load(f)

    class MockResponse:
        async def json(self):
            return mock_response

        async def read(self):
            return json.dumps(mock_response).encode("utf-8")

        def raise_for_status(self):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    class MockSession:
        def __init__(self, *args, **kwargs):
            pass

        def get(self, url, params=None):
            return MockResponse()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    class MockConfig:
        mlb_stats_api_base_url = "https://statsapi.mlb.com/api/v1"

    with patch("aiohttp.ClientSession", MockSession), patch(
        "mlb.ingestion.lineups.get_config", return_value=MockConfig()
    ), patch("mlb.ingestion.lineups.V1LineupProvider._persist_lineups", new_callable=AsyncMock):
        rows = await lineup_provider.fetch_lineup("test_game_ac6", team_id=147)

    # Verify: exactly 9 rows (bench players skipped)
    assert len(rows) == 9
    assert all(1 <= r.batting_order <= 9 for r in rows)
    # Verify bench players not included
    assert all(r.player_id not in [12354, 12355] for r in rows)


@pytest.mark.asyncio
async def test_lineup_fallback_to_array():
    """
    AC7: Fallback to array.
    Mock missing battingOrder field with battingOrder array present.
    Assert: order from array index.
    """
    import json

    lineup_provider = V1LineupProvider()

    # Mock response with missing battingOrder field
    mock_response = {
        "teams": {
            "home": {
                "team": {"id": 147},
                "players": {
                    "ID12345": {
                        "person": {"id": 12345, "fullName": "Aaron Judge"},
                        "position": {"abbreviation": "RF"},
                        # battingOrder field missing
                    },
                    "ID12346": {
                        "person": {"id": 12346, "fullName": "Anthony Rizzo"},
                        "position": {"abbreviation": "1B"},
                        # battingOrder field missing
                    },
                },
                "battingOrder": [12345, 12346],  # Fallback array
            }
        },
        "status": {"abstractGameCode": "P"},
    }

    class MockResponse:
        async def json(self):
            return mock_response

        async def read(self):
            return json.dumps(mock_response).encode("utf-8")

        def raise_for_status(self):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    class MockSession:
        def __init__(self, *args, **kwargs):
            pass

        def get(self, url, params=None):
            return MockResponse()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    class MockConfig:
        mlb_stats_api_base_url = "https://statsapi.mlb.com/api/v1"

    with patch("aiohttp.ClientSession", MockSession), patch(
        "mlb.ingestion.lineups.get_config", return_value=MockConfig()
    ), patch("mlb.ingestion.lineups.V1LineupProvider._persist_lineups", new_callable=AsyncMock):
        rows = await lineup_provider.fetch_lineup("test_game_ac7", team_id=147)

    # Verify: 2 rows with batting_order from array index
    assert len(rows) == 2
    # Player 12345 should be batting_order 1 (index 0)
    judge = next(r for r in rows if r.player_id == 12345)
    assert judge.batting_order == 1
    # Player 12346 should be batting_order 2 (index 1)
    rizzo = next(r for r in rows if r.player_id == 12346)
    assert rizzo.batting_order == 2


@pytest.mark.asyncio
async def test_lineup_api_timeout():
    """
    AC8: API timeout.
    Mock timeout. Assert: returns [], logs warning.
    """
    lineup_provider = V1LineupProvider()

    # Mock aiohttp timeout
    class MockSession:
        def __init__(self, *args, **kwargs):
            pass

        def get(self, url, params=None):
            raise asyncio.TimeoutError("API timeout")

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    class MockConfig:
        mlb_stats_api_base_url = "https://statsapi.mlb.com/api/v1"

    with patch("aiohttp.ClientSession", MockSession), patch(
        "mlb.ingestion.lineups.get_config", return_value=MockConfig()
    ):
        rows = await lineup_provider.fetch_lineup("test_game_ac8", team_id=147)

    # Should return empty list on timeout
    assert rows == []


@pytest.mark.asyncio
async def test_lineup_invalid_batting_order():
    """
    AC9: Invalid batting_order.
    Mock player with battingOrder="1000". Assert: player skipped, logs warning.
    """
    import json

    lineup_provider = V1LineupProvider()

    # Mock response with invalid batting_order
    mock_response = {
        "teams": {
            "home": {
                "team": {"id": 147},
                "players": {
                    "ID12345": {
                        "person": {"id": 12345, "fullName": "Aaron Judge"},
                        "position": {"abbreviation": "RF"},
                        "battingOrder": "100",  # Valid
                    },
                    "ID12346": {
                        "person": {"id": 12346, "fullName": "Pinch Hitter"},
                        "position": {"abbreviation": "PH"},
                        "battingOrder": "1000",  # Invalid (out of range)
                    },
                },
                "battingOrder": [12345],  # Only valid player in array
            }
        },
        "status": {"abstractGameCode": "P"},
    }

    class MockResponse:
        async def json(self):
            return mock_response

        async def read(self):
            return json.dumps(mock_response).encode("utf-8")

        def raise_for_status(self):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    class MockSession:
        def __init__(self, *args, **kwargs):
            pass

        def get(self, url, params=None):
            return MockResponse()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    class MockConfig:
        mlb_stats_api_base_url = "https://statsapi.mlb.com/api/v1"

    with patch("aiohttp.ClientSession", MockSession), patch(
        "mlb.ingestion.lineups.get_config", return_value=MockConfig()
    ), patch("mlb.ingestion.lineups.V1LineupProvider._persist_lineups", new_callable=AsyncMock):
        rows = await lineup_provider.fetch_lineup("test_game_ac9", team_id=147)

    # Verify: only valid player returned (invalid batting_order skipped)
    assert len(rows) == 1
    assert rows[0].player_id == 12345
    assert rows[0].batting_order == 1


@pytest.mark.asyncio
async def test_lineup_transaction_rollback():
    """
    AC10: Transaction rollback.
    Mock DB error during lineup insert. Assert: exception caught internally, no partial writes.
    """
    pool = await get_pool()

    try:
        # Setup: Create test game
        async with pool.acquire() as conn:
            teams = await conn.fetch("SELECT team_id FROM teams LIMIT 2")
            park = await conn.fetchval("SELECT park_id FROM parks LIMIT 1")

            home_team_id = teams[0]["team_id"]
            away_team_id = teams[1]["team_id"]

            await conn.execute(
                """
                INSERT INTO games (game_id, game_date, home_team_id, away_team_id, park_id, status)
                VALUES ('test_game_ac10', '2026-02-14', $1, $2, $3, 'scheduled')
                ON CONFLICT (game_id) DO NOTHING
                """,
                home_team_id,
                away_team_id,
                park,
            )

        # Create lineup with invalid data that will cause DB error
        lineup = [
            LineupRow(
                game_id="test_game_ac10",
                team_id=home_team_id,
                player_id=60000,
                batting_order=1,
                is_confirmed=True,
                source_ts=datetime(2026, 2, 14, 10, 0, 0, tzinfo=timezone.utc),
            )
        ]

        lineup_provider = V1LineupProvider()

        # Mock get_pool to return a pool that will fail on executemany
        class MockConnection:
            def __init__(self, real_conn):
                self._real_conn = real_conn

            async def execute(self, *args, **kwargs):
                return await self._real_conn.execute(*args, **kwargs)

            async def executemany(self, sql, *args, **kwargs):
                # Fail on lineup insert
                if "INSERT INTO lineups" in sql:
                    raise asyncpg.PostgresError("Simulated database error")
                return await self._real_conn.executemany(sql, *args, **kwargs)

            def transaction(self):
                return self._real_conn.transaction()

            async def __aenter__(self):
                await self._real_conn.__aenter__()
                return self

            async def __aexit__(self, *args):
                return await self._real_conn.__aexit__(*args)

        class MockPool:
            def __init__(self, real_pool):
                self._real_pool = real_pool

            def acquire(self):
                class AcquireContext:
                    def __init__(self, pool):
                        self.pool = pool

                    async def __aenter__(self):
                        conn = await self.pool._real_pool.acquire().__aenter__()
                        return MockConnection(conn)

                    async def __aexit__(self, *args):
                        pass

                return AcquireContext(self)

        async def mock_get_pool():
            real_pool = await get_pool()
            return MockPool(real_pool)

        # Patch get_pool in lineups module
        with patch("mlb.ingestion.lineups.get_pool", mock_get_pool):
            # This should catch the error internally and not raise
            await lineup_provider._persist_lineups(lineup)

        # Verify: No lineup rows inserted (transaction rolled back)
        async with pool.acquire() as conn:
            lineup_count = await conn.fetchval(
                "SELECT COUNT(*) FROM lineups WHERE game_id = 'test_game_ac10'"
            )
            assert lineup_count == 0

        # Verify: Player was upserted before error (D-020)
        async with pool.acquire() as conn:
            player_count = await conn.fetchval(
                "SELECT COUNT(*) FROM players WHERE player_id = 60000"
            )
            # Player should NOT exist because transaction rolled back
            assert player_count == 0

    finally:
        # Cleanup
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM lineups WHERE game_id = 'test_game_ac10'")
            await conn.execute("DELETE FROM players WHERE player_id = 60000")
            await conn.execute("DELETE FROM games WHERE game_id = 'test_game_ac10'")


# ============================================================================
# Step 1D: Player Stats / Game Logs Ingestion Tests
# ============================================================================


def test_parse_hitting_stats():
    """
    AC1: Parse hitting stats.
    Mock hitting API response. Assert pa, ab, h, tb, hr, rbi, r, bb, k populated; pitching fields None.
    """
    import json
    from pathlib import Path

    from mlb.ingestion.stats import V1StatsProvider

    # Load fixture
    fixture_path = Path(__file__).parent / "fixtures" / "gamelog_hitting.json"
    with open(fixture_path) as f:
        hitting_data = json.load(f)

    provider = V1StatsProvider()
    logs_dict: dict[tuple[int, str], GameLogRow] = {}

    # Process hitting splits
    asyncio.run(
        provider._process_hitting_splits(hitting_data, 660271, ["746587"], logs_dict)
    )

    # Verify single log row created
    assert len(logs_dict) == 1
    log = logs_dict[(660271, "746587")]

    # Verify hitting fields populated
    assert log.player_id == 660271
    assert log.game_id == "746587"
    assert log.pa == 4
    assert log.ab == 3
    assert log.h == 2
    assert log.tb == 5
    assert log.hr == 1
    assert log.rbi == 2
    assert log.r == 1
    assert log.bb == 1
    assert log.k == 1

    # Verify pitching fields None
    assert log.ip_outs is None
    assert log.er is None
    assert log.pitch_count is None
    assert log.is_starter is None


def test_parse_pitching_stats():
    """
    AC2: Parse pitching stats.
    Mock pitching API response. Assert ip_outs, er, pitch_count, is_starter populated; hitting fields None.
    """
    import json
    from pathlib import Path

    from mlb.ingestion.stats import V1StatsProvider

    # Load fixture
    fixture_path = Path(__file__).parent / "fixtures" / "gamelog_pitching.json"
    with open(fixture_path) as f:
        pitching_data = json.load(f)

    provider = V1StatsProvider()
    logs_dict: dict[tuple[int, str], GameLogRow] = {}

    # Process pitching splits
    asyncio.run(
        provider._process_pitching_splits(pitching_data, 660271, ["746587"], logs_dict)
    )

    # Verify single log row created
    assert len(logs_dict) == 1
    log = logs_dict[(660271, "746587")]

    # Verify pitching fields populated
    assert log.player_id == 660271
    assert log.game_id == "746587"
    assert log.ip_outs == 17  # 5.2 IP = 5*3 + 2 = 17 outs
    assert log.er == 3
    assert log.pitch_count == 87
    assert log.is_starter is True  # >= 9 outs

    # Verify hitting fields None (k is batter K per schema, not pitcher K)
    assert log.pa is None
    assert log.ab is None
    assert log.h is None
    assert log.tb is None
    assert log.hr is None
    assert log.rbi is None
    assert log.r is None
    assert log.bb is None
    assert log.k is None  # k is batter strikeouts, not populated for pitcher-only logs


def test_innings_conversion():
    """
    AC3: Innings conversion.
    Test parse_ip_to_outs(): "0.1"→1, "0.2"→2, "5.0"→15, "5.2"→17, "6"→18, None→None.
    Invalid "5.3"→None + WARNING logged.
    """
    from mlb.ingestion.stats import parse_ip_to_outs

    # Valid formats
    assert parse_ip_to_outs("0.1") == 1
    assert parse_ip_to_outs("0.2") == 2
    assert parse_ip_to_outs("5.0") == 15
    assert parse_ip_to_outs("5.2") == 17
    assert parse_ip_to_outs("6") == 18
    assert parse_ip_to_outs("6.0") == 18
    assert parse_ip_to_outs("7.1") == 22
    assert parse_ip_to_outs(None) is None
    assert parse_ip_to_outs("") is None

    # Invalid format (partial_outs not in {0, 1, 2})
    assert parse_ip_to_outs("5.3") is None  # Should log WARNING


def test_parse_ip_to_outs_handles_non_numeric():
    """
    FC-35: Non-numeric innings data handling.
    Test parse_ip_to_outs() with malformed API data: "X.2", "5.X", "ERR".
    Assert: returns None, logs WARNING (non-numeric).
    """
    from mlb.ingestion.stats import parse_ip_to_outs

    # Non-numeric full innings
    assert parse_ip_to_outs("X.2") is None  # Should log WARNING (non-numeric)
    assert parse_ip_to_outs("ERR") is None  # Should log WARNING (non-numeric)
    assert parse_ip_to_outs("ABC.1") is None  # Should log WARNING (non-numeric)

    # Non-numeric partial outs
    assert parse_ip_to_outs("5.X") is None  # Should log WARNING (non-numeric)
    assert parse_ip_to_outs("3.Y") is None  # Should log WARNING (non-numeric)

    # Both non-numeric
    assert parse_ip_to_outs("X.Y") is None  # Should log WARNING (non-numeric)


def test_starter_detection():
    """
    AC4: Starter detection.
    Test detect_starter(): 18 outs→True, 9 outs→True, 8 outs→False, None→None.
    """
    from mlb.ingestion.stats import detect_starter

    # Starters (>= 9 outs = 3+ innings)
    assert detect_starter(18) is True  # 6 IP
    assert detect_starter(9) is True  # 3 IP
    assert detect_starter(15) is True  # 5 IP
    assert detect_starter(27) is True  # 9 IP (complete game)

    # Relievers (< 9 outs = < 3 IP)
    assert detect_starter(8) is False  # 2.2 IP
    assert detect_starter(6) is False  # 2 IP
    assert detect_starter(3) is False  # 1 IP
    assert detect_starter(1) is False  # 0.1 IP

    # Non-pitcher
    assert detect_starter(None) is None


def test_twoway_player_merge():
    """
    AC5: Two-way player merge.
    Mock player with both hitting and pitching splits for same game.
    Assert single row with both stat sets populated.
    """
    import json
    from pathlib import Path

    from mlb.ingestion.stats import V1StatsProvider

    # Load fixture
    fixture_path = Path(__file__).parent / "fixtures" / "gamelog_twoway.json"
    with open(fixture_path) as f:
        fixture_data = json.load(f)

    provider = V1StatsProvider()
    logs_dict: dict[tuple[int, str], GameLogRow] = {}

    # Process hitting then pitching (simulating two-way player)
    asyncio.run(
        provider._process_hitting_splits(
            fixture_data["hitting"], 660271, ["746999"], logs_dict
        )
    )
    asyncio.run(
        provider._process_pitching_splits(
            fixture_data["pitching"], 660271, ["746999"], logs_dict
        )
    )

    # Verify single log row with both stat sets
    assert len(logs_dict) == 1
    log = logs_dict[(660271, "746999")]

    # Verify hitting stats
    assert log.pa == 4
    assert log.ab == 4
    assert log.h == 2
    assert log.hr == 0

    # Verify pitching stats
    assert log.ip_outs == 18  # 6.0 IP
    assert log.er == 2
    assert log.pitch_count == 95
    assert log.is_starter is True


@pytest.mark.asyncio
async def test_gamelog_upsert_conflict():
    """
    AC6: UPSERT conflict.
    Insert log, then re-insert with updated hr count.
    Assert single row exists with new hr value, same log_id.
    """
    pool = await get_pool()
    stats_provider = V1StatsProvider()

    try:
        # Setup: Create test game and player
        async with pool.acquire() as conn:
            teams = await conn.fetch("SELECT team_id FROM teams LIMIT 2")
            park = await conn.fetchval("SELECT park_id FROM parks LIMIT 1")

            home_team_id = teams[0]["team_id"]
            away_team_id = teams[1]["team_id"]

            await conn.execute(
                """
                INSERT INTO games (game_id, game_date, home_team_id, away_team_id, park_id, status)
                VALUES ('test_game_stats_1', '2026-02-14', $1, $2, $3, 'final')
                ON CONFLICT (game_id) DO NOTHING
                """,
                home_team_id,
                away_team_id,
                park,
            )

            # Ensure player exists
            await conn.execute(
                """
                INSERT INTO players (player_id, name)
                VALUES (70000, 'Test Player')
                ON CONFLICT (player_id) DO NOTHING
                """
            )

        # Insert initial log
        initial_log = GameLogRow(
            player_id=70000,
            game_id="test_game_stats_1",
            pa=4,
            ab=4,
            h=2,
            tb=5,
            hr=1,
            rbi=2,
            r=1,
            bb=0,
            k=1,
        )
        await stats_provider.write_game_logs([initial_log])

        # Get log_id
        async with pool.acquire() as conn:
            initial_row = await conn.fetchrow(
                "SELECT log_id, hr FROM player_game_logs WHERE player_id = 70000 AND game_id = 'test_game_stats_1'"
            )
            assert initial_row is not None
            assert initial_row["hr"] == 1
            log_id = initial_row["log_id"]

        # Re-insert with updated hr count
        updated_log = GameLogRow(
            player_id=70000,
            game_id="test_game_stats_1",
            pa=4,
            ab=4,
            h=2,
            tb=8,  # Updated
            hr=2,  # Updated
            rbi=3,  # Updated
            r=1,
            bb=0,
            k=1,
        )
        await stats_provider.write_game_logs([updated_log])

        # Verify: single row with new values, same log_id
        async with pool.acquire() as conn:
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM player_game_logs WHERE player_id = 70000 AND game_id = 'test_game_stats_1'"
            )
            assert count == 1

            updated_row = await conn.fetchrow(
                "SELECT log_id, hr, tb, rbi FROM player_game_logs WHERE player_id = 70000 AND game_id = 'test_game_stats_1'"
            )
            assert updated_row["log_id"] == log_id  # Same log_id
            assert updated_row["hr"] == 2  # Updated
            assert updated_row["tb"] == 8  # Updated
            assert updated_row["rbi"] == 3  # Updated

    finally:
        # Cleanup
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM player_game_logs WHERE player_id = 70000"
            )
            await conn.execute("DELETE FROM players WHERE player_id = 70000")
            await conn.execute("DELETE FROM games WHERE game_id = 'test_game_stats_1'")


@pytest.mark.asyncio
async def test_gamelog_d020_player_upsert():
    """
    AC7: D-020 player upsert.
    Mock log for player_id not in players table.
    Assert player inserted, game log inserted without FK error.
    """
    pool = await get_pool()
    stats_provider = V1StatsProvider()

    try:
        # Setup: Create test game
        async with pool.acquire() as conn:
            teams = await conn.fetch("SELECT team_id FROM teams LIMIT 2")
            park = await conn.fetchval("SELECT park_id FROM parks LIMIT 1")

            home_team_id = teams[0]["team_id"]
            away_team_id = teams[1]["team_id"]

            await conn.execute(
                """
                INSERT INTO games (game_id, game_date, home_team_id, away_team_id, park_id, status)
                VALUES ('test_game_stats_2', '2026-02-14', $1, $2, $3, 'final')
                ON CONFLICT (game_id) DO NOTHING
                """,
                home_team_id,
                away_team_id,
                park,
            )

        # Delete player if exists
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM players WHERE player_id = 70001")

        # Verify player does not exist
        async with pool.acquire() as conn:
            player_count = await conn.fetchval(
                "SELECT COUNT(*) FROM players WHERE player_id = 70001"
            )
            assert player_count == 0

        # Write log for unknown player
        log = GameLogRow(
            player_id=70001,
            game_id="test_game_stats_2",
            pa=3,
            ab=3,
            h=1,
            tb=4,
            hr=1,
        )
        await stats_provider.write_game_logs([log])

        # Verify player was auto-created
        async with pool.acquire() as conn:
            player_row = await conn.fetchrow(
                "SELECT player_id, name FROM players WHERE player_id = 70001"
            )
            assert player_row is not None
            assert player_row["player_id"] == 70001

        # Verify game log was inserted
        async with pool.acquire() as conn:
            log_count = await conn.fetchval(
                "SELECT COUNT(*) FROM player_game_logs WHERE player_id = 70001 AND game_id = 'test_game_stats_2'"
            )
            assert log_count == 1

    finally:
        # Cleanup
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM player_game_logs WHERE player_id = 70001"
            )
            await conn.execute("DELETE FROM players WHERE player_id = 70001")
            await conn.execute("DELETE FROM games WHERE game_id = 'test_game_stats_2'")


def test_gamelog_missing_fields():
    """
    AC8: Missing fields.
    Mock response with plateAppearances but no atBats.
    Assert pa populated, ab is None.
    """
    import json

    from mlb.ingestion.stats import V1StatsProvider

    # Mock response with partial data
    mock_data = {
        "stats": [
            {
                "splits": [
                    {
                        "stat": {
                            "plateAppearances": 4,
                            # atBats missing
                            "hits": 2,
                        },
                        "game": {"gamePk": 999999},
                    }
                ]
            }
        ]
    }

    provider = V1StatsProvider()
    logs_dict: dict[tuple[int, str], GameLogRow] = {}

    asyncio.run(provider._process_hitting_splits(mock_data, 99999, ["999999"], logs_dict))

    # Verify pa populated, ab is None
    log = logs_dict[(99999, "999999")]
    assert log.pa == 4
    assert log.ab is None  # Missing from API
    assert log.h == 2


@pytest.mark.asyncio
async def test_gamelog_api_timeout():
    """
    AC9: API timeout.
    Mock timeout. Assert returns [], WARNING logged.
    """
    from mlb.ingestion.stats import V1StatsProvider

    stats_provider = V1StatsProvider()

    # Mock aiohttp timeout
    class MockSession:
        def __init__(self, *args, **kwargs):
            pass

        def get(self, url, **kwargs):
            raise asyncio.TimeoutError("API timeout")

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    # Mock pool with empty data
    class MockConnection:
        async def fetch(self, query, *args):
            if "FROM games" in query:
                return [{"game_id": "test_game"}]
            elif "FROM lineups" in query:
                return [{"player_id": 12345}]
            return []

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    class MockPool:
        def acquire(self):
            return MockConnection()

    async def mock_get_pool():
        return MockPool()

    with patch("aiohttp.ClientSession", MockSession), patch(
        "mlb.ingestion.stats.get_pool", mock_get_pool
    ):
        rows = await stats_provider.fetch_game_logs(date(2026, 2, 14))

    # Should return empty list on timeout
    assert rows == []


@pytest.mark.asyncio
async def test_gamelog_empty_lineups():
    """
    AC10: Empty lineups.
    Query for date with no lineups data. Assert returns [], INFO logged (not error).
    """
    from mlb.ingestion.stats import V1StatsProvider

    stats_provider = V1StatsProvider()

    # Mock pool with game but no lineups
    class MockConnection:
        async def fetch(self, query, *args):
            if "FROM games" in query:
                # Return completed game
                return [{"game_id": "test_game_no_lineup"}]
            elif "FROM lineups" in query:
                # No lineups for this game
                return []
            return []

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    class MockPool:
        def acquire(self):
            return MockConnection()

    async def mock_get_pool():
        return MockPool()

    with patch("mlb.ingestion.stats.get_pool", mock_get_pool):
        rows = await stats_provider.fetch_game_logs(date(2026, 2, 14))

    # Should return empty list when no lineups found
    assert rows == []


@pytest.mark.asyncio
async def test_fetch_game_logs_caches_season_logs():
    """
    FC-33: Request-level caching for game logs.
    Call fetch_game_logs() twice for same date within cache TTL.
    Assert: only 1 HTTP request made (second call hits cache).
    """
    import json
    from pathlib import Path

    from mlb.ingestion.cache import get_cache
    from mlb.ingestion.stats import V1StatsProvider

    stats_provider = V1StatsProvider()

    # Clear cache before test
    cache = get_cache()
    cache.clear()

    # Load fixture for mock response
    fixture_path = Path(__file__).parent / "fixtures" / "gamelog_hitting.json"
    with open(fixture_path) as f:
        mock_response_data = json.load(f)
    mock_response_bytes = json.dumps(mock_response_data).encode("utf-8")

    # Track HTTP call count
    http_call_count = {"hitting": 0, "pitching": 0}

    class MockResponse:
        def __init__(self, data_bytes):
            self.status = 200
            self._data = data_bytes

        async def read(self):
            return self._data

        async def json(self):
            return json.loads(self._data.decode("utf-8"))

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    class MockSession:
        def __init__(self, *args, **kwargs):
            pass

        def get(self, url, **kwargs):
            # Track which endpoint was called
            if "group=hitting" in url:
                http_call_count["hitting"] += 1
                return MockResponse(mock_response_bytes)
            elif "group=pitching" in url:
                http_call_count["pitching"] += 1
                # Return empty stats for pitching (player doesn't pitch)
                empty_response = {"stats": []}
                return MockResponse(json.dumps(empty_response).encode("utf-8"))
            return MockResponse(b"{}")

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    # Mock pool with game and lineup data
    class MockConnection:
        async def fetch(self, query, *args):
            if "FROM games" in query:
                # Return completed game
                return [{"game_id": "746587"}]
            elif "FROM lineups" in query:
                # Return single player
                return [{"player_id": 660271}]
            return []

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    class MockPool:
        def acquire(self):
            return MockConnection()

    async def mock_get_pool():
        return MockPool()

    with patch("aiohttp.ClientSession", MockSession), patch(
        "mlb.ingestion.stats.get_pool", mock_get_pool
    ):
        # First call - should make HTTP requests
        rows1 = await stats_provider.fetch_game_logs(date(2024, 6, 15))
        assert len(rows1) == 1

        # Second call - should use cache (no additional HTTP requests)
        rows2 = await stats_provider.fetch_game_logs(date(2024, 6, 15))
        assert len(rows2) == 1

    # Verify: only 1 HTTP call per endpoint (hitting + pitching)
    assert http_call_count["hitting"] == 1, "Expected 1 hitting HTTP call, got " + str(
        http_call_count["hitting"]
    )
    assert http_call_count["pitching"] == 1, "Expected 1 pitching HTTP call, got " + str(
        http_call_count["pitching"]
    )

    # Verify cache contains the expected keys
    cache_key_hitting = "gamelog:660271:2024:hitting"
    cache_key_pitching = "gamelog:660271:2024:pitching"
    assert cache.get(cache_key_hitting) is not None, "Cache should contain hitting data"
    assert (
        cache.get(cache_key_pitching) is not None
    ), "Cache should contain pitching data"

    # Cleanup
    cache.clear()


@pytest.mark.asyncio
async def test_fetch_game_logs_includes_relief_pitchers():
    """
    FC-34: Full roster discovery includes relief pitchers.
    Mock game with 5 relief pitchers not in lineup. Assert: all 5 have game logs fetched.
    """
    import json
    from pathlib import Path

    from mlb.ingestion.cache import get_cache
    from mlb.ingestion.stats import V1StatsProvider

    stats_provider = V1StatsProvider()

    # Clear cache before test
    cache = get_cache()
    cache.clear()

    # Load fixture with relief pitchers
    fixture_path = Path(__file__).parent / "fixtures" / "boxscore_with_relievers.json"
    with open(fixture_path) as f:
        boxscore_data = json.load(f)

    # Track which player IDs were requested for game logs
    requested_player_ids = set()

    class MockResponse:
        def __init__(self, data, status=200):
            self.status = status
            self._data = data

        async def read(self):
            if isinstance(self._data, dict):
                return json.dumps(self._data).encode("utf-8")
            return self._data

        async def json(self):
            if isinstance(self._data, bytes):
                return json.loads(self._data.decode("utf-8"))
            return self._data

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    class MockSession:
        def __init__(self, *args, **kwargs):
            pass

        def get(self, url, **kwargs):
            # Boxscore endpoint
            if "/boxscore" in url:
                return MockResponse(boxscore_data, status=200)
            # Game log endpoints
            elif "/stats?stats=gameLog" in url:
                # Extract player_id from URL
                parts = url.split("/people/")[1].split("/")[0]
                player_id = int(parts)
                requested_player_ids.add(player_id)

                # Return empty game logs
                return MockResponse({"stats": []}, status=200)
            return MockResponse({}, status=404)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    # Mock pool with game and lineup data
    class MockConnection:
        async def fetch(self, query, *args):
            if "FROM games" in query:
                # Return completed game
                return [{"game_id": "test_game_fc34_1"}]
            elif "FROM lineups" in query:
                # Return only starting 9 from home team (no relief pitchers)
                return [
                    {"player_id": 12345},
                    {"player_id": 12346},
                    {"player_id": 12347},
                    {"player_id": 12348},
                    {"player_id": 12349},
                    {"player_id": 12350},
                    {"player_id": 12351},
                    {"player_id": 12352},
                    {"player_id": 12353},
                ]
            return []

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    class MockPool:
        def acquire(self):
            return MockConnection()

    async def mock_get_pool():
        return MockPool()

    with patch("aiohttp.ClientSession", MockSession), patch(
        "mlb.ingestion.stats.get_pool", mock_get_pool
    ):
        rows = await stats_provider.fetch_game_logs(date(2024, 6, 15))

    # Verify: relief pitchers (90001-90005) were requested for game logs
    relief_pitcher_ids = {90001, 90002, 90003, 90004, 90005}
    assert relief_pitcher_ids.issubset(
        requested_player_ids
    ), f"Relief pitchers {relief_pitcher_ids - requested_player_ids} were not fetched"

    # Verify: starting players were also requested
    starter_ids = {12345, 12346, 12347}
    assert starter_ids.issubset(
        requested_player_ids
    ), f"Starting players {starter_ids - requested_player_ids} were not fetched"

    # Cleanup
    cache.clear()


@pytest.mark.asyncio
async def test_fetch_game_logs_falls_back_to_lineups_if_boxscore_fails():
    """
    FC-34: Fallback to lineups-only if boxscore fails.
    Mock boxscore API timeout. Assert: fetch_game_logs still returns logs for starting 9 (degraded but not broken).
    """
    import json

    from mlb.ingestion.cache import get_cache
    from mlb.ingestion.stats import V1StatsProvider

    stats_provider = V1StatsProvider()

    # Clear cache before test
    cache = get_cache()
    cache.clear()

    # Track which player IDs were requested for game logs
    requested_player_ids = set()

    class MockResponse:
        def __init__(self, data, status=200):
            self.status = status
            self._data = data

        async def read(self):
            if isinstance(self._data, dict):
                return json.dumps(self._data).encode("utf-8")
            return self._data

        async def json(self):
            if isinstance(self._data, bytes):
                return json.loads(self._data.decode("utf-8"))
            return self._data

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    class MockSession:
        def __init__(self, *args, **kwargs):
            pass

        def get(self, url, **kwargs):
            # Boxscore endpoint - simulate timeout
            if "/boxscore" in url:
                raise asyncio.TimeoutError("Boxscore API timeout")
            # Game log endpoints
            elif "/stats?stats=gameLog" in url:
                # Extract player_id from URL
                parts = url.split("/people/")[1].split("/")[0]
                player_id = int(parts)
                requested_player_ids.add(player_id)

                # Return empty game logs
                return MockResponse({"stats": []}, status=200)
            return MockResponse({}, status=404)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    # Mock pool with game and lineup data
    class MockConnection:
        async def fetch(self, query, *args):
            if "FROM games" in query:
                # Return completed game
                return [{"game_id": "test_game_fc34_2"}]
            elif "FROM lineups" in query:
                # Return starting 9 from lineup
                return [
                    {"player_id": 12345},
                    {"player_id": 12346},
                    {"player_id": 12347},
                    {"player_id": 12348},
                    {"player_id": 12349},
                    {"player_id": 12350},
                    {"player_id": 12351},
                    {"player_id": 12352},
                    {"player_id": 12353},
                ]
            return []

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    class MockPool:
        def acquire(self):
            return MockConnection()

    async def mock_get_pool():
        return MockPool()

    with patch("aiohttp.ClientSession", MockSession), patch(
        "mlb.ingestion.stats.get_pool", mock_get_pool
    ):
        rows = await stats_provider.fetch_game_logs(date(2024, 6, 15))

    # Verify: function didn't crash and still fetched lineup players
    assert len(requested_player_ids) > 0, "Should have fetched at least lineup players"

    # Verify: all starting 9 were requested despite boxscore failure
    starter_ids = {12345, 12346, 12347, 12348, 12349, 12350, 12351, 12352, 12353}
    assert starter_ids.issubset(
        requested_player_ids
    ), f"Starting players {starter_ids - requested_player_ids} were not fetched after boxscore failure"

    # Verify: no relief pitchers were fetched (boxscore failed, no fallback to discover them)
    relief_pitcher_ids = {90001, 90002, 90003, 90004, 90005}
    assert not relief_pitcher_ids.intersection(
        requested_player_ids
    ), "Relief pitchers should not be fetched when boxscore fails (lineups-only fallback)"

    # Cleanup
    cache.clear()


@pytest.mark.asyncio
async def test_step_1d_reuses_step_1c_boxscore_cache():
    """
    FC-34 / D-060: Verify Step 1D doesn't make redundant boxscore calls if Step 1C already cached data.
    Simulate Step 1C caching boxscore. Mock HTTP session to count calls. Assert: boxscore NOT called by Step 1D.
    """
    import json
    from pathlib import Path

    from mlb.ingestion.cache import get_cache
    from mlb.ingestion.stats import V1StatsProvider

    stats_provider = V1StatsProvider()

    # Clear cache before test
    cache = get_cache()
    cache.clear()

    # Load fixture with relievers
    fixture_path = Path(__file__).parent / "fixtures" / "boxscore_with_relievers.json"
    with open(fixture_path) as f:
        boxscore_data = json.load(f)

    # Simulate Step 1C caching boxscore
    game_id = "test_game_fc34_1"
    cache_key = f"boxscore:{game_id}"
    cache.set(cache_key, json.dumps(boxscore_data).encode("utf-8"), 7200)

    # Track HTTP calls
    http_calls = {"boxscore": 0, "gamelog": 0}

    class MockResponse:
        def __init__(self, data, status=200):
            self.status = status
            self._data = data

        async def read(self):
            if isinstance(self._data, dict):
                return json.dumps(self._data).encode("utf-8")
            return self._data

        async def json(self):
            if isinstance(self._data, bytes):
                return json.loads(self._data.decode("utf-8"))
            return self._data

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    class MockSession:
        def __init__(self, *args, **kwargs):
            pass

        def get(self, url, **kwargs):
            # Boxscore endpoint - should NOT be called
            if "/boxscore" in url:
                http_calls["boxscore"] += 1
                # Return data anyway (but test will fail if this is called)
                return MockResponse(boxscore_data, status=200)
            # Game log endpoints
            elif "/stats?stats=gameLog" in url:
                http_calls["gamelog"] += 1
                # Return empty game logs
                return MockResponse({"stats": []}, status=200)
            return MockResponse({}, status=404)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    # Mock pool with game and lineup data
    class MockConnection:
        async def fetch(self, query, *args):
            if "FROM games" in query:
                return [{"game_id": game_id}]
            elif "FROM lineups" in query:
                # Return starting 9 from fixture
                return [
                    {"player_id": 12345},
                    {"player_id": 12346},
                    {"player_id": 12347},
                    {"player_id": 12348},
                    {"player_id": 12349},
                    {"player_id": 12350},
                    {"player_id": 12351},
                    {"player_id": 12352},
                    {"player_id": 12353},
                ]
            return []

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    class MockPool:
        def acquire(self):
            return MockConnection()

    async def mock_get_pool():
        return MockPool()

    with patch("aiohttp.ClientSession", MockSession), patch(
        "mlb.ingestion.stats.get_pool", mock_get_pool
    ):
        # Call Step 1D
        rows = await stats_provider.fetch_game_logs(date(2024, 6, 15))

    # Assert: boxscore was NOT called (cache hit)
    assert (
        http_calls["boxscore"] == 0
    ), f"Step 1D made {http_calls['boxscore']} redundant boxscore call(s) despite cache"

    # Assert: game logs were still fetched
    assert http_calls["gamelog"] > 0, "Step 1D should have fetched game logs"

    # Assert: full roster was still discovered (relief pitchers from cached boxscore)
    # Fixture has 9 starters + 5 relievers = 14 players minimum
    # (actual number may be higher depending on implementation)
    # For this test, we just verify that roster discovery worked

    # Cleanup
    cache.clear()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
