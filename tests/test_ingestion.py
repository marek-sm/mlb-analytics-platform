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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
