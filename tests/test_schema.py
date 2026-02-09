"""Tests for database schema and migrations."""

from decimal import Decimal

import pytest
import asyncpg

from mlb.db.schema import migrate, schema_version


@pytest.fixture(scope="function")
async def clean_db(pool):
    """Clean database before each test."""
    async with pool.acquire() as conn:
        # Drop all tables
        await conn.execute("""
            DROP SCHEMA public CASCADE;
            CREATE SCHEMA public;
        """)
    yield
    # Cleanup after test
    async with pool.acquire() as conn:
        await conn.execute("""
            DROP SCHEMA public CASCADE;
            CREATE SCHEMA public;
        """)


class TestMigrations:
    """Test migration functionality."""

    @pytest.mark.asyncio
    async def test_migrate_fresh_database(self, pool, clean_db):
        """Test running migrations on a fresh database."""
        applied = await migrate()
        assert applied == 4, "Should apply 4 migrations (001, 002, 003, and 004)"

        version = await schema_version()
        assert version == 4, "Schema version should be 4"

    @pytest.mark.asyncio
    async def test_migrate_idempotent(self, pool, clean_db):
        """Test that re-running migrations is idempotent."""
        # First run
        applied1 = await migrate()
        assert applied1 == 4

        # Second run should apply nothing
        applied2 = await migrate()
        assert applied2 == 0, "Should apply 0 migrations on second run"

        version = await schema_version()
        assert version == 4

    @pytest.mark.asyncio
    async def test_schema_version_before_migrations(self, pool, clean_db):
        """Test schema_version() before any migrations."""
        version = await schema_version()
        assert version is None, "Schema version should be None before migrations"

    @pytest.mark.asyncio
    async def test_schema_version_after_migrations(self, pool, clean_db):
        """Test schema_version() after migrations."""
        await migrate()
        version = await schema_version()
        assert version == 4


class TestSchema:
    """Test database schema structure."""

    @pytest.fixture(autouse=True)
    async def setup(self, pool, clean_db):
        """Run migrations before each test."""
        await migrate()

    @pytest.mark.asyncio
    async def test_all_tables_exist(self, pool):
        """Test that all required tables exist."""
        expected_tables = {
            "teams", "parks", "games", "players", "lineups",
            "player_game_logs", "odds_snapshots", "projections",
            "sim_market_probs", "player_projections", "eval_results",
            "subscriptions", "weather", "schema_migrations"
        }

        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
            """)
            actual_tables = {row["table_name"] for row in rows}

        assert expected_tables == actual_tables

    @pytest.mark.asyncio
    async def test_foreign_keys_exist(self, pool):
        """Test that all required foreign key relationships exist."""
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT
                    tc.table_name,
                    kcu.column_name,
                    ccu.table_name AS foreign_table_name,
                    ccu.column_name AS foreign_column_name
                FROM information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu
                    ON tc.constraint_name = kcu.constraint_name
                    AND tc.table_schema = kcu.table_schema
                JOIN information_schema.constraint_column_usage AS ccu
                    ON ccu.constraint_name = tc.constraint_name
                    AND ccu.table_schema = tc.table_schema
                WHERE tc.constraint_type = 'FOREIGN KEY'
                    AND tc.table_schema = 'public'
                ORDER BY tc.table_name, kcu.column_name
            """)

        # Build map of foreign keys
        fk_map = {
            (row["table_name"], row["column_name"]): row["foreign_table_name"]
            for row in rows
        }

        # Verify all required foreign key relationships
        # (All 17 FK constraints defined in schema migrations)
        required_fks = {
            ("parks", "team_id"): "teams",
            ("games", "home_team_id"): "teams",
            ("games", "away_team_id"): "teams",
            ("games", "park_id"): "parks",
            ("players", "team_id"): "teams",
            ("lineups", "game_id"): "games",
            ("lineups", "player_id"): "players",
            ("lineups", "team_id"): "teams",
            ("player_game_logs", "player_id"): "players",
            ("player_game_logs", "game_id"): "games",
            ("odds_snapshots", "game_id"): "games",
            ("projections", "game_id"): "games",
            ("sim_market_probs", "projection_id"): "projections",
            ("player_projections", "projection_id"): "projections",
            ("player_projections", "player_id"): "players",
            ("player_projections", "game_id"): "games",
            ("weather", "game_id"): "games",
        }

        for (table, column), foreign_table in required_fks.items():
            assert fk_map.get((table, column)) == foreign_table, \
                f"Missing or incorrect FK: {table}.{column} -> {foreign_table}"

    @pytest.mark.asyncio
    async def test_unique_constraints_exist(self, pool):
        """Test that unique constraints exist."""
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT
                    tc.table_name,
                    tc.constraint_name,
                    kcu.column_name
                FROM information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu
                    ON tc.constraint_name = kcu.constraint_name
                    AND tc.table_schema = kcu.table_schema
                WHERE tc.constraint_type = 'UNIQUE'
                    AND tc.table_schema = 'public'
                ORDER BY tc.table_name, kcu.column_name
            """)

        constraint_map = {}
        for row in rows:
            key = (row["table_name"], row["constraint_name"])
            if key not in constraint_map:
                constraint_map[key] = []
            constraint_map[key].append(row["column_name"])

        # Check for specific unique constraints
        team_abbr_unique = any(
            cols == ["abbr"] for (table, _), cols in constraint_map.items()
            if table == "teams"
        )
        assert team_abbr_unique, "teams.abbr should have UNIQUE constraint"

        # Check multi-column unique on player_game_logs
        player_game_unique = any(
            set(cols) == {"player_id", "game_id"}
            for (table, _), cols in constraint_map.items()
            if table == "player_game_logs"
        )
        assert player_game_unique, "player_game_logs (player_id, game_id) should be UNIQUE"

    @pytest.mark.asyncio
    async def test_indexes_exist(self, pool):
        """Test that required indexes exist."""
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT
                    schemaname,
                    tablename,
                    indexname
                FROM pg_indexes
                WHERE schemaname = 'public'
                ORDER BY tablename, indexname
            """)

        index_names = {row["indexname"] for row in rows}
        assert "idx_odds_game_market" in index_names


class TestSeedData:
    """Test seeded reference data."""

    @pytest.fixture(autouse=True)
    async def setup(self, pool, clean_db):
        """Run migrations before each test."""
        await migrate()

    @pytest.mark.asyncio
    async def test_teams_seeded(self, pool):
        """Test that 30 teams are seeded."""
        async with pool.acquire() as conn:
            count = await conn.fetchval("SELECT COUNT(*) FROM teams")
            assert count == 30, f"Expected 30 teams, found {count}"

            # Check specific teams exist
            yankees = await conn.fetchrow(
                "SELECT * FROM teams WHERE abbr = 'NYY'"
            )
            assert yankees is not None
            assert yankees["name"] == "New York Yankees"
            assert yankees["league"] == "AL"
            assert yankees["division"] == "East"

            dodgers = await conn.fetchrow(
                "SELECT * FROM teams WHERE abbr = 'LAD'"
            )
            assert dodgers is not None
            assert dodgers["name"] == "Los Angeles Dodgers"
            assert dodgers["league"] == "NL"
            assert dodgers["division"] == "West"

    @pytest.mark.asyncio
    async def test_parks_seeded(self, pool):
        """Test that 30 parks are seeded with park factors."""
        async with pool.acquire() as conn:
            count = await conn.fetchval("SELECT COUNT(*) FROM parks")
            assert count == 30, f"Expected 30 parks, found {count}"

            # Check Coors Field (known hitter-friendly park)
            coors = await conn.fetchrow(
                "SELECT * FROM parks WHERE name = 'Coors Field'"
            )
            assert coors is not None
            assert coors["is_outdoor"] is True
            assert coors["is_retractable"] is False
            assert coors["park_factor"] == Decimal("1.200")

            # Check Oracle Park (known pitcher-friendly park)
            oracle = await conn.fetchrow(
                "SELECT * FROM parks WHERE name = 'Oracle Park'"
            )
            assert oracle is not None
            assert oracle["park_factor"] == Decimal("0.920")

            # Check retractable roof park
            minute_maid = await conn.fetchrow(
                "SELECT * FROM parks WHERE name = 'Minute Maid Park'"
            )
            assert minute_maid is not None
            assert minute_maid["is_retractable"] is True
            assert minute_maid["park_factor"] == Decimal("1.000")

            # All parks should have park_factor
            null_factors = await conn.fetchval(
                "SELECT COUNT(*) FROM parks WHERE park_factor IS NULL"
            )
            assert null_factors == 0


class TestColumnTypes:
    """Test that columns have correct types and constraints."""

    @pytest.fixture(autouse=True)
    async def setup(self, pool, clean_db):
        """Run migrations before each test."""
        await migrate()

    @pytest.mark.asyncio
    async def test_timestamp_defaults(self, pool):
        """Test that created_at and updated_at have defaults."""
        async with pool.acquire() as conn:
            # Insert a game without timestamps
            await conn.execute("""
                INSERT INTO games (game_id, game_date, status)
                VALUES ('test_game_1', '2024-04-01', 'scheduled')
            """)

            row = await conn.fetchrow(
                "SELECT created_at, updated_at FROM games WHERE game_id = 'test_game_1'"
            )

            assert row["created_at"] is not None
            assert row["updated_at"] is not None

    @pytest.mark.asyncio
    async def test_numeric_precision(self, pool):
        """Test that numeric columns have correct precision."""
        async with pool.acquire() as conn:
            # Check park_factor precision
            result = await conn.fetchrow("""
                SELECT
                    numeric_precision,
                    numeric_scale
                FROM information_schema.columns
                WHERE table_name = 'parks'
                    AND column_name = 'park_factor'
            """)

            assert result["numeric_precision"] == 5
            assert result["numeric_scale"] == 3

    @pytest.mark.asyncio
    async def test_nullable_fields(self, pool):
        """Test that nullable fields are correctly defined."""
        async with pool.acquire() as conn:
            # first_pitch should be nullable (TBD games)
            nullable = await conn.fetchval("""
                SELECT is_nullable
                FROM information_schema.columns
                WHERE table_name = 'games'
                    AND column_name = 'first_pitch'
            """)
            assert nullable == "YES"

            # odds side should be nullable (full-game totals)
            nullable = await conn.fetchval("""
                SELECT is_nullable
                FROM information_schema.columns
                WHERE table_name = 'odds_snapshots'
                    AND column_name = 'side'
            """)
            assert nullable == "YES"

    @pytest.mark.asyncio
    async def test_odds_price_check_constraint(self, pool):
        """Test that odds_snapshots.price CHECK constraint enforces >= 1.0."""
        async with pool.acquire() as conn:
            # Insert a test game to satisfy foreign key constraint
            await conn.execute("""
                INSERT INTO games (game_id, game_date, status)
                VALUES ('test_game_fc01', '2024-04-01', 'scheduled')
            """)

            # Test 1: INSERT with price = 0.5 should be rejected
            with pytest.raises(asyncpg.CheckViolationError):
                await conn.execute("""
                    INSERT INTO odds_snapshots
                    (game_id, book, market, price, snapshot_ts)
                    VALUES ('test_game_fc01', 'DraftKings', 'ml', 0.5, now())
                """)

            # Test 2: INSERT with price = 1.91 should succeed
            await conn.execute("""
                INSERT INTO odds_snapshots
                (game_id, book, market, price, snapshot_ts)
                VALUES ('test_game_fc01', 'DraftKings', 'ml', 1.91, now())
            """)

            # Verify the valid insert succeeded
            count = await conn.fetchval("""
                SELECT COUNT(*) FROM odds_snapshots
                WHERE game_id = 'test_game_fc01' AND price = 1.91
            """)
            assert count == 1

    @pytest.mark.asyncio
    async def test_lineup_confirmed_unique_index(self, pool):
        """Test that partial unique index enforces one confirmed lineup per slot."""
        async with pool.acquire() as conn:
            # Insert test game, team, and player
            await conn.execute("""
                INSERT INTO games (game_id, game_date, status)
                VALUES ('test_game_fc03', '2024-04-01', 'scheduled')
            """)

            await conn.execute("""
                INSERT INTO teams (team_id, abbr, name, league, division)
                VALUES (999, 'TST', 'Test Team', 'AL', 'East')
            """)

            await conn.execute("""
                INSERT INTO players (player_id, name, team_id)
                VALUES (9001, 'Test Player 1', 999),
                       (9002, 'Test Player 2', 999)
            """)

            # Test 1: Two unconfirmed lineups for same slot should succeed
            await conn.execute("""
                INSERT INTO lineups
                (game_id, team_id, player_id, batting_order, is_confirmed, source_ts)
                VALUES
                ('test_game_fc03', 999, 9001, 1, FALSE, '2024-04-01 10:00:00'),
                ('test_game_fc03', 999, 9002, 1, FALSE, '2024-04-01 11:00:00')
            """)

            unconfirmed_count = await conn.fetchval("""
                SELECT COUNT(*) FROM lineups
                WHERE game_id = 'test_game_fc03' AND batting_order = 1
                AND is_confirmed = FALSE
            """)
            assert unconfirmed_count == 2, "Two unconfirmed lineups should be allowed"

            # Test 2: First confirmed lineup should succeed
            await conn.execute("""
                INSERT INTO lineups
                (game_id, team_id, player_id, batting_order, is_confirmed, source_ts)
                VALUES ('test_game_fc03', 999, 9001, 2, TRUE, '2024-04-01 12:00:00')
            """)

            # Test 3: Second confirmed lineup for same slot should be rejected
            with pytest.raises(asyncpg.UniqueViolationError):
                await conn.execute("""
                    INSERT INTO lineups
                    (game_id, team_id, player_id, batting_order, is_confirmed, source_ts)
                    VALUES ('test_game_fc03', 999, 9002, 2, TRUE, '2024-04-01 13:00:00')
                """)

    @pytest.mark.asyncio
    async def test_updated_at_trigger(self, pool):
        """Test that updated_at trigger fires on UPDATE for mutable tables."""
        async with pool.acquire() as conn:
            # Test games table
            await conn.execute("""
                INSERT INTO games (game_id, game_date, status)
                VALUES ('test_game_fc04', '2024-04-01', 'scheduled')
            """)

            row = await conn.fetchrow("""
                SELECT created_at, updated_at
                FROM games
                WHERE game_id = 'test_game_fc04'
            """)

            created_at = row["created_at"]
            initial_updated_at = row["updated_at"]

            # Small delay to ensure timestamp difference
            await conn.execute("SELECT pg_sleep(0.01)")

            # Update the game status
            await conn.execute("""
                UPDATE games
                SET status = 'final'
                WHERE game_id = 'test_game_fc04'
            """)

            row = await conn.fetchrow("""
                SELECT created_at, updated_at
                FROM games
                WHERE game_id = 'test_game_fc04'
            """)

            new_updated_at = row["updated_at"]

            # Verify updated_at was updated by trigger
            assert new_updated_at > created_at, "updated_at should be after created_at"
            assert new_updated_at > initial_updated_at, "updated_at should be newer after UPDATE"

            # Test subscriptions table
            await conn.execute("""
                INSERT INTO subscriptions (discord_user_id, tier, status)
                VALUES ('test_user_fc04', 'free', 'active')
            """)

            row = await conn.fetchrow("""
                SELECT created_at, updated_at
                FROM subscriptions
                WHERE discord_user_id = 'test_user_fc04'
            """)

            created_at = row["created_at"]
            initial_updated_at = row["updated_at"]

            # Small delay
            await conn.execute("SELECT pg_sleep(0.01)")

            await conn.execute("""
                UPDATE subscriptions
                SET tier = 'paid'
                WHERE discord_user_id = 'test_user_fc04'
            """)

            row = await conn.fetchrow("""
                SELECT created_at, updated_at
                FROM subscriptions
                WHERE discord_user_id = 'test_user_fc04'
            """)

            new_updated_at = row["updated_at"]

            assert new_updated_at > created_at
            assert new_updated_at > initial_updated_at

    @pytest.mark.asyncio
    async def test_sim_market_probs_edge_computed_column(self, pool):
        """Test that sim_market_probs.edge_computed_at column exists and is nullable TIMESTAMPTZ."""
        async with pool.acquire() as conn:
            # Check column exists and has correct type
            result = await conn.fetchrow("""
                SELECT
                    column_name,
                    data_type,
                    is_nullable
                FROM information_schema.columns
                WHERE table_name = 'sim_market_probs'
                    AND column_name = 'edge_computed_at'
            """)

            assert result is not None, "edge_computed_at column should exist"
            assert result["data_type"] == "timestamp with time zone", \
                "edge_computed_at should be TIMESTAMPTZ"
            assert result["is_nullable"] == "YES", \
                "edge_computed_at should be nullable"

    @pytest.mark.asyncio
    async def test_updated_at_trigger_players(self, pool):
        """Test that updated_at trigger fires on UPDATE for players table."""
        async with pool.acquire() as conn:
            # Insert a player
            await conn.execute("""
                INSERT INTO players (player_id, name, team_id, position)
                VALUES (8888, 'Test Player FC08', 147, 'SS')
            """)

            row = await conn.fetchrow("""
                SELECT created_at, updated_at
                FROM players
                WHERE player_id = 8888
            """)

            created_at = row["created_at"]
            initial_updated_at = row["updated_at"]

            # Small delay to ensure timestamp difference
            await conn.execute("SELECT pg_sleep(0.01)")

            # Update team_id (simulating mid-season trade)
            await conn.execute("""
                UPDATE players
                SET team_id = 121
                WHERE player_id = 8888
            """)

            row = await conn.fetchrow("""
                SELECT created_at, updated_at
                FROM players
                WHERE player_id = 8888
            """)

            new_updated_at = row["updated_at"]

            # Verify updated_at was updated by trigger
            assert new_updated_at > created_at, "updated_at should be after created_at"
            assert new_updated_at > initial_updated_at, "updated_at should be newer after UPDATE"
