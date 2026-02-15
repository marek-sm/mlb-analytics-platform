"""Tests for Unit 4: Team Run-Scoring Models.

This test suite validates all acceptance criteria from the mini-spec:
1. build_game_features() returns fully populated GameFeatures
2. train() completes on ≥30 games and serializes models
3. predict() returns valid TeamRunParams
4. Park factor test (D-010): Coors vs neutral park
5. Shrinkage test: low-IP vs high-IP pitcher ERA estimates
6. Weather None handling: dome parks return None weather fields
7. Model artifacts can be loaded and produce identical output
"""

import numpy as np
import pytest
import pytest_asyncio
from datetime import date, timedelta

from mlb.db.models import Table
from mlb.models.features import build_game_features, GameFeatures, _get_pitcher_features
from mlb.models.team_runs import train, predict, TeamRunParams


# Test data constants (using actual seeded team/park IDs from Unit 2)
TEST_HOME_TEAM_ID = 147  # Yankees
TEST_AWAY_TEAM_ID = 139  # Rays
TEST_PARK_ID = 3313  # Yankee Stadium
TEST_PARK_COORS_ID = 19  # Coors Field (from seed data)
TEST_PARK_DOME_ID = 3394  # Tropicana Field (dome)


@pytest_asyncio.fixture
async def sample_game_data(pool):
    """Insert sample data for a game with confirmed lineups and weather."""
    game_id = "test_game_001"
    game_date = date(2026, 6, 15)

    async with pool.acquire() as conn:
        # Insert game (upsert to handle re-runs)
        await conn.execute(
            f"""
            INSERT INTO {Table.GAMES} (game_id, game_date, home_team_id, away_team_id,
                                         park_id, status, home_score, away_score)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (game_id) DO UPDATE SET
                home_score = EXCLUDED.home_score,
                away_score = EXCLUDED.away_score
            """,
            game_id,
            game_date,
            TEST_HOME_TEAM_ID,
            TEST_AWAY_TEAM_ID,
            TEST_PARK_ID,
            'final',
            5,
            3,
        )

        # Insert weather (outdoor park) - use fixed timestamp to avoid conflicts
        await conn.execute(
            f"""
            INSERT INTO {Table.WEATHER} (game_id, temp_f, wind_speed_mph, wind_dir,
                                          precip_pct, fetched_at)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (game_id, fetched_at) DO UPDATE SET
                temp_f = EXCLUDED.temp_f
            """,
            game_id,
            75,
            10,
            'NW',
            5,
            game_date,  # Use game_date as fixed timestamp
        )

        # Insert starting pitchers (players 101, 102)
        for player_id in [101, 102]:
            await conn.execute(
                f"""
                INSERT INTO {Table.PLAYERS} (player_id, name, team_id, position,
                                              bats, throws)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (player_id) DO NOTHING
                """,
                player_id,
                f"Pitcher {player_id}",
                TEST_HOME_TEAM_ID if player_id == 101 else TEST_AWAY_TEAM_ID,
                'P',
                'R',
                'R',
            )

        # Insert confirmed lineups (batting_order 1-9 for each team)
        for i in range(1, 10):
            # Home team lineup (team_id=TEST_HOME_TEAM_ID)
            player_id = 200 + i
            await conn.execute(
                f"""
                INSERT INTO {Table.PLAYERS} (player_id, name, team_id, position,
                                              bats, throws)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (player_id) DO NOTHING
                """,
                player_id,
                f"Batter {player_id}",
                TEST_HOME_TEAM_ID,
                'CF',
                'R',
                'R',
            )
            # Delete existing confirmed lineup for this slot
            await conn.execute(
                f"""
                DELETE FROM {Table.LINEUPS}
                WHERE game_id = $1 AND team_id = $2 AND batting_order = $3
                  AND is_confirmed = TRUE
                """,
                game_id,
                TEST_HOME_TEAM_ID,
                i,
            )
            await conn.execute(
                f"""
                INSERT INTO {Table.LINEUPS} (game_id, team_id, player_id, batting_order,
                                              is_confirmed, source_ts)
                VALUES ($1, $2, $3, $4, $5, NOW())
                """,
                game_id,
                TEST_HOME_TEAM_ID,
                player_id,
                i,
                True,
            )

            # Away team lineup (team_id=TEST_AWAY_TEAM_ID)
            player_id = 300 + i
            await conn.execute(
                f"""
                INSERT INTO {Table.PLAYERS} (player_id, name, team_id, position,
                                              bats, throws)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (player_id) DO NOTHING
                """,
                player_id,
                f"Batter {player_id}",
                TEST_AWAY_TEAM_ID,
                'CF',
                'L',
                'L',
            )
            # Delete existing confirmed lineup for this slot
            await conn.execute(
                f"""
                DELETE FROM {Table.LINEUPS}
                WHERE game_id = $1 AND team_id = $2 AND batting_order = $3
                  AND is_confirmed = TRUE
                """,
                game_id,
                TEST_AWAY_TEAM_ID,
                i,
            )
            await conn.execute(
                f"""
                INSERT INTO {Table.LINEUPS} (game_id, team_id, player_id, batting_order,
                                              is_confirmed, source_ts)
                VALUES ($1, $2, $3, $4, $5, NOW())
                """,
                game_id,
                TEST_AWAY_TEAM_ID,
                player_id,
                i,
                True,
            )

        # Add starting pitchers to lineups (batting_order 1 for simplicity)
        await conn.execute(
            f"""
            DELETE FROM {Table.LINEUPS}
            WHERE game_id = $1 AND team_id = $2 AND batting_order = $3
            """,
            game_id,
            TEST_HOME_TEAM_ID,
            1,
        )
        await conn.execute(
            f"""
            INSERT INTO {Table.LINEUPS} (game_id, team_id, player_id, batting_order,
                                          is_confirmed, source_ts)
            VALUES ($1, $2, $3, $4, $5, NOW())
            """,
            game_id,
            TEST_HOME_TEAM_ID,
            101,
            1,
            True,
        )

        await conn.execute(
            f"""
            DELETE FROM {Table.LINEUPS}
            WHERE game_id = $1 AND team_id = $2 AND batting_order = $3
            """,
            game_id,
            TEST_AWAY_TEAM_ID,
            1,
        )
        await conn.execute(
            f"""
            INSERT INTO {Table.LINEUPS} (game_id, team_id, player_id, batting_order,
                                          is_confirmed, source_ts)
            VALUES ($1, $2, $3, $4, $5, NOW())
            """,
            game_id,
            TEST_AWAY_TEAM_ID,
            102,
            1,
            True,
        )

        # Insert pitcher game logs (recent starts)
        for days_ago in [5, 10, 15, 20]:
            past_date = game_date - timedelta(days=days_ago)
            past_game_id = f"past_game_{days_ago}"

            await conn.execute(
                f"""
                INSERT INTO {Table.GAMES} (game_id, game_date, home_team_id, away_team_id,
                                            park_id, status)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (game_id) DO NOTHING
                """,
                past_game_id,
                past_date,
                TEST_HOME_TEAM_ID,
                TEST_AWAY_TEAM_ID,
                TEST_PARK_ID,
                'final',
            )

            # Pitcher 101: 21 outs (7 IP), 3 ER, 95 pitches
            await conn.execute(
                f"""
                INSERT INTO {Table.PLAYER_GAME_LOGS} (player_id, game_id, ip_outs, er,
                                                        pitch_count, is_starter)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (player_id, game_id) DO NOTHING
                """,
                101,
                past_game_id,
                21,
                3,
                95,
                True,
            )

            # Pitcher 102: 18 outs (6 IP), 4 ER, 90 pitches
            await conn.execute(
                f"""
                INSERT INTO {Table.PLAYER_GAME_LOGS} (player_id, game_id, ip_outs, er,
                                                        pitch_count, is_starter)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (player_id, game_id) DO NOTHING
                """,
                102,
                past_game_id,
                18,
                4,
                90,
                True,
            )

        # Insert batting stats for lineup players
        for player_id in list(range(201, 210)) + list(range(301, 310)):
            for days_ago in range(1, 30, 3):  # 10 recent games
                past_date = game_date - timedelta(days=days_ago)
                past_game_id = f"bat_game_{player_id}_{days_ago}"

                # Create game if not exists
                await conn.execute(
                    f"""
                    INSERT INTO {Table.GAMES} (game_id, game_date, home_team_id,
                                                away_team_id, park_id, status)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (game_id) DO NOTHING
                    """,
                    past_game_id,
                    past_date,
                    TEST_HOME_TEAM_ID,
                    TEST_AWAY_TEAM_ID,
                    TEST_PARK_ID,
                    'final',
                )

                # Insert batting stats: 4 PA, 3 AB, 1 H, 2 TB, 1 BB
                await conn.execute(
                    f"""
                    INSERT INTO {Table.PLAYER_GAME_LOGS} (player_id, game_id, pa, ab, h,
                                                            tb, bb, is_starter)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (player_id, game_id) DO NOTHING
                    """,
                    player_id,
                    past_game_id,
                    4,
                    3,
                    1,
                    2,
                    1,
                    False,
                )

    return game_id


@pytest.mark.asyncio
async def test_build_game_features_returns_populated_features(pool, sample_game_data):
    """AC1: build_game_features() returns fully populated GameFeatures."""
    game_id = sample_game_data

    async with pool.acquire() as conn:
        features = await build_game_features(conn, game_id)

        assert isinstance(features, GameFeatures)
        assert features.game_id == game_id
        assert features.park_factor > 0
        assert features.is_outdoor in [True, False]

        # Weather fields should be populated for outdoor parks
        if features.is_outdoor:
            assert features.temp_f is not None
            assert features.wind_speed_mph is not None
            assert features.wind_dir is not None
            assert features.precip_pct is not None

        # Starter IDs should be set
        assert features.home_starter_id > 0
        assert features.away_starter_id > 0

        # Rest, pitch count, ERA should be reasonable
        assert features.home_starter_rest >= 0
        assert features.away_starter_rest >= 0
        assert features.home_starter_pitch_ct_avg > 0
        assert features.away_starter_pitch_ct_avg > 0
        assert features.home_starter_era_recent > 0
        assert features.away_starter_era_recent > 0

        # Lineup OPS should be in plausible range
        assert 0.3 <= features.home_lineup_ops <= 1.5
        assert 0.3 <= features.away_lineup_ops <= 1.5

        # Bullpen usage and run env should be non-negative
        assert features.home_bullpen_usage >= 0
        assert features.away_bullpen_usage >= 0
        assert features.home_run_env > 0
        assert features.away_run_env > 0


@pytest.mark.asyncio
async def test_weather_none_handling_for_dome_parks(pool):
    """AC6: Weather fields are None for indoor/retractable parks."""
    game_id = "dome_game"
    game_date = date(2026, 6, 15)

    async with pool.acquire() as conn:
        # Insert game at Tropicana Field (dome)
        await conn.execute(
            f"""
            INSERT INTO {Table.GAMES} (game_id, game_date, home_team_id, away_team_id,
                                         park_id, status)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (game_id) DO NOTHING
            """,
            game_id,
            game_date,
            TEST_HOME_TEAM_ID,
            TEST_AWAY_TEAM_ID,
            TEST_PARK_DOME_ID,
            'scheduled',
        )

        # Insert players and lineups
        for team_id in [TEST_HOME_TEAM_ID, TEST_AWAY_TEAM_ID]:
            pitcher_id = 1000 + team_id
            await conn.execute(
                f"""
                INSERT INTO {Table.PLAYERS} (player_id, name, team_id, position,
                                              bats, throws)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (player_id) DO NOTHING
                """,
                pitcher_id,
                f"Pitcher {pitcher_id}",
                team_id,
                'P',
                'R',
                'R',
            )

            # Delete existing confirmed lineup for this slot
            await conn.execute(
                f"""
                DELETE FROM {Table.LINEUPS}
                WHERE game_id = $1 AND team_id = $2 AND batting_order = $3
                  AND is_confirmed = TRUE
                """,
                game_id,
                team_id,
                1,
            )

            await conn.execute(
                f"""
                INSERT INTO {Table.LINEUPS} (game_id, team_id, player_id, batting_order,
                                              is_confirmed, source_ts)
                VALUES ($1, $2, $3, $4, $5, NOW())
                """,
                game_id,
                team_id,
                pitcher_id,
                1,
                True,
            )

        features = await build_game_features(conn, game_id)

        # Verify weather fields are None for retractable park
        assert features.temp_f is None
        assert features.wind_speed_mph is None
        assert features.wind_dir is None
        assert features.precip_pct is None


@pytest.mark.asyncio
async def test_train_completes_and_serializes_models(pool, sample_game_data):
    """AC2: train() completes on ≥30 games and serializes models."""
    async with pool.acquire() as conn:
        # Insert 30 additional games with final scores
        for i in range(30):
            game_id = f"train_game_{i}"
            game_date = date(2026, 6, 1) + timedelta(days=i)

            await conn.execute(
                f"""
                INSERT INTO {Table.GAMES} (game_id, game_date, home_team_id, away_team_id,
                                            park_id, status, home_score, away_score)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (game_id) DO UPDATE SET
                    game_date = EXCLUDED.game_date,
                    home_team_id = EXCLUDED.home_team_id,
                    away_team_id = EXCLUDED.away_team_id,
                    park_id = EXCLUDED.park_id,
                    status = EXCLUDED.status,
                    home_score = EXCLUDED.home_score,
                    away_score = EXCLUDED.away_score
                """,
                game_id,
                game_date,
                TEST_HOME_TEAM_ID,
                TEST_AWAY_TEAM_ID,
                TEST_PARK_ID,
                'final',
                np.random.randint(2, 8),
                np.random.randint(2, 8),
            )

            # Insert minimal lineups and starters
            for team_id in [TEST_HOME_TEAM_ID, TEST_AWAY_TEAM_ID]:
                pitcher_id = 2000 + i * 10 + team_id
                await conn.execute(
                    f"""
                    INSERT INTO {Table.PLAYERS} (player_id, name, team_id, position,
                                                  bats, throws)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (player_id) DO NOTHING
                    """,
                    pitcher_id,
                    f"Pitcher {pitcher_id}",
                    team_id,
                    'P',
                    'R',
                    'R',
                )

                # Delete existing lineup for this slot to prevent duplicates
                await conn.execute(
                    f"""
                    DELETE FROM {Table.LINEUPS}
                    WHERE game_id = $1 AND team_id = $2 AND batting_order = $3
                    """,
                    game_id,
                    team_id,
                    1,
                )

                await conn.execute(
                    f"""
                    INSERT INTO {Table.LINEUPS} (game_id, team_id, player_id, batting_order,
                                                  is_confirmed, source_ts)
                    VALUES ($1, $2, $3, $4, $5, NOW())
                    """,
                    game_id,
                    team_id,
                    pitcher_id,
                    1,
                    True,
                )

        # Train models
        model_version = await train(conn)

        assert model_version is not None
        assert len(model_version) > 0

        # Verify models were saved to disk
        from mlb.models.registry import _get_artifacts_dir

        artifacts_dir = _get_artifacts_dir()
        assert (artifacts_dir / f"home_mu_{model_version}.pkl").exists()
        assert (artifacts_dir / f"away_mu_{model_version}.pkl").exists()
        assert (artifacts_dir / f"home_disp_{model_version}.pkl").exists()
        assert (artifacts_dir / f"away_disp_{model_version}.pkl").exists()


@pytest.mark.asyncio
async def test_predict_returns_valid_team_run_params(pool, sample_game_data):
    """AC3: predict() returns TeamRunParams with valid ranges."""
    async with pool.acquire() as conn:
        # Train models first
        model_version = await train(conn)

        # Run inference
        game_id = sample_game_data
        params = await predict(conn, game_id, model_version)

        assert isinstance(params, TeamRunParams)
        assert params.game_id == game_id
        assert 1.0 <= params.home_mu <= 12.0
        assert 1.0 <= params.away_mu <= 12.0
        assert params.home_disp > 0
        assert params.away_disp > 0
        assert params.model_version == model_version


@pytest.mark.asyncio
async def test_park_factor_applied_exactly_once(pool):
    """AC4: Park factor test (Coors vs neutral park)."""
    game_date = date(2026, 7, 1)

    async with pool.acquire() as conn:
        # Insert 30 training games first (required for train() to work)
        for i in range(30):
            training_game_id = f"park_train_game_{i}"
            training_date = date(2026, 6, 1) + timedelta(days=i)

            await conn.execute(
                f"""
                INSERT INTO {Table.GAMES} (game_id, game_date, home_team_id, away_team_id,
                                            park_id, status, home_score, away_score)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (game_id) DO NOTHING
                """,
                training_game_id,
                training_date,
                TEST_HOME_TEAM_ID,
                TEST_AWAY_TEAM_ID,
                TEST_PARK_ID,
                'final',
                np.random.randint(2, 8),
                np.random.randint(2, 8),
            )

            # Insert minimal lineups for training games
            for team_id in [TEST_HOME_TEAM_ID, TEST_AWAY_TEAM_ID]:
                pitcher_id = 5000 + i * 10 + team_id
                await conn.execute(
                    f"""
                    INSERT INTO {Table.PLAYERS} (player_id, name, team_id, position,
                                                  bats, throws)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (player_id) DO NOTHING
                    """,
                    pitcher_id,
                    f"Train Pitcher {pitcher_id}",
                    team_id,
                    'P',
                    'R',
                    'R',
                )

                # Delete existing confirmed lineup for this slot
                await conn.execute(
                    f"""
                    DELETE FROM {Table.LINEUPS}
                    WHERE game_id = $1 AND team_id = $2 AND batting_order = $3
                      AND is_confirmed = TRUE
                    """,
                    training_game_id,
                    team_id,
                    1,
                )

                await conn.execute(
                    f"""
                    INSERT INTO {Table.LINEUPS} (game_id, team_id, player_id, batting_order,
                                                  is_confirmed, source_ts)
                    VALUES ($1, $2, $3, $4, $5, NOW())
                    """,
                    training_game_id,
                    team_id,
                    pitcher_id,
                    1,
                    True,
                )

        # Create two identical games except park
        coors_game_id = "coors_game"
        neutral_game_id = "neutral_game"

        # Coors Field: park_id=19 (park_factor=1.200)
        await conn.execute(
            f"""
            INSERT INTO {Table.GAMES} (game_id, game_date, home_team_id, away_team_id,
                                         park_id, status, home_score, away_score)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (game_id) DO UPDATE SET
                home_score = EXCLUDED.home_score,
                away_score = EXCLUDED.away_score
            """,
            coors_game_id,
            game_date,
            TEST_HOME_TEAM_ID,
            TEST_AWAY_TEAM_ID,
            TEST_PARK_COORS_ID,
            'final',
            6,
            5,
        )

        # Yankee Stadium: park_id=3313 (park_factor=1.050, close to neutral)
        await conn.execute(
            f"""
            INSERT INTO {Table.GAMES} (game_id, game_date, home_team_id, away_team_id,
                                         park_id, status, home_score, away_score)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (game_id) DO UPDATE SET
                home_score = EXCLUDED.home_score,
                away_score = EXCLUDED.away_score
            """,
            neutral_game_id,
            game_date,
            TEST_HOME_TEAM_ID,
            TEST_AWAY_TEAM_ID,
            TEST_PARK_ID,
            'final',
            5,
            4,
        )

        # Insert identical lineups and starters for both games
        for game_id in [coors_game_id, neutral_game_id]:
            for team_id in [TEST_HOME_TEAM_ID, TEST_AWAY_TEAM_ID]:
                pitcher_id = 3000 + team_id
                await conn.execute(
                    f"""
                    INSERT INTO {Table.PLAYERS} (player_id, name, team_id, position,
                                                  bats, throws)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (player_id) DO NOTHING
                    """,
                    pitcher_id,
                    f"Pitcher {pitcher_id}",
                    team_id,
                    'P',
                    'R',
                    'R',
                )

                # Delete existing confirmed lineup for this slot first
                await conn.execute(
                    f"""
                    DELETE FROM {Table.LINEUPS}
                    WHERE game_id = $1 AND team_id = $2 AND batting_order = $3
                      AND is_confirmed = TRUE
                    """,
                    game_id,
                    team_id,
                    1,
                )

                await conn.execute(
                    f"""
                    INSERT INTO {Table.LINEUPS} (game_id, team_id, player_id, batting_order,
                                                  is_confirmed, source_ts)
                    VALUES ($1, $2, $3, $4, $5, NOW())
                    """,
                    game_id,
                    team_id,
                    pitcher_id,
                    1,
                    True,
                )

        # Train models
        model_version = await train(conn)

        # Predict for both parks
        coors_params = await predict(conn, coors_game_id, model_version)
        neutral_params = await predict(conn, neutral_game_id, model_version)

        # Verify park factor effect: Coors should be ~1.2× neutral
        # Allow some tolerance due to model variance
        ratio = coors_params.home_mu / neutral_params.home_mu
        assert 1.1 <= ratio <= 1.3, f"Coors/neutral ratio {ratio:.2f} out of range"


@pytest.mark.asyncio
async def test_shrinkage_for_low_vs_high_ip_pitchers(pool):
    """AC5: Shrinkage test: 10 IP vs 150 IP pitcher ERA estimates."""
    game_date = date(2026, 7, 15)
    league_mean_era = 4.50

    async with pool.acquire() as conn:
        # Create two pitchers: one with 10 IP, one with 150 IP
        low_ip_pitcher_id = 4001
        high_ip_pitcher_id = 4002

        await conn.execute(
            f"""
            INSERT INTO {Table.PLAYERS} (player_id, name, team_id, position,
                                          bats, throws)
            VALUES ($1, $2, $3, $4, $5, $6),
                   ($7, $8, $9, $10, $11, $12)
            ON CONFLICT (player_id) DO NOTHING
            """,
            low_ip_pitcher_id,
            'Low IP Pitcher',
            TEST_HOME_TEAM_ID,
            'P',
            'R',
            'R',
            high_ip_pitcher_id,
            'High IP Pitcher',
            TEST_AWAY_TEAM_ID,
            'P',
            'R',
            'R',
        )

        # Low IP pitcher: 10 IP (30 outs), 2 ER → raw ERA = 1.80
        for i in range(2):  # 2 starts × 5 IP each
            past_game_id = f"low_ip_game_{i}"
            past_date = game_date - timedelta(days=10 + i * 5)

            await conn.execute(
                f"""
                INSERT INTO {Table.GAMES} (game_id, game_date, home_team_id, away_team_id,
                                            park_id, status)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (game_id) DO NOTHING
                """,
                past_game_id,
                past_date,
                TEST_HOME_TEAM_ID,
                TEST_AWAY_TEAM_ID,
                TEST_PARK_ID,
                'final',
            )

            await conn.execute(
                f"""
                INSERT INTO {Table.PLAYER_GAME_LOGS} (player_id, game_id, ip_outs, er,
                                                        pitch_count, is_starter)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (player_id, game_id) DO NOTHING
                """,
                low_ip_pitcher_id,
                past_game_id,
                15,
                1,
                80,
                True,
            )

        # High IP pitcher: 150 IP (450 outs), 30 ER → raw ERA = 1.80
        for i in range(25):  # 25 starts × 6 IP each
            past_game_id = f"high_ip_game_{i}"
            past_date = game_date - timedelta(days=25 - i)

            await conn.execute(
                f"""
                INSERT INTO {Table.GAMES} (game_id, game_date, home_team_id, away_team_id,
                                            park_id, status)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (game_id) DO NOTHING
                """,
                past_game_id,
                past_date,
                TEST_HOME_TEAM_ID,
                TEST_AWAY_TEAM_ID,
                TEST_PARK_ID,
                'final',
            )

            await conn.execute(
                f"""
                INSERT INTO {Table.PLAYER_GAME_LOGS} (player_id, game_id, ip_outs, er,
                                                        pitch_count, is_starter)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (player_id, game_id) DO NOTHING
                """,
                high_ip_pitcher_id,
                past_game_id,
                18,
                1,
                95,
                True,
            )

        # Calculate ERA estimates using feature engineering
        low_ip_features = await _get_pitcher_features(conn, low_ip_pitcher_id, game_date)
        high_ip_features = await _get_pitcher_features(conn, high_ip_pitcher_id, game_date)

        low_ip_era = low_ip_features["era_recent"]
        high_ip_era = high_ip_features["era_recent"]

        # Assert shrinkage: low IP should be closer to league mean
        assert abs(low_ip_era - league_mean_era) < abs(
            high_ip_era - league_mean_era
        ), f"Shrinkage failed: low_ip={low_ip_era:.2f}, high_ip={high_ip_era:.2f}"


@pytest.mark.asyncio
async def test_model_artifacts_load_and_produce_identical_output(pool, sample_game_data):
    """AC7: Model artifacts can be loaded and produce identical output."""
    async with pool.acquire() as conn:
        # Train models
        model_version = await train(conn)

        # Run inference once
        game_id = sample_game_data
        params1 = await predict(conn, game_id, model_version)

        # Run inference again (loads from disk)
        params2 = await predict(conn, game_id, model_version)

        # Verify identical output
        assert params1.home_mu == params2.home_mu
        assert params1.away_mu == params2.away_mu
        assert params1.home_disp == params2.home_disp
        assert params1.away_disp == params2.away_disp
        assert params1.model_version == params2.model_version


# FC-16: Test feature name stability
@pytest.mark.asyncio
async def test_feature_name_stability(pool, sample_game_data):
    """
    Test that GameFeatures.feature_names() matches trained model feature names (FC-16, D-026).

    Verifies:
    - Feature names from GameFeatures.feature_names() match model's booster_.feature_name()
    - This prevents silent feature misalignment when the feature set evolves
    """
    async with pool.acquire() as conn:
        # Train models
        model_version = await train(conn)

        # Load trained models
        from mlb.models.registry import load_model

        home_mu_model = load_model(f"home_mu_{model_version}")
        away_mu_model = load_model(f"away_mu_{model_version}")

        # Get feature names from trained models
        home_model_features = home_mu_model.booster_.feature_name()
        away_model_features = away_mu_model.booster_.feature_name()

        # Get expected feature names from GameFeatures
        home_expected_features = GameFeatures.feature_names(is_home=True)
        away_expected_features = GameFeatures.feature_names(is_home=False)

        # Assert exact match (order and names)
        assert home_model_features == home_expected_features, (
            f"Home feature name mismatch!\n"
            f"Model:    {home_model_features}\n"
            f"Expected: {home_expected_features}"
        )

        assert away_model_features == away_expected_features, (
            f"Away feature name mismatch!\n"
            f"Model:    {away_model_features}\n"
            f"Expected: {away_expected_features}"
        )
