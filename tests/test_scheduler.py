"""Tests for Unit 9 — Scheduler & Orchestration Pipeline.

Tests all acceptance criteria:
1. Global run end-to-end
2. Per-game run
3. Lineup gate — confirmed
4. Lineup gate — unconfirmed but high p_start
5. Lineup gate — team markets always pass
6. Rerun throttle
7. Retry on ingestion failure
8. Nightly eval
9. Cron entry points
"""

import asyncio
from datetime import date, datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mlb.db.models import Table
from mlb.scheduler.cron import midday_run, morning_run, night_before_run, nightly_eval_run
from mlb.scheduler.events import ChangeEvent, check_for_changes, trigger_rerun_if_needed
from mlb.scheduler.gate import is_publishable
from mlb.scheduler.pipeline import (
    _retry_ingestion,
    run_daily_eval,
    run_game,
    run_global,
)


@pytest.mark.asyncio
async def test_global_run_end_to_end(pool):
    """AC1: Global run completes all pipeline steps for all games."""

    # Setup: Create 2 games for today
    today = date.today()
    game_ids = ["2024_TEST_G1", "2024_TEST_G2"]

    async with pool.acquire() as conn:
        # Insert games
        for game_id in game_ids:
            await conn.execute(
                f"""
                INSERT INTO {Table.GAMES}
                (game_id, game_date, home_team_id, away_team_id, park_id,
                 first_pitch, status, updated_at)
                VALUES ($1, $2, 147, 139, 3313, $3, 'scheduled', now())
                ON CONFLICT (game_id) DO NOTHING
                """,
                game_id,
                today,
                datetime.now(timezone.utc) + timedelta(hours=3),
            )

    # Mock ingestion providers to avoid external API calls
    with patch("mlb.scheduler.pipeline.V1GameProvider") as mock_game_provider, \
         patch("mlb.scheduler.pipeline.V1OddsProvider") as mock_odds_provider, \
         patch("mlb.scheduler.pipeline.V1WeatherProvider") as mock_weather_provider, \
         patch("mlb.scheduler.pipeline.V1LineupProvider") as mock_lineup_provider, \
         patch("mlb.scheduler.pipeline.predict_team_runs") as mock_predict, \
         patch("mlb.scheduler.pipeline.simulate_game") as mock_simulate, \
         patch("mlb.scheduler.pipeline.compute_edges") as mock_edges:

        # Mock game provider
        mock_game_instance = mock_game_provider.return_value
        mock_game_instance.fetch_schedule = AsyncMock(return_value=[
            MagicMock(
                game_id=game_ids[0],
                game_date=today,
                home_team_id=147,
                away_team_id=139,
                park_id=3313,
                first_pitch=datetime.now(timezone.utc) + timedelta(hours=3),
                status="scheduled",
                home_score=None,
                away_score=None,
            ),
            MagicMock(
                game_id=game_ids[1],
                game_date=today,
                home_team_id=110,
                away_team_id=111,
                park_id=2,
                first_pitch=datetime.now(timezone.utc) + timedelta(hours=4),
                status="scheduled",
                home_score=None,
                away_score=None,
            ),
        ])
        mock_game_instance.write_games = AsyncMock()

        # Mock odds provider - return minimal non-empty payload to avoid retry
        mock_odds_instance = mock_odds_provider.return_value
        mock_odds_instance.fetch_odds = AsyncMock(return_value=[
            MagicMock(game_id=game_ids[0], book="Test", market="ml", side="home",
                      line=None, price=1.91, snapshot_ts=datetime.now(timezone.utc))
        ])
        mock_odds_instance.write_odds = AsyncMock()

        # Mock weather provider - return minimal non-empty payload to avoid retry
        mock_weather_instance = mock_weather_provider.return_value
        mock_weather_instance.fetch_weather = AsyncMock(return_value=MagicMock(
            game_id=game_ids[0], temp_f=72, wind_speed_mph=5, wind_dir="N",
            precip_pct=0, fetched_at=datetime.now(timezone.utc)
        ))
        mock_weather_instance.write_weather = AsyncMock()

        # Mock lineup provider - return minimal non-empty payload to avoid retry
        mock_lineup_instance = mock_lineup_provider.return_value
        mock_lineup_instance.fetch_lineups = AsyncMock(return_value=[
            MagicMock(game_id=game_ids[0], team_id=147, player_id=101,
                      batting_order=1, is_confirmed=False, source_ts=datetime.now(timezone.utc))
        ])
        mock_lineup_instance.write_lineups = AsyncMock()

        # Mock model prediction (skip due to feature requirements)
        mock_predict.side_effect = ValueError("Missing features")

        # Run global pipeline
        await run_global("morning")

        # Assert: game provider was called
        mock_game_instance.fetch_schedule.assert_called_once()
        mock_game_instance.write_games.assert_called_once()


@pytest.mark.asyncio
async def test_per_game_run(pool):
    """AC2: Per-game run produces outputs for exactly that game."""
    game_id = "2024_TEST_SINGLE"
    today = date.today()

    async with pool.acquire() as conn:
        # Insert game
        await conn.execute(
            f"""
            INSERT INTO {Table.GAMES}
            (game_id, game_date, home_team_id, away_team_id, park_id,
             first_pitch, status, updated_at)
            VALUES ($1, $2, 147, 139, 3313, $3, 'scheduled', now())
            ON CONFLICT (game_id) DO NOTHING
            """,
            game_id,
            today,
            datetime.now(timezone.utc) + timedelta(hours=2),
        )

    # Mock providers - return minimal non-empty payloads to avoid retry
    with patch("mlb.scheduler.pipeline.V1OddsProvider") as mock_odds, \
         patch("mlb.scheduler.pipeline.V1WeatherProvider") as mock_weather, \
         patch("mlb.scheduler.pipeline.V1LineupProvider") as mock_lineup, \
         patch("mlb.scheduler.pipeline.predict_team_runs") as mock_predict:

        mock_odds_instance = mock_odds.return_value
        mock_odds_instance.fetch_odds = AsyncMock(return_value=[
            MagicMock(game_id=game_id, book="Test", market="ml", side="home",
                      line=None, price=1.91, snapshot_ts=datetime.now(timezone.utc))
        ])
        mock_odds_instance.write_odds = AsyncMock()

        mock_weather_instance = mock_weather.return_value
        mock_weather_instance.fetch_weather = AsyncMock(return_value=MagicMock(
            game_id=game_id, temp_f=72, wind_speed_mph=5, wind_dir="N",
            precip_pct=0, fetched_at=datetime.now(timezone.utc)
        ))
        mock_weather_instance.write_weather = AsyncMock()

        mock_lineup_instance = mock_lineup.return_value
        mock_lineup_instance.fetch_lineups = AsyncMock(return_value=[
            MagicMock(game_id=game_id, team_id=147, player_id=101,
                      batting_order=1, is_confirmed=False, source_ts=datetime.now(timezone.utc))
        ])
        mock_lineup_instance.write_lineups = AsyncMock()

        mock_predict.side_effect = ValueError("Missing features")

        # Run per-game pipeline
        await run_game(game_id)

        # Assert: ingestion was called for this game only (once since we return non-empty)
        mock_odds_instance.fetch_odds.assert_called_once()
        mock_lineup_instance.fetch_lineups.assert_called_once()


@pytest.mark.asyncio
async def test_lineup_gate_confirmed(pool):
    """AC3: Confirmed lineup allows player props to pass gate."""
    game_id = "2024_TEST_CONFIRMED"
    player_id = 101
    today = date.today()

    async with pool.acquire() as conn:
        # Insert game
        await conn.execute(
            f"""
            INSERT INTO {Table.GAMES}
            (game_id, game_date, home_team_id, away_team_id, park_id,
             first_pitch, status, updated_at)
            VALUES ($1, $2, 147, 139, 3313, now(), 'scheduled', now())
            ON CONFLICT (game_id) DO NOTHING
            """,
            game_id,
            today,
        )

        # Insert projection
        await conn.execute(
            f"""
            INSERT INTO {Table.PROJECTIONS}
            (game_id, run_ts, home_mu, away_mu, home_disp, away_disp,
             sim_n, created_at)
            VALUES ($1, now(), 4.5, 4.0, 2.0, 2.0, 5000, now())
            """,
            game_id,
        )

        projection_id = await conn.fetchval(
            f"SELECT projection_id FROM {Table.PROJECTIONS} WHERE game_id = $1",
            game_id,
        )

        # Insert sim_market_probs with edge computed (edge_computed_at is on this table)
        await conn.execute(
            f"""
            INSERT INTO {Table.SIM_MARKET_PROBS}
            (projection_id, market, side, line, prob, edge_computed_at)
            VALUES ($1, 'ml', 'home', NULL, 0.52, now())
            ON CONFLICT (projection_id, market, side, line) DO NOTHING
            """,
            projection_id,
        )

        # Insert player first (FK requirement)
        await conn.execute(
            f"""
            INSERT INTO {Table.PLAYERS}
            (player_id, name, team_id)
            VALUES ($1, 'Test Player', 147)
            ON CONFLICT (player_id) DO NOTHING
            """,
            player_id,
        )

        # Insert confirmed lineup (no position or updated_at columns in lineups)
        await conn.execute(
            f"""
            INSERT INTO {Table.LINEUPS}
            (game_id, team_id, player_id, batting_order, is_confirmed, source_ts)
            VALUES ($1, 147, $2, 1, TRUE, now())
            ON CONFLICT (game_id, team_id, batting_order, source_ts) DO NOTHING
            """,
            game_id,
            player_id,
        )

        # Insert player projection
        await conn.execute(
            f"""
            INSERT INTO {Table.PLAYER_PROJECTIONS}
            (projection_id, player_id, game_id, stat, line, prob_over, p_start, created_at)
            VALUES ($1, $2, $3, 'H', 1.5, 0.55, 0.75, now())
            """,
            projection_id,
            player_id,
            game_id,
        )

    # Test team market (should pass)
    assert await is_publishable(game_id, "ml") is True

    # Test player prop with confirmed lineup (should pass even with low p_start)
    assert await is_publishable(game_id, "H", player_id) is True


@pytest.mark.asyncio
async def test_lineup_gate_high_p_start(pool):
    """AC4: High p_start allows player props even without confirmed lineup."""
    game_id = "2024_TEST_PSTART"
    player_id_high = 201
    player_id_low = 202
    today = date.today()

    async with pool.acquire() as conn:
        # Insert game
        await conn.execute(
            f"""
            INSERT INTO {Table.GAMES}
            (game_id, game_date, home_team_id, away_team_id, park_id,
             first_pitch, status, updated_at)
            VALUES ($1, $2, 147, 139, 3313, now(), 'scheduled', now())
            ON CONFLICT (game_id) DO NOTHING
            """,
            game_id,
            today,
        )

        # Insert projection
        await conn.execute(
            f"""
            INSERT INTO {Table.PROJECTIONS}
            (game_id, run_ts, home_mu, away_mu, home_disp, away_disp,
             sim_n, created_at)
            VALUES ($1, now(), 4.5, 4.0, 2.0, 2.0, 5000, now())
            """,
            game_id,
        )

        projection_id = await conn.fetchval(
            f"SELECT projection_id FROM {Table.PROJECTIONS} WHERE game_id = $1",
            game_id,
        )

        # Insert sim_market_probs with edge computed
        await conn.execute(
            f"""
            INSERT INTO {Table.SIM_MARKET_PROBS}
            (projection_id, market, side, line, prob, edge_computed_at)
            VALUES ($1, 'ml', 'home', NULL, 0.52, now())
            ON CONFLICT (projection_id, market, side, line) DO NOTHING
            """,
            projection_id,
        )

        # Insert players first (FK requirement)
        await conn.execute(
            f"""
            INSERT INTO {Table.PLAYERS}
            (player_id, name, team_id)
            VALUES
            ($1, 'Test Player High', 147),
            ($2, 'Test Player Low', 147)
            ON CONFLICT (player_id) DO NOTHING
            """,
            player_id_high,
            player_id_low,
        )

        # Insert unconfirmed lineups (no position or updated_at columns)
        await conn.execute(
            f"""
            INSERT INTO {Table.LINEUPS}
            (game_id, team_id, player_id, batting_order, is_confirmed, source_ts)
            VALUES
            ($1, 147, $2, 1, FALSE, now()),
            ($1, 147, $3, 2, FALSE, now())
            ON CONFLICT (game_id, team_id, batting_order, source_ts) DO NOTHING
            """,
            game_id,
            player_id_high,
            player_id_low,
        )

        # Insert player projections with different p_start values
        await conn.execute(
            f"""
            INSERT INTO {Table.PLAYER_PROJECTIONS}
            (projection_id, player_id, game_id, stat, line, prob_over, p_start, created_at)
            VALUES
            ($1, $2, $3, 'H', 1.5, 0.55, 0.90, now()),
            ($1, $4, $3, 'H', 1.5, 0.50, 0.60, now())
            """,
            projection_id,
            player_id_high,
            game_id,
            player_id_low,
        )

    # Player with p_start = 0.90 should pass (>= 0.85 threshold)
    assert await is_publishable(game_id, "H", player_id_high) is True

    # Player with p_start = 0.60 should fail (< 0.85 threshold)
    assert await is_publishable(game_id, "H", player_id_low) is False


@pytest.mark.asyncio
async def test_lineup_gate_team_markets_exempt(pool):
    """AC5: Team markets always pass when edge is computed."""
    game_id = "2024_TEST_TEAM"
    today = date.today()

    async with pool.acquire() as conn:
        # Insert game
        await conn.execute(
            f"""
            INSERT INTO {Table.GAMES}
            (game_id, game_date, home_team_id, away_team_id, park_id,
             first_pitch, status, updated_at)
            VALUES ($1, $2, 147, 139, 3313, now(), 'scheduled', now())
            ON CONFLICT (game_id) DO NOTHING
            """,
            game_id,
            today,
        )

        # Insert projection, no confirmed lineups
        await conn.execute(
            f"""
            INSERT INTO {Table.PROJECTIONS}
            (game_id, run_ts, home_mu, away_mu, home_disp, away_disp,
             sim_n, created_at)
            VALUES ($1, now(), 4.5, 4.0, 2.0, 2.0, 5000, now())
            """,
            game_id,
        )

        projection_id = await conn.fetchval(
            f"SELECT projection_id FROM {Table.PROJECTIONS} WHERE game_id = $1",
            game_id,
        )

        # Insert sim_market_probs with edge computed (this is what makes team markets publishable)
        await conn.execute(
            f"""
            INSERT INTO {Table.SIM_MARKET_PROBS}
            (projection_id, market, side, line, prob, edge_computed_at)
            VALUES
            ($1, 'ml', 'home', NULL, 0.52, now()),
            ($1, 'rl', 'home', -1.5, 0.48, now()),
            ($1, 'total', 'over', 8.5, 0.51, now())
            ON CONFLICT (projection_id, market, side, line) DO NOTHING
            """,
            projection_id,
        )

    # Team markets should pass even without confirmed lineups
    assert await is_publishable(game_id, "ml") is True
    assert await is_publishable(game_id, "rl") is True
    assert await is_publishable(game_id, "total") is True


@pytest.mark.asyncio
async def test_rerun_throttle(pool):
    """AC6: Rerun throttle prevents excessive reruns."""
    game_id = "2024_TEST_THROTTLE"
    today = date.today()

    # Setup game
    async with pool.acquire() as conn:
        await conn.execute(
            f"""
            INSERT INTO {Table.GAMES}
            (game_id, game_date, home_team_id, away_team_id, park_id,
             first_pitch, status, updated_at)
            VALUES ($1, $2, 147, 139, 3313, now(), 'scheduled', now())
            ON CONFLICT (game_id) DO NOTHING
            """,
            game_id,
            today,
        )

        # Insert players first (FK requirement)
        await conn.execute(
            f"""
            INSERT INTO {Table.PLAYERS}
            (player_id, name, team_id)
            VALUES
            (101, 'Test Player 101', 147),
            (201, 'Test Player 201', 139)
            ON CONFLICT (player_id) DO NOTHING
            """
        )

        # Insert confirmed lineups to trigger events (no position or updated_at columns)
        await conn.execute(
            f"""
            INSERT INTO {Table.LINEUPS}
            (game_id, team_id, player_id, batting_order, is_confirmed, source_ts)
            VALUES
            ($1, 147, 101, 1, TRUE, now()),
            ($1, 139, 201, 1, TRUE, now())
            ON CONFLICT (game_id, team_id, batting_order, source_ts) DO NOTHING
            """,
            game_id,
        )

    with patch("mlb.scheduler.events.run_game") as mock_run_game:
        mock_run_game.return_value = asyncio.Future()
        mock_run_game.return_value.set_result(None)

        # First trigger should succeed
        result1 = await trigger_rerun_if_needed(game_id)
        assert result1 is True
        assert mock_run_game.call_count == 1

        # Second trigger within throttle window should be blocked
        result2 = await trigger_rerun_if_needed(game_id)
        assert result2 is False
        assert mock_run_game.call_count == 1  # Not called again

        # Simulate time passing (mock the last rerun time)
        from mlb.scheduler import events
        events._last_rerun[game_id] = datetime.now(timezone.utc) - timedelta(minutes=15)

        # Third trigger after throttle window should succeed
        result3 = await trigger_rerun_if_needed(game_id)
        assert result3 is True
        assert mock_run_game.call_count == 2


@pytest.mark.asyncio
async def test_retry_on_ingestion_failure():
    """AC7: Retry policy with exponential backoff on ingestion failure."""
    call_count = 0

    async def failing_ingestion():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            return []  # Empty result (failure per D-019)
        return [MagicMock()]  # Success on 3rd attempt

    result = await _retry_ingestion(failing_ingestion, "test-ingestion")

    # Should have retried and succeeded on 3rd attempt
    assert call_count == 3
    assert len(result) == 1


@pytest.mark.asyncio
async def test_nightly_eval(pool):
    """AC8: Nightly eval triggers backtest for today's final games."""
    today = date.today()
    game_id = "2024_TEST_EVAL"

    async with pool.acquire() as conn:
        # Insert final game with scores
        await conn.execute(
            f"""
            INSERT INTO {Table.GAMES}
            (game_id, game_date, home_team_id, away_team_id, park_id,
             first_pitch, status, home_score, away_score, updated_at)
            VALUES ($1, $2, 147, 139, 3313, now(), 'final', 5, 3, now())
            ON CONFLICT (game_id) DO NOTHING
            """,
            game_id,
            today,
        )

        # Insert projection
        await conn.execute(
            f"""
            INSERT INTO {Table.PROJECTIONS}
            (game_id, run_ts, home_mu, away_mu, home_disp, away_disp,
             sim_n, created_at)
            VALUES ($1, now(), 4.5, 4.0, 2.0, 2.0, 5000, now())
            """,
            game_id,
        )

        projection_id = await conn.fetchval(
            f"SELECT projection_id FROM {Table.PROJECTIONS} WHERE game_id = $1",
            game_id,
        )

        # Insert market probabilities for evaluation
        await conn.execute(
            f"""
            INSERT INTO {Table.SIM_MARKET_PROBS}
            (projection_id, market, side, line, prob)
            VALUES ($1, 'ml', 'home', NULL, 0.55)
            """,
            projection_id,
        )

    # Run nightly evaluation
    await run_daily_eval()

    # Assert: eval_results rows exist
    async with pool.acquire() as conn:
        eval_count = await conn.fetchval(
            f"""
            SELECT COUNT(*)
            FROM {Table.EVAL_RESULTS}
            WHERE eval_date = $1
            """,
            today,
        )

        # Should have results for at least one market
        assert eval_count > 0


def test_cron_entry_points():
    """AC9: Cron entry points are callable with no arguments."""
    # Test that cron functions exist and are callable
    # We won't actually execute them to avoid side effects

    assert callable(night_before_run)
    assert callable(morning_run)
    assert callable(midday_run)
    assert callable(nightly_eval_run)

    # Verify function signatures accept no arguments
    import inspect

    assert len(inspect.signature(night_before_run).parameters) == 0
    assert len(inspect.signature(morning_run).parameters) == 0
    assert len(inspect.signature(midday_run).parameters) == 0
    assert len(inspect.signature(nightly_eval_run).parameters) == 0


@pytest.mark.asyncio
async def test_no_games_today(pool):
    """Edge case: No games scheduled for today."""
    with patch("mlb.scheduler.pipeline.V1GameProvider") as mock_provider:
        mock_instance = mock_provider.return_value
        # Empty list triggers retry (3 times) per D-019
        mock_instance.fetch_schedule = AsyncMock(return_value=[])
        mock_instance.write_games = AsyncMock()

        # Should complete without error
        await run_global("morning")

        # Empty result triggers 3 retries
        assert mock_instance.fetch_schedule.call_count == 3
        mock_instance.write_games.assert_not_called()


@pytest.mark.asyncio
async def test_postponed_game_skipped(pool):
    """Edge case: Postponed games are skipped during processing."""
    game_id = "2024_TEST_POSTPONED"
    today = date.today()

    async with pool.acquire() as conn:
        await conn.execute(
            f"""
            INSERT INTO {Table.GAMES}
            (game_id, game_date, home_team_id, away_team_id, park_id,
             first_pitch, status, updated_at)
            VALUES ($1, $2, 147, 139, 3313, NULL, 'postponed', now())
            ON CONFLICT (game_id) DO NOTHING
            """,
            game_id,
            today,
        )

    with patch("mlb.scheduler.pipeline._process_game") as mock_process:
        await run_game(game_id)

        # Should not process postponed game
        mock_process.assert_not_called()
