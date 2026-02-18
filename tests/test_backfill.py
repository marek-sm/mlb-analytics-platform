"""Tests for Step 3 — Historical Backfill orchestration.

Verifies acceptance criteria using mock adapters; no real database or API
calls are made.
"""

from datetime import date, datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from mlb.ingestion.base import GameRow, WeatherRow
from mlb.operations.backfill import run_backfill

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BACKFILL = "mlb.operations.backfill"


def _make_game(
    game_id: str,
    game_date: date,
    home_team_id: int = 143,
    away_team_id: int = 144,
    park_id: int = 2681,
    status: str = "final",
) -> GameRow:
    """Build a minimal GameRow for testing."""
    return GameRow(
        game_id=game_id,
        game_date=game_date,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        park_id=park_id,
        first_pitch=datetime(2025, 6, 1, 18, 0, 0, tzinfo=timezone.utc),
        status=status,
    )


def _make_final_games(n: int, base_date: date = date(2025, 6, 1)) -> list[GameRow]:
    """Return n GameRow objects all with status='final'."""
    games = []
    for i in range(n):
        d = base_date + timedelta(days=i // 15)
        games.append(_make_game(str(700_000 + i), d))
    return games


def _make_mock_pool(fetchval_side_effects: list):
    """
    Build a (mock_pool, mock_conn) pair.

    mock_conn.fetchval.side_effect is set to fetchval_side_effects so callers
    can verify each sequential SQL scalar query.
    """
    mock_conn = AsyncMock()
    mock_conn.fetchval.side_effect = list(fetchval_side_effects)

    # Emulate `async with pool.acquire() as conn:`
    acquire_cm = MagicMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=mock_conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=False)

    mock_pool = MagicMock()
    mock_pool.acquire.return_value = acquire_cm

    return mock_pool, mock_conn


# fetchval_side_effects for a successful 7-phase run (6 scalar queries):
#   Phase 1 count, Phase 6 × 5 counts
_HAPPY_FETCHVALS = [31, 31, 100, 200, 300, 50]


# ---------------------------------------------------------------------------
# Test 1: fetch_schedule is called in ascending date order
# ---------------------------------------------------------------------------


async def test_backfill_ingests_games_in_date_order():
    """fetch_schedule is called once per date in chronological order."""
    start = date(2025, 6, 1)
    end = date(2025, 6, 3)  # 3 dates
    dates = [date(2025, 6, 1), date(2025, 6, 2), date(2025, 6, 3)]
    # 11 final games per date → 33 total, satisfying the ≥30 gate
    games_per_date = [_make_final_games(11, d) for d in dates]
    mock_pool, _ = _make_mock_pool([33, 33, 100, 200, 300, 50])

    with (
        patch(f"{_BACKFILL}.V1GameProvider") as MockGame,
        patch(f"{_BACKFILL}.V1LineupProvider") as MockLineup,
        patch(f"{_BACKFILL}.V1StatsProvider") as MockStats,
        patch(f"{_BACKFILL}.V1OddsProvider") as MockOdds,
        patch(f"{_BACKFILL}.V1WeatherProvider") as MockWeather,
        patch(f"{_BACKFILL}.get_pool", new_callable=AsyncMock, return_value=mock_pool),
        patch(
            f"{_BACKFILL}.team_runs_module.train",
            new_callable=AsyncMock,
            return_value="v1",
        ),
        patch(
            f"{_BACKFILL}.player_props_module.train",
            new_callable=AsyncMock,
            return_value="v1",
        ),
        patch("os.listdir", return_value=["home_mu_v1.pkl", "away_mu_v1.pkl"]),
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        mock_game = MockGame.return_value
        mock_game.fetch_schedule = AsyncMock(side_effect=games_per_date)
        mock_game.write_games = AsyncMock()
        MockLineup.return_value.fetch_lineup = AsyncMock(return_value=[])
        MockStats.return_value.fetch_game_logs = AsyncMock(return_value=[])
        MockStats.return_value.write_game_logs = AsyncMock()
        MockOdds.return_value.fetch_odds = AsyncMock(return_value=[])
        MockWeather.return_value.fetch_weather = AsyncMock(return_value=None)

        await run_backfill(start, end)

    scheduled_calls = mock_game.fetch_schedule.call_args_list
    assert len(scheduled_calls) == 3, "fetch_schedule must be called once per date"
    assert scheduled_calls[0] == call(date(2025, 6, 1))
    assert scheduled_calls[1] == call(date(2025, 6, 2))
    assert scheduled_calls[2] == call(date(2025, 6, 3))


# ---------------------------------------------------------------------------
# Test 2: fetch_lineup called twice per game (home + away)
# ---------------------------------------------------------------------------


async def test_backfill_calls_lineup_for_both_teams():
    """fetch_lineup is called exactly twice per final game: once per team."""
    start = end = date(2025, 6, 1)
    games = _make_final_games(31, start)
    mock_pool, _ = _make_mock_pool(_HAPPY_FETCHVALS)

    with (
        patch(f"{_BACKFILL}.V1GameProvider") as MockGame,
        patch(f"{_BACKFILL}.V1LineupProvider") as MockLineup,
        patch(f"{_BACKFILL}.V1StatsProvider") as MockStats,
        patch(f"{_BACKFILL}.V1OddsProvider") as MockOdds,
        patch(f"{_BACKFILL}.V1WeatherProvider") as MockWeather,
        patch(f"{_BACKFILL}.get_pool", new_callable=AsyncMock, return_value=mock_pool),
        patch(
            f"{_BACKFILL}.team_runs_module.train",
            new_callable=AsyncMock,
            return_value="v1",
        ),
        patch(
            f"{_BACKFILL}.player_props_module.train",
            new_callable=AsyncMock,
            return_value="v1",
        ),
        patch("os.listdir", return_value=["home_mu_v1.pkl"]),
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        MockGame.return_value.fetch_schedule = AsyncMock(return_value=games)
        MockGame.return_value.write_games = AsyncMock()

        mock_lineup = MockLineup.return_value
        mock_lineup.fetch_lineup = AsyncMock(return_value=[])

        MockStats.return_value.fetch_game_logs = AsyncMock(return_value=[])
        MockStats.return_value.write_game_logs = AsyncMock()
        MockOdds.return_value.fetch_odds = AsyncMock(return_value=[])
        MockWeather.return_value.fetch_weather = AsyncMock(return_value=None)

        await run_backfill(start, end)

    # 31 games × 2 teams = 62 calls
    assert mock_lineup.fetch_lineup.call_count == 62

    # Confirm both home and away team IDs appear across all calls
    called_team_ids = {c.args[1] for c in mock_lineup.fetch_lineup.call_args_list}
    assert 143 in called_team_ids  # home_team_id
    assert 144 in called_team_ids  # away_team_id


# ---------------------------------------------------------------------------
# Test 3: fetch_odds called once per final game
# ---------------------------------------------------------------------------


async def test_backfill_ingests_odds_for_all_games():
    """fetch_odds is called exactly once for every final game."""
    start = end = date(2025, 6, 1)
    n = 31
    games = _make_final_games(n, start)
    mock_pool, _ = _make_mock_pool(_HAPPY_FETCHVALS)

    with (
        patch(f"{_BACKFILL}.V1GameProvider") as MockGame,
        patch(f"{_BACKFILL}.V1LineupProvider") as MockLineup,
        patch(f"{_BACKFILL}.V1StatsProvider") as MockStats,
        patch(f"{_BACKFILL}.V1OddsProvider") as MockOdds,
        patch(f"{_BACKFILL}.V1WeatherProvider") as MockWeather,
        patch(f"{_BACKFILL}.get_pool", new_callable=AsyncMock, return_value=mock_pool),
        patch(
            f"{_BACKFILL}.team_runs_module.train",
            new_callable=AsyncMock,
            return_value="v1",
        ),
        patch(
            f"{_BACKFILL}.player_props_module.train",
            new_callable=AsyncMock,
            return_value="v1",
        ),
        patch("os.listdir", return_value=["home_mu_v1.pkl"]),
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        MockGame.return_value.fetch_schedule = AsyncMock(return_value=games)
        MockGame.return_value.write_games = AsyncMock()
        MockLineup.return_value.fetch_lineup = AsyncMock(return_value=[])
        MockStats.return_value.fetch_game_logs = AsyncMock(return_value=[])
        MockStats.return_value.write_game_logs = AsyncMock()

        mock_odds = MockOdds.return_value
        mock_odds.fetch_odds = AsyncMock(return_value=[])

        MockWeather.return_value.fetch_weather = AsyncMock(return_value=None)

        await run_backfill(start, end)

    assert mock_odds.fetch_odds.call_count == n


# ---------------------------------------------------------------------------
# Test 4: Weather None result does not trigger write_weather
# ---------------------------------------------------------------------------


async def test_backfill_weather_skips_none():
    """If fetch_weather returns None, write_weather must not be called."""
    start = end = date(2025, 6, 1)
    games = _make_final_games(31, start)
    mock_pool, _ = _make_mock_pool(_HAPPY_FETCHVALS)

    with (
        patch(f"{_BACKFILL}.V1GameProvider") as MockGame,
        patch(f"{_BACKFILL}.V1LineupProvider") as MockLineup,
        patch(f"{_BACKFILL}.V1StatsProvider") as MockStats,
        patch(f"{_BACKFILL}.V1OddsProvider") as MockOdds,
        patch(f"{_BACKFILL}.V1WeatherProvider") as MockWeather,
        patch(f"{_BACKFILL}.get_pool", new_callable=AsyncMock, return_value=mock_pool),
        patch(
            f"{_BACKFILL}.team_runs_module.train",
            new_callable=AsyncMock,
            return_value="v1",
        ),
        patch(
            f"{_BACKFILL}.player_props_module.train",
            new_callable=AsyncMock,
            return_value="v1",
        ),
        patch("os.listdir", return_value=["home_mu_v1.pkl"]),
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        MockGame.return_value.fetch_schedule = AsyncMock(return_value=games)
        MockGame.return_value.write_games = AsyncMock()
        MockLineup.return_value.fetch_lineup = AsyncMock(return_value=[])
        MockStats.return_value.fetch_game_logs = AsyncMock(return_value=[])
        MockStats.return_value.write_game_logs = AsyncMock()
        MockOdds.return_value.fetch_odds = AsyncMock(return_value=[])

        mock_weather = MockWeather.return_value
        mock_weather.fetch_weather = AsyncMock(return_value=None)  # always None
        mock_weather.write_weather = AsyncMock()

        await run_backfill(start, end)

    mock_weather.write_weather.assert_not_called()


# ---------------------------------------------------------------------------
# Test 5: RuntimeError raised when DB reports < 30 final games
# ---------------------------------------------------------------------------


async def test_backfill_fails_under_30_games():
    """run_backfill raises RuntimeError when fewer than 30 final games exist."""
    start = end = date(2025, 6, 1)
    games = _make_final_games(10, start)  # only 10 games in memory
    mock_pool, _ = _make_mock_pool([10])  # DB count also returns 10

    with (
        patch(f"{_BACKFILL}.V1GameProvider") as MockGame,
        patch(f"{_BACKFILL}.V1LineupProvider"),
        patch(f"{_BACKFILL}.V1StatsProvider"),
        patch(f"{_BACKFILL}.V1OddsProvider"),
        patch(f"{_BACKFILL}.V1WeatherProvider"),
        patch(f"{_BACKFILL}.get_pool", new_callable=AsyncMock, return_value=mock_pool),
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        MockGame.return_value.fetch_schedule = AsyncMock(return_value=games)
        MockGame.return_value.write_games = AsyncMock()

        with pytest.raises(RuntimeError, match="Insufficient final games: 10"):
            await run_backfill(start, end)


# ---------------------------------------------------------------------------
# Test 6: Backfill continues when an adapter raises for one date/game
# ---------------------------------------------------------------------------


async def test_backfill_continues_on_adapter_failure():
    """A single adapter exception is logged and skipped; the run completes."""
    start = date(2025, 6, 1)
    end = date(2025, 6, 2)  # two dates

    games_date1 = _make_final_games(31, date(2025, 6, 1))
    # date 2 fetch raises — its games are never added to all_game_rows
    mock_pool, _ = _make_mock_pool(_HAPPY_FETCHVALS)

    with (
        patch(f"{_BACKFILL}.V1GameProvider") as MockGame,
        patch(f"{_BACKFILL}.V1LineupProvider") as MockLineup,
        patch(f"{_BACKFILL}.V1StatsProvider") as MockStats,
        patch(f"{_BACKFILL}.V1OddsProvider") as MockOdds,
        patch(f"{_BACKFILL}.V1WeatherProvider") as MockWeather,
        patch(f"{_BACKFILL}.get_pool", new_callable=AsyncMock, return_value=mock_pool),
        patch(
            f"{_BACKFILL}.team_runs_module.train",
            new_callable=AsyncMock,
            return_value="v1",
        ),
        patch(
            f"{_BACKFILL}.player_props_module.train",
            new_callable=AsyncMock,
            return_value="v1",
        ),
        patch("os.listdir", return_value=["home_mu_v1.pkl"]),
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        mock_game = MockGame.return_value
        # Date 1 succeeds; date 2 raises a transient error
        mock_game.fetch_schedule = AsyncMock(
            side_effect=[games_date1, RuntimeError("API timeout")]
        )
        mock_game.write_games = AsyncMock()

        mock_lineup = MockLineup.return_value
        mock_lineup.fetch_lineup = AsyncMock(return_value=[])

        MockStats.return_value.fetch_game_logs = AsyncMock(return_value=[])
        MockStats.return_value.write_game_logs = AsyncMock()
        MockOdds.return_value.fetch_odds = AsyncMock(return_value=[])
        MockWeather.return_value.fetch_weather = AsyncMock(return_value=None)

        # Must complete without raising despite date 2 failure
        await run_backfill(start, end)

    # Date 1's 31 games × 2 teams = 62 lineup calls still happen
    assert mock_lineup.fetch_lineup.call_count == 62


# ---------------------------------------------------------------------------
# Test 7: fetch_odds receives event_date matching each game's first_pitch
# ---------------------------------------------------------------------------


async def test_backfill_passes_event_date_to_odds():
    """Phase 4 passes event_date=game.first_pitch to fetch_odds for every game (D-067)."""
    start = end = date(2025, 6, 1)
    games = _make_final_games(31, start)
    mock_pool, _ = _make_mock_pool(_HAPPY_FETCHVALS)

    with (
        patch(f"{_BACKFILL}.V1GameProvider") as MockGame,
        patch(f"{_BACKFILL}.V1LineupProvider") as MockLineup,
        patch(f"{_BACKFILL}.V1StatsProvider") as MockStats,
        patch(f"{_BACKFILL}.V1OddsProvider") as MockOdds,
        patch(f"{_BACKFILL}.V1WeatherProvider") as MockWeather,
        patch(f"{_BACKFILL}.get_pool", new_callable=AsyncMock, return_value=mock_pool),
        patch(
            f"{_BACKFILL}.team_runs_module.train",
            new_callable=AsyncMock,
            return_value="v1",
        ),
        patch(
            f"{_BACKFILL}.player_props_module.train",
            new_callable=AsyncMock,
            return_value="v1",
        ),
        patch("os.listdir", return_value=["home_mu_v1.pkl"]),
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        MockGame.return_value.fetch_schedule = AsyncMock(return_value=games)
        MockGame.return_value.write_games = AsyncMock()
        MockLineup.return_value.fetch_lineup = AsyncMock(return_value=[])
        MockStats.return_value.fetch_game_logs = AsyncMock(return_value=[])
        MockStats.return_value.write_game_logs = AsyncMock()

        mock_odds = MockOdds.return_value
        mock_odds.fetch_odds = AsyncMock(return_value=[])

        MockWeather.return_value.fetch_weather = AsyncMock(return_value=None)

        await run_backfill(start, end)

    assert mock_odds.fetch_odds.call_count == len(games)

    # Every call must supply keyword event_date matching that game's first_pitch
    for i, c in enumerate(mock_odds.fetch_odds.call_args_list):
        kwargs = c.kwargs
        assert "event_date" in kwargs, (
            f"Call {i}: fetch_odds missing event_date kwarg"
        )
        assert kwargs["event_date"] == games[i].first_pitch, (
            f"Call {i}: event_date mismatch — "
            f"got {kwargs['event_date']}, want {games[i].first_pitch}"
        )
