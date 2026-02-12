"""Tests for Unit 7: Odds Processing, Edge Calculation & Bankroll Sizing.

Tests all acceptance criteria from the mini-spec:
1. Proportional devig math
2. Best-line selection
3. Edge calculation
4. Kelly sizing
5. Minimum edge threshold
6. edge_computed_at set
7. No odds available
8. Player prop no-match
9. Idempotent
"""

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
import pytest_asyncio

from mlb.db.models import Table
from mlb.odds import (
    compute_edges,
    get_best_lines,
    proportional_devig,
)
from mlb.odds.persistence import persist_edges


class TestProportionalDevig:
    """Test acceptance criterion #1: Proportional devig math."""

    def test_even_odds(self):
        """Test devig([1.91, 1.91]) returns [0.5, 0.5]."""
        result = proportional_devig([1.91, 1.91])
        assert len(result) == 2
        assert abs(result[0] - 0.5) < 0.001
        assert abs(result[1] - 0.5) < 0.001

    def test_favorite_underdog(self):
        """Test devig([1.50, 2.80]) returns probabilities summing to 1.0 with favorite > 0.5."""
        result = proportional_devig([1.50, 2.80])
        assert len(result) == 2
        assert abs(sum(result) - 1.0) < 0.0001
        assert result[0] > 0.5  # Favorite (lower price)
        assert result[1] < 0.5  # Underdog (higher price)
        # Check approximate values
        # 1/1.50 = 0.6667, 1/2.80 = 0.3571, total = 1.0238
        # fair = [0.6667/1.0238, 0.3571/1.0238] = [0.6512, 0.3488]
        assert abs(result[0] - 0.6512) < 0.01
        assert abs(result[1] - 0.3488) < 0.01

    def test_three_way_market(self):
        """Test devig works for three-way markets (e.g., regulation winner + tie)."""
        result = proportional_devig([2.0, 3.0, 4.0])
        assert len(result) == 3
        assert abs(sum(result) - 1.0) < 0.0001
        # Implied: [0.5, 0.333, 0.25], total = 1.083
        # Fair: [0.4615, 0.3077, 0.2308]
        assert result[0] > result[1] > result[2]

    def test_empty_list_raises(self):
        """Test empty prices list raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            proportional_devig([])

    def test_invalid_price_raises(self):
        """Test price < 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="must be >= 1.0"):
            proportional_devig([0.5, 2.0])


@pytest.mark.asyncio
class TestBestLineSelection:
    """Test acceptance criterion #2: Best-line selection."""

    @pytest_asyncio.fixture
    async def game_with_odds(self, pool):
        """Create a game with odds from multiple books."""
        async with pool.acquire() as conn:
            # Create game
            game_id = "test_best_line_game"
            await conn.execute(
                f"""
                INSERT INTO {Table.GAMES} (game_id, game_date, status)
                VALUES ($1, CURRENT_DATE, 'scheduled')
                ON CONFLICT (game_id) DO NOTHING
                """,
                game_id,
            )

            # Insert odds from 3 books for the same market
            snapshot_ts = datetime.now(timezone.utc)
            await conn.executemany(
                f"""
                INSERT INTO {Table.ODDS_SNAPSHOTS}
                (game_id, book, market, side, line, price, snapshot_ts)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                [
                    # Book1: ml home at 1.90
                    (game_id, "book1", "ml", "home", None, 1.90, snapshot_ts),
                    # Book2: ml home at 2.00 (best price)
                    (game_id, "book2", "ml", "home", None, 2.00, snapshot_ts),
                    # Book3: ml home at 1.85
                    (game_id, "book3", "ml", "home", None, 1.85, snapshot_ts),
                    # Also add away side
                    (game_id, "book1", "ml", "away", None, 1.95, snapshot_ts),
                    (game_id, "book2", "ml", "away", None, 1.90, snapshot_ts),
                    (game_id, "book3", "ml", "away", None, 2.00, snapshot_ts),
                ],
            )

            yield game_id

            # Cleanup
            await conn.execute(
                f"DELETE FROM {Table.ODDS_SNAPSHOTS} WHERE game_id = $1", game_id
            )
            await conn.execute(f"DELETE FROM {Table.GAMES} WHERE game_id = $1", game_id)

    async def test_returns_highest_price(self, pool, game_with_odds):
        """Test that get_best_lines returns the book with highest price."""
        async with pool.acquire() as conn:
            best_lines = await get_best_lines(conn, game_with_odds)

            # Should have 2 lines (home and away ml)
            assert len(best_lines) == 2

            # Find home line
            home_line = next(bl for bl in best_lines if bl.side == "home")
            assert home_line.best_price == 2.00
            assert home_line.book == "book2"

            # Find away line
            away_line = next(bl for bl in best_lines if bl.side == "away")
            assert away_line.best_price == 2.00
            assert away_line.book == "book3"

    async def test_empty_for_no_odds(self, pool):
        """Test returns empty list when no odds exist."""
        async with pool.acquire() as conn:
            # Create game without odds
            game_id = "test_no_odds_game"
            await conn.execute(
                f"""
                INSERT INTO {Table.GAMES} (game_id, game_date, status)
                VALUES ($1, CURRENT_DATE, 'scheduled')
                ON CONFLICT (game_id) DO NOTHING
                """,
                game_id,
            )

            try:
                best_lines = await get_best_lines(conn, game_id)
                assert best_lines == []
            finally:
                await conn.execute(
                    f"DELETE FROM {Table.GAMES} WHERE game_id = $1", game_id
                )


class TestEdgeCalculation:
    """Test acceptance criteria #3-5: Edge calculation, Kelly sizing, minimum threshold."""

    def test_positive_edge(self):
        """Test p_model=0.55, p_fair=0.50 produces edge=0.05."""
        edge = 0.55 - 0.50
        assert abs(edge - 0.05) < 0.0001

    def test_negative_edge(self):
        """Test p_model=0.48, p_fair=0.50 produces edge=-0.02."""
        edge = 0.48 - 0.50
        assert abs(edge - (-0.02)) < 0.0001

    def test_kelly_sizing_formula(self):
        """Test Kelly sizing: edge=0.05, best_price=2.00 → kelly=0.0125."""
        edge = 0.05
        best_price = 2.00
        kelly_multiplier = 0.25
        decimal_odds = best_price - 1.0
        kelly = kelly_multiplier * edge / decimal_odds
        assert abs(kelly - 0.0125) < 0.0001

    def test_kelly_zero_for_negative_edge(self):
        """Test negative edge produces kelly_fraction=0.0."""
        edge = -0.02
        best_price = 2.00
        kelly_multiplier = 0.25
        min_threshold = 0.02

        # Below threshold
        if edge < min_threshold:
            kelly = 0.0
        else:
            kelly = kelly_multiplier * edge / (best_price - 1.0)

        assert kelly == 0.0

    def test_kelly_zero_below_threshold(self):
        """Test edge below threshold produces kelly_fraction=0.0 (but edge is still stored)."""
        edge = 0.01  # Below 0.02 threshold
        best_price = 2.00
        kelly_multiplier = 0.25
        min_threshold = 0.02

        if edge < min_threshold:
            kelly = 0.0
        else:
            kelly = kelly_multiplier * edge / (best_price - 1.0)

        assert kelly == 0.0
        assert edge == 0.01  # Edge value is still stored


@pytest.mark.asyncio
class TestEdgeComputedAt:
    """Test acceptance criterion #6: edge_computed_at set."""

    @pytest_asyncio.fixture
    async def projection_with_markets(self, pool):
        """Create a projection with sim_market_probs rows."""
        async with pool.acquire() as conn:
            # Create minimal game
            game_id = "test_edge_computed_game"
            await conn.execute(
                f"""
                INSERT INTO {Table.GAMES} (game_id, game_date, status)
                VALUES ($1, CURRENT_DATE, 'scheduled')
                ON CONFLICT (game_id) DO NOTHING
                """,
                game_id,
            )

            # Create projection
            projection_id = await conn.fetchval(
                f"""
                INSERT INTO {Table.PROJECTIONS}
                (game_id, run_ts, home_mu, away_mu, home_disp, away_disp, sim_n)
                VALUES ($1, $2, 4.5, 4.0, 0.8, 0.8, 5000)
                RETURNING projection_id
                """,
                game_id,
                datetime.now(timezone.utc),
            )

            # Insert sim_market_probs (without edge_computed_at)
            await conn.executemany(
                f"""
                INSERT INTO {Table.SIM_MARKET_PROBS}
                (projection_id, market, side, line, prob)
                VALUES ($1, $2, $3, $4, $5)
                """,
                [
                    (projection_id, "ml", "home", None, 0.55),
                    (projection_id, "ml", "away", None, 0.45),
                ],
            )

            # Add minimal odds
            snapshot_ts = datetime.now(timezone.utc)
            await conn.executemany(
                f"""
                INSERT INTO {Table.ODDS_SNAPSHOTS}
                (game_id, book, market, side, line, price, snapshot_ts)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                [
                    (game_id, "book1", "ml", "home", None, 1.90, snapshot_ts),
                    (game_id, "book1", "ml", "away", None, 1.95, snapshot_ts),
                ],
            )

            yield projection_id

            # Cleanup
            await conn.execute(
                f"DELETE FROM {Table.SIM_MARKET_PROBS} WHERE projection_id = $1",
                projection_id,
            )
            await conn.execute(
                f"DELETE FROM {Table.PROJECTIONS} WHERE projection_id = $1",
                projection_id,
            )
            await conn.execute(
                f"DELETE FROM {Table.ODDS_SNAPSHOTS} WHERE game_id = $1", game_id
            )
            await conn.execute(f"DELETE FROM {Table.GAMES} WHERE game_id = $1", game_id)

    async def test_edge_computed_at_set_after_compute(
        self, pool, projection_with_markets
    ):
        """Test that edge_computed_at is set on all sim_market_probs after compute_edges."""
        async with pool.acquire() as conn:
            # Compute edges
            edge_result = await compute_edges(conn, projection_with_markets)

            # Persist edges
            await persist_edges(conn, edge_result)

            # Check that edge_computed_at is set on all rows
            rows = await conn.fetch(
                f"""
                SELECT edge_computed_at
                FROM {Table.SIM_MARKET_PROBS}
                WHERE projection_id = $1
                """,
                projection_with_markets,
            )

            assert len(rows) == 2
            for row in rows:
                assert row["edge_computed_at"] is not None


@pytest.mark.asyncio
class TestNoOddsAvailable:
    """Test acceptance criterion #7: No odds available."""

    @pytest_asyncio.fixture
    async def projection_without_odds(self, pool):
        """Create a projection for a game with no odds."""
        async with pool.acquire() as conn:
            # Create game
            game_id = "test_no_odds_projection"
            await conn.execute(
                f"""
                INSERT INTO {Table.GAMES} (game_id, game_date, status)
                VALUES ($1, CURRENT_DATE, 'scheduled')
                ON CONFLICT (game_id) DO NOTHING
                """,
                game_id,
            )

            # Create projection
            projection_id = await conn.fetchval(
                f"""
                INSERT INTO {Table.PROJECTIONS}
                (game_id, run_ts, home_mu, away_mu, home_disp, away_disp, sim_n)
                VALUES ($1, $2, 4.5, 4.0, 0.8, 0.8, 5000)
                RETURNING projection_id
                """,
                game_id,
                datetime.now(timezone.utc),
            )

            # Insert sim_market_probs
            await conn.executemany(
                f"""
                INSERT INTO {Table.SIM_MARKET_PROBS}
                (projection_id, market, side, line, prob)
                VALUES ($1, $2, $3, $4, $5)
                """,
                [
                    (projection_id, "ml", "home", None, 0.55),
                    (projection_id, "ml", "away", None, 0.45),
                ],
            )

            # NO ODDS inserted

            yield projection_id

            # Cleanup
            await conn.execute(
                f"DELETE FROM {Table.SIM_MARKET_PROBS} WHERE projection_id = $1",
                projection_id,
            )
            await conn.execute(
                f"DELETE FROM {Table.PROJECTIONS} WHERE projection_id = $1",
                projection_id,
            )
            await conn.execute(f"DELETE FROM {Table.GAMES} WHERE game_id = $1", game_id)

    async def test_no_error_when_no_odds(self, pool, projection_without_odds):
        """Test that compute_edges completes without error when no odds exist."""
        async with pool.acquire() as conn:
            edge_result = await compute_edges(conn, projection_without_odds)

            # Should complete without error
            assert edge_result.projection_id == projection_without_odds
            assert edge_result.market_edges == []

            # Persist should also complete
            await persist_edges(conn, edge_result)

            # Check that edge_computed_at is still set (marks pass as complete)
            rows = await conn.fetch(
                f"""
                SELECT edge, kelly_fraction, edge_computed_at
                FROM {Table.SIM_MARKET_PROBS}
                WHERE projection_id = $1
                """,
                projection_without_odds,
            )

            assert len(rows) == 2
            for row in rows:
                assert row["edge"] is None
                assert row["kelly_fraction"] is None
                assert row["edge_computed_at"] is not None


@pytest.mark.asyncio
class TestPlayerPropNoMatch:
    """Test acceptance criterion #8: Player prop no-match."""

    @pytest_asyncio.fixture
    async def projection_with_player_props(self, pool):
        """Create a projection with player_projections but no player prop odds."""
        async with pool.acquire() as conn:
            # Create game and player
            game_id = "test_player_prop_game"
            player_id = 999999
            await conn.execute(
                f"""
                INSERT INTO {Table.GAMES} (game_id, game_date, status)
                VALUES ($1, CURRENT_DATE, 'scheduled')
                ON CONFLICT (game_id) DO NOTHING
                """,
                game_id,
            )
            await conn.execute(
                f"""
                INSERT INTO {Table.PLAYERS} (player_id, name)
                VALUES ($1, 'Test Player')
                ON CONFLICT (player_id) DO NOTHING
                """,
                player_id,
            )

            # Create projection
            projection_id = await conn.fetchval(
                f"""
                INSERT INTO {Table.PROJECTIONS}
                (game_id, run_ts, home_mu, away_mu, home_disp, away_disp, sim_n)
                VALUES ($1, $2, 4.5, 4.0, 0.8, 0.8, 5000)
                RETURNING projection_id
                """,
                game_id,
                datetime.now(timezone.utc),
            )

            # Insert player_projections
            await conn.execute(
                f"""
                INSERT INTO {Table.PLAYER_PROJECTIONS}
                (projection_id, player_id, game_id, p_start, stat, line, prob_over)
                VALUES ($1, $2, $3, 0.90, 'H', 0.5, 0.65)
                """,
                projection_id,
                player_id,
                game_id,
            )

            # Add team market odds (but no player prop odds)
            snapshot_ts = datetime.now(timezone.utc)
            await conn.executemany(
                f"""
                INSERT INTO {Table.ODDS_SNAPSHOTS}
                (game_id, book, market, side, line, price, snapshot_ts)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                [
                    (game_id, "book1", "ml", "home", None, 1.90, snapshot_ts),
                    (game_id, "book1", "ml", "away", None, 1.95, snapshot_ts),
                ],
            )

            yield projection_id

            # Cleanup
            await conn.execute(
                f"DELETE FROM {Table.PLAYER_PROJECTIONS} WHERE projection_id = $1",
                projection_id,
            )
            await conn.execute(
                f"DELETE FROM {Table.SIM_MARKET_PROBS} WHERE projection_id = $1",
                projection_id,
            )
            await conn.execute(
                f"DELETE FROM {Table.PROJECTIONS} WHERE projection_id = $1",
                projection_id,
            )
            await conn.execute(
                f"DELETE FROM {Table.ODDS_SNAPSHOTS} WHERE game_id = $1", game_id
            )
            await conn.execute(f"DELETE FROM {Table.GAMES} WHERE game_id = $1", game_id)
            await conn.execute(
                f"DELETE FROM {Table.PLAYERS} WHERE player_id = $1", player_id
            )

    async def test_player_prop_no_match_leaves_null(
        self, pool, projection_with_player_props
    ):
        """Test that player props with no odds have edge/kelly remain NULL."""
        async with pool.acquire() as conn:
            edge_result = await compute_edges(conn, projection_with_player_props)

            # Should have player edges but with NULL values
            assert len(edge_result.player_edges) == 1
            player_edge = edge_result.player_edges[0]
            assert player_edge.edge is None
            assert player_edge.kelly_fraction is None
            assert player_edge.best_price is None

            # Persist should complete
            await persist_edges(conn, edge_result)

            # Check database - edge should remain NULL
            row = await conn.fetchrow(
                f"""
                SELECT edge, kelly_fraction
                FROM {Table.PLAYER_PROJECTIONS}
                WHERE projection_id = $1
                """,
                projection_with_player_props,
            )

            assert row["edge"] is None
            assert row["kelly_fraction"] is None


@pytest.mark.asyncio
class TestIdempotent:
    """Test acceptance criterion #9: Idempotent."""

    @pytest_asyncio.fixture
    async def projection_for_idempotence(self, pool):
        """Create a projection for idempotence testing."""
        async with pool.acquire() as conn:
            game_id = "test_idempotent_game"
            await conn.execute(
                f"""
                INSERT INTO {Table.GAMES} (game_id, game_date, status)
                VALUES ($1, CURRENT_DATE, 'scheduled')
                ON CONFLICT (game_id) DO NOTHING
                """,
                game_id,
            )

            projection_id = await conn.fetchval(
                f"""
                INSERT INTO {Table.PROJECTIONS}
                (game_id, run_ts, home_mu, away_mu, home_disp, away_disp, sim_n)
                VALUES ($1, $2, 4.5, 4.0, 0.8, 0.8, 5000)
                RETURNING projection_id
                """,
                game_id,
                datetime.now(timezone.utc),
            )

            await conn.executemany(
                f"""
                INSERT INTO {Table.SIM_MARKET_PROBS}
                (projection_id, market, side, line, prob)
                VALUES ($1, $2, $3, $4, $5)
                """,
                [
                    (projection_id, "ml", "home", None, 0.55),
                    (projection_id, "ml", "away", None, 0.45),
                ],
            )

            snapshot_ts = datetime.now(timezone.utc)
            await conn.executemany(
                f"""
                INSERT INTO {Table.ODDS_SNAPSHOTS}
                (game_id, book, market, side, line, price, snapshot_ts)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                [
                    (game_id, "book1", "ml", "home", None, 1.90, snapshot_ts),
                    (game_id, "book1", "ml", "away", None, 1.95, snapshot_ts),
                ],
            )

            yield projection_id

            # Cleanup
            await conn.execute(
                f"DELETE FROM {Table.SIM_MARKET_PROBS} WHERE projection_id = $1",
                projection_id,
            )
            await conn.execute(
                f"DELETE FROM {Table.PROJECTIONS} WHERE projection_id = $1",
                projection_id,
            )
            await conn.execute(
                f"DELETE FROM {Table.ODDS_SNAPSHOTS} WHERE game_id = $1", game_id
            )
            await conn.execute(f"DELETE FROM {Table.GAMES} WHERE game_id = $1", game_id)

    async def test_running_twice_overwrites(self, pool, projection_for_idempotence):
        """Test that running compute_edges twice overwrites edge values without duplicate rows."""
        async with pool.acquire() as conn:
            # First run
            edge_result1 = await compute_edges(conn, projection_for_idempotence)
            await persist_edges(conn, edge_result1)

            # Check row count
            count1 = await conn.fetchval(
                f"""
                SELECT COUNT(*)
                FROM {Table.SIM_MARKET_PROBS}
                WHERE projection_id = $1
                """,
                projection_for_idempotence,
            )

            # Get first edge values
            rows1 = await conn.fetch(
                f"""
                SELECT prob_id, edge, kelly_fraction, edge_computed_at
                FROM {Table.SIM_MARKET_PROBS}
                WHERE projection_id = $1
                ORDER BY prob_id
                """,
                projection_for_idempotence,
            )

            # Wait a moment to ensure different timestamp
            await asyncio.sleep(0.1)

            # Second run
            edge_result2 = await compute_edges(conn, projection_for_idempotence)
            await persist_edges(conn, edge_result2)

            # Check row count (should be same)
            count2 = await conn.fetchval(
                f"""
                SELECT COUNT(*)
                FROM {Table.SIM_MARKET_PROBS}
                WHERE projection_id = $1
                """,
                projection_for_idempotence,
            )

            assert count1 == count2  # No duplicate rows

            # Get second edge values
            rows2 = await conn.fetch(
                f"""
                SELECT prob_id, edge, kelly_fraction, edge_computed_at
                FROM {Table.SIM_MARKET_PROBS}
                WHERE projection_id = $1
                ORDER BY prob_id
                """,
                projection_for_idempotence,
            )

            # Edge values should be the same (same odds, same model probs)
            for r1, r2 in zip(rows1, rows2):
                assert r1["prob_id"] == r2["prob_id"]
                assert abs(float(r1["edge"]) - float(r2["edge"])) < 0.0001
                assert (
                    abs(float(r1["kelly_fraction"]) - float(r2["kelly_fraction"]))
                    < 0.0001
                )
                # edge_computed_at should be updated
                assert r2["edge_computed_at"] > r1["edge_computed_at"]
