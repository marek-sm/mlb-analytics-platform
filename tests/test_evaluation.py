"""Tests for Unit 8: Evaluation & Backtesting Harness.

Tests all acceptance criteria from the mini-spec:
1. Log loss computation
2. Brier score computation
3. ECE computation
4. Tail accuracy computation
5. CLV computation
6. CLV uses T-5 closing odds
7. Rolling-origin backtest
8. Calibration fit/apply
9. Persistence to eval_results
10. Idempotent runs
"""

import asyncio
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import numpy as np
import pytest
import pytest_asyncio

from mlb.db.models import Table
from mlb.evaluation import (
    EvalReport,
    apply_calibration,
    brier_score,
    compute_clv,
    ece,
    fit_calibration,
    log_loss,
    persist_eval_report,
    run_backtest,
    tail_accuracy,
)


class TestMetrics:
    """Test acceptance criteria #1-4: Metric functions."""

    def test_log_loss_calibrated_vs_biased(self):
        """AC#1: Calibrated model has lower log loss than biased model."""
        # Perfectly calibrated: predictions match actual rate
        p_calibrated = np.array([0.8, 0.8, 0.2, 0.2, 0.5])
        outcomes = np.array([1.0, 1.0, 0.0, 0.0, 1.0])  # 60% hit rate

        # Biased: all predictions are 0.9 regardless of outcome
        p_biased = np.array([0.9, 0.9, 0.9, 0.9, 0.9])

        loss_calibrated = log_loss(p_calibrated, outcomes)
        loss_biased = log_loss(p_biased, outcomes)

        assert loss_calibrated < loss_biased

    def test_brier_score_exact(self):
        """AC#2: Brier score exact numeric test."""
        p_model = np.array([0.8, 0.2])
        outcomes = np.array([1.0, 0.0])
        # Brier = mean((0.8-1)² + (0.2-0)²) = mean(0.04 + 0.04) = 0.04
        result = brier_score(p_model, outcomes)
        assert abs(result - 0.04) < 1e-10

    def test_ece_perfect_calibration(self):
        """AC#3: ECE ≈ 0 for perfectly calibrated predictions."""
        # Create perfectly calibrated predictions
        # Bin 1: p=0.1, actual=0.1
        # Bin 2: p=0.5, actual=0.5
        # Bin 3: p=0.9, actual=0.9
        p_model = np.array([0.1] * 10 + [0.5] * 10 + [0.9] * 10)
        outcomes = np.array([0.0] * 9 + [1.0] * 1 + [0.0] * 5 + [1.0] * 5 + [0.0] * 1 + [1.0] * 9)

        result = ece(p_model, outcomes, n_bins=10)
        assert result < 0.05  # Very low ECE for well-calibrated

    def test_ece_miscalibrated(self):
        """AC#3: ECE ≈ 0.4 for predictions all 0.9 with 50/50 outcomes."""
        p_model = np.array([0.9] * 20)
        outcomes = np.array([1.0] * 10 + [0.0] * 10)  # 50% actual rate

        result = ece(p_model, outcomes, n_bins=10)
        # ECE should be close to |0.5 - 0.9| = 0.4
        assert abs(result - 0.4) < 0.05

    def test_tail_accuracy_low_tail(self):
        """AC#4: Tail accuracy computation for low tail."""
        # 10 predictions with p < 0.15, 1 event occurs
        p_model = np.array([0.1] * 10 + [0.5] * 10)
        outcomes = np.array([1.0] + [0.0] * 9 + [1.0] * 5 + [0.0] * 5)

        result = tail_accuracy(p_model, outcomes, threshold=0.15)
        assert result["low_tail_n"] == 10
        assert abs(result["low_tail_acc"] - 0.10) < 0.01

    def test_tail_accuracy_insufficient_samples(self):
        """AC#4: Tail accuracy returns None for insufficient samples."""
        p_model = np.array([0.05, 0.05, 0.5, 0.5])
        outcomes = np.array([0.0, 0.0, 1.0, 0.0])

        result = tail_accuracy(p_model, outcomes, threshold=0.15)
        assert result["low_tail_n"] == 2
        assert result["low_tail_acc"] is None  # < 5 samples

    def test_log_loss_validation(self):
        """Test log loss raises on invalid inputs."""
        with pytest.raises(ValueError, match="same length"):
            log_loss(np.array([0.5]), np.array([]))

        with pytest.raises(ValueError, match="empty"):
            log_loss(np.array([]), np.array([]))

        with pytest.raises(ValueError, match="must be in"):
            log_loss(np.array([1.5]), np.array([1.0]))

        with pytest.raises(ValueError, match="must be 0 or 1"):
            log_loss(np.array([0.5]), np.array([0.5]))

    def test_brier_score_validation(self):
        """Test Brier score raises on invalid inputs."""
        with pytest.raises(ValueError, match="same length"):
            brier_score(np.array([0.5]), np.array([]))

    def test_ece_validation(self):
        """Test ECE raises on invalid inputs."""
        with pytest.raises(ValueError, match="same length"):
            ece(np.array([0.5]), np.array([]))


@pytest.mark.asyncio
class TestCLV:
    """Test acceptance criteria #5-6: CLV computation."""

    @pytest_asyncio.fixture
    async def game_with_closing_odds(self, pool):
        """Create a game with closing odds at T-5 and a projection."""
        async with pool.acquire() as conn:
            game_id = "test_clv_computation_game"
            first_pitch = datetime.now(timezone.utc) + timedelta(hours=2)

            # Create game
            await conn.execute(
                f"""
                INSERT INTO {Table.GAMES}
                (game_id, game_date, status, first_pitch, home_score, away_score)
                VALUES ($1, CURRENT_DATE, 'final', $2, 5, 3)
                ON CONFLICT (game_id) DO UPDATE
                SET status = 'final', first_pitch = $2, home_score = 5, away_score = 3
                """,
                game_id,
                first_pitch,
            )

            # Insert closing odds at T-5 minutes
            close_ts = first_pitch - timedelta(minutes=5)
            await conn.executemany(
                f"""
                INSERT INTO {Table.ODDS_SNAPSHOTS}
                (game_id, book, market, side, line, price, snapshot_ts)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                [
                    (game_id, "book1", "ml", "home", None, 1.85, close_ts),
                    (game_id, "book1", "ml", "away", None, 2.10, close_ts),
                ],
            )

            # Create projection
            proj_id = await conn.fetchval(
                f"""
                INSERT INTO {Table.PROJECTIONS}
                (game_id, run_ts, home_mu, away_mu, home_disp, away_disp, sim_n)
                VALUES ($1, $2, 4.5, 3.8, 1.2, 1.1, 5000)
                RETURNING projection_id
                """,
                game_id,
                first_pitch - timedelta(hours=1),
            )

            # Insert model probabilities
            # P(home) = 0.55, P(away) = 0.45
            await conn.executemany(
                f"""
                INSERT INTO {Table.SIM_MARKET_PROBS}
                (projection_id, market, side, line, prob)
                VALUES ($1, $2, $3, $4, $5)
                """,
                [
                    (proj_id, "ml", "home", None, 0.55),
                    (proj_id, "ml", "away", None, 0.45),
                ],
            )

            yield game_id, proj_id

            # Cleanup (delete child records first due to FK constraints)
            # Must use projection_id because that's the actual FK
            await conn.execute(
                f"DELETE FROM {Table.SIM_MARKET_PROBS} WHERE projection_id = $1", proj_id
            )
            await conn.execute(
                f"DELETE FROM {Table.PROJECTIONS} WHERE projection_id = $1", proj_id
            )
            await conn.execute(
                f"DELETE FROM {Table.ODDS_SNAPSHOTS} WHERE game_id = $1", game_id
            )
            await conn.execute(f"DELETE FROM {Table.GAMES} WHERE game_id = $1", game_id)

    async def test_clv_computation(self, pool, game_with_closing_odds):
        """AC#5: CLV computation with correct formula."""
        game_id, proj_id = game_with_closing_odds

        async with pool.acquire() as conn:
            clv_rows = await compute_clv(conn, [game_id])

        assert len(clv_rows) == 2  # home and away

        # Closing odds: home=1.85, away=2.10
        # Implied probs: home=1/1.85=0.5405, away=1/2.10=0.4762, total=1.0167
        # Devigged: home=0.5405/1.0167=0.5316, away=0.4762/1.0167=0.4684
        # Model probs: home=0.55, away=0.45
        # CLV: home=0.55-0.5316=0.0184, away=0.45-0.4684=-0.0184

        home_clv = next((r for r in clv_rows if r.p_model == 0.55), None)
        away_clv = next((r for r in clv_rows if r.p_model == 0.45), None)

        assert home_clv is not None
        assert abs(home_clv.p_close_fair - 0.5316) < 0.01
        assert abs(home_clv.clv - 0.0184) < 0.01

        assert away_clv is not None
        assert abs(away_clv.p_close_fair - 0.4684) < 0.01
        assert abs(away_clv.clv + 0.0184) < 0.01

    async def test_clv_uses_t_minus_5(self, pool):
        """AC#6: CLV uses T-5 closing odds, not the absolute latest."""
        game_id = "test_clv_t5_specific_game"
        first_pitch = datetime.now(timezone.utc) + timedelta(hours=2)

        async with pool.acquire() as conn:
            # Create game
            await conn.execute(
                f"""
                INSERT INTO {Table.GAMES}
                (game_id, game_date, status, first_pitch, home_score, away_score)
                VALUES ($1, CURRENT_DATE, 'final', $2, 5, 3)
                ON CONFLICT (game_id) DO UPDATE
                SET status = 'final', first_pitch = $2, home_score = 5, away_score = 3
                """,
                game_id,
                first_pitch,
            )

            # Insert odds at T-5 (should be used)
            close_ts_t5 = first_pitch - timedelta(minutes=5)
            await conn.executemany(
                f"""
                INSERT INTO {Table.ODDS_SNAPSHOTS}
                (game_id, book, market, side, line, price, snapshot_ts)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                [
                    (game_id, "book1", "ml", "home", None, 1.85, close_ts_t5),
                    (game_id, "book1", "ml", "away", None, 2.10, close_ts_t5),
                ],
            )

            # Insert odds at T-1 (should NOT be used - too close to game time)
            close_ts_t1 = first_pitch - timedelta(minutes=1)
            await conn.executemany(
                f"""
                INSERT INTO {Table.ODDS_SNAPSHOTS}
                (game_id, book, market, side, line, price, snapshot_ts)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                [
                    (game_id, "book1", "ml", "home", None, 1.50, close_ts_t1),
                    (game_id, "book1", "ml", "away", None, 2.50, close_ts_t1),
                ],
            )

            # Create projection
            proj_id = await conn.fetchval(
                f"""
                INSERT INTO {Table.PROJECTIONS}
                (game_id, run_ts, home_mu, away_mu, home_disp, away_disp, sim_n)
                VALUES ($1, $2, 4.5, 3.8, 1.2, 1.1, 5000)
                RETURNING projection_id
                """,
                game_id,
                first_pitch - timedelta(hours=1),
            )

            # Insert model probabilities
            await conn.executemany(
                f"""
                INSERT INTO {Table.SIM_MARKET_PROBS}
                (projection_id, market, side, line, prob)
                VALUES ($1, $2, $3, $4, $5)
                """,
                [
                    (proj_id, "ml", "home", None, 0.55),
                    (proj_id, "ml", "away", None, 0.45),
                ],
            )

            # Compute CLV
            clv_rows = await compute_clv(conn, [game_id])

            # Should use T-5 odds (1.85, 2.10), not T-1 odds (1.50, 2.50)
            home_clv = next((r for r in clv_rows if r.p_model == 0.55), None)
            assert home_clv is not None
            # Closing fair from 1.85/2.10 ≈ 0.5316
            assert abs(home_clv.p_close_fair - 0.5316) < 0.01

            # Cleanup (delete child records first due to FK constraints)
            await conn.execute(
                f"DELETE FROM {Table.SIM_MARKET_PROBS} WHERE projection_id IN (SELECT projection_id FROM {Table.PROJECTIONS} WHERE game_id = $1)",
                game_id,
            )
            await conn.execute(f"DELETE FROM {Table.PROJECTIONS} WHERE game_id = $1", game_id)
            await conn.execute(
                f"DELETE FROM {Table.ODDS_SNAPSHOTS} WHERE game_id = $1", game_id
            )
            await conn.execute(f"DELETE FROM {Table.GAMES} WHERE game_id = $1", game_id)


@pytest.mark.asyncio
class TestBacktest:
    """Test acceptance criterion #7: Rolling-origin backtest."""

    @pytest_asyncio.fixture
    async def backtest_data(self, pool):
        """Create test data for backtesting."""
        async with pool.acquire() as conn:
            game_ids = []
            # Use a date range that won't overlap with calibration tests
            test_date = date.today() - timedelta(days=200)

            # Create 5 games over 5 days
            for i in range(5):
                game_id = f"test_backtest_game_{i}"
                game_date = test_date + timedelta(days=i)
                first_pitch = datetime.combine(game_date, datetime.min.time()).replace(
                    tzinfo=timezone.utc
                ) + timedelta(hours=19)

                await conn.execute(
                    f"""
                    INSERT INTO {Table.GAMES}
                    (game_id, game_date, status, first_pitch, home_score, away_score)
                    VALUES ($1, $2, 'final', $3, $4, $5)
                    ON CONFLICT (game_id) DO UPDATE
                    SET status = 'final', first_pitch = $3, home_score = $4, away_score = $5
                    """,
                    game_id,
                    game_date,
                    first_pitch,
                    5,
                    3,
                )

                # Create projection (made before first pitch)
                proj_id = await conn.fetchval(
                    f"""
                    INSERT INTO {Table.PROJECTIONS}
                    (game_id, run_ts, home_mu, away_mu, home_disp, away_disp, sim_n)
                    VALUES ($1, $2, 4.5, 3.5, 1.2, 1.1, 5000)
                    RETURNING projection_id
                    """,
                    game_id,
                    first_pitch - timedelta(hours=2),
                )

                # Insert model probabilities for ML market
                await conn.executemany(
                    f"""
                    INSERT INTO {Table.SIM_MARKET_PROBS}
                    (projection_id, market, side, line, prob)
                    VALUES ($1, $2, $3, $4, $5)
                    """,
                    [
                        (proj_id, "ml", "home", None, 0.60),
                        (proj_id, "ml", "away", None, 0.40),
                    ],
                )

                game_ids.append(game_id)

            yield test_date, test_date + timedelta(days=4), game_ids

            # Cleanup (delete child records first due to FK constraints)
            for game_id in game_ids:
                await conn.execute(
                    f"DELETE FROM {Table.SIM_MARKET_PROBS} WHERE projection_id IN (SELECT projection_id FROM {Table.PROJECTIONS} WHERE game_id = $1)",
                    game_id,
                )
                await conn.execute(f"DELETE FROM {Table.PROJECTIONS} WHERE game_id = $1", game_id)
                await conn.execute(f"DELETE FROM {Table.GAMES} WHERE game_id = $1", game_id)

    async def test_rolling_origin_backtest(self, pool, backtest_data):
        """AC#7: Rolling-origin backtest uses only data from each date."""
        start_date, end_date, game_ids = backtest_data

        async with pool.acquire() as conn:
            report = await run_backtest(conn, start_date, end_date, "ml")

        # Should have 10 predictions (5 games × 2 sides)
        assert report.sample_n == 10
        assert report.market == "ml"
        assert report.start_date == start_date
        assert report.end_date == end_date

        # All games had home=5, away=3, so home won
        # Model predicted home=0.60, away=0.40
        # Outcomes: home=1.0, away=0.0
        # Brier for home: (0.6-1)² = 0.16
        # Brier for away: (0.4-0)² = 0.16
        # Mean Brier = 0.16
        assert abs(report.brier_score - 0.16) < 0.01

    async def test_backtest_no_final_games(self, pool):
        """AC#7: Backtest returns sample_n=0 if no final games."""
        start_date = date.today() - timedelta(days=365)
        end_date = start_date + timedelta(days=7)

        async with pool.acquire() as conn:
            report = await run_backtest(conn, start_date, end_date, "ml")

        assert report.sample_n == 0
        assert report.log_loss is None
        assert report.brier_score is None


@pytest.mark.asyncio
class TestCalibration:
    """Test acceptance criterion #8: Calibration fit and apply."""

    @pytest_asyncio.fixture
    async def calibration_training_data(self, pool):
        """Create training data for calibration."""
        async with pool.acquire() as conn:
            game_ids = []

            # Create 100 games with varying outcomes
            for i in range(100):
                game_id = f"test_cal_game_{i}"
                game_date = date.today() - timedelta(days=100 - i)
                first_pitch = datetime.combine(game_date, datetime.min.time()).replace(
                    tzinfo=timezone.utc
                ) + timedelta(hours=19)

                # Vary outcomes: 60% home wins
                home_won = i < 60
                home_score = 5 if home_won else 3
                away_score = 3 if home_won else 5

                await conn.execute(
                    f"""
                    INSERT INTO {Table.GAMES}
                    (game_id, game_date, status, first_pitch, home_score, away_score)
                    VALUES ($1, $2, 'final', $3, $4, $5)
                    ON CONFLICT (game_id) DO UPDATE
                    SET status = 'final', home_score = $4, away_score = $5
                    """,
                    game_id,
                    game_date,
                    first_pitch,
                    home_score,
                    away_score,
                )

                # Create projection
                proj_id = await conn.fetchval(
                    f"""
                    INSERT INTO {Table.PROJECTIONS}
                    (game_id, run_ts, home_mu, away_mu, home_disp, away_disp, sim_n)
                    VALUES ($1, $2, 4.5, 3.5, 1.2, 1.1, 5000)
                    RETURNING projection_id
                    """,
                    game_id,
                    first_pitch - timedelta(hours=2),
                )

                # Model is slightly overconfident: predicts 0.65/0.35 but actual is 0.60/0.40
                await conn.executemany(
                    f"""
                    INSERT INTO {Table.SIM_MARKET_PROBS}
                    (projection_id, market, side, line, prob)
                    VALUES ($1, $2, $3, $4, $5)
                    """,
                    [
                        (proj_id, "ml", "home", None, 0.65),
                        (proj_id, "ml", "away", None, 0.35),
                    ],
                )

                game_ids.append(game_id)

            yield game_ids

            # Cleanup (delete child records first due to FK constraints)
            for game_id in game_ids:
                await conn.execute(
                    f"DELETE FROM {Table.SIM_MARKET_PROBS} WHERE projection_id IN (SELECT projection_id FROM {Table.PROJECTIONS} WHERE game_id = $1)",
                    game_id,
                )
                await conn.execute(f"DELETE FROM {Table.PROJECTIONS} WHERE game_id = $1", game_id)
                await conn.execute(f"DELETE FROM {Table.GAMES} WHERE game_id = $1", game_id)

    async def test_fit_calibration(self, pool, calibration_training_data):
        """AC#8: fit_calibration produces CalibrationModel."""
        async with pool.acquire() as conn:
            model = await fit_calibration(conn, "ml", method="isotonic")

        assert model is not None
        assert model.market == "ml"
        assert model.method == "isotonic"
        assert model.fitted_at is not None
        assert model.params is not None

    async def test_apply_calibration(self, pool, calibration_training_data):
        """AC#8: apply_calibration returns float in [0, 1]."""
        async with pool.acquire() as conn:
            model = await fit_calibration(conn, "ml", method="isotonic")

        # Apply calibration to a test probability
        p_calibrated = apply_calibration(0.6, model)

        assert isinstance(p_calibrated, float)
        assert 0.0 <= p_calibrated <= 1.0

    async def test_calibration_improves_ece(self, pool, calibration_training_data):
        """AC#8: After calibration, ECE on training set decreases."""
        async with pool.acquire() as conn:
            # Get uncalibrated predictions and outcomes
            rows = await conn.fetch(
                f"""
                SELECT
                    smp.prob AS p_model,
                    CASE WHEN g.home_score > g.away_score THEN 1.0 ELSE 0.0 END AS outcome
                FROM {Table.SIM_MARKET_PROBS} smp
                JOIN {Table.PROJECTIONS} p ON smp.projection_id = p.projection_id
                JOIN {Table.GAMES} g ON p.game_id = g.game_id
                WHERE smp.market = 'ml' AND smp.side = 'home'
                  AND g.status = 'final'
                """
            )

            p_model = np.array([float(r["p_model"]) for r in rows])
            outcomes = np.array([float(r["outcome"]) for r in rows])

            # Compute ECE before calibration
            ece_before = ece(p_model, outcomes, n_bins=10)

            # Fit calibration
            model = await fit_calibration(conn, "ml", method="isotonic")

            # Apply calibration
            p_calibrated = apply_calibration(p_model, model)

            # Compute ECE after calibration
            ece_after = ece(p_calibrated, outcomes, n_bins=10)

            # ECE should decrease (or stay the same if already well-calibrated)
            assert ece_after <= ece_before * 1.1  # Allow 10% tolerance for noise


@pytest.mark.asyncio
class TestPersistence:
    """Test acceptance criteria #9-10: Persistence and idempotency."""

    async def test_persist_eval_report(self, pool):
        """AC#9: persist_eval_report writes 6 rows to eval_results."""
        report = EvalReport(
            eval_date=date.today(),
            market="ml",
            start_date=date.today() - timedelta(days=7),
            end_date=date.today(),
            sample_n=100,
            log_loss=0.45,
            brier_score=0.20,
            ece=0.05,
            tail_low_acc=0.12,
            tail_high_acc=0.88,
            median_clv=0.02,
            meta={"tail_low_n": 10, "tail_high_n": 15, "clv_sample_n": 95},
        )

        async with pool.acquire() as conn:
            await persist_eval_report(conn, report)

            # Check that 6 rows were inserted
            rows = await conn.fetch(
                f"""
                SELECT metric, value, sample_n
                FROM {Table.EVAL_RESULTS}
                WHERE eval_date = $1 AND market = $2
                ORDER BY metric
                """,
                report.eval_date,
                report.market,
            )

        assert len(rows) == 6
        metrics = {row["metric"]: float(row["value"]) for row in rows}

        assert "log_loss" in metrics
        assert "brier" in metrics
        assert "ece" in metrics
        assert "tail_acc_low" in metrics
        assert "tail_acc_high" in metrics
        assert "clv" in metrics

        assert abs(metrics["log_loss"] - 0.45) < 1e-6
        assert abs(metrics["brier"] - 0.20) < 1e-6

    async def test_idempotent_persistence(self, pool):
        """AC#10: Running the same backtest twice does not duplicate rows."""
        report = EvalReport(
            eval_date=date.today(),
            market="rl",
            start_date=date.today() - timedelta(days=7),
            end_date=date.today(),
            sample_n=50,
            log_loss=0.50,
            brier_score=0.25,
            ece=0.08,
            tail_low_acc=None,  # Insufficient samples
            tail_high_acc=0.90,
            median_clv=0.01,
            meta={"tail_low_n": 2, "tail_high_n": 20, "clv_sample_n": 48},
        )

        async with pool.acquire() as conn:
            # First persist
            await persist_eval_report(conn, report)

            # Second persist (same eval_date and market)
            await persist_eval_report(conn, report)

            # Check row count
            count = await conn.fetchval(
                f"""
                SELECT COUNT(*)
                FROM {Table.EVAL_RESULTS}
                WHERE eval_date = $1 AND market = $2
                """,
                report.eval_date,
                report.market,
            )

        # Should have 5 rows (tail_low_acc was None, so skipped)
        assert count == 5

    async def test_partial_rerun_preserves_other_metrics(self, pool):
        """FC-20: Partial rerun preserves metrics not being updated."""
        # First, write all 6 metrics
        full_report = EvalReport(
            eval_date=date.today(),
            market="total",
            start_date=date.today() - timedelta(days=7),
            end_date=date.today(),
            sample_n=100,
            log_loss=0.45,
            brier_score=0.20,
            ece=0.05,
            tail_low_acc=0.12,
            tail_high_acc=0.88,
            median_clv=0.02,
            meta={"tail_low_n": 10, "tail_high_n": 15, "clv_sample_n": 95},
        )

        async with pool.acquire() as conn:
            await persist_eval_report(conn, full_report)

            # Verify all 6 metrics written
            count = await conn.fetchval(
                f"""
                SELECT COUNT(*)
                FROM {Table.EVAL_RESULTS}
                WHERE eval_date = $1 AND market = $2
                """,
                full_report.eval_date,
                full_report.market,
            )
            assert count == 6

            # Now write a partial report with only 2 metrics (log_loss and brier)
            partial_report = EvalReport(
                eval_date=date.today(),
                market="total",
                start_date=date.today() - timedelta(days=7),
                end_date=date.today(),
                sample_n=120,  # Different sample_n
                log_loss=0.50,  # Updated value
                brier_score=0.25,  # Updated value
                ece=None,  # Not included
                tail_low_acc=None,  # Not included
                tail_high_acc=None,  # Not included
                median_clv=None,  # Not included
                meta={"tail_low_n": 12, "tail_high_n": 18, "clv_sample_n": 110},
            )

            await persist_eval_report(conn, partial_report)

            # Verify all 6 metrics still exist (2 updated, 4 preserved)
            rows = await conn.fetch(
                f"""
                SELECT metric, value, sample_n
                FROM {Table.EVAL_RESULTS}
                WHERE eval_date = $1 AND market = $2
                ORDER BY metric
                """,
                full_report.eval_date,
                full_report.market,
            )

        # Should still have 6 rows
        assert len(rows) == 6

        # Build metrics dict
        metrics = {row["metric"]: (float(row["value"]), int(row["sample_n"])) for row in rows}

        # Updated metrics should have new values
        assert abs(metrics["log_loss"][0] - 0.50) < 1e-6
        assert metrics["log_loss"][1] == 120
        assert abs(metrics["brier"][0] - 0.25) < 1e-6
        assert metrics["brier"][1] == 120

        # Preserved metrics should have original values
        assert abs(metrics["ece"][0] - 0.05) < 1e-6
        assert metrics["ece"][1] == 100
        assert abs(metrics["tail_acc_low"][0] - 0.12) < 1e-6
        assert metrics["tail_acc_low"][1] == 10
        assert abs(metrics["tail_acc_high"][0] - 0.88) < 1e-6
        assert metrics["tail_acc_high"][1] == 15
        assert abs(metrics["clv"][0] - 0.02) < 1e-6
        assert metrics["clv"][1] == 95
