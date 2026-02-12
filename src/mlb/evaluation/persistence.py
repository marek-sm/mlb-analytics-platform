"""Persistence layer for evaluation results.

Writes evaluation metrics to the eval_results table with idempotent upserts.
"""

import json
import logging

import asyncpg

from mlb.db.models import Table
from mlb.evaluation.backtest import EvalReport

logger = logging.getLogger(__name__)


async def persist_eval_report(
    conn: asyncpg.Connection,
    report: EvalReport,
) -> None:
    """Persist an evaluation report to the database.

    Writes 6 rows to eval_results, one for each metric:
    - log_loss
    - brier
    - ece
    - tail_acc_low
    - tail_acc_high
    - clv

    Args:
        conn: Database connection
        report: EvalReport to persist

    Notes:
        - Upserts on (eval_date, market, metric) to ensure idempotency (D-042)
        - Skips metrics that are None (e.g., insufficient tail samples)
        - Meta field stores JSON-serializable metadata
    """
    # Prepare rows to insert
    rows_to_insert = []

    if report.log_loss is not None:
        rows_to_insert.append(
            (
                report.eval_date,
                report.market,
                "log_loss",
                report.log_loss,
                report.sample_n,
                json.dumps(report.meta),
            )
        )

    if report.brier_score is not None:
        rows_to_insert.append(
            (
                report.eval_date,
                report.market,
                "brier",
                report.brier_score,
                report.sample_n,
                json.dumps(report.meta),
            )
        )

    if report.ece is not None:
        rows_to_insert.append(
            (
                report.eval_date,
                report.market,
                "ece",
                report.ece,
                report.sample_n,
                json.dumps(report.meta),
            )
        )

    if report.tail_low_acc is not None:
        rows_to_insert.append(
            (
                report.eval_date,
                report.market,
                "tail_acc_low",
                report.tail_low_acc,
                report.meta.get("tail_low_n", 0),
                json.dumps(report.meta),
            )
        )

    if report.tail_high_acc is not None:
        rows_to_insert.append(
            (
                report.eval_date,
                report.market,
                "tail_acc_high",
                report.tail_high_acc,
                report.meta.get("tail_high_n", 0),
                json.dumps(report.meta),
            )
        )

    if report.median_clv is not None:
        rows_to_insert.append(
            (
                report.eval_date,
                report.market,
                "clv",
                report.median_clv,
                report.meta.get("clv_sample_n", 0),
                json.dumps(report.meta),
            )
        )

    if not rows_to_insert:
        logger.warning(
            f"No metrics to persist for {report.market} {report.start_date} to {report.end_date}"
        )
        return

    # Upsert rows using INSERT ... ON CONFLICT (FC-20)
    # Per-metric upsert prevents data loss on partial writes
    await conn.executemany(
        f"""
        INSERT INTO {Table.EVAL_RESULTS}
        (eval_date, market, metric, value, sample_n, meta)
        VALUES ($1, $2, $3, $4, $5, $6::jsonb)
        ON CONFLICT (eval_date, market, metric)
        DO UPDATE SET
            value = EXCLUDED.value,
            sample_n = EXCLUDED.sample_n,
            meta = EXCLUDED.meta
        """,
        rows_to_insert,
    )

    logger.info(
        f"Persisted {len(rows_to_insert)} metrics for {report.market} "
        f"{report.start_date} to {report.end_date} (n={report.sample_n})"
    )
