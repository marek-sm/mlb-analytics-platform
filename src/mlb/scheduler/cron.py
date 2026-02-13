"""Cron-compatible entry points for scheduled runs.

Provides callable functions with no arguments for cron integration:
- night_before_run()
- morning_run()
- midday_run()
- nightly_eval_run()

Each function is importable and callable for crontab or scheduler integration.
"""

import asyncio
import logging

from mlb.scheduler.pipeline import run_daily_eval, run_global

logger = logging.getLogger(__name__)


def night_before_run() -> None:
    """Entry point for night-before global run (~10 PM ET).

    Callable with no arguments for cron integration.
    """
    logger.info("Cron: night_before_run triggered")
    asyncio.run(run_global("night_before"))


def morning_run() -> None:
    """Entry point for morning global run (~8 AM ET).

    Callable with no arguments for cron integration.
    """
    logger.info("Cron: morning_run triggered")
    asyncio.run(run_global("morning"))


def midday_run() -> None:
    """Entry point for midday global run (~12 PM ET).

    Callable with no arguments for cron integration.
    """
    logger.info("Cron: midday_run triggered")
    asyncio.run(run_global("midday"))


def nightly_eval_run() -> None:
    """Entry point for nightly evaluation run (after last game).

    Callable with no arguments for cron integration.
    Should be scheduled after typical last game time (e.g., 2 AM ET).
    """
    logger.info("Cron: nightly_eval_run triggered")
    asyncio.run(run_daily_eval())
