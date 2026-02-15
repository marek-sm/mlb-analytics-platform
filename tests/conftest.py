"""Pytest configuration and fixtures for database tests."""

import asyncio

import pytest_asyncio

from mlb.db.pool import get_pool, close_pool
from mlb.db.schema.migrate import migrate


@pytest_asyncio.fixture(scope="session")
async def pool():
    """
    Get database pool for tests (session-scoped).

    The pool is created once per test session and properly closed
    at the end to prevent connection leaks and event loop issues.

    Migrations are run after creating the pool to ensure all tables exist.
    """
    # Get the pool
    pool = await get_pool()

    # Apply migrations to ensure tables exist
    await migrate()

    yield pool

    # Cleanup: close the pool after all tests with defensive timeout
    try:
        await asyncio.wait_for(close_pool(), timeout=10.0)
    except asyncio.TimeoutError:
        # close_pool() itself has timeout+terminate logic, but this is extra defense
        pass
