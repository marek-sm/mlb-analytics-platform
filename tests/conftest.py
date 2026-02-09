"""Pytest configuration and fixtures for database tests."""

import pytest_asyncio

from mlb.db.pool import get_pool, close_pool


@pytest_asyncio.fixture(scope="session")
async def pool():
    """
    Get database pool for tests (session-scoped).

    The pool is created once per test session and properly closed
    at the end to prevent connection leaks and event loop issues.
    """
    # Get the pool
    pool = await get_pool()
    yield pool

    # Cleanup: close the pool after all tests
    await close_pool()
