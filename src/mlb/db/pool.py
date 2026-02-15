"""Database connection pool factory and health check."""

import asyncio
import logging
from typing import Optional

import asyncpg

from mlb.config import get_config

logger = logging.getLogger(__name__)


_pool: Optional[asyncpg.Pool] = None


async def get_pool() -> asyncpg.Pool:
    """
    Get or create the database connection pool.

    On first call, initializes the pool and runs a health check.
    Subsequent calls return the existing pool.

    Returns:
        asyncpg.Pool: Initialized database connection pool

    Raises:
        asyncpg.PostgresError: If database is unreachable or health check fails
        asyncio.TimeoutError: If connection attempt exceeds 5 seconds
    """
    global _pool

    if _pool is not None:
        return _pool

    config = get_config()

    # Extract connection string
    dsn = str(config.db_dsn)

    # Create pool with timeout
    try:
        _pool = await asyncio.wait_for(
            asyncpg.create_pool(
                dsn,
                min_size=config.db_pool_min,
                max_size=config.db_pool_max,
            ),
            timeout=5.0,
        )
    except asyncio.TimeoutError:
        raise asyncio.TimeoutError(
            "Database connection timed out after 5 seconds. "
            "Ensure PostgreSQL is running and accessible."
        )

    if _pool is None:
        raise RuntimeError("Failed to create database pool")

    # Health check
    try:
        async with _pool.acquire() as conn:
            result = await conn.fetchval("SELECT 1")
            if result != 1:
                raise RuntimeError(f"Health check failed: expected 1, got {result}")
    except Exception as e:
        await _pool.close()
        _pool = None
        raise RuntimeError(f"Database health check failed: {e}") from e

    return _pool


async def close_pool() -> None:
    """
    Close the database connection pool if it exists.

    Attempts graceful close with a 5-second timeout. If the timeout occurs
    (e.g., due to leaked connections), forces termination to prevent hangs.
    """
    global _pool
    if _pool is not None:
        try:
            # Try graceful close with timeout
            await asyncio.wait_for(_pool.close(), timeout=5.0)
        except asyncio.TimeoutError:
            # Force termination if graceful close hangs
            logger.warning(
                "Pool close timed out after 5 seconds. "
                "Forcing termination (likely leaked connection)."
            )
            _pool.terminate()
        finally:
            # Always clear the pool reference
            _pool = None
