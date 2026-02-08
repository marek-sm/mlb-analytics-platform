"""Application entry point."""

import asyncio
import logging
import sys

from mlb.config import get_config
from mlb.db import get_pool
from mlb.db.pool import close_pool


async def boot() -> None:
    """
    Boot sequence: load config → validate → initialize pool → shutdown.

    Raises:
        SystemExit: On configuration or database errors
    """
    logger = logging.getLogger(__name__)

    try:
        # Load and validate configuration
        config = get_config()
        logger.info(f"Configuration loaded: env={config.env}")

        # Initialize database pool with health check
        pool = await get_pool()
        logger.info(
            f"Database pool initialized: min={config.db_pool_min}, max={config.db_pool_max}"
        )

        # Clean shutdown
        await close_pool()
        logger.info("Application shutdown complete")

    except Exception as e:
        logger.error(f"Boot sequence failed: {e}")
        raise SystemExit(1) from e


def main() -> None:
    """Main entry point with logging configuration."""
    # Configure logging
    config = get_config()
    logging.basicConfig(
        level=config.log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run boot sequence
    try:
        asyncio.run(boot())
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
        sys.exit(0)
    except SystemExit:
        raise
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
