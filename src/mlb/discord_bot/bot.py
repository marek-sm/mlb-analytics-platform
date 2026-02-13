"""Discord bot lifecycle management.

Handles bot startup, shutdown, guild/channel setup, and event loop.
Provides interface for scheduler to trigger pick publishing.
"""

import asyncio
import logging
import signal
from typing import Optional

import discord
from discord.ext import commands

from mlb.config.settings import get_config
from mlb.discord_bot.channels import ensure_channels
from mlb.discord_bot.publisher import Publisher

logger = logging.getLogger(__name__)


class MLBPicksBot(commands.Bot):
    """Discord bot for publishing MLB picks.

    Publish-only bot in v1 - no user commands or interactive features.
    Connects to configured guild, ensures channels exist, and provides
    publishing interface for the scheduler.
    """

    def __init__(self):
        """Initialize bot with minimal intents (no message content needed)."""
        intents = discord.Intents.default()
        intents.guilds = True
        intents.members = True  # Needed for permission sync

        super().__init__(
            command_prefix="!",  # Unused in v1 but required
            intents=intents,
            help_command=None,  # Disable default help
        )

        self.config = get_config()
        self.publisher: Optional[Publisher] = None
        self._ready = asyncio.Event()
        self._shutdown = asyncio.Event()

    async def setup_hook(self) -> None:
        """Called when bot is setting up. Initialize resources here."""
        logger.info("Bot setup hook called")

    async def on_ready(self) -> None:
        """Called when bot has connected to Discord and is ready.

        Sets up guild channels and initializes publisher.
        """
        logger.info(f"Bot connected as {self.user}")

        # Get target guild
        if not self.config.discord_guild_id:
            logger.error("DISCORD_GUILD_ID not configured")
            await self.close()
            return

        guild = self.get_guild(int(self.config.discord_guild_id))
        if not guild:
            logger.error(
                f"Guild {self.config.discord_guild_id} not found. "
                "Ensure bot is invited to the server."
            )
            await self.close()
            return

        logger.info(f"Connected to guild: {guild.name}")

        # Ensure channels exist
        try:
            channels = await ensure_channels(guild)
            logger.info(f"Channels ready: {list(channels.keys())}")

            # Initialize publisher
            self.publisher = Publisher(channels)
            logger.info("Publisher initialized")

            # Mark bot as ready
            self._ready.set()
            logger.info("Bot ready for publishing")

        except Exception as e:
            logger.exception(f"Error during channel setup: {e}")
            await self.close()

    async def on_error(self, event: str, *args, **kwargs) -> None:
        """Called when an event handler raises an exception."""
        logger.exception(f"Error in event {event}")

    async def publish_picks_for_game(self, game_id: str) -> int:
        """Publish picks for a game (called by scheduler).

        Args:
            game_id: Game identifier

        Returns:
            Number of picks published

        Raises:
            RuntimeError: If bot is not ready
        """
        if not self.publisher:
            raise RuntimeError("Bot not ready - publisher not initialized")

        return await self.publisher.publish_picks(game_id)

    async def publish_daily_free_pick(self) -> bool:
        """Publish the daily free pick (called by scheduler).

        Returns:
            True if a free pick was posted, False otherwise

        Raises:
            RuntimeError: If bot is not ready
        """
        if not self.publisher:
            raise RuntimeError("Bot not ready - publisher not initialized")

        return await self.publisher.publish_free_pick()

    async def wait_ready(self, timeout: float = 30.0) -> bool:
        """Wait for bot to be ready.

        Args:
            timeout: Maximum seconds to wait

        Returns:
            True if bot became ready, False if timeout
        """
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            logger.error(f"Bot did not become ready within {timeout}s")
            return False

    def shutdown(self) -> None:
        """Signal bot to shut down gracefully."""
        logger.info("Shutdown signal received")
        self._shutdown.set()

    async def run_until_shutdown(self) -> None:
        """Run bot until shutdown signal received."""
        # Start bot in background
        bot_task = asyncio.create_task(self.start(self.config.discord_token.get_secret_value()))

        # Wait for shutdown signal
        await self._shutdown.wait()

        # Close bot
        logger.info("Shutting down bot...")
        await self.close()

        # Wait for bot task to complete
        try:
            await asyncio.wait_for(bot_task, timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("Bot task did not complete within timeout")
            bot_task.cancel()


def run_bot() -> None:
    """Run the Discord bot with signal handling.

    Blocks until SIGTERM/SIGINT received or bot exits.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    bot = MLBPicksBot()

    # Register signal handlers
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}")
        bot.shutdown()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        loop.run_until_complete(bot.run_until_shutdown())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        loop.close()
        logger.info("Bot stopped")


if __name__ == "__main__":
    run_bot()
