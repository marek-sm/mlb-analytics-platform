"""Lightweight HTTP server for Stripe webhook endpoint."""

import asyncio
import logging
import signal
from typing import Optional

import discord
from aiohttp import web

from mlb.config.settings import get_config
from mlb.payments.webhooks import handle_webhook

logger = logging.getLogger(__name__)


async def webhook_endpoint(request: web.Request) -> web.Response:
    """Handle POST /webhooks/stripe.

    Args:
        request: aiohttp request

    Returns:
        aiohttp.web.Response
    """
    # Get signature header
    sig_header = request.headers.get("Stripe-Signature")
    if not sig_header:
        logger.error("Missing Stripe-Signature header")
        return web.Response(status=400, text="Missing signature")

    # Read raw payload
    payload = await request.read()

    # Get bot client from app state (if available)
    bot_client = request.app.get("bot_client")

    # Handle webhook
    return await handle_webhook(payload, sig_header, bot_client)


async def create_app(bot_client: Optional[discord.Client] = None) -> web.Application:
    """Create aiohttp application with webhook route.

    Args:
        bot_client: Optional Discord bot client for role sync

    Returns:
        Configured aiohttp Application
    """
    app = web.Application()
    app.router.add_post("/webhooks/stripe", webhook_endpoint)

    # Store bot client in app state if provided
    if bot_client:
        app["bot_client"] = bot_client

    return app


async def run_server(
    bot_client: Optional[discord.Client] = None,
    shutdown_event: Optional[asyncio.Event] = None,
) -> None:
    """Run webhook server until shutdown signal.

    Args:
        bot_client: Optional Discord bot client for role sync
        shutdown_event: Optional event to signal shutdown
    """
    config = get_config()
    app = await create_app(bot_client)

    runner = web.AppRunner(app)
    await runner.setup()

    site = web.TCPSite(runner, "0.0.0.0", config.webhook_server_port)
    await site.start()

    logger.info(f"Webhook server listening on port {config.webhook_server_port}")

    # Wait for shutdown signal
    if shutdown_event:
        await shutdown_event.wait()
    else:
        # Run forever if no shutdown event provided
        await asyncio.Event().wait()

    # Cleanup
    logger.info("Shutting down webhook server...")
    await runner.cleanup()


def main() -> None:
    """Run webhook server as standalone process.

    Blocks until SIGTERM/SIGINT received.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    shutdown_event = asyncio.Event()

    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}")
        shutdown_event.set()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        loop.run_until_complete(run_server(shutdown_event=shutdown_event))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        loop.close()
        logger.info("Webhook server stopped")


if __name__ == "__main__":
    main()
