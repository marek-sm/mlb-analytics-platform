"""Stripe webhook handler and event processing."""

import logging
from datetime import datetime
from typing import Optional

import discord
import stripe
from aiohttp import web

from mlb.config.settings import get_config
from mlb.payments.sync import sync_subscription

logger = logging.getLogger(__name__)


async def handle_webhook(
    payload: bytes,
    sig_header: str,
    bot_client: Optional[discord.Client] = None,
) -> web.Response:
    """Handle and verify Stripe webhook events.

    Verifies webhook signature, routes events to appropriate handlers,
    and returns appropriate HTTP responses.

    Args:
        payload: Raw webhook payload bytes
        sig_header: Stripe-Signature header value
        bot_client: Optional Discord bot client for role sync

    Returns:
        aiohttp.web.Response (200 for success, 400 for errors)
    """
    config = get_config()

    # Verify webhook signature
    try:
        event = stripe.Webhook.construct_event(
            payload,
            sig_header,
            config.stripe_webhook_secret.get_secret_value(),
        )
    except ValueError:
        logger.error("Invalid webhook payload")
        return web.Response(status=400, text="Invalid payload")
    except stripe.SignatureVerificationError:
        logger.error("Invalid webhook signature")
        return web.Response(status=400, text="Invalid signature")

    event_type = event["type"]
    logger.info(f"Received webhook: {event_type}")

    # Route event to handler
    try:
        if event_type == "checkout.session.completed":
            await _handle_checkout_completed(event["data"]["object"], bot_client)
        elif event_type == "invoice.paid":
            await _handle_invoice_paid(event["data"]["object"], bot_client)
        elif event_type == "invoice.payment_failed":
            await _handle_invoice_payment_failed(event["data"]["object"], bot_client)
        elif event_type == "customer.subscription.updated":
            await _handle_subscription_updated(event["data"]["object"], bot_client)
        elif event_type == "customer.subscription.deleted":
            await _handle_subscription_deleted(event["data"]["object"], bot_client)
        else:
            # Unknown event type - acknowledge but don't process
            logger.info(f"Unhandled event type: {event_type}")

        return web.Response(status=200, text="OK")

    except Exception as e:
        logger.exception(f"Error processing webhook {event_type}: {e}")
        # Return 500 so Stripe will retry
        return web.Response(status=500, text="Internal error")


async def _handle_checkout_completed(
    session: dict,
    bot_client: Optional[discord.Client],
) -> None:
    """Handle checkout.session.completed event.

    Creates initial subscription record with paid tier.

    Args:
        session: Stripe Checkout Session object
        bot_client: Optional Discord bot client
    """
    discord_user_id = session.get("client_reference_id")
    if not discord_user_id:
        logger.warning("checkout.session.completed missing client_reference_id - skipping")
        return

    stripe_customer_id = session.get("customer")
    subscription_id = session.get("subscription")

    # Fetch subscription details to get current_period_end
    subscription = stripe.Subscription.retrieve(subscription_id)

    await sync_subscription(
        discord_user_id=discord_user_id,
        stripe_customer_id=stripe_customer_id,
        tier="paid",
        status="active",
        current_period_end=datetime.fromtimestamp(subscription["current_period_end"]),
        bot_client=bot_client,
    )


async def _handle_invoice_paid(
    invoice: dict,
    bot_client: Optional[discord.Client],
) -> None:
    """Handle invoice.paid event.

    Updates subscription status to active and refreshes period end.

    Args:
        invoice: Stripe Invoice object
        bot_client: Optional Discord bot client
    """
    stripe_customer_id = invoice.get("customer")
    subscription_id = invoice.get("subscription")

    if not stripe_customer_id or not subscription_id:
        logger.warning("invoice.paid missing customer or subscription - skipping")
        return

    # Fetch subscription to get discord_user_id from metadata and period end
    subscription = stripe.Subscription.retrieve(subscription_id)

    # Get discord_user_id from customer metadata or subscription metadata
    customer = stripe.Customer.retrieve(stripe_customer_id)
    discord_user_id = customer.get("metadata", {}).get("discord_user_id")

    if not discord_user_id:
        # Try to get from database by stripe_customer_id
        from mlb.db.models import Table
        from mlb.db.pool import get_pool

        pool = await get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT discord_user_id
                FROM {Table.SUBSCRIPTIONS}
                WHERE stripe_customer_id = $1
                """,
                stripe_customer_id,
            )
            if row:
                discord_user_id = row["discord_user_id"]

    if not discord_user_id:
        logger.warning(
            f"invoice.paid: cannot find discord_user_id for customer {stripe_customer_id}"
        )
        return

    await sync_subscription(
        discord_user_id=discord_user_id,
        stripe_customer_id=stripe_customer_id,
        tier="paid",
        status="active",
        current_period_end=datetime.fromtimestamp(subscription["current_period_end"]),
        bot_client=bot_client,
    )


async def _handle_invoice_payment_failed(
    invoice: dict,
    bot_client: Optional[discord.Client],
) -> None:
    """Handle invoice.payment_failed event.

    Updates subscription status to past_due.

    Args:
        invoice: Stripe Invoice object
        bot_client: Optional Discord bot client
    """
    stripe_customer_id = invoice.get("customer")
    subscription_id = invoice.get("subscription")

    if not stripe_customer_id or not subscription_id:
        logger.warning("invoice.payment_failed missing customer or subscription - skipping")
        return

    # Fetch subscription
    subscription = stripe.Subscription.retrieve(subscription_id)

    # Get discord_user_id from database
    from mlb.db.models import Table
    from mlb.db.pool import get_pool

    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            f"""
            SELECT discord_user_id
            FROM {Table.SUBSCRIPTIONS}
            WHERE stripe_customer_id = $1
            """,
            stripe_customer_id,
        )

    if not row:
        logger.warning(
            f"invoice.payment_failed: customer {stripe_customer_id} not in database"
        )
        return

    discord_user_id = row["discord_user_id"]

    await sync_subscription(
        discord_user_id=discord_user_id,
        stripe_customer_id=stripe_customer_id,
        tier="paid",  # Keep tier as paid
        status="past_due",
        current_period_end=datetime.fromtimestamp(subscription["current_period_end"]),
        bot_client=bot_client,
    )


async def _handle_subscription_updated(
    subscription: dict,
    bot_client: Optional[discord.Client],
) -> None:
    """Handle customer.subscription.updated event.

    Syncs subscription status from Stripe.

    Args:
        subscription: Stripe Subscription object
        bot_client: Optional Discord bot client
    """
    stripe_customer_id = subscription.get("customer")

    if not stripe_customer_id:
        logger.warning("subscription.updated missing customer - skipping")
        return

    # Get discord_user_id from database
    from mlb.db.models import Table
    from mlb.db.pool import get_pool

    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            f"""
            SELECT discord_user_id
            FROM {Table.SUBSCRIPTIONS}
            WHERE stripe_customer_id = $1
            """,
            stripe_customer_id,
        )

    if not row:
        logger.warning(
            f"subscription.updated: customer {stripe_customer_id} not in database"
        )
        return

    discord_user_id = row["discord_user_id"]

    # Map Stripe status to our status
    stripe_status = subscription.get("status")
    if stripe_status in ("active", "trialing"):
        status = "active"
        tier = "paid"
    elif stripe_status in ("canceled", "unpaid"):
        status = "cancelled"
        tier = "free"
    elif stripe_status == "past_due":
        status = "past_due"
        tier = "paid"
    else:
        logger.warning(f"Unknown Stripe status: {stripe_status}")
        status = "cancelled"
        tier = "free"

    await sync_subscription(
        discord_user_id=discord_user_id,
        stripe_customer_id=stripe_customer_id,
        tier=tier,
        status=status,
        current_period_end=datetime.fromtimestamp(subscription["current_period_end"]),
        bot_client=bot_client,
    )


async def _handle_subscription_deleted(
    subscription: dict,
    bot_client: Optional[discord.Client],
) -> None:
    """Handle customer.subscription.deleted event.

    Sets tier to free and status to cancelled.

    Args:
        subscription: Stripe Subscription object
        bot_client: Optional Discord bot client
    """
    stripe_customer_id = subscription.get("customer")

    if not stripe_customer_id:
        logger.warning("subscription.deleted missing customer - skipping")
        return

    # Get discord_user_id from database
    from mlb.db.models import Table
    from mlb.db.pool import get_pool

    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            f"""
            SELECT discord_user_id
            FROM {Table.SUBSCRIPTIONS}
            WHERE stripe_customer_id = $1
            """,
            stripe_customer_id,
        )

    if not row:
        logger.warning(
            f"subscription.deleted: customer {stripe_customer_id} not in database"
        )
        return

    discord_user_id = row["discord_user_id"]

    await sync_subscription(
        discord_user_id=discord_user_id,
        stripe_customer_id=stripe_customer_id,
        tier="free",
        status="cancelled",
        current_period_end=None,
        bot_client=bot_client,
    )
