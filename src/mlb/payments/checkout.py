"""Stripe Checkout session creation for subscription signup."""

import logging

import stripe

from mlb.config.settings import get_config

logger = logging.getLogger(__name__)


def create_checkout_url(discord_user_id: str) -> str:
    """Create a Stripe Checkout Session for subscription signup.

    Creates a Checkout Session with the configured subscription price,
    passing the discord_user_id as client_reference_id so the webhook
    can link the Stripe payment to the Discord user.

    Args:
        discord_user_id: Discord user ID (as string)

    Returns:
        Stripe Checkout Session URL

    Raises:
        stripe.StripeError: On Stripe API errors
        ValueError: If required config is missing
    """
    config = get_config()

    # Validate required config
    if not config.stripe_secret.get_secret_value():
        raise ValueError("stripe_secret not configured")
    if not config.stripe_price_id:
        raise ValueError("stripe_price_id not configured")

    # Set Stripe API key
    stripe.api_key = config.stripe_secret.get_secret_value()

    # Create Checkout Session
    session = stripe.checkout.Session.create(
        mode="subscription",
        client_reference_id=discord_user_id,
        line_items=[
            {
                "price": config.stripe_price_id,
                "quantity": 1,
            }
        ],
        success_url=config.checkout_success_url,
        cancel_url=config.checkout_cancel_url,
    )

    logger.info(f"Created checkout session {session.id} for user {discord_user_id}")

    return session.url
