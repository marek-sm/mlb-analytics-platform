"""Subscription state synchronization and Discord role management."""

import logging
from datetime import datetime
from typing import Optional

import discord

from mlb.config.settings import get_config
from mlb.db.models import Table
from mlb.db.pool import get_pool

logger = logging.getLogger(__name__)


async def sync_subscription(
    discord_user_id: str,
    stripe_customer_id: str,
    tier: str,
    status: str,
    current_period_end: Optional[datetime],
    bot_client: Optional[discord.Client] = None,
) -> None:
    """Sync subscription state to database and Discord roles.

    Upserts the subscriptions table with current subscription data and
    grants/revokes Discord roles based on tier and status.

    Args:
        discord_user_id: Discord user ID
        stripe_customer_id: Stripe customer ID
        tier: Subscription tier ('free' or 'paid')
        status: Subscription status ('active', 'cancelled', 'past_due')
        current_period_end: End of current billing period (None for free tier)
        bot_client: Optional Discord bot client for role sync. If None, only
                   database is updated (role sync skipped with warning).

    Raises:
        asyncpg.PostgresError: On database errors
    """
    config = get_config()
    pool = await get_pool()

    # Upsert subscription in database
    async with pool.acquire() as conn:
        await conn.execute(
            f"""
            INSERT INTO {Table.SUBSCRIPTIONS}
                (discord_user_id, stripe_customer_id, tier, status, current_period_end)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (discord_user_id) DO UPDATE SET
                stripe_customer_id = EXCLUDED.stripe_customer_id,
                tier = EXCLUDED.tier,
                status = EXCLUDED.status,
                current_period_end = EXCLUDED.current_period_end,
                updated_at = now()
            """,
            discord_user_id,
            stripe_customer_id,
            tier,
            status,
            current_period_end,
        )

    logger.info(
        f"Synced subscription for user {discord_user_id}: "
        f"tier={tier}, status={status}, customer={stripe_customer_id}"
    )

    # Sync Discord role
    if bot_client is None:
        logger.warning(
            f"Discord bot client not provided - role sync skipped for {discord_user_id}"
        )
        return

    try:
        await _sync_discord_role(
            bot_client=bot_client,
            discord_user_id=discord_user_id,
            should_have_role=(tier == "paid" and status == "active"),
        )
    except Exception as e:
        # Log error but don't fail the webhook - database is source of truth
        logger.error(f"Failed to sync Discord role for {discord_user_id}: {e}")


async def _sync_discord_role(
    bot_client: discord.Client,
    discord_user_id: str,
    should_have_role: bool,
) -> None:
    """Grant or revoke Discord paid subscriber role.

    Args:
        bot_client: Discord bot client
        discord_user_id: Discord user ID
        should_have_role: True to grant role, False to revoke

    Raises:
        discord.HTTPException: On Discord API errors
        ValueError: If guild or role not found
    """
    config = get_config()

    # Get guild
    if not config.discord_guild_id:
        raise ValueError("discord_guild_id not configured")

    guild = bot_client.get_guild(int(config.discord_guild_id))
    if not guild:
        raise ValueError(f"Guild {config.discord_guild_id} not found")

    # Get member
    member = guild.get_member(int(discord_user_id))
    if not member:
        # Member not in guild - log warning and skip
        logger.warning(
            f"User {discord_user_id} not found in guild - role sync skipped"
        )
        return

    # Get or create paid subscriber role
    role_name = config.discord_paid_role_name
    role = discord.utils.get(guild.roles, name=role_name)

    if not role:
        # Create role if it doesn't exist
        role = await guild.create_role(
            name=role_name,
            reason="Paid subscriber role created by webhook handler",
        )
        logger.info(f"Created Discord role: {role_name}")

    # Check if member already has role
    has_role = role in member.roles

    # Grant or revoke role
    if should_have_role and not has_role:
        await member.add_roles(role, reason="Subscription activated")
        logger.info(f"Granted {role_name} role to {member.name}")
    elif not should_have_role and has_role:
        await member.remove_roles(role, reason="Subscription cancelled/expired")
        logger.info(f"Revoked {role_name} role from {member.name}")
    else:
        logger.debug(
            f"Role {role_name} already {'granted' if has_role else 'revoked'} "
            f"for {member.name}"
        )
