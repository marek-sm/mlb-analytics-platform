"""Discord channel creation and tier-based permission management."""

import logging

import discord

from mlb.config.settings import get_config
from mlb.db.models import Table
from mlb.db.pool import get_pool

logger = logging.getLogger(__name__)


async def ensure_channels(guild: discord.Guild) -> dict[str, discord.TextChannel]:
    """Ensure all required channels exist in the guild.

    Creates missing channels with appropriate permissions. Paid channels
    are restricted to bot role + paid subscriber role. Free channels are
    visible to everyone.

    Args:
        guild: Discord guild to set up channels in

    Returns:
        Dictionary mapping channel names to TextChannel objects

    Raises:
        discord.HTTPException: On Discord API errors
    """
    config = get_config()

    # Define required channels with their visibility
    channel_specs = {
        config.free_pick_channel: "public",
        config.announcements_channel: "public",
    }

    # Add paid channels
    for channel_name in config.paid_channels:
        channel_specs[channel_name] = "paid"

    # Get existing channels
    existing_channels = {ch.name: ch for ch in guild.text_channels}

    # Dictionary to return
    channels = {}

    for channel_name, visibility in channel_specs.items():
        if channel_name in existing_channels:
            # Channel exists
            channels[channel_name] = existing_channels[channel_name]
            logger.info(f"Channel #{channel_name} already exists")
        else:
            # Create channel with appropriate permissions
            overwrites = {}

            if visibility == "paid":
                # Paid channels: deny @everyone, allow bot
                overwrites[guild.default_role] = discord.PermissionOverwrite(
                    read_messages=False
                )
                # Bot needs to be able to send messages
                if guild.me:
                    overwrites[guild.me] = discord.PermissionOverwrite(
                        read_messages=True,
                        send_messages=True,
                        embed_links=True,
                    )

            # Create the channel
            channel = await guild.create_text_channel(
                name=channel_name,
                overwrites=overwrites,
            )
            channels[channel_name] = channel
            logger.info(f"Created channel #{channel_name} ({visibility})")

    return channels


async def sync_member_permissions(
    guild: discord.Guild,
    member: discord.Member,
) -> None:
    """Sync a member's channel permissions based on subscription tier.

    Queries the subscriptions table and grants/removes access to paid channels
    based on tier and status.

    Args:
        guild: Discord guild
        member: Discord member to sync permissions for

    Raises:
        discord.HTTPException: On Discord API errors
    """
    config = get_config()
    pool = await get_pool()

    # Query subscription status
    async with pool.acquire() as conn:
        sub = await conn.fetchrow(
            f"""
            SELECT tier, status
            FROM {Table.SUBSCRIPTIONS}
            WHERE discord_user_id = $1
            """,
            str(member.id),
        )

    # Determine if user should have paid access
    has_paid_access = False
    if sub:
        has_paid_access = sub["tier"] == "paid" and sub["status"] == "active"

    # Get paid channels
    paid_channel_names = set(config.paid_channels)
    paid_channels = [
        ch for ch in guild.text_channels if ch.name in paid_channel_names
    ]

    # Update permissions for each paid channel
    for channel in paid_channels:
        if has_paid_access:
            # Grant read access
            await channel.set_permissions(
                member,
                read_messages=True,
                overwrite=True,
            )
            logger.debug(f"Granted {member.name} access to #{channel.name}")
        else:
            # Remove any existing overwrites (fall back to @everyone deny)
            await channel.set_permissions(member, overwrite=None)
            logger.debug(f"Revoked {member.name} access to #{channel.name}")

    logger.info(
        f"Synced permissions for {member.name} (paid_access={has_paid_access})"
    )
