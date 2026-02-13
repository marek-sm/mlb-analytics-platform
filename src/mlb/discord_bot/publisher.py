"""Pick publishing logic for Discord channels.

Publishes team markets and player props after pipeline runs.
Respects publishing gate, anti-spam rules, and tier gating.
"""

import logging
from datetime import datetime, timedelta, timezone

import discord

from mlb.config.settings import get_config
from mlb.db.models import Market, PlayerStat, Side, Table
from mlb.db.pool import get_pool
from mlb.discord_bot.formatter import (
    format_player_prop_embed,
    format_team_market_embed,
)
from mlb.scheduler.gate import is_publishable

logger = logging.getLogger(__name__)


class Publisher:
    """Manages Discord message publishing with anti-spam tracking.

    Tracks published messages in-memory to avoid duplicates and enable
    message editing on projection updates.
    """

    def __init__(self, channels: dict[str, discord.TextChannel]):
        """Initialize publisher with channel references.

        Args:
            channels: Dictionary mapping channel names to TextChannel objects
        """
        self.channels = channels
        # Anti-spam: track (game_id, market, side, line) -> message_id
        self._message_cache: dict[tuple[str, str, str, float | None], int] = {}
        # Track daily free pick posted flag
        self._free_pick_posted_date: str | None = None

    async def publish_picks(self, game_id: str) -> int:
        """Publish all publishable picks for a game.

        Queries sim_market_probs and player_projections for the latest
        projection, filters by publishing gate and positive edge, and
        sends/updates Discord messages.

        Args:
            game_id: Game identifier

        Returns:
            Number of picks published (new messages + edited messages)

        Raises:
            discord.HTTPException: On Discord API errors
        """
        config = get_config()
        pool = await get_pool()
        published_count = 0

        async with pool.acquire() as conn:
            # Get latest projection for this game
            projection = await conn.fetchrow(
                f"""
                SELECT projection_id, edge_computed_at, sim_n
                FROM {Table.PROJECTIONS}
                WHERE game_id = $1
                ORDER BY run_ts DESC
                LIMIT 1
                """,
                game_id,
            )

            if not projection:
                logger.warning(f"No projection found for game {game_id}")
                return 0

            if not projection["edge_computed_at"]:
                logger.info(
                    f"Skipping game {game_id} - edge not computed (D-012)"
                )
                return 0

            projection_id = projection["projection_id"]
            sim_n = projection["sim_n"]

            # Get game details
            game = await conn.fetchrow(
                f"""
                SELECT
                    g.home_team_id,
                    g.away_team_id,
                    g.first_pitch,
                    ht.abbr as home_abbr,
                    at.abbr as away_abbr
                FROM {Table.GAMES} g
                JOIN {Table.TEAMS} ht ON g.home_team_id = ht.team_id
                JOIN {Table.TEAMS} at ON g.away_team_id = at.team_id
                WHERE g.game_id = $1
                """,
                game_id,
            )

            if not game:
                logger.warning(f"Game {game_id} not found in database")
                return 0

            # Publish team markets
            published_count += await self._publish_team_markets(
                projection_id=projection_id,
                game_id=game_id,
                home_team=game["home_abbr"],
                away_team=game["away_abbr"],
                first_pitch=game["first_pitch"],
                sim_n=sim_n,
                conn=conn,
            )

            # Publish player props
            published_count += await self._publish_player_props(
                projection_id=projection_id,
                game_id=game_id,
                home_team=game["home_abbr"],
                away_team=game["away_abbr"],
                first_pitch=game["first_pitch"],
                sim_n=sim_n,
                conn=conn,
            )

        return published_count

    async def _publish_team_markets(
        self,
        projection_id: int,
        game_id: str,
        home_team: str,
        away_team: str,
        first_pitch: datetime,
        sim_n: int,
        conn,
    ) -> int:
        """Publish team market picks for a projection."""
        config = get_config()
        published = 0

        # Query sim_market_probs for positive-edge plays
        rows = await conn.fetch(
            f"""
            SELECT
                market,
                side,
                line,
                prob,
                edge,
                kelly_fraction
            FROM {Table.SIM_MARKET_PROBS}
            WHERE projection_id = $1
              AND edge > 0
              AND kelly_fraction > 0
            ORDER BY edge DESC
            """,
            projection_id,
        )

        for row in rows:
            market = row["market"]
            side = row["side"]
            line = row["line"]

            # Check publishing gate
            if not await is_publishable(game_id, market, player_id=None):
                continue

            # Get best odds (stubbed for v1 - use prob to derive price)
            # In production, query odds_snapshots for best book/price
            model_prob = row["prob"]
            best_book = "Placeholder Book"
            best_price = 1.0 / model_prob if model_prob > 0 else 2.0

            # Determine target channel
            channel_name = self._get_team_market_channel(market)
            if not channel_name or channel_name not in self.channels:
                logger.warning(f"No channel configured for market {market}")
                continue

            channel = self.channels[channel_name]

            # Format embed
            embed = format_team_market_embed(
                game_id=game_id,
                home_team=home_team,
                away_team=away_team,
                market=market,
                side=side,
                line=line,
                model_prob=model_prob,
                edge=row["edge"],
                kelly_fraction=row["kelly_fraction"],
                best_book=best_book,
                best_price=best_price,
                first_pitch=first_pitch,
                sim_n=sim_n,
            )

            # Anti-spam: check if already posted
            cache_key = (game_id, market, side, line)

            if cache_key in self._message_cache:
                # Edit existing message
                try:
                    message = await channel.fetch_message(
                        self._message_cache[cache_key]
                    )
                    await message.edit(embed=embed)
                    logger.debug(f"Edited message for {cache_key}")
                    published += 1
                except discord.NotFound:
                    # Message was deleted - post new one
                    del self._message_cache[cache_key]
                    message = await channel.send(embed=embed)
                    self._message_cache[cache_key] = message.id
                    logger.info(f"Posted team market: {cache_key}")
                    published += 1
            else:
                # Post new message
                message = await channel.send(embed=embed)
                self._message_cache[cache_key] = message.id
                logger.info(f"Posted team market: {cache_key}")
                published += 1

        return published

    async def _publish_player_props(
        self,
        projection_id: int,
        game_id: str,
        home_team: str,
        away_team: str,
        first_pitch: datetime,
        sim_n: int,
        conn,
    ) -> int:
        """Publish player prop picks for a projection."""
        config = get_config()
        published = 0

        # Query player_projections for positive-edge plays
        rows = await conn.fetch(
            f"""
            SELECT
                pp.player_id,
                pp.p_start,
                pp.stat,
                pp.line,
                pp.prob_over,
                pp.edge,
                pp.kelly_fraction,
                pl.name as player_name
            FROM {Table.PLAYER_PROJECTIONS} pp
            JOIN {Table.PLAYERS} pl ON pp.player_id = pl.player_id
            WHERE pp.projection_id = $1
              AND pp.edge > 0
              AND pp.kelly_fraction > 0
            ORDER BY pp.edge DESC
            """,
            projection_id,
        )

        for row in rows:
            player_id = row["player_id"]
            stat = row["stat"]
            line = row["line"]

            # Check publishing gate (player props require lineup confirmation or high p_start)
            if not await is_publishable(game_id, stat, player_id=player_id):
                continue

            # Determine side (publish over if prob_over > 0.5, else under)
            prob_over = row["prob_over"]
            if prob_over > 0.5:
                side = "over"
                model_prob = prob_over
            else:
                side = "under"
                model_prob = 1.0 - prob_over

            # Get best odds (stubbed for v1)
            best_book = "Placeholder Book"
            best_price = 1.0 / model_prob if model_prob > 0 else 2.0

            # Determine target channel (hitter vs pitcher props)
            channel_name = self._get_player_prop_channel(stat)
            if not channel_name or channel_name not in self.channels:
                logger.warning(f"No channel configured for stat {stat}")
                continue

            channel = self.channels[channel_name]

            # Format embed
            embed = format_player_prop_embed(
                player_name=row["player_name"],
                stat=stat,
                line=line,
                side=side,
                p_start=row["p_start"],
                model_prob=model_prob,
                edge=row["edge"],
                kelly_fraction=row["kelly_fraction"],
                best_book=best_book,
                best_price=best_price,
                home_team=home_team,
                away_team=away_team,
                first_pitch=first_pitch,
                sim_n=sim_n,
            )

            # Anti-spam: cache key for player props
            cache_key = (game_id, f"prop_{stat}", str(player_id), line)

            if cache_key in self._message_cache:
                # Edit existing message
                try:
                    message = await channel.fetch_message(
                        self._message_cache[cache_key]
                    )
                    await message.edit(embed=embed)
                    logger.debug(f"Edited message for {cache_key}")
                    published += 1
                except discord.NotFound:
                    # Message was deleted - post new one
                    del self._message_cache[cache_key]
                    message = await channel.send(embed=embed)
                    self._message_cache[cache_key] = message.id
                    logger.info(f"Posted player prop: {cache_key}")
                    published += 1
            else:
                # Post new message
                message = await channel.send(embed=embed)
                self._message_cache[cache_key] = message.id
                logger.info(f"Posted player prop: {cache_key}")
                published += 1

        return published

    async def publish_free_pick(self) -> bool:
        """Publish the daily free pick to #free-picks channel.

        Selects the highest-edge team market play from games with first_pitch
        in the configured time window (default 60-90 minutes from now).
        Posts at most once per day.

        Returns:
            True if a free pick was posted, False otherwise

        Raises:
            discord.HTTPException: On Discord API errors
        """
        config = get_config()
        pool = await get_pool()

        # Check if already posted today
        today = datetime.now(timezone.utc).date().isoformat()
        if self._free_pick_posted_date == today:
            logger.info("Free pick already posted today")
            return False

        # Get free-picks channel
        if config.free_pick_channel not in self.channels:
            logger.error(f"Channel {config.free_pick_channel} not found")
            return False

        channel = self.channels[config.free_pick_channel]

        # Calculate time window
        now = datetime.now(timezone.utc)
        window_start = now + timedelta(minutes=config.free_pick_window_min)
        window_end = now + timedelta(minutes=config.free_pick_window_max)

        async with pool.acquire() as conn:
            # Query for highest-edge team market in time window
            result = await conn.fetchrow(
                f"""
                SELECT
                    p.projection_id,
                    p.game_id,
                    p.sim_n,
                    g.first_pitch,
                    ht.abbr as home_abbr,
                    at.abbr as away_abbr,
                    smp.market,
                    smp.side,
                    smp.line,
                    smp.prob,
                    smp.edge,
                    smp.kelly_fraction
                FROM {Table.PROJECTIONS} p
                JOIN {Table.GAMES} g ON p.game_id = g.game_id
                JOIN {Table.TEAMS} ht ON g.home_team_id = ht.team_id
                JOIN {Table.TEAMS} at ON g.away_team_id = at.team_id
                JOIN {Table.SIM_MARKET_PROBS} smp ON p.projection_id = smp.projection_id
                WHERE p.edge_computed_at IS NOT NULL
                  AND g.first_pitch BETWEEN $1 AND $2
                  AND smp.edge > 0
                  AND smp.kelly_fraction > 0
                ORDER BY smp.edge DESC
                LIMIT 1
                """,
                window_start,
                window_end,
            )

            if not result:
                logger.info(
                    f"No positive-edge plays in free pick window ({config.free_pick_window_min}-{config.free_pick_window_max} min)"
                )
                return False

            # Check publishing gate
            if not await is_publishable(
                result["game_id"], result["market"], player_id=None
            ):
                logger.info("Free pick candidate not publishable (lineup gate)")
                return False

            # Get best odds (stubbed for v1)
            model_prob = result["prob"]
            best_book = "Placeholder Book"
            best_price = 1.0 / model_prob if model_prob > 0 else 2.0

            # Format embed
            embed = format_team_market_embed(
                game_id=result["game_id"],
                home_team=result["home_abbr"],
                away_team=result["away_abbr"],
                market=result["market"],
                side=result["side"],
                line=result["line"],
                model_prob=model_prob,
                edge=result["edge"],
                kelly_fraction=result["kelly_fraction"],
                best_book=best_book,
                best_price=best_price,
                first_pitch=result["first_pitch"],
                sim_n=result["sim_n"],
            )

            # Post to free-picks channel
            await channel.send(embed=embed)
            logger.info(
                f"Posted free pick: {result['game_id']} {result['market']} {result['side']}"
            )

            # Mark as posted
            self._free_pick_posted_date = today

            return True

    def _get_team_market_channel(self, market: str) -> str | None:
        """Map team market to channel name."""
        config = get_config()
        mapping = {
            "ml": "team-moneyline",
            "rl": "team-runline",
            "total": "team-totals",
            "team_total": "team-totals",
        }
        channel_name = mapping.get(market)
        # Validate it's in paid channels
        if channel_name and channel_name in config.paid_channels:
            return channel_name
        return None

    def _get_player_prop_channel(self, stat: str) -> str | None:
        """Map player stat to channel name."""
        config = get_config()

        # Hitter stats
        hitter_stats = {"H", "TB", "HR", "RBI", "R", "BB"}
        # Pitcher stats
        pitcher_stats = {"K", "OUTS", "ER"}

        if stat in hitter_stats:
            channel_name = "player-props-h"
        elif stat in pitcher_stats:
            channel_name = "player-props-p"
        else:
            return None

        # Validate it's in paid channels
        if channel_name in config.paid_channels:
            return channel_name
        return None
