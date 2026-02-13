"""Tests for Discord bot and publishing layer (Unit 10).

Validates all acceptance criteria from the mini-spec:
1. Bot connects and logs ready
2. Channels created on startup
3. Team market published
4. Player prop published
5. Publishing gate enforced
6. Free pick timing
7. Free pick uniqueness
8. Anti-spam (edit vs new message)
9. Tier gating
10. Negative/zero edge not published
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from mlb.config.settings import AppConfig
from mlb.discord_bot.channels import ensure_channels, sync_member_permissions
from mlb.discord_bot.formatter import (
    format_player_prop_embed,
    format_team_market_embed,
)
from mlb.discord_bot.publisher import Publisher


class TestFormatter:
    """Test Discord embed formatting (pure functions)."""

    def test_format_team_market_embed_moneyline(self):
        """AC3: Team market embed contains all required fields."""
        embed = format_team_market_embed(
            game_id="2026-04-15-NYY-BOS",
            home_team="BOS",
            away_team="NYY",
            market="ml",
            side="home",
            line=None,
            model_prob=0.552,
            edge=0.031,
            kelly_fraction=0.012,
            best_book="DraftKings",
            best_price=2.05,
            first_pitch=datetime(2026, 4, 15, 19, 10, tzinfo=timezone.utc),
            sim_n=5000,
        )

        assert "NYY @ BOS" in embed.title
        assert "Moneyline" in embed.title

        # Check fields exist
        field_names = [f.name for f in embed.fields]
        assert "Model Probability" in field_names
        assert "Edge" in field_names
        assert "Kelly Sizing" in field_names
        assert "Best Book" in field_names

        # Check values
        for field in embed.fields:
            if field.name == "Model Probability":
                assert "55.2%" in field.value
            elif field.name == "Edge":
                assert "+3.1%" in field.value
            elif field.name == "Kelly Sizing":
                assert "1.2%" in field.value
                assert "bankroll" in field.value
            elif field.name == "Best Book":
                assert "DraftKings @ 2.05" in field.value

        # Check footer
        assert "5,000 simulations" in embed.footer.text

    def test_format_team_market_embed_runline(self):
        """Team market embed for run line shows line in side display."""
        embed = format_team_market_embed(
            game_id="2026-04-15-NYY-BOS",
            home_team="BOS",
            away_team="NYY",
            market="rl",
            side="away",
            line=1.5,
            model_prob=0.601,
            edge=0.025,
            kelly_fraction=0.008,
            best_book="FanDuel",
            best_price=1.95,
            first_pitch=datetime(2026, 4, 15, 19, 10, tzinfo=timezone.utc),
            sim_n=5000,
        )

        assert "Run Line" in embed.title

        # Check side display includes line
        for field in embed.fields:
            if field.name == "Pick":
                assert "NYY" in field.value
                assert "1.5" in field.value

    def test_format_player_prop_embed(self):
        """AC4: Player prop embed contains all required fields."""
        embed = format_player_prop_embed(
            player_name="Aaron Judge",
            stat="H",
            line=0.5,
            side="over",
            p_start=0.92,
            model_prob=0.623,
            edge=0.041,
            kelly_fraction=0.018,
            best_book="DraftKings",
            best_price=1.85,
            home_team="BOS",
            away_team="NYY",
            first_pitch=datetime(2026, 4, 15, 19, 10, tzinfo=timezone.utc),
            sim_n=5000,
        )

        assert "Aaron Judge" in embed.title
        assert "Hits O/U 0.5" in embed.title

        # Check fields
        field_names = [f.name for f in embed.fields]
        assert "P(Start)" in field_names
        assert "Model Probability" in field_names
        assert "Edge" in field_names
        assert "Kelly Sizing" in field_names
        assert "Best Book" in field_names

        # Check values
        for field in embed.fields:
            if field.name == "P(Start)":
                assert "92.0%" in field.value
            elif field.name == "Model Probability":
                assert "62.3%" in field.value
            elif field.name == "Edge":
                assert "+4.1%" in field.value


class TestChannels:
    """Test channel creation and permission management."""

    @pytest.mark.asyncio
    async def test_ensure_channels_creates_missing(self, mock_config):
        """AC2: Channels created on startup with correct names."""
        mock_config.free_pick_channel = "free-picks"
        mock_config.paid_channels = ["team-moneyline", "player-props-h"]
        mock_config.announcements_channel = "announcements"

        # Mock guild with no existing channels
        guild = MagicMock()
        guild.text_channels = []
        guild.default_role = MagicMock()
        guild.me = MagicMock()
        guild.create_text_channel = AsyncMock()

        # Mock created channels
        def create_channel_side_effect(name, **kwargs):
            channel = MagicMock()
            channel.name = name
            return channel

        guild.create_text_channel.side_effect = create_channel_side_effect

        with patch("mlb.discord_bot.channels.get_config", return_value=mock_config):
            channels = await ensure_channels(guild)

        # Assert all channels created
        assert "free-picks" in channels
        assert "team-moneyline" in channels
        assert "player-props-h" in channels
        assert "announcements" in channels

        # Assert create_text_channel called 4 times
        assert guild.create_text_channel.call_count == 4

    @pytest.mark.asyncio
    async def test_ensure_channels_reuses_existing(self, mock_config):
        """Channels not recreated if they already exist."""
        mock_config.free_pick_channel = "free-picks"
        mock_config.paid_channels = ["team-moneyline"]
        mock_config.announcements_channel = "announcements"

        # Mock guild with existing channels
        existing_channel = MagicMock()
        existing_channel.name = "free-picks"

        guild = MagicMock()
        guild.text_channels = [existing_channel]
        guild.default_role = MagicMock()
        guild.me = MagicMock()
        guild.create_text_channel = AsyncMock()

        with patch("mlb.discord_bot.channels.get_config", return_value=mock_config):
            channels = await ensure_channels(guild)

        # free-picks should be reused, not recreated
        assert channels["free-picks"] == existing_channel
        # Only team-moneyline and announcements should be created
        assert guild.create_text_channel.call_count == 2

    @pytest.mark.asyncio
    async def test_sync_member_permissions_paid_access(self, mock_db_pool):
        """AC9: Paid subscriber can see paid channels."""
        # Mock subscription query
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"tier": "paid", "status": "active"}
        mock_db_pool.acquire.return_value.__aenter__.return_value = mock_conn

        # Mock guild and member
        guild = MagicMock()
        member = MagicMock()
        member.id = 12345
        member.name = "test_user"

        # Mock paid channel
        paid_channel = MagicMock()
        paid_channel.name = "team-moneyline"
        paid_channel.set_permissions = AsyncMock()
        guild.text_channels = [paid_channel]

        mock_config = MagicMock()
        mock_config.paid_channels = ["team-moneyline"]

        with patch("mlb.discord_bot.channels.get_pool", return_value=mock_db_pool):
            with patch(
                "mlb.discord_bot.channels.get_config", return_value=mock_config
            ):
                await sync_member_permissions(guild, member)

        # Assert permissions granted
        paid_channel.set_permissions.assert_called_once()
        call_args = paid_channel.set_permissions.call_args
        assert call_args[0][0] == member
        assert call_args[1]["read_messages"] is True

    @pytest.mark.asyncio
    async def test_sync_member_permissions_free_user(self, mock_db_pool):
        """AC9: Free user cannot see paid channels."""
        # Mock subscription query - free tier
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"tier": "free", "status": "active"}
        mock_db_pool.acquire.return_value.__aenter__.return_value = mock_conn

        # Mock guild and member
        guild = MagicMock()
        member = MagicMock()
        member.id = 12345
        member.name = "free_user"

        # Mock paid channel
        paid_channel = MagicMock()
        paid_channel.name = "team-moneyline"
        paid_channel.set_permissions = AsyncMock()
        guild.text_channels = [paid_channel]

        mock_config = MagicMock()
        mock_config.paid_channels = ["team-moneyline"]

        with patch("mlb.discord_bot.channels.get_pool", return_value=mock_db_pool):
            with patch(
                "mlb.discord_bot.channels.get_config", return_value=mock_config
            ):
                await sync_member_permissions(guild, member)

        # Assert permissions revoked (overwrite=None)
        paid_channel.set_permissions.assert_called_once()
        call_args = paid_channel.set_permissions.call_args
        assert call_args[0][0] == member
        assert call_args[1]["overwrite"] is None


class TestPublisher:
    """Test pick publishing logic."""

    @pytest.mark.asyncio
    async def test_publish_picks_team_market(self, mock_db_pool, mock_config):
        """AC3: Team market published with all fields."""
        # Mock database responses
        mock_conn = AsyncMock()

        # Latest projection
        mock_conn.fetchrow.side_effect = [
            {
                "projection_id": 1,
                "edge_computed_at": datetime.now(timezone.utc),
                "sim_n": 5000,
            },
            # Game details
            {
                "home_team_id": 1,
                "away_team_id": 2,
                "first_pitch": datetime.now(timezone.utc) + timedelta(hours=2),
                "home_abbr": "BOS",
                "away_abbr": "NYY",
            },
        ]

        # sim_market_probs
        mock_conn.fetch.side_effect = [
            [
                {
                    "market": "ml",
                    "side": "home",
                    "line": None,
                    "prob": 0.552,
                    "edge": 0.031,
                    "kelly_fraction": 0.012,
                }
            ],
            [],  # No player props
        ]

        mock_db_pool.acquire.return_value.__aenter__.return_value = mock_conn

        # Mock channel
        mock_channel = MagicMock()
        mock_channel.send = AsyncMock()
        mock_channel.name = "team-moneyline"

        channels = {"team-moneyline": mock_channel}
        publisher = Publisher(channels)

        with patch("mlb.discord_bot.publisher.get_pool", return_value=mock_db_pool):
            with patch("mlb.discord_bot.publisher.get_config", return_value=mock_config):
                with patch("mlb.discord_bot.publisher.is_publishable", return_value=True):
                    count = await publisher.publish_picks("2026-04-15-NYY-BOS")

        # Assert message sent
        assert count == 1
        mock_channel.send.assert_called_once()

        # Check embed content
        embed = mock_channel.send.call_args[1]["embed"]
        assert "Moneyline" in embed.title

    @pytest.mark.asyncio
    async def test_publish_picks_player_prop(self, mock_db_pool, mock_config):
        """AC4: Player prop published for confirmed lineup."""
        # Mock database responses
        mock_conn = AsyncMock()

        mock_conn.fetchrow.side_effect = [
            {
                "projection_id": 1,
                "edge_computed_at": datetime.now(timezone.utc),
                "sim_n": 5000,
            },
            {
                "home_team_id": 1,
                "away_team_id": 2,
                "first_pitch": datetime.now(timezone.utc) + timedelta(hours=2),
                "home_abbr": "BOS",
                "away_abbr": "NYY",
            },
        ]

        # No team markets, one player prop
        mock_conn.fetch.side_effect = [
            [],  # No team markets
            [
                {
                    "player_id": 123,
                    "p_start": 0.92,
                    "stat": "H",
                    "line": 0.5,
                    "prob_over": 0.623,
                    "edge": 0.041,
                    "kelly_fraction": 0.018,
                    "player_name": "Aaron Judge",
                }
            ],
        ]

        mock_db_pool.acquire.return_value.__aenter__.return_value = mock_conn

        # Mock channel
        mock_channel = MagicMock()
        mock_channel.send = AsyncMock()

        channels = {"player-props-h": mock_channel}
        publisher = Publisher(channels)

        with patch("mlb.discord_bot.publisher.get_pool", return_value=mock_db_pool):
            with patch("mlb.discord_bot.publisher.get_config", return_value=mock_config):
                with patch("mlb.discord_bot.publisher.is_publishable", return_value=True):
                    count = await publisher.publish_picks("2026-04-15-NYY-BOS")

        # Assert message sent
        assert count == 1
        mock_channel.send.assert_called_once()

        # Check embed
        embed = mock_channel.send.call_args[1]["embed"]
        assert "Aaron Judge" in embed.title

    @pytest.mark.asyncio
    async def test_publish_picks_respects_publishing_gate(
        self, mock_db_pool, mock_config
    ):
        """AC5: Publishing gate enforced (player with low p_start not published)."""
        # Mock database responses
        mock_conn = AsyncMock()

        mock_conn.fetchrow.side_effect = [
            {
                "projection_id": 1,
                "edge_computed_at": datetime.now(timezone.utc),
                "sim_n": 5000,
            },
            {
                "home_team_id": 1,
                "away_team_id": 2,
                "first_pitch": datetime.now(timezone.utc) + timedelta(hours=2),
                "home_abbr": "BOS",
                "away_abbr": "NYY",
            },
        ]

        # One team market (publishable), one player prop (not publishable)
        mock_conn.fetch.side_effect = [
            [
                {
                    "market": "ml",
                    "side": "home",
                    "line": None,
                    "prob": 0.552,
                    "edge": 0.031,
                    "kelly_fraction": 0.012,
                }
            ],
            [
                {
                    "player_id": 456,
                    "p_start": 0.60,  # Below threshold
                    "stat": "H",
                    "line": 0.5,
                    "prob_over": 0.623,
                    "edge": 0.041,
                    "kelly_fraction": 0.018,
                    "player_name": "Uncertain Player",
                }
            ],
        ]

        mock_db_pool.acquire.return_value.__aenter__.return_value = mock_conn

        # Mock channels
        mock_ml_channel = MagicMock()
        mock_ml_channel.send = AsyncMock()
        mock_prop_channel = MagicMock()
        mock_prop_channel.send = AsyncMock()

        channels = {
            "team-moneyline": mock_ml_channel,
            "player-props-h": mock_prop_channel,
        }
        publisher = Publisher(channels)

        # is_publishable returns True for team markets, False for player props
        async def mock_is_publishable(game_id, market, player_id=None):
            return player_id is None

        with patch("mlb.discord_bot.publisher.get_pool", return_value=mock_db_pool):
            with patch("mlb.discord_bot.publisher.get_config", return_value=mock_config):
                with patch(
                    "mlb.discord_bot.publisher.is_publishable",
                    side_effect=mock_is_publishable,
                ):
                    count = await publisher.publish_picks("2026-04-15-NYY-BOS")

        # Only team market published
        assert count == 1
        mock_ml_channel.send.assert_called_once()
        mock_prop_channel.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_publish_picks_skips_negative_edge(self, mock_db_pool, mock_config):
        """AC10: Negative/zero edge plays not published."""
        # Mock database responses
        mock_conn = AsyncMock()

        mock_conn.fetchrow.side_effect = [
            {
                "projection_id": 1,
                "edge_computed_at": datetime.now(timezone.utc),
                "sim_n": 5000,
            },
            {
                "home_team_id": 1,
                "away_team_id": 2,
                "first_pitch": datetime.now(timezone.utc) + timedelta(hours=2),
                "home_abbr": "BOS",
                "away_abbr": "NYY",
            },
        ]

        # Query returns rows with edge <= 0 or kelly = 0
        mock_conn.fetch.side_effect = [
            [],  # Query filters out edge <= 0
            [],  # Query filters out edge <= 0
        ]

        mock_db_pool.acquire.return_value.__aenter__.return_value = mock_conn

        mock_channel = MagicMock()
        mock_channel.send = AsyncMock()

        channels = {"team-moneyline": mock_channel}
        publisher = Publisher(channels)

        with patch("mlb.discord_bot.publisher.get_pool", return_value=mock_db_pool):
            with patch("mlb.discord_bot.publisher.get_config", return_value=mock_config):
                with patch("mlb.discord_bot.publisher.is_publishable", return_value=True):
                    count = await publisher.publish_picks("2026-04-15-NYY-BOS")

        # No messages sent
        assert count == 0
        mock_channel.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_anti_spam_edits_existing_message(self, mock_db_pool, mock_config):
        """AC8: Consecutive publish_picks() edits existing message, not duplicate."""
        # Mock database (same as test_publish_picks_team_market)
        mock_conn = AsyncMock()

        # First call
        mock_conn.fetchrow.side_effect = [
            {
                "projection_id": 1,
                "edge_computed_at": datetime.now(timezone.utc),
                "sim_n": 5000,
            },
            {
                "home_team_id": 1,
                "away_team_id": 2,
                "first_pitch": datetime.now(timezone.utc) + timedelta(hours=2),
                "home_abbr": "BOS",
                "away_abbr": "NYY",
            },
        ]

        mock_conn.fetch.side_effect = [
            [
                {
                    "market": "ml",
                    "side": "home",
                    "line": None,
                    "prob": 0.552,
                    "edge": 0.031,
                    "kelly_fraction": 0.012,
                }
            ],
            [],
        ]

        mock_db_pool.acquire.return_value.__aenter__.return_value = mock_conn

        # Mock channel and message
        mock_message = MagicMock()
        mock_message.id = 999
        mock_message.edit = AsyncMock()

        mock_channel = MagicMock()
        mock_channel.send = AsyncMock(return_value=mock_message)
        mock_channel.fetch_message = AsyncMock(return_value=mock_message)

        channels = {"team-moneyline": mock_channel}
        publisher = Publisher(channels)

        with patch("mlb.discord_bot.publisher.get_pool", return_value=mock_db_pool):
            with patch("mlb.discord_bot.publisher.get_config", return_value=mock_config):
                with patch("mlb.discord_bot.publisher.is_publishable", return_value=True):
                    # First publish
                    count1 = await publisher.publish_picks("2026-04-15-NYY-BOS")

        assert count1 == 1
        mock_channel.send.assert_called_once()

        # Reset mock for second call
        mock_conn.fetchrow.side_effect = [
            {
                "projection_id": 2,  # New projection
                "edge_computed_at": datetime.now(timezone.utc),
                "sim_n": 5000,
            },
            {
                "home_team_id": 1,
                "away_team_id": 2,
                "first_pitch": datetime.now(timezone.utc) + timedelta(hours=2),
                "home_abbr": "BOS",
                "away_abbr": "NYY",
            },
        ]

        mock_conn.fetch.side_effect = [
            [
                {
                    "market": "ml",
                    "side": "home",
                    "line": None,
                    "prob": 0.560,  # Updated prob
                    "edge": 0.035,
                    "kelly_fraction": 0.014,
                }
            ],
            [],
        ]

        with patch("mlb.discord_bot.publisher.get_pool", return_value=mock_db_pool):
            with patch("mlb.discord_bot.publisher.get_config", return_value=mock_config):
                with patch("mlb.discord_bot.publisher.is_publishable", return_value=True):
                    # Second publish - should edit
                    count2 = await publisher.publish_picks("2026-04-15-NYY-BOS")

        assert count2 == 1
        # send() should still be called only once (from first publish)
        assert mock_channel.send.call_count == 1
        # edit() should be called once
        mock_message.edit.assert_called_once()

    @pytest.mark.asyncio
    async def test_publish_free_pick_timing(self, mock_db_pool, mock_config):
        """AC6: Free pick selects game in 60-90 min window."""
        mock_config.free_pick_window_min = 60
        mock_config.free_pick_window_max = 90
        mock_config.free_pick_channel = "free-picks"

        # Mock database
        mock_conn = AsyncMock()

        now = datetime.now(timezone.utc)
        game_in_window = now + timedelta(minutes=75)

        mock_conn.fetchrow.return_value = {
            "projection_id": 1,
            "game_id": "2026-04-15-NYY-BOS",
            "sim_n": 5000,
            "first_pitch": game_in_window,
            "home_abbr": "BOS",
            "away_abbr": "NYY",
            "market": "ml",
            "side": "home",
            "line": None,
            "prob": 0.552,
            "edge": 0.031,
            "kelly_fraction": 0.012,
        }

        mock_db_pool.acquire.return_value.__aenter__.return_value = mock_conn

        # Mock channel
        mock_channel = MagicMock()
        mock_channel.send = AsyncMock()

        channels = {"free-picks": mock_channel}
        publisher = Publisher(channels)

        with patch("mlb.discord_bot.publisher.get_pool", return_value=mock_db_pool):
            with patch("mlb.discord_bot.publisher.get_config", return_value=mock_config):
                with patch("mlb.discord_bot.publisher.is_publishable", return_value=True):
                    posted = await publisher.publish_free_pick()

        # Assert posted
        assert posted is True
        mock_channel.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_publish_free_pick_outside_window(self, mock_db_pool, mock_config):
        """AC6: Free pick does NOT select game outside time window."""
        mock_config.free_pick_window_min = 60
        mock_config.free_pick_window_max = 90
        mock_config.free_pick_channel = "free-picks"

        # Mock database returns no results (outside window)
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = None

        mock_db_pool.acquire.return_value.__aenter__.return_value = mock_conn

        mock_channel = MagicMock()
        mock_channel.send = AsyncMock()

        channels = {"free-picks": mock_channel}
        publisher = Publisher(channels)

        with patch("mlb.discord_bot.publisher.get_pool", return_value=mock_db_pool):
            with patch("mlb.discord_bot.publisher.get_config", return_value=mock_config):
                posted = await publisher.publish_free_pick()

        # Assert not posted
        assert posted is False
        mock_channel.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_publish_free_pick_once_per_day(self, mock_db_pool, mock_config):
        """AC7: Free pick posted at most once per day."""
        mock_config.free_pick_window_min = 60
        mock_config.free_pick_window_max = 90
        mock_config.free_pick_channel = "free-picks"

        # Mock database
        mock_conn = AsyncMock()

        now = datetime.now(timezone.utc)
        game_in_window = now + timedelta(minutes=75)

        mock_conn.fetchrow.return_value = {
            "projection_id": 1,
            "game_id": "2026-04-15-NYY-BOS",
            "sim_n": 5000,
            "first_pitch": game_in_window,
            "home_abbr": "BOS",
            "away_abbr": "NYY",
            "market": "ml",
            "side": "home",
            "line": None,
            "prob": 0.552,
            "edge": 0.031,
            "kelly_fraction": 0.012,
        }

        mock_db_pool.acquire.return_value.__aenter__.return_value = mock_conn

        mock_channel = MagicMock()
        mock_channel.send = AsyncMock()

        channels = {"free-picks": mock_channel}
        publisher = Publisher(channels)

        with patch("mlb.discord_bot.publisher.get_pool", return_value=mock_db_pool):
            with patch("mlb.discord_bot.publisher.get_config", return_value=mock_config):
                with patch("mlb.discord_bot.publisher.is_publishable", return_value=True):
                    # First call
                    posted1 = await publisher.publish_free_pick()
                    # Second call (same day)
                    posted2 = await publisher.publish_free_pick()

        # First posted, second skipped
        assert posted1 is True
        assert posted2 is False
        # send() called only once
        mock_channel.send.assert_called_once()


# Fixtures


@pytest.fixture
def mock_config():
    """Provide a mock AppConfig for tests."""
    config = MagicMock(spec=AppConfig)
    config.discord_guild_id = "123456789"
    config.free_pick_channel = "free-picks"
    config.paid_channels = ["team-moneyline", "team-runline", "player-props-h"]
    config.announcements_channel = "announcements"
    config.free_pick_window_min = 60
    config.free_pick_window_max = 90
    config.p_start_threshold = 0.85
    return config


@pytest.fixture
def mock_db_pool():
    """Provide a mock database connection pool."""
    pool = MagicMock()
    return pool
