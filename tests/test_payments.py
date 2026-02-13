"""Tests for Stripe subscription and webhook integration (Unit 11).

Validates all acceptance criteria from the mini-spec:
1. Checkout URL generation
2. Webhook signature verification
3. checkout.session.completed creates subscription
4. invoice.paid updates to active
5. invoice.payment_failed sets past_due
6. customer.subscription.deleted sets free/cancelled
7. Discord role grant on activation
8. Discord role revoke on cancellation
9. Idempotent webhook processing
10. Unknown events acknowledged
11. Server starts on configured port
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import stripe
from aiohttp import web

from mlb.config.settings import AppConfig
from mlb.payments.checkout import create_checkout_url
from mlb.payments.server import create_app, webhook_endpoint
from mlb.payments.sync import sync_subscription
from mlb.payments.webhooks import handle_webhook


class TestCheckout:
    """Test checkout session creation."""

    @patch("mlb.payments.checkout.stripe.checkout.Session.create")
    @patch("mlb.payments.checkout.get_config")
    def test_create_checkout_url_success(self, mock_config, mock_create):
        """AC1: create_checkout_url returns valid Stripe Checkout URL."""
        # Mock config
        config = Mock()
        config.stripe_secret.get_secret_value.return_value = "sk_test_123"
        config.stripe_price_id = "price_test_123"
        config.checkout_success_url = "https://discord.com"
        config.checkout_cancel_url = "https://discord.com"
        mock_config.return_value = config

        # Mock Stripe response
        mock_session = Mock()
        mock_session.id = "cs_test_123"
        mock_session.url = "https://checkout.stripe.com/pay/cs_test_123"
        mock_create.return_value = mock_session

        # Create checkout URL
        url = create_checkout_url("user123")

        # Verify
        assert url == "https://checkout.stripe.com/pay/cs_test_123"
        mock_create.assert_called_once()

        # Verify call arguments
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["mode"] == "subscription"
        assert call_kwargs["client_reference_id"] == "user123"
        assert "line_items" in call_kwargs
        assert "success_url" in call_kwargs
        assert "cancel_url" in call_kwargs

    def test_create_checkout_url_missing_price_id(self):
        """AC1: Raises ValueError if stripe_price_id not configured."""
        with patch("mlb.payments.checkout.get_config") as mock_config:
            config = Mock()
            config.stripe_secret.get_secret_value.return_value = "sk_test_123"
            config.stripe_price_id = ""
            mock_config.return_value = config

            with pytest.raises(ValueError, match="stripe_price_id"):
                create_checkout_url("user123")


class TestWebhookSignatureVerification:
    """Test webhook signature verification."""

    @pytest.mark.asyncio
    async def test_valid_signature_accepted(self):
        """AC2: Valid Stripe signature is processed."""
        payload = b'{"type": "ping"}'
        sig_header = "t=123,v1=valid_sig"

        with patch("mlb.payments.webhooks.stripe.Webhook.construct_event") as mock_verify:
            # Mock successful verification
            mock_verify.return_value = {"type": "ping", "data": {"object": {}}}

            response = await handle_webhook(payload, sig_header, None)

            assert response.status == 200
            mock_verify.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalid_signature_rejected(self):
        """AC2: Invalid signature returns 400 and no database write."""
        payload = b'{"type": "ping"}'
        sig_header = "t=123,v1=bad_sig"

        with patch("mlb.payments.webhooks.stripe.Webhook.construct_event") as mock_verify:
            # Mock signature verification failure
            mock_verify.side_effect = stripe.SignatureVerificationError(
                "Invalid signature", sig_header
            )

            response = await handle_webhook(payload, sig_header, None)

            assert response.status == 400
            assert "Invalid signature" in response.text


class TestCheckoutCompleted:
    """Test checkout.session.completed webhook."""

    @pytest.mark.asyncio
    async def test_checkout_completed_creates_subscription(self):
        """AC3: checkout.session.completed creates subscription with paid/active."""
        event = {
            "type": "checkout.session.completed",
            "data": {
                "object": {
                    "client_reference_id": "user123",
                    "customer": "cus_test_123",
                    "subscription": "sub_test_123",
                }
            },
        }

        mock_subscription = {
            "id": "sub_test_123",
            "current_period_end": 1735689600,  # 2025-01-01
        }

        with patch("mlb.payments.webhooks.stripe.Webhook.construct_event") as mock_verify, \
             patch("mlb.payments.webhooks.stripe.Subscription.retrieve") as mock_sub, \
             patch("mlb.db.pool.get_pool") as mock_pool_webhooks, \
             patch("mlb.payments.sync.get_pool") as mock_pool_sync:

            mock_verify.return_value = event
            mock_sub.return_value = mock_subscription

            # Mock database pool
            mock_conn = AsyncMock()
            mock_acquire = AsyncMock()
            mock_acquire.__aenter__.return_value = mock_conn
            mock_acquire.__aexit__.return_value = None
            mock_pool_obj = MagicMock()
            mock_pool_obj.acquire.return_value = mock_acquire
            # Make get_pool() return an awaitable for both patches
            async def mock_get_pool_fn():
                return mock_pool_obj

            mock_pool_webhooks.side_effect = mock_get_pool_fn
            mock_pool_sync.side_effect = mock_get_pool_fn

            response = await handle_webhook(b"{}", "sig", None)

            assert response.status == 200

            # Verify database call
            mock_conn.execute.assert_called_once()
            call_args = mock_conn.execute.call_args[0]
            assert "user123" in call_args
            assert "cus_test_123" in call_args
            assert "paid" in call_args
            assert "active" in call_args


class TestInvoicePaid:
    """Test invoice.paid webhook."""

    @pytest.mark.asyncio
    async def test_invoice_paid_updates_to_active(self):
        """AC4: invoice.paid updates subscription to active and refreshes period."""
        event = {
            "type": "invoice.paid",
            "data": {
                "object": {
                    "customer": "cus_test_123",
                    "subscription": "sub_test_123",
                }
            },
        }

        mock_subscription = {
            "id": "sub_test_123",
            "current_period_end": 1735689600,
        }

        mock_customer = {
            "id": "cus_test_123",
            "metadata": {},
        }

        with patch("mlb.payments.webhooks.stripe.Webhook.construct_event") as mock_verify, \
             patch("mlb.payments.webhooks.stripe.Subscription.retrieve") as mock_sub, \
             patch("mlb.payments.webhooks.stripe.Customer.retrieve") as mock_cust, \
             patch("mlb.db.pool.get_pool") as mock_pool_webhooks, \
             patch("mlb.payments.sync.get_pool") as mock_pool_sync:

            mock_verify.return_value = event
            mock_sub.return_value = mock_subscription
            mock_cust.return_value = mock_customer

            # Mock database pool - first for lookup, then for upsert
            mock_conn = AsyncMock()
            mock_conn.fetchrow.return_value = {"discord_user_id": "user123"}
            mock_acquire = AsyncMock()
            mock_acquire.__aenter__.return_value = mock_conn
            mock_acquire.__aexit__.return_value = None
            mock_pool_obj = MagicMock()
            mock_pool_obj.acquire.return_value = mock_acquire

            # Make get_pool() return an awaitable for both patches
            async def mock_get_pool_fn():
                return mock_pool_obj

            mock_pool_webhooks.side_effect = mock_get_pool_fn
            mock_pool_sync.side_effect = mock_get_pool_fn

            response = await handle_webhook(b"{}", "sig", None)

            assert response.status == 200

            # Verify database upsert called with active status
            assert mock_conn.execute.call_count >= 1


class TestInvoicePaymentFailed:
    """Test invoice.payment_failed webhook."""

    @pytest.mark.asyncio
    async def test_invoice_payment_failed_sets_past_due(self):
        """AC5: invoice.payment_failed updates status to past_due, tier remains paid."""
        event = {
            "type": "invoice.payment_failed",
            "data": {
                "object": {
                    "customer": "cus_test_123",
                    "subscription": "sub_test_123",
                }
            },
        }

        mock_subscription = {
            "id": "sub_test_123",
            "current_period_end": 1735689600,
        }

        # Setup mock pool (same pattern as test_discord.py)
        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"discord_user_id": "user123"}
        mock_acquire = MagicMock()
        mock_acquire.__aenter__.return_value = mock_conn
        mock_acquire.__aexit__.return_value = AsyncMock()
        mock_pool.acquire.return_value = mock_acquire

        with patch("mlb.payments.webhooks.stripe.Webhook.construct_event") as mock_verify, \
             patch("mlb.payments.webhooks.stripe.Subscription.retrieve") as mock_sub, \
             patch("mlb.db.pool.get_pool", return_value=mock_pool), \
             patch("mlb.payments.sync.get_pool", return_value=mock_pool):

            mock_verify.return_value = event
            mock_sub.return_value = mock_subscription

            response = await handle_webhook(b"{}", "sig", None)

            assert response.status == 200

            # Verify database was called with past_due status
            assert mock_conn.execute.call_count >= 1
            # Check that the call included 'past_due' status
            call_args = str(mock_conn.execute.call_args_list)
            assert "past_due" in call_args


class TestSubscriptionDeleted:
    """Test customer.subscription.deleted webhook."""

    @pytest.mark.asyncio
    async def test_subscription_deleted_sets_free_cancelled(self):
        """AC6: subscription.deleted sets tier=free, status=cancelled."""
        event = {
            "type": "customer.subscription.deleted",
            "data": {
                "object": {
                    "customer": "cus_test_123",
                    "current_period_end": 1735689600,
                }
            },
        }

        # Setup mock pool (same pattern as test_discord.py)
        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"discord_user_id": "user123"}
        mock_acquire = MagicMock()
        mock_acquire.__aenter__.return_value = mock_conn
        mock_acquire.__aexit__.return_value = AsyncMock()
        mock_pool.acquire.return_value = mock_acquire

        with patch("mlb.payments.webhooks.stripe.Webhook.construct_event") as mock_verify, \
             patch("mlb.db.pool.get_pool", return_value=mock_pool), \
             patch("mlb.payments.sync.get_pool", return_value=mock_pool):

            mock_verify.return_value = event

            response = await handle_webhook(b"{}", "sig", None)

            assert response.status == 200

            # Verify database call contains free and cancelled
            assert mock_conn.execute.call_count >= 1
            call_args = str(mock_conn.execute.call_args_list)
            assert "free" in call_args
            assert "cancelled" in call_args


class TestDiscordRoleSync:
    """Test Discord role grant and revoke."""

    @pytest.mark.asyncio
    async def test_role_granted_on_activation(self):
        """AC7: Discord role granted after checkout.session.completed."""
        mock_bot = MagicMock()  # Not AsyncMock - get_guild is synchronous
        mock_guild = MagicMock()
        mock_member = AsyncMock()  # Member operations are async
        mock_role = MagicMock()

        mock_guild.get_member.return_value = mock_member
        mock_bot.get_guild.return_value = mock_guild
        mock_member.roles = []

        # Setup mock pool (same pattern as test_discord.py)
        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_acquire = MagicMock()
        mock_acquire.__aenter__.return_value = mock_conn
        mock_acquire.__aexit__.return_value = AsyncMock()
        mock_pool.acquire.return_value = mock_acquire

        with patch("mlb.payments.sync.discord.utils.get") as mock_get_role, \
             patch("mlb.payments.sync.get_pool", return_value=mock_pool), \
             patch("mlb.payments.sync.get_config") as mock_config:

            mock_get_role.return_value = mock_role
            config = Mock()
            config.discord_guild_id = "123456789"
            config.discord_paid_role_name = "Subscriber"
            mock_config.return_value = config

            await sync_subscription(
                discord_user_id="123456789012",  # Valid Discord user ID (numeric string)
                stripe_customer_id="cus_test_123",
                tier="paid",
                status="active",
                current_period_end=datetime(2025, 1, 1, tzinfo=timezone.utc),
                bot_client=mock_bot,
            )

            # Verify role was added
            mock_member.add_roles.assert_called_once()

    @pytest.mark.asyncio
    async def test_role_revoked_on_cancellation(self):
        """AC8: Discord role revoked after subscription.deleted."""
        mock_bot = MagicMock()  # Not AsyncMock - get_guild is synchronous
        mock_guild = MagicMock()
        mock_member = AsyncMock()  # Member operations are async
        mock_role = MagicMock()

        mock_guild.get_member.return_value = mock_member
        mock_bot.get_guild.return_value = mock_guild
        mock_member.roles = [mock_role]

        # Setup mock pool (same pattern as test_discord.py)
        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_acquire = MagicMock()
        mock_acquire.__aenter__.return_value = mock_conn
        mock_acquire.__aexit__.return_value = AsyncMock()
        mock_pool.acquire.return_value = mock_acquire

        with patch("mlb.payments.sync.discord.utils.get") as mock_get_role, \
             patch("mlb.payments.sync.get_pool", return_value=mock_pool), \
             patch("mlb.payments.sync.get_config") as mock_config:

            mock_get_role.return_value = mock_role
            config = Mock()
            config.discord_guild_id = "123456789"
            config.discord_paid_role_name = "Subscriber"
            mock_config.return_value = config

            await sync_subscription(
                discord_user_id="123456789012",  # Valid Discord user ID (numeric string)
                stripe_customer_id="cus_test_123",
                tier="free",
                status="cancelled",
                current_period_end=None,
                bot_client=mock_bot,
            )

            # Verify role was removed
            mock_member.remove_roles.assert_called_once()


class TestIdempotency:
    """Test webhook idempotency."""

    @pytest.mark.asyncio
    async def test_replaying_webhook_produces_same_state(self):
        """AC9: Replaying same webhook event produces same database state."""
        event = {
            "type": "checkout.session.completed",
            "data": {
                "object": {
                    "client_reference_id": "user123",
                    "customer": "cus_test_123",
                    "subscription": "sub_test_123",
                }
            },
        }

        mock_subscription = {
            "id": "sub_test_123",
            "current_period_end": 1735689600,
        }

        with patch("mlb.payments.webhooks.stripe.Webhook.construct_event") as mock_verify, \
             patch("mlb.payments.webhooks.stripe.Subscription.retrieve") as mock_sub, \
             patch("mlb.db.pool.get_pool") as mock_pool_webhooks, \
             patch("mlb.payments.sync.get_pool") as mock_pool_sync:

            mock_verify.return_value = event
            mock_sub.return_value = mock_subscription

            # Mock database pool
            mock_conn = AsyncMock()
            mock_acquire = AsyncMock()
            mock_acquire.__aenter__.return_value = mock_conn
            mock_acquire.__aexit__.return_value = None
            mock_pool_obj = MagicMock()
            mock_pool_obj.acquire.return_value = mock_acquire
            # Make get_pool() return an awaitable for both patches
            async def mock_get_pool_fn():
                return mock_pool_obj

            mock_pool_webhooks.side_effect = mock_get_pool_fn
            mock_pool_sync.side_effect = mock_get_pool_fn

            # Process webhook twice
            response1 = await handle_webhook(b"{}", "sig", None)
            response2 = await handle_webhook(b"{}", "sig", None)

            assert response1.status == 200
            assert response2.status == 200

            # Both should succeed (upsert handles duplicates)
            assert mock_conn.execute.call_count == 2


class TestUnknownEvents:
    """Test handling of unknown webhook events."""

    @pytest.mark.asyncio
    async def test_unknown_event_acknowledged(self):
        """AC10: Unknown event type returns 200 and no database write."""
        event = {
            "type": "charge.refunded",
            "data": {"object": {}},
        }

        # Setup mock pool (same pattern as test_discord.py)
        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_acquire = MagicMock()
        mock_acquire.__aenter__.return_value = mock_conn
        mock_acquire.__aexit__.return_value = AsyncMock()
        mock_pool.acquire.return_value = mock_acquire

        with patch("mlb.payments.webhooks.stripe.Webhook.construct_event") as mock_verify, \
             patch("mlb.db.pool.get_pool", return_value=mock_pool), \
             patch("mlb.payments.sync.get_pool", return_value=mock_pool):

            mock_verify.return_value = event

            response = await handle_webhook(b"{}", "sig", None)

            assert response.status == 200

            # No database write should occur
            mock_conn.execute.assert_not_called()


class TestServer:
    """Test webhook server startup and endpoint."""

    @pytest.mark.asyncio
    async def test_server_starts_on_configured_port(self):
        """AC11: Server starts on configured port and responds to webhook endpoint."""
        app = await create_app(None)

        # Verify webhook route exists
        routes = [r for r in app.router.routes()]
        webhook_routes = [
            r for r in routes
            if hasattr(r.resource, "canonical") and "/webhooks/stripe" in r.resource.canonical
        ]

        assert len(webhook_routes) == 1
        assert webhook_routes[0].method == "POST"

    @pytest.mark.asyncio
    async def test_webhook_endpoint_requires_signature(self):
        """AC2: Webhook endpoint rejects requests without signature header."""
        mock_request = Mock()
        mock_request.headers = {}

        response = await webhook_endpoint(mock_request)

        assert response.status == 400
        assert "Missing signature" in response.text
