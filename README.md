# MLB Analytics Platform

MLB analytics platform with Monte Carlo simulation and Discord publishing.

## Overview

This platform provides analytical capabilities for Major League Baseball data through probabilistic modeling (Monte Carlo simulations) with automated insights delivery. Built with Python 3.11+ using async patterns throughout.

**Current Status:** Unit 11 complete (Stripe subscription & webhook integration). See [DEVLOG.md](docs/DEVLOG.md) for implementation details and [DECISIONS.md](docs/DECISIONS.md) for architectural choices.

## Installation

```bash
# Clone and enter the repository
git clone https://github.com/marek-sm/mlb-analytics-platform.git
cd mlb-analytics-platform

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install with development dependencies
pip install -e ".[dev]"

# Configure environment
cp .env.example .env
# Edit .env with your database DSN and other settings
```

## Configuration

Configuration is via environment variables (12-factor). See [.env.example](.env.example) for all options.

**Required:**

- `ENV`: Environment (dev/staging/prod)
- `DB_DSN`: PostgreSQL connection string

**Optional:**

- `DB_POOL_MIN`, `DB_POOL_MAX`: Connection pool sizing
- `DEFAULT_SIM_N`: Monte Carlo simulation count (2000-10000, default: 5000)
- `SHRINKAGE_K_BATTER`: Empirical Bayes shrinkage for batters in PA (50-500, default: 200)
- `SHRINKAGE_K_PITCHER`: Empirical Bayes shrinkage for pitchers in IP (20-200, default: 80)
- `ROLLING_WINDOW_BATTING_DAYS`: Rolling window for batting stats in days (14-120, default: 60)
- `ROLLING_WINDOW_PITCHING_DAYS`: Rolling window for pitching stats in days (7-60, default: 30)
- `BF_PER_OUT_RATIO`: League-average batters faced per out for K/BF approximation (1.0-2.0, default: 1.35)
- `MIN_EDGE_THRESHOLD`: Minimum edge for kelly_fraction > 0 (0.0-0.10, default: 0.02)
- `KELLY_FRACTION_MULTIPLIER`: Fractional Kelly multiplier (0.05-1.0, default: 0.25)
- `SCHEDULE_NIGHT_BEFORE_ET`: ET time for night-before global run (default: "22:00")
- `SCHEDULE_MORNING_ET`: ET time for morning global run (default: "08:00")
- `SCHEDULE_MIDDAY_ET`: ET time for midday global run (default: "12:00")
- `GAME_RUN_T_MINUS_MINUTES`: Minutes before first_pitch for per-game runs (default: [90, 30])
- `RERUN_THROTTLE_MINUTES`: Minimum gap between reruns per game (1-60, default: 10)
- `P_START_THRESHOLD`: Publishing gate threshold for p_start (0.0-1.0, default: 0.85)
- `MAX_RETRY_ATTEMPTS`: Maximum ingestion retry attempts (0-5, default: 2)
- `DISCORD_GUILD_ID`: Discord guild (server) ID for bot operations
- `FREE_PICK_CHANNEL`: Channel name for daily free picks (default: "free-picks")
- `PAID_CHANNELS`: Channel names for paid-tier picks (default: ["team-moneyline", "team-runline", "team-totals", "player-props-h", "player-props-p"])
- `ANNOUNCEMENTS_CHANNEL`: Channel name for bot announcements (default: "announcements")
- `FREE_PICK_WINDOW_MIN`: Earliest time before first_pitch to post free pick in minutes (30-120, default: 60)
- `FREE_PICK_WINDOW_MAX`: Latest time before first_pitch to post free pick in minutes (30-120, default: 90)
- `STRIPE_WEBHOOK_SECRET`: Stripe webhook signing secret
- `STRIPE_PRICE_ID`: Stripe Price ID for subscription product
- `CHECKOUT_SUCCESS_URL`: Stripe Checkout success redirect URL (default: "https://discord.com")
- `CHECKOUT_CANCEL_URL`: Stripe Checkout cancel redirect URL (default: "https://discord.com")
- `WEBHOOK_SERVER_PORT`: Port for Stripe webhook HTTP server (1024-65535, default: 8080)
- `DISCORD_PAID_ROLE_NAME`: Discord role name for paid subscribers (default: "Subscriber")
- `LOG_LEVEL`: Logging verbosity

## Usage

```bash
# Run database migrations
python -m mlb.db.schema.migrate

# Run the application
python -m mlb.main

# Run tests
pytest

# Run tests with coverage
pytest --cov=mlb --cov-report=term-missing

# Code formatting
black src tests
ruff check src tests

# Type checking
mypy src
```

## V1 Scope

**Included in V1:**

- Repository structure and packaging (src/mlb namespace)
- Configuration management via Pydantic BaseSettings
- Async PostgreSQL connection pooling (asyncpg)
- Database schema and migrations (Unit 2)
- Provider-agnostic data ingestion (Unit 3)
  - Odds with American → European decimal conversion
  - Lineups with confirmation flip logic
  - Player stats with auto-upsert
  - Game schedules
  - Weather with park filtering
- Team run-scoring models (Unit 4)
  - Game-level feature engineering with park factors
  - LightGBM models for μ (mean) and dispersion
  - Empirical Bayes shrinkage for player stats
  - Bullpen fatigue and rolling performance metrics
  - Conservative weather fallbacks for dome parks
- Player prop models (Unit 5)
  - Player-level feature engineering (hitters and pitchers)
  - P(start) model for lineup uncertainty
  - PA distribution and pitcher outs distribution models
  - Event-rate models with shrinkage (H, TB, HR, RBI, R, BB, K, ER)
  - Top-7 lineup filter and platoon matchup analysis
  - K/BF approximation using configurable batters-faced-per-out ratio
- Monte Carlo simulation engine (Unit 6)
  - Negative Binomial score sampling with adaptive trial counts (2k-10k)
  - Correlated noise via bivariate normal copula (configurable ρ)
  - Extra-innings tie-break with bullpen fatigue differential
  - Player prop sampling (hitters: PA/H/TB/HR/RBI/R/BB; pitchers: outs/K/ER)
  - Team market derivation (ML, RL ±1.5, Total, Team Totals)
  - Player prop probability derivation with hardcoded main lines
  - Persistence to projections, sim_market_probs, player_projections tables
- Odds processing, edge calculation & bankroll sizing (Unit 7)
  - Best-line selection across multiple sportsbooks
  - Proportional (multiplicative) devig for fair probability calculation
  - Edge calculation (p_model − p_fair) for all team markets
  - Fractional Kelly sizing with configurable threshold and multiplier
  - Idempotent persistence with edge_computed_at timestamp tracking
  - Handles missing odds, stale odds, and player prop no-match cases
- Evaluation & backtesting harness (Unit 8)
  - Pure metric functions: log loss, Brier score, ECE, tail accuracy
  - Closing Line Value (CLV) computation with T-5 closing odds
  - Rolling-origin backtesting for model performance evaluation
  - Market-specific calibration models (isotonic regression, Platt scaling)
  - Per-metric upsert to eval_results table (FC-20 safe partial writes)
  - Comprehensive test coverage (19 tests) for all evaluation contracts
- Scheduler & orchestration pipeline (Unit 9)
  - Global scheduled runs (night-before, morning, midday) with full end-to-end pipeline
  - Per-game runs at T-90 and T-30 minutes before first pitch
  - Event-driven reruns with configurable throttle window (default 10 minutes)
  - Retry policy with exponential backoff for ingestion failures
  - Publishing gate logic enforcing lineup uncertainty policy
  - Team markets always publishable when edge computed
  - Player props require confirmed lineup OR p_start >= threshold (default 0.85)
  - Cron-compatible entry points (4 zero-argument functions)
  - Nightly evaluation trigger for completed games
- Discord bot & publishing layer (Unit 10)
  - Bot lifecycle management with graceful startup/shutdown (SIGTERM/SIGINT handling)
  - 7-channel structure: #free-picks (public), 5 paid channels, #announcements
  - Tier-based permission sync (paid vs free subscribers)
  - Pick publishing with anti-spam (message editing on reruns, in-memory cache)
  - Free pick selection (highest-edge team market, 60-90 min window)
  - Structured Discord embeds (team markets & player props with all fields)
  - Publish-only in v1 (no user commands or interactive features)
  - Respects publishing gate and positive-edge filtering (edge > 0, kelly > 0)
- Stripe subscription & webhook integration (Unit 11)
  - Checkout session creation with discord_user_id linking via client_reference_id
  - Webhook signature verification using Stripe signing secret
  - Event handlers for 5 subscription lifecycle events (checkout completed, invoice paid/failed, subscription updated/deleted)
  - Idempotent subscription state sync to subscriptions table (tier, status, stripe_customer_id, current_period_end)
  - Best-effort Discord role management (grant "Subscriber" role on paid/active, revoke on free/cancelled)
  - Standalone aiohttp webhook server (POST /webhooks/stripe) decoupled from Discord bot
  - Single subscription tier (free and paid) with two-way state transitions

**Non-Goals for V1:**

- Multi-database pool support (single pool only)
- Real-time streaming data ingestion
- Web UI or REST API (Discord-only output)
- Historical data backfill beyond current season
- Advanced monetization features (multiple tiers, proration, refund automation, customer portal)
- Mobile applications

## Project Structure

```
mlb-analytics-platform/
├── src/mlb/               # Source code (namespace package)
│   ├── config/           # Configuration and settings
│   │   └── settings.py  # Pydantic settings classes
│   ├── db/               # Database layer
│   │   ├── models.py    # Table constants and column enums
│   │   ├── pool.py      # Connection pooling
│   │   └── schema/      # Migrations and schema management
│   │       ├── migrate.py           # Migration runner
│   │       └── migrations/          # SQL migration files
│   │           ├── 001_initial.sql         # Create all tables
│   │           ├── 002_seed_teams.sql      # Seed reference data
│   │           ├── 003_fix_cards.sql       # Fix cards table constraints
│   │           ├── 004_players_trigger.sql # Add player audit trigger
│   │           ├── 005_game_scores.sql     # Add home_score/away_score columns
│   │           └── 006_eval_upsert.sql     # Add unique index for eval upserts
│   ├── ingestion/        # Data ingestion layer
│   │   ├── base.py      # Abstract providers and canonical schemas
│   │   ├── cache.py     # TTL-based HTTP response cache
│   │   ├── odds.py      # Odds provider with format conversion
│   │   ├── lineups.py   # Lineup provider with confirmation logic
│   │   ├── stats.py     # Stats provider with upsert logic
│   │   ├── games.py     # Game schedule provider
│   │   └── weather.py   # Weather provider with park filtering
│   ├── models/           # Projection models
│   │   ├── features.py         # Game-level feature engineering
│   │   ├── team_runs.py        # Team run-scoring models (μ + dispersion)
│   │   ├── player_features.py  # Player-level feature engineering
│   │   ├── player_props.py     # Player prop models (P(start), PA, outs, event rates)
│   │   ├── registry.py         # Model serialization and versioning
│   │   └── artifacts/          # Serialized model files
│   ├── simulation/       # Monte Carlo simulation engine
│   │   ├── engine.py    # Simulation kernel (NB sampling, tie-break, player props)
│   │   ├── markets.py   # Market probability derivation
│   │   └── persistence.py # Database persistence layer
│   ├── odds/             # Odds processing and edge calculation
│   │   ├── best_line.py # Best-line selection across books
│   │   ├── devig.py     # Proportional devig for fair probabilities
│   │   ├── edge.py      # Edge calculation and Kelly sizing
│   │   └── persistence.py # Edge value persistence
│   ├── evaluation/       # Model evaluation and backtesting
│   │   ├── metrics.py    # Pure metric functions (log loss, Brier, ECE, tail accuracy)
│   │   ├── clv.py        # Closing Line Value computation
│   │   ├── backtest.py   # Rolling-origin backtest orchestration
│   │   ├── calibration.py # Market-specific calibration models
│   │   └── persistence.py # Eval results persistence
│   ├── scheduler/        # Scheduler and orchestration pipeline
│   │   ├── pipeline.py   # Pipeline orchestration (run_global, run_game, run_daily_eval)
│   │   ├── cron.py       # Cron-compatible entry points
│   │   ├── events.py     # Change detection and rerun throttle
│   │   └── gate.py       # Publishing gate logic (is_publishable)
│   ├── discord_bot/      # Discord bot and publishing layer
│   │   ├── bot.py        # Bot lifecycle (MLBPicksBot class, startup/shutdown)
│   │   ├── channels.py   # Channel creation and permission sync
│   │   ├── publisher.py  # Pick publishing (Publisher class, anti-spam, free pick)
│   │   └── formatter.py  # Discord embed formatting (pure functions)
│   ├── payments/         # Stripe subscription and webhook integration
│   │   ├── checkout.py   # Checkout session creation
│   │   ├── webhooks.py   # Webhook handler and event routing
│   │   ├── sync.py       # Subscription state sync and Discord role management
│   │   └── server.py     # Standalone aiohttp webhook server
│   └── main.py           # Application entry point
├── tests/                # Test suite
│   ├── conftest.py         # Pytest fixtures and configuration
│   ├── test_config.py      # Configuration tests
│   ├── test_schema.py      # Schema and migration tests
│   ├── test_ingestion.py   # Ingestion provider tests
│   ├── test_team_runs.py   # Team run-scoring model tests
│   ├── test_player_props.py # Player prop model tests
│   ├── test_simulation.py  # Monte Carlo simulation tests
│   ├── test_edge.py        # Odds processing and edge calculation tests
│   ├── test_evaluation.py  # Evaluation and backtesting tests
│   ├── test_scheduler.py   # Scheduler and orchestration tests
│   ├── test_discord.py     # Discord bot and publishing tests
│   └── test_payments.py    # Stripe subscription and webhook tests
├── docs/                 # Documentation
│   ├── DEVLOG.md        # Development log
│   └── DECISIONS.md     # Architecture decision records
├── check_db.py          # Database connection verification utility
└── pyproject.toml       # Project metadata and dependencies
```

## Development

See [DEVLOG.md](docs/DEVLOG.md) for:

- Unit-by-unit implementation progress
- What's been shipped
- Known limitations
- What's next

See [DECISIONS.md](docs/DECISIONS.md) for:

- Architecture decision records
- Technical rationale

## Requirements

- Python >= 3.11
- PostgreSQL database
- Environment variables configured per .env.example
