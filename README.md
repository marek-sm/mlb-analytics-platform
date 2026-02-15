# MLB Analytics Platform

**Probabilistic MLB betting platform with Monte Carlo simulation, negative binomial run modeling, and Discord-based delivery.**

---

## Features

- **Team Markets:** Moneyline, run line (±1.5), game totals (over/under 8.5), team totals (over/under 4.5)
- **Player Props:** Hitter stats (H, TB, HR, RBI, R, BB) and pitcher stats (K, outs, ER) for starting pitchers and top-7 hitters
- **Simulation-Based Projections:** 2,000–10,000 Monte Carlo trials per game using Negative Binomial run distributions
- **Edge Calculation:** Proportional devig across multiple sportsbooks with fractional Kelly bankroll sizing
- **Discord Publishing:** Tiered delivery (free daily pick + 5 paid channels for full coverage)
- **Stripe Integration:** Subscription management with automated Discord role sync
- **Daily Evaluation:** Rolling-origin backtesting with log loss, Brier score, ECE, tail accuracy, and CLV metrics

---

## System Architecture

The platform follows a pipeline architecture with scheduled runs and event-driven reruns:

```
Data Providers → Ingestion → Database → Feature Engineering → Models → Simulation → Edge Calculation → Discord Publishing
                                                                                                    ↓
                                                                             Stripe Webhooks → Subscription Sync
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed Mermaid diagrams and module dependencies.

---

## Data Flow

### Data Sources

- **The Odds API**: Odds data (moneyline, run line, totals) with American-to-decimal conversion (D-056)
- **MLB Stats API**: Game schedules, team rosters, and official game data (D-055)
- **Weather API**: Outdoor park weather conditions (temperature, wind, precipitation)
- **Stats Provider**: Player statistics and performance data

1. **Ingestion** (Unit 3): Fetch odds, lineups, player stats, game schedules, and weather from external providers. Convert American odds to European decimal format. Store with source timestamps for temporal tracking.
2. **Feature Engineering** (Unit 4): Build game-level features including park factors, weather conditions, starting pitcher metrics (rest, rolling pitch count, shrunk ERA), lineup strength (OPS with empirical Bayes shrinkage), and bullpen fatigue.
3. **Team Run Models** (Unit 4): Train LightGBM models to predict mean (μ) and dispersion (r) for home and away team run distributions. Apply park factors as multiplicative adjustments to μ (exactly once).
4. **Player Prop Models** (Unit 5): Train P(start) classifier, PA distribution (multiclass), pitcher outs distribution (multiclass), and event-rate models (shrunk rolling means for H/PA, HR/PA, K/BF, ER/out, etc.).
5. **Monte Carlo Simulation** (Unit 6): Sample team runs from Negative Binomial distributions with optional correlated noise (bivariate normal copula). Sample player stats (PA → H/TB/HR/RBI/R/BB for hitters; outs → K/ER for pitchers). Derive market probabilities from simulated outcomes.
6. **Edge Calculation** (Unit 7): Fetch best available odds per market across sportsbooks. Devig using proportional method to obtain fair probabilities. Calculate edge = p_model − p_fair. Compute fractional Kelly sizing (0.25× by default) with minimum edge threshold (2%).
7. **Publishing** (Unit 10): Filter projections through publishing gate (lineup confirmed OR p_start ≥ 0.85 for player props; team markets always publishable when edge computed). Publish positive-edge plays (edge > 0, kelly > 0) to Discord channels with anti-spam message editing.
8. **Evaluation** (Unit 8): Nightly backtesting on completed games. Compute metrics (log loss, Brier, ECE, tail accuracy, CLV vs. T-5 closing odds) and persist to eval_results table. Train market-specific calibration models (isotonic regression).

---

## Modeling Approach

### Negative Binomial Run Scoring

Team run distributions are modeled as Negative Binomial with mean μ and dispersion r, both predicted by LightGBM models. This captures the overdispersion inherent in MLB run scoring (variance > mean), unlike Poisson models which assume variance = mean.

**Why Negative Binomial over Poisson?** (Decision D-024)
MLB run distributions exhibit significant variance beyond what Poisson can capture. Negative Binomial allows the dispersion parameter r to be fitted to data, producing more accurate tail probabilities for game totals and run lines.

### Team Run Model Features

- Park factors (static seasonal, applied exactly once as multiplicative adjustment to μ)
- Weather: temperature, wind speed/direction, precipitation probability (outdoor parks only; domes/retractable treated as neutral)
- Starting pitcher: rest days, rolling pitch count average, shrunk ERA (empirical Bayes with k=80 IP prior)
- Lineup strength: OPS with empirical Bayes shrinkage (k=200 PA prior)
- Bullpen fatigue: 7-day IP usage
- Team run environment: 30-day rolling R/G

### Player Prop Models

**Hitter Props (top-7 lineup only):**
- **P(start):** LightGBM binary classifier using platoon matchup (switch hitters always advantaged), days rest, starts last 7/14 days, batting order history
- **PA distribution:** LightGBM multiclass (7 classes: 0, 1, 2, 3, 4, 5, 6+ PA)
- **Event rates:** Shrunk rolling means for H/PA, TB/PA, HR/PA, RBI/PA, R/PA, BB/PA (60-day batting window)

**Pitcher Props (starters only):**
- **Outs distribution:** LightGBM multiclass (10 classes: 0–3, 4–6, …, 27+ outs)
- **Event rates:** K/BF (approximated as K / (ip_outs × 1.35)), ER/out (30-day pitching window)

**Why shrunk means over ML rate models in v1?** (Decision D-029)
For per-opportunity event rates (H/PA, K/BF), empirical Bayes shrinkage toward league mean provides sufficient signal for v1 sample sizes while avoiding overfitting. Upgrading to gradient boosting for rate models is a v2 option.

### Monte Carlo Simulation

- **Adaptive trial counts:** 2,000–10,000 simulations per game (configurable)
- **Correlated noise:** Bivariate normal copula with correlation ρ=0.15 (configurable) to capture same-game dependencies
- **Extra-innings tie-break:** Simplified probabilistic resolution P(home_win | tie) = home_μ / (home_μ + away_μ), adjusted by ±0.02 for bullpen fatigue differential. No full inning-by-inning extras in v1.
- **Player sampling:** Independent Bernoulli per PA for hits/HR/BB, Poisson for RBI/R (v1 independence assumption; joint distribution deferred to v2)

### Devig and Edge Calculation

**Proportional (multiplicative) devig only** (Decision D-036):
Fair probability for outcome i = (1/price_i) / Σ(1/price_j). Requires both sides of a two-way market from the same book. No power devig, Shin, or additive methods in v1.

**Main lines hardcoded** (Decision D-034):
Team totals 4.5, game total 8.5, run line ±1.5. Player props: H=0.5, TB=1.5, HR=0.5, RBI=0.5, R=0.5, BB=0.5, K=4.5, OUTS=16.5, ER=2.5. Dynamic line detection deferred to v2.

**Quarter-Kelly sizing** (Decision D-038):
Kelly fraction = 0.25 × edge / (decimal_price − 1). Minimum edge threshold 2% (configurable); edges below threshold stored but kelly_fraction set to 0.0.

---

## Evaluation

### Metrics

- **Log Loss:** Cross-entropy loss with epsilon clipping (1e-15) to prevent log(0)
- **Brier Score:** Mean squared error between probabilities and binary outcomes
- **ECE (Expected Calibration Error):** Calibration diagnostic with 10 bins (default)
- **Tail Accuracy:** Separate calibration for extreme probabilities (p < 0.15, p > 0.85)
- **CLV (Closing Line Value):** p_model − p_close_fair, where p_close_fair is devigged odds at T-5 minutes before first pitch. Median CLV > 0 indicates beating the closing market.

### Rolling-Origin Backtesting

- Evaluate model performance over historical date ranges for each market (ml, rl, total, team_total)
- Only final games with non-null scores (no future leakage)
- Uses most recent projection per game (by run_ts)
- Results persisted to eval_results table with upsert on (eval_date, market, metric)

### Market-Specific Calibration

- **Isotonic regression (default):** Nonparametric, handles non-monotonic miscalibration
- **Platt scaling (optional):** Parametric logistic regression, smoother for small samples
- Requires minimum 50 samples to fit (configurable)
- Team markets only in v1 (player prop calibration deferred to v2)

---

## Key Tradeoffs & Design Decisions

### Why Negative Binomial over Poisson?

MLB run distributions exhibit overdispersion (variance > mean). Negative Binomial models both mean and dispersion explicitly, capturing tail probabilities more accurately than Poisson (which assumes variance = mean). This is critical for totals and run line markets where tail accuracy drives edge.

### Why shrunk means over ML rate models for player event rates (v1)?

For per-opportunity rates (H/PA, K/BF, BB/PA), empirical Bayes shrinkage toward league mean balances signal and noise effectively on small samples (50–200 PA). Gradient boosting would add complexity and risk overfitting in v1. Upgrading to ML-based rate models is a v2 option.

### Why proportional devig only?

Proportional devig is simple, deterministic, and widely used in sports betting. Power devig, Shin, and additive methods require additional assumptions (bettor sharpness, insider knowledge) that add complexity without clear v1 benefit. Single-method keeps the system auditable.

### Why hardcoded main lines?

Dynamic line detection from odds requires line-matching logic and handling of alternate lines, both out-of-scope for v1. Hardcoded main lines (e.g., total 8.5, run line ±1.5) cover 80%+ of MLB market volume and simplify probability derivation. Alternate lines deferred to v2.

### Why quarter-Kelly (0.25×)?

Full Kelly is known to be aggressive and produces high variance in sports betting contexts. Quarter-Kelly (0.25×) is a standard fractional Kelly multiplier that balances growth and risk. Configurable via `KELLY_FRACTION_MULTIPLIER` (range 0.05–1.0).

### Why top-7 lineup only for hitter props?

Bottom-of-order hitters (positions 8–9) typically receive 2–3 PA per game with low volume, producing unreliable projections and limited market availability. Filtering to top-7 reduces noise and focuses coverage on high-confidence plays.

### Why no reliever props?

Reliever usage is highly volatile and game-state dependent (e.g., blowouts vs. close games). Modeling requires inning-by-inning simulation and bullpen strategy prediction, both out-of-scope for v1. Starting pitchers only.

---

## Tech Stack

- **Language:** Python 3.11+
- **Database:** PostgreSQL (asyncpg driver)
- **Models:** LightGBM for classification/regression, scikit-learn for calibration
- **Simulation:** NumPy, SciPy (Negative Binomial sampling, copula transformations)
- **Discord Bot:** discord.py 2.3+
- **Payments:** Stripe SDK 7.0+, aiohttp 3.9+ (webhook server)
- **Testing:** pytest with async fixtures
- **Deployment:** Cron-compatible entry points, standalone webhook server

---

## Setup & Deployment

### Prerequisites

- Python >= 3.11
- PostgreSQL database
- Discord bot token (create at [Discord Developer Portal](https://discord.com/developers/applications))
- Stripe account with webhook secret and Price ID
- Environment variables configured per `.env.example`

### Installation

```bash
# Clone repository
git clone https://github.com/marek-sm/mlb-analytics-platform.git
cd mlb-analytics-platform

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install with development dependencies
pip install -e ".[dev]"

# Configure environment
cp .env.example .env
# Edit .env with your database DSN, API keys, Discord token, Stripe keys, etc.
```

### Database Migration

```bash
# Run migrations to create tables and seed reference data
python -m mlb.db.schema.migrate
```

### Cron Schedule Setup

Add the following to your crontab for scheduled pipeline runs (adjust times to your timezone):

```cron
# Global runs (ET timezone)
0 22 * * * cd /path/to/mlb-analytics-platform && /path/to/.venv/bin/python -c "from mlb.scheduler.cron import night_before_run; night_before_run()"
0 8 * * * cd /path/to/mlb-analytics-platform && /path/to/.venv/bin/python -c "from mlb.scheduler.cron import morning_run; morning_run()"
0 12 * * * cd /path/to/mlb-analytics-platform && /path/to/.venv/bin/python -c "from mlb.scheduler.cron import midday_run; midday_run()"

# Nightly evaluation (runs after midnight for yesterday's games)
30 2 * * * cd /path/to/mlb-analytics-platform && /path/to/.venv/bin/python -c "from mlb.scheduler.cron import nightly_eval_run; nightly_eval_run()"
```

Per-game runs at T-90 and T-30 minutes require custom scheduling or manual triggering via `run_game(game_id)`.

### Discord Bot Startup

The Discord bot must run as a persistent process:

```bash
# Run bot in foreground
python -c "from mlb.discord_bot.bot import MLBPicksBot; import asyncio; bot = MLBPicksBot(); asyncio.run(bot.run_until_shutdown())"

# Or use a process manager (e.g., systemd, supervisord)
```

**Systemd example:**

```ini
[Unit]
Description=MLB Analytics Discord Bot
After=network.target

[Service]
Type=simple
User=mlb
WorkingDirectory=/path/to/mlb-analytics-platform
Environment="PATH=/path/to/.venv/bin"
ExecStart=/path/to/.venv/bin/python -c "from mlb.discord_bot.bot import MLBPicksBot; import asyncio; bot = MLBPicksBot(); asyncio.run(bot.run_until_shutdown())"
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

### Stripe Webhook Server Startup

The webhook server runs as a standalone HTTP service (decoupled from Discord bot per D-052):

```bash
# Run webhook server in foreground
python -m mlb.payments.server

# Or use systemd
```

**Systemd example:**

```ini
[Unit]
Description=MLB Analytics Stripe Webhook Server
After=network.target

[Service]
Type=simple
User=mlb
WorkingDirectory=/path/to/mlb-analytics-platform
Environment="PATH=/path/to/.venv/bin"
ExecStart=/path/to/.venv/bin/python -m mlb.payments.server
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

**Configure Stripe webhook endpoint:** Point Stripe webhook to `https://yourdomain.com/webhooks/stripe` (configure reverse proxy with SSL termination to forward to localhost:8080).

---

## What This System Does Not Do Yet

- Live betting or in-game probability updates
- Alternate lines or derivative markets beyond main lines
- Any UI beyond Discord-based delivery
- Multi-sport support (MLB only)
- Reliever props
- Bench-only hitter props (positions 8–9)
- Full inning-by-inning extra-innings simulation (uses simplified tie-break)
- Retractable-roof inference (treated as neutral)
- Dynamic main-line detection from odds (hardcoded lines)
- Joint hitter stat distribution (independent sampling in v1)
- ML-based event rate models (shrunk rolling means in v1)
- Player prop calibration (team markets only)
- Multi-provider data failover
- Power/Shin/additive devig methods
- Interactive Discord commands
- Admin dashboard or subscription management UI
- Automated refunds
- Batters-faced column in schema (uses BF approximation)

---

## Project Structure

```
mlb-analytics-platform/
├── src/mlb/                          # Source code (namespace package)
│   ├── __init__.py                   # Package root
│   ├── main.py                       # Application entry point
│   ├── config/                       # Configuration and settings
│   │   ├── __init__.py
│   │   └── settings.py               # Pydantic BaseSettings (AppConfig)
│   ├── db/                           # Database layer
│   │   ├── __init__.py
│   │   ├── models.py                 # Table constants and column enums
│   │   ├── pool.py                   # asyncpg connection pooling
│   │   └── schema/                   # Migrations and schema management
│   │       ├── __init__.py
│   │       ├── migrate.py            # Migration runner
│   │       └── migrations/           # SQL migration files
│   │           ├── 001_initial.sql           # Create all tables
│   │           ├── 002_seed_teams.sql        # Seed teams and parks
│   │           ├── 003_fix_cards.sql         # Fix cards table constraints
│   │           ├── 004_players_trigger.sql   # Player audit trigger
│   │           ├── 005_game_scores.sql       # Add home_score/away_score
│   │           └── 006_eval_upsert.sql       # Add eval_results unique index
│   ├── ingestion/                    # Data ingestion layer
│   │   ├── __init__.py
│   │   ├── base.py                   # Abstract providers and canonical schemas
│   │   ├── cache.py                  # TTL-based HTTP response cache
│   │   ├── odds.py                   # Odds provider with American→decimal conversion
│   │   ├── lineups.py                # Lineup provider with confirmation flip logic
│   │   ├── stats.py                  # Stats provider with upsert logic
│   │   ├── games.py                  # Game schedule provider
│   │   └── weather.py                # Weather provider with park filtering
│   ├── models/                       # Projection models
│   │   ├── __init__.py
│   │   ├── features.py               # Game-level feature engineering
│   │   ├── team_runs.py              # Team run-scoring models (μ + dispersion)
│   │   ├── player_features.py        # Player-level feature engineering
│   │   ├── player_props.py           # Player prop models (P(start), PA, outs, event rates)
│   │   ├── registry.py               # Model serialization and versioning
│   │   └── artifacts/                # Serialized model files (.pkl)
│   ├── simulation/                   # Monte Carlo simulation engine
│   │   ├── __init__.py
│   │   ├── engine.py                 # Simulation kernel (NB sampling, player props)
│   │   ├── markets.py                # Market probability derivation
│   │   └── persistence.py            # Database persistence layer
│   ├── odds/                         # Odds processing and edge calculation
│   │   ├── __init__.py
│   │   ├── best_line.py              # Best-line selection across books
│   │   ├── devig.py                  # Proportional devig for fair probabilities
│   │   ├── edge.py                   # Edge calculation and Kelly sizing
│   │   └── persistence.py            # Edge value persistence
│   ├── evaluation/                   # Model evaluation and backtesting
│   │   ├── __init__.py
│   │   ├── metrics.py                # Pure metric functions (log loss, Brier, ECE, tail accuracy)
│   │   ├── clv.py                    # Closing Line Value computation
│   │   ├── backtest.py               # Rolling-origin backtest orchestration
│   │   ├── calibration.py            # Market-specific calibration models
│   │   └── persistence.py            # Eval results persistence
│   ├── scheduler/                    # Scheduler and orchestration pipeline
│   │   ├── __init__.py
│   │   ├── pipeline.py               # Pipeline orchestration (run_global, run_game, run_daily_eval)
│   │   ├── cron.py                   # Cron-compatible entry points
│   │   ├── events.py                 # Change detection and rerun throttle
│   │   └── gate.py                   # Publishing gate logic (is_publishable)
│   ├── discord_bot/                  # Discord bot and publishing layer
│   │   ├── __init__.py
│   │   ├── bot.py                    # Bot lifecycle (MLBPicksBot class, startup/shutdown)
│   │   ├── channels.py               # Channel creation and permission sync
│   │   ├── publisher.py              # Pick publishing (Publisher class, anti-spam, free pick)
│   │   └── formatter.py              # Discord embed formatting (pure functions)
│   └── payments/                     # Stripe subscription and webhook integration
│       ├── __init__.py
│       ├── checkout.py               # Checkout session creation
│       ├── webhooks.py               # Webhook handler and event routing
│       ├── sync.py                   # Subscription state sync and Discord role management
│       └── server.py                 # Standalone aiohttp webhook server
├── tests/                            # Test suite
│   ├── conftest.py                   # Pytest fixtures and configuration
│   ├── test_config.py                # Configuration tests
│   ├── test_schema.py                # Schema and migration tests
│   ├── test_ingestion.py             # Ingestion provider tests
│   ├── test_team_runs.py             # Team run-scoring model tests
│   ├── test_player_props.py          # Player prop model tests
│   ├── test_simulation.py            # Monte Carlo simulation tests
│   ├── test_edge.py                  # Odds processing and edge calculation tests
│   ├── test_evaluation.py            # Evaluation and backtesting tests
│   ├── test_scheduler.py             # Scheduler and orchestration tests
│   ├── test_discord.py               # Discord bot and publishing tests
│   └── test_payments.py              # Stripe subscription and webhook tests
├── docs/                             # Documentation
│   ├── ARCHITECTURE.md               # System architecture and data flow diagrams
│   ├── DEVLOG.md                     # Development log (unit-by-unit progress)
│   └── DECISIONS.md                  # Architecture decision records (D-001 through D-054)
├── .env.example                      # Environment variable template
├── .gitignore                        # Git ignore rules
├── check_db.py                       # Database connection verification utility
├── pyproject.toml                    # Project metadata and dependencies
└── README.md                         # This file
```

---

## License

Proprietary. All rights reserved.

---

## Development

See [DEVLOG.md](docs/DEVLOG.md) for:
- Unit-by-unit implementation progress
- What's been shipped
- Known limitations
- What's next

See [DECISIONS.md](docs/DECISIONS.md) for:
- Architecture decision records (D-001 through D-054)
- Technical rationale and tradeoffs
- Out-of-scope features and v2 deferments

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for:
- System flow Mermaid diagrams
- Module dependency map
- Detailed data flow narrative
