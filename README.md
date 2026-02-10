# MLB Analytics Platform

MLB analytics platform with Monte Carlo simulation and Discord publishing.

## Overview

This platform provides analytical capabilities for Major League Baseball data through probabilistic modeling (Monte Carlo simulations) with automated insights delivery. Built with Python 3.11+ using async patterns throughout.

**Current Status:** Unit 2 complete (database schema and migrations). See [DEVLOG.md](docs/DEVLOG.md) for implementation details and [DECISIONS.md](docs/DECISIONS.md) for architectural choices.

## Installation

```bash
# Clone and enter the repository
git clone <repository-url>
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
- Raw data ingestion for odds, weather, rosters (Unit 2+)
- Monte Carlo simulation engine with adaptive simulation counts
- Discord publishing of analytical insights

**Non-Goals for V1:**

- Multi-database pool support (single pool only)
- Real-time streaming data ingestion
- Web UI or REST API (Discord-only output)
- Historical data backfill beyond current season
- Advanced monetization features (Stripe integration placeholder only)
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
│   │           └── 004_players_trigger.sql # Add player audit trigger
│   └── main.py           # Application entry point
├── tests/                # Test suite
│   ├── conftest.py      # Pytest fixtures and configuration
│   ├── test_config.py   # Configuration tests
│   └── test_schema.py   # Schema and migration tests
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
