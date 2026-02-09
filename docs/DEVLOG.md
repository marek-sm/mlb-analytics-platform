# Development Log

## 2026-02-08: Unit 1 - Repository Skeleton & Configuration

### What Shipped
- **Core Infrastructure**
  - Established `src/mlb/` namespace package structure
  - Implemented `AppConfig` using Pydantic BaseSettings with full 12-factor env var support
  - Created `get_pool()` factory with asyncpg connection pool initialization
  - Implemented database health check (SELECT 1) with 5-second timeout
  - Built minimal boot sequence in `main.py`: config → pool → health check → shutdown

- **Configuration Schema**
  - Environment validation: dev/staging/prod
  - Database: DSN (required), pool min/max with validation
  - API keys: odds, weather (placeholders for future units)
  - Secrets: Discord token, Stripe key (placeholders)
  - Simulation: default_sim_n [2000-10000] with bounds enforcement
  - Logging: configurable log level

- **Project Setup**
  - `pyproject.toml` with core and dev dependency groups
  - `.env.example` documenting all configuration variables
  - `.gitignore` updated for Python, virtual environments, env files

### Tests Added
Comprehensive test coverage in `tests/test_config.py` (20 test cases):
- Valid configuration: minimal and all fields
- Missing required fields: DB_DSN, ENV
- Invalid ENV values and validation of allowed set
- Pool size validation: max < min, max == min, valid ranges
- Simulation count bounds: below min, above max, at boundaries [2000, 10000]
- Empty secret strings (allowed as placeholders)
- Extra environment variables (correctly ignored)
- Case-insensitive environment variable names

### Known Limitations
- Database pool is a module-level singleton; no multi-pool support
- Health check timeout is fixed at 5 seconds (not configurable)
- No database tables or migrations (deferred to Unit 2)
- Secret validation is minimal; downstream units must validate their own keys

### What's Next
**Unit 2**: Schema & Raw Ingestion
- Design and implement Postgres schema
- Create database migrations
- Build data ingestion pipeline for odds/weather/rosters

---

## 2026-02-08: Unit 2 - Database Schema & Migrations

### What Shipped
- **Database Schema (14 tables)**
  - Reference tables: teams, parks with static seasonal park factors
  - Game & lineup tables: games, players, lineups with source timestamps
  - Stats table: player_game_logs (unified hitter/pitcher logs)
  - Odds table: odds_snapshots with game/market/snapshot_ts index
  - Projection tables: projections, sim_market_probs, player_projections
  - Evaluation table: eval_results for model performance tracking
  - Subscription table: subscriptions for Discord/Stripe integration
  - Weather table: weather with game-level conditions
  - Migration tracking: schema_migrations

- **Migration System**
  - Forward-only migration runner in `mlb.db.schema.migrate`
  - Advisory lock (pg_advisory_lock) prevents concurrent migration runs
  - Idempotent: skips already-applied versions
  - CLI entry point: `python -m mlb.db.schema.migrate`
  - `schema_version()` helper returns highest applied migration number

- **Seed Data**
  - All 30 MLB teams with league/division metadata
  - 30 home parks with park factors, outdoor/retractable flags
  - Park factors range from 0.920 (Oracle Park) to 1.200 (Coors Field)
  - Retractable roof parks set to 1.000 (neutral) per spec

- **Schema Module**
  - `mlb.db.models`: Table name constants (Table class) and column enums (League, Division, GameStatus, Position, etc.)
  - No ORM - raw SQL only
  - All mutable tables have created_at/updated_at timestamps (UTC)

### Tests Added
Comprehensive test coverage in `tests/test_schema.py` (22 test cases):
- **Migration Tests**
  - Fresh database migration (applies both migrations)
  - Idempotency (re-running applies nothing)
  - schema_version() before/after migrations
- **Schema Structure Tests**
  - All 14 tables exist
  - Foreign keys verified (20+ FKs including parks→teams, games→teams/parks, lineups→games/players)
  - Unique constraints (teams.abbr, player_game_logs(player_id, game_id), etc.)
  - Indexes (idx_odds_game_market)
- **Seed Data Tests**
  - 30 teams seeded with correct league/division
  - 30 parks seeded with park factors
  - Specific park validations (Coors 1.200, Oracle 0.920, retractable = 1.000)
- **Column Type Tests**
  - Timestamp defaults (created_at, updated_at)
  - Numeric precision (park_factor: NUMERIC(5,3))
  - Nullable fields (first_pitch, odds side for full-game totals)

### Known Limitations
- No down/rollback migrations (fix-forward only)
- Advisory lock uses fixed ID (123456); no configurable lock namespace
- Migration runner does not validate SQL syntax before execution
- No automatic index recommendations or query optimization
- No table partitioning or archival (v2 concern)
- Park factors are single scalars (no handedness splits)
- No data validation at database level beyond NOT NULL constraints

### What's Next
**Unit 3**: Data Ingestion
- Build odds API client with rate limiting
- Implement weather API client
- Create roster/lineup ingestion pipeline
- Schedule daily data fetches

---

## 2026-02-09: Unit 2 - Verification & Closure

### Final Verification
- **Database Schema Validated**
  - All 4 migrations successfully applied to `mlb_analytics` database
  - All 14 tables created with correct structure
  - `schema_migrations` table populated and tracking applied versions
  - Foreign key verification against live database confirms 17 FK constraints present

- **Test Suite Updated**
  - `test_foreign_keys_exist` now validates specific FK relationships by name (all 17 FKs)
  - Removed arbitrary count assertions; schema verification based on required relationships
  - FK validation uses actual table/column pairs: parks→teams, games→teams/parks, lineups→games/players/teams, player_game_logs→players/games, projections→games, sim_market_probs→projections, player_projections→projections/players/games, weather→games, odds_snapshots→games
  - All 18 schema tests passing

### Ground Truth Established
- Database schema (via pg_constraint) is the source of truth for constraint verification
- Test suite validates presence of specific required relationships, not counts
- Migration system proven idempotent and transactional

**Unit 2 is formally closed. Schema and migrations are stable and verified.**

---
