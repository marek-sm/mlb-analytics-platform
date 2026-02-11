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

## 2026-02-09: Unit 3 - Data Ingestion: Provider-Agnostic Interfaces

### What Shipped
- **Abstract Base Classes (ABCs)**
  - `OddsProvider`: fetch_odds() returning canonical OddsRow with European decimal prices
  - `LineupProvider`: fetch_lineup() and is_confirmed() for lineup data
  - `StatsProvider`: fetch_game_logs() for player statistics
  - `GameProvider`: fetch_schedule() for game schedules
  - `WeatherProvider`: fetch_weather() with park type filtering

- **Canonical Row Schemas**
  - `OddsRow`: game_id, book, market, side, line, price (decimal ≥1.0), snapshot_ts
  - `LineupRow`: game_id, team_id, player_id, batting_order, is_confirmed, source_ts
  - `GameLogRow`: player_id, game_id, batting/pitching stats (nullable), is_starter
  - `GameRow`: game_id, game_date, home/away teams, park, first_pitch, status
  - `WeatherRow`: game_id, temp_f, wind_speed/dir, precip_pct, fetched_at

- **Concrete V1 Implementations**
  - `V1OddsProvider`: Stub with American→European decimal conversion logic
  - `V1LineupProvider`: Implements confirmation flip logic per D-011
  - `V1StatsProvider`: Upsert game logs with player existence check (D-020)
  - `V1GameProvider`: Upsert games with status change handling
  - `V1WeatherProvider`: Park type filtering (returns None for indoor/retractable)

- **Odds Conversion System**
  - `american_to_decimal()`: Converts American odds to European decimal
  - `detect_and_convert_odds()`: Auto-detects format with range-based heuristics (D-017)
  - Positive American: decimal = (american / 100) + 1
  - Negative American: decimal = (100 / abs(american)) + 1
  - Ambiguous values logged and skipped

- **Lineup Confirmation Contract**
  - `write_lineup()`: Flips prior confirmed rows to is_confirmed=FALSE before inserting new confirmed lineup
  - Implements D-011: at most one confirmed lineup per (game_id, team_id, batting_order)
  - Handles lineup re-confirmation (rare but valid)

- **TTL-Based Cache**
  - In-memory cache with expiration checks
  - `CacheEntry` dataclass: key, payload (bytes), fetched_at, ttl_seconds
  - `get_cache()` singleton accessor
  - Prune expired entries on demand

- **Conservative Fallback Policy**
  - All providers return empty list or None on error (never fabricate data)
  - Errors logged with warnings (exc_info=True)
  - No partial writes on failure
  - Retry logic delegated to scheduler (D-019)

### Tests Added
Comprehensive test coverage in `tests/test_ingestion.py` (16 test cases):
- **ABC Implementation Tests**
  - All 5 ABCs have concrete implementations
- **Odds Conversion Tests**
  - American +150 → 2.50, −110 → 1.909
  - Edge cases: +100, −100, +200, −200
  - Zero raises ValueError
  - Format detection for decimal (1.0-50.0) vs American (≥100 or ≤−100)
  - Ambiguous values (e.g., 75, −50) raise with clear error
- **Lineup Confirmation Flip Test**
  - Insert initial confirmed lineup (9 players)
  - Insert new confirmed lineup (different players)
  - Verify old lineup flipped to is_confirmed=FALSE
  - Verify new lineup is confirmed
- **Stats Upsert Test**
  - Initial game log insert
  - Updated game log with same (player_id, game_id)
  - Verify upsert (not duplicate insert)
  - Verify stats updated correctly
- **Weather Park Filtering Test**
  - Returns None for retractable-roof parks
  - Returns None for invalid park_id
  - Outdoor parks proceed to fetch (stub returns None in v1)
- **Fallback Behavior Tests**
  - All providers return empty/None on errors (not exceptions)
  - No data written on fetch failure
- **Cache Tests**
  - TTL prevents duplicate fetches
  - Expiration removes stale entries
  - Cache miss returns None
  - Prune removes expired entries
  - Singleton behavior verified
- **Player Upsert Test (D-020)**
  - Unknown player upserted with minimal metadata
  - Missing position/bats/throws are NULL
  - Update with team_id works correctly

### Known Limitations
- V1 providers are stubs (no real API integration)
- No HTTP client implementation (deferred to provider-specific unit)
- No rate limiting or quota tracking
- Cache is in-memory only (cleared on restart)
- No cache size limits or eviction policy
- No multi-provider failover (single provider assumption per D-014)
- No Parquet export (deferred)
- Weather for retractable-roof parks treated as neutral (roof state not inferred)

### What's Next
**Unit 4**: Feature Engineering
- Aggregate player stats (rolling averages, weighted splits)
- Calculate matchup-specific adjustments (vs. LHP/RHP)
- Apply park factors to expected runs
- Build feature vectors for projection models

---

## 2026-02-10: Unit 4 - Team Run-Scoring Models

### What Shipped
- **Feature Engineering Module (`models/features.py`)**
  - `build_game_features()`: Builds game-level feature vectors from ingested data
  - Fetches game metadata, park factors, weather (with dome park handling)
  - Extracts starting pitchers from confirmed lineups (D-011)
  - Calculates pitcher features: rest days, rolling pitch count avg, shrunk ERA
  - Calculates lineup OPS with empirical Bayes shrinkage (D-021)
  - Computes bullpen fatigue (7-day IP usage) and team run environment (30-day R/G)
  - Conservative weather fallback for outdoor parks (72°F, 5mph wind, 0% precip) when data missing (D-023)

- **Model Training & Inference (`models/team_runs.py`)**
  - `train()`: Trains four LightGBM models (home/away μ, home/away dispersion)
  - Separate μ and r models with identical feature schemas (D-024)
  - Trains on historical final games with confirmed lineups
  - Serializes models to disk with timestamp-based versioning
  - Logs training metrics (RMSE on training set)
  - `predict()`: Loads models, builds features, returns `TeamRunParams`
  - Park factor applied exactly once as multiplicative adjustment to μ (D-010)
  - μ clamped to [0.5, 15.0], dispersion clamped to ≥0.1 for stability

- **Model Registry (`models/registry.py`)**
  - `save_model()` / `load_model()`: Pickle-based serialization to `models/artifacts/`
  - Versioned by timestamp (e.g., `home_mu_20260210_153000.pkl`)

- **Configuration Extensions**
  - Added `shrinkage_k_batter` (default 200 PA) and `shrinkage_k_pitcher` (default 80 IP) to AppConfig
  - Added `rolling_window_batting_days` (default 60) and `rolling_window_pitching_days` (default 30)
  - All configurable via environment variables

- **Contracts**
  - `GameFeatures`: 23 fields including park factor, weather, starters, lineup strength, bullpen usage
  - `TeamRunParams`: game_id, home/away μ, home/away dispersion, model_version

### Tests Added
Comprehensive test coverage in `tests/test_team_runs.py` (7 test cases):
- **AC1**: `build_game_features()` returns fully populated GameFeatures with all fields non-null (except weather for domes)
- **AC2**: `train()` completes on ≥30 games and serializes all four models to disk
- **AC3**: `predict()` returns valid TeamRunParams with μ in [1.0, 12.0] and dispersion > 0
- **AC4**: Park factor test: Coors Field (1.200) vs neutral park (1.000) produces ~1.2× μ ratio
- **AC5**: Shrinkage test: 10 IP pitcher ERA estimate closer to league mean than 150 IP pitcher
- **AC6**: Weather None handling: dome/retractable parks return None for all weather fields, model succeeds
- **AC7**: Model artifacts load from disk and produce identical output to in-memory models

### Known Limitations
- V1 uses LightGBM only (no XGBoost or neural nets)
- Dispersion model trains on absolute residuals as proxy (not true Negative Binomial MLE)
- No cross-validation or held-out test set (RMSE reported on training data)
- No feature importance logging or model interpretability tools
- No automatic model retraining or drift detection (deferred to Unit 9)
- Lineup OPS calculation assumes batting-order-agnostic contribution (no lineup-slot weighting)
- Wind direction simplified to 8 categorical bins (no continuous degrees)
- No handedness splits for park factors (deferred to v2)
- No caching of feature engineering results (每 game rebuilt from scratch)
- Model artifacts stored as pickle files (no MLflow or versioning beyond timestamps)
- No model ensemble or stacking (single model per target)
- Insufficient history fallback (< 3 starts) uses league averages without player priors

### Dependency Notes
- **scikit-learn added for training metrics (RMSE). Not used in inference path.**

### What's Next
**Unit 5**: Player Prop Models
- Share feature engineering with Unit 4 for game-level covariates
- Build player-specific features (recent performance, matchup splits)
- Train models for H, TB, HR, RBI, R, BB, K, OUTS, ER
- Write `player_projections` table

---
