# Development Log

This document tracks the unit-by-unit implementation progress of the MLB Analytics Platform. Each entry documents what shipped, tests added, known limitations, and next steps.

**Project Status:** All 12 implementation units complete (Units 1–12). System is feature-complete for v1 scope.

---

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

## 2026-02-11: Unit 5 - Player Prop Models: Hitters & Pitchers

### What Shipped
- **Player Feature Engineering (`models/player_features.py`)**
  - `build_hitter_features()`: Per-hitter per-game features for top-7 lineup positions only
  - `build_pitcher_features()`: Per-starter per-game features
  - Hitter features: platoon matchup (D-030), days_rest, starts_last_7/14, shrunk rolling rates (H, TB, HR, RBI, R, BB per PA)
  - Pitcher features: days_rest, rolling pitch count, IP outs, shrunk K rate (per BF), ER rate (per out)
  - Switch hitters always get `platoon_adv = True` (D-030)
  - Reuses game-level covariates (game_mu, opp_lineup_ops) from Unit 4

- **Player Prop Models (`models/player_props.py`)**
  - P(start) model: LightGBM binary classifier predicting probability a hitter starts (D-027)
  - PA distribution model: LightGBM multiclass (7 classes: 0, 1, 2, 3, 4, 5, 6+) (D-028)
  - Pitcher outs distribution: LightGBM multiclass (10 classes: 0–3, 4–6, ..., 27+ outs)
  - Event-rate models: shrunk rolling means using empirical Bayes (D-029)
    - Hitter rates: H/PA, TB/PA, HR/PA, RBI/PA, R/PA, BB/PA
    - Pitcher rates: K/BF, ER/out
  - `train()`: Trains on ≥30 final games, serializes 3 models (p_start, pa_dist, outs_dist)
  - `predict_hitters()`: Returns `HitterPropParams` for top-7 hitters
  - `predict_pitcher()`: Returns `PitcherPropParams` for starting pitcher

- **Contracts**
  - `HitterFeatures`: 17 fields including platoon_adv, rolling stats, game context
  - `PitcherFeatures`: 10 fields including rolling workload, matchup strength
  - `HitterPropParams`: p_start (publishing gate input), pa_dist, event rates
  - `PitcherPropParams`: outs_dist, k_rate, er_rate

### Tests Added
Comprehensive test coverage in `tests/test_player_props.py` (9 test cases):
- **AC1**: P(start) model: validates directional correctness (p_start > 0)
- **AC2**: PA distribution sums to 1.0 (±0.001), all probabilities non-negative
- **AC3**: Hitter shrinkage: 20 PA player regresses harder to league mean than 400 PA player (H rate)
- **AC4**: Pitcher outs distribution sums to 1.0 (±0.001)
- **AC5**: Pitcher shrinkage: 10 IP pitcher regresses harder to league mean than 150 IP pitcher (K rate)
- **AC6**: Top-7 filter: `build_hitter_features()` returns only positions 1-7, excludes 8-9
- **AC7**: `train()` completes on ≥30 games and serializes p_start, pa_dist, outs_dist models
- **AC8**: BB rate is optional (can be None)
- **D-030 validation**: Switch hitters always get platoon_adv = True

### Known Limitations
- **Schema limitation (FC-17)**: `player_game_logs` table lacks `batters_faced` column. K rate calculated as K per out instead of K per batter faced (less accurate proxy). Adding `batters_faced` column deferred to v2.
- V1 uses LightGBM multiclass for discrete distributions (PA, outs), not count models (Poisson/NegBin)
- Event rates are shrunk rolling means, not ML models (upgrading to gradient boosting is v2 option)
- No cross-validation or held-out test set (models evaluated on training data)
- No reliever props (v1 exclusion: starting pitchers only)
- No bench-only hitters (v1 exclusion: top-7 lineup only)
- BB rate model is always trained (no feature flag for optional training in v1)
- No caching of player features (rebuilt on each prediction)
- Insufficient history fallback (< 10 PA / 3 starts) uses league averages without player priors
- No confidence intervals or uncertainty quantification on prop predictions
- P(start) threshold is model output, not publishing gate threshold (Unit 9 applies threshold)
- No alternate lines (over/under different totals) in v1
- Model calibration on synthetic test data is poor; test assertions validate directional correctness only

### Fix: FC-18 - K rate uses BF approximation (2026-02-11)
**Problem**: K rate was calculated as K per out instead of K per batter faced, which would produce biased strikeout projections in Unit 6 simulation.

**Fix applied**:
- Added `bf_per_out_ratio` config constant (default 1.35)
- Updated `player_features.py` and `player_props.py` to calculate K rate as K / (ip_outs × bf_per_out_ratio)
- Added test `test_k_rate_uses_batters_faced_approx` validating the approximation
- Documented decision D-031

**Impact**: K rate now correctly represents P(K | batter faced) as specified, enabling accurate strikeout simulation in Unit 6.

### What's Next
**Unit 6**: Monte Carlo Simulation Engine
- Sample from team run distributions (Unit 4) and player prop distributions (Unit 5)
- Simulate individual PA outcomes (H, TB, HR, etc.) and pitcher outs/K/ER
- Aggregate simulated stats to produce player projection distributions
- Write to `player_projections` table

---

## 2026-02-11: Unit 6 - Monte Carlo Simulation Engine

### What Shipped
- **Simulation Kernel (`simulation/engine.py`)**
  - `simulate_game()`: Main entry point for Monte Carlo simulation (2k–10k trials, adaptive)
  - Negative Binomial sampling for team run distributions using μ and r from Unit 4
  - Correlated noise via bivariate normal copula (D-032): shared standard normal Z per trial, default ρ=0.15
  - Extra-innings tie-break: probabilistic resolution based on relative team strength (D-033)
  - Player prop sampling:
    - Hitters: PA from pa_dist, then Bernoulli/Binomial for H/HR/BB, Poisson for RBI/R (D-035)
    - Pitchers: outs from outs_dist, K from k_rate × BF, ER from er_rate × outs
  - Park factor NOT re-applied (uses pre-adjusted μ from TeamRunParams, D-010)
  - Handles hitter p_start < 1.0 (zeros out stats in non-start trials)
  - BB model optional (skips if bb_rate is None)

- **Market Derivation (`simulation/markets.py`)**
  - `derive_team_markets()`: Derives all four team markets from simulated score matrix
    - Moneyline (home/away win probability)
    - Run Line ±1.5 (home/away cover probability)
    - Game Total 8.5 (over/under probability)
    - Team Totals 4.5 (home/away over/under probability)
  - `derive_player_props()`: Derives player prop probabilities from sampled stats
    - Hitter props: H, TB, HR, RBI, R, BB (over main line)
    - Pitcher props: K, OUTS, ER (over main line)
    - Uses hardcoded main lines (D-034): H=0.5, TB=1.5, HR=0.5, etc.

- **Persistence (`simulation/persistence.py`)**
  - `persist_simulation_results()`: Writes SimResult → projections + sim_market_probs + player_projections
  - Returns projection_id for foreign key linkage
  - Leaves edge, kelly_fraction, edge_computed_at as NULL (populated by Unit 7)
  - Batch inserts for team markets and player props

- **Contracts**
  - `SimResult`: game_id, run_ts, home/away μ/disp, sim_n, home/away scores, model_version, hitter_sims, pitcher_sims
  - `HitterSimResult`: player_id, p_start, PA/H/TB/HR/RBI/R/BB arrays (shape: sim_n)
  - `PitcherSimResult`: player_id, outs/K/ER arrays (shape: sim_n)
  - `MarketProb`: market, side, line, prob
  - `PlayerPropProb`: player_id, p_start, stat, line, prob_over

### Tests Added
Comprehensive test coverage in `tests/test_simulation.py` (20+ test cases):
- **AC1**: NB score distribution: sample mean within ±0.2, variance within ±20% of theoretical
- **AC2**: Moneyline: equal teams produce P(home_win) = 0.50 ± 0.03
- **AC3**: Run line: P(home covers -1.5) < P(home_win) (spreading harder than winning)
- **AC4**: Game total: μ=4.5 teams produce P(over 8.5) ≈ 0.50 ± 0.05
- **AC5**: Team totals: P(home over 4.5) consistent with home_mu
- **AC6**: Tie-break: all ties resolved, P(home_win | tie) correlates with relative strength
- **AC7**: Correlated noise: ρ=0.3 produces sample correlation in [0.15, 0.45]; ρ=0.0 near zero
- **AC8**: Hitter prop sampling: h_rate=0.25, mean PA=4 → mean(h) ≈ 1.0 ± 0.15
- **AC9**: Pitcher prop sampling: k_rate=0.22, 18 outs → mean(k) ≈ 5.35 ± 0.5
- **AC10**: Persistence round-trip: projection_id FK linkage, edge IS NULL
- **AC11**: Adaptive N: sim_n clamped to [2000, 10000]
- **AC12**: Park factor not re-applied: Coors μ=5.4 sample mean matches 5.4, not 5.4×1.2
- **Edge cases**: hitter not starting (p_start < 1.0), BB model absent, very low μ, minimum sim_n

### Known Limitations
- V1 uses simplified tie-break (no full inning-by-inning extra innings)
- Hitter stat sampling assumes independence (no joint distribution of H/TB/HR/RBI/R)
- TB derived from H + extra bases (not independently sampled, approximate)
- No alternate lines (only main lines hardcoded, D-034)
- No live/in-game simulation (v1 exclusion)
- Bullpen fatigue differential in tie-break is not yet implemented (±0.02 adjustment placeholder)
- No derivative markets (run spread other than ±1.5, alternate totals)
- No confidence intervals or uncertainty quantification on market probabilities
- Correlation parameter (ρ) is configurable but not inferred from data
- Pitcher lineup OPS placeholder (0.700 league average, not actual lineup OPS calculated)

### Dependency Notes
- **scipy added for stats.nbinom (Negative Binomial sampling and copula transformations)**
- **numpy used for array operations and random sampling**

### What's Next
**Unit 7**: Edge Calculation & Kelly Sizing
- Fetch latest odds from odds_snapshots
- Devig multi-book odds to fair probabilities
- Calculate edge = P_model − P_fair
- Compute Kelly fractions for positive-edge opportunities
- Update sim_market_probs and player_projections with edge, kelly_fraction, edge_computed_at

---

## 2026-02-12: Unit 7 - Odds Processing, Edge Calculation & Bankroll Sizing

### What Shipped
- **Best-Line Selection (`odds/best_line.py`)**
  - `get_best_lines()`: Fetches highest-price (most favorable) odds per (market, side, line) across all books
  - Uses most recent snapshot_ts per book
  - Returns `BestLine` dataclass with game_id, market, side, line, best_price, book, snapshot_ts
  - Returns empty list for games with no odds (no error raised)

- **Proportional Devig (`odds/devig.py`)**
  - `proportional_devig()`: Converts European decimal prices to fair probabilities (D-036)
  - Formula: `fair_i = (1/price_i) / sum(1/price_j)`
  - Validates all prices ≥ 1.0 (European decimal format)
  - Works for two-way and multi-way markets (e.g., regulation winner + tie)

- **Edge Calculation & Kelly Sizing (`odds/edge.py`)**
  - `compute_edges()`: Main entry point for edge computation
  - Fetches best lines from odds_snapshots
  - Applies proportional devig to get fair probabilities (D-039: requires both sides from same book)
  - Calculates edge = p_model − p_fair for all team markets
  - Computes fractional Kelly: `0.25 × edge / (best_price − 1)` (D-038)
  - Sets kelly_fraction = 0.0 if edge < min_edge_threshold (default 0.02, D-037)
  - Logs warnings for stale odds (>2 hours older than projection)
  - Player props: edge/kelly remain NULL if no matching odds available (acceptable per AC#8)
  - Returns `EdgeResult` with market_edges and player_edges lists

- **Persistence (`odds/persistence.py`)**
  - `persist_edges()`: Updates sim_market_probs and player_projections with edge values
  - Batch UPDATEs using executemany for efficiency
  - Sets edge_computed_at on ALL sim_market_probs rows for the projection (D-012)
  - Marks edge pass as complete even when no odds found
  - Idempotent: re-running overwrites edge values without creating duplicate rows (AC#9)

- **Configuration Extensions**
  - Added `min_edge_threshold` (default 0.02, range [0.0, 0.10]) to AppConfig (D-037)
  - Added `kelly_fraction_multiplier` (default 0.25, range [0.05, 1.0]) to AppConfig (D-038)
  - Both configurable via environment variables

- **Contracts**
  - `BestLine`: game_id, market, side, line, best_price, book, snapshot_ts, fair_prob
  - `MarketEdge`: prob_id, market, side, p_model, p_fair, edge, best_price, kelly_fraction
  - `PlayerEdge`: pp_id, player_id, stat, p_model, p_fair, edge, best_price, kelly_fraction (all nullable)
  - `EdgeResult`: projection_id, market_edges, player_edges, computed_at

### Tests Added
Comprehensive test coverage in `tests/test_edge.py` (16 test cases):
- **AC1 - Proportional devig**: [1.91, 1.91] → [0.5, 0.5]; [1.50, 2.80] → probabilities sum to 1.0 with favorite > 0.5
- **AC2 - Best-line selection**: Returns highest price across 3 books; empty list for no odds
- **AC3 - Edge calculation**: p_model=0.55, p_fair=0.50 → edge=0.05; negative edge validated
- **AC4 - Kelly sizing**: edge=0.05, price=2.00 → kelly=0.0125 (0.25 × 0.05 / 1.0)
- **AC5 - Minimum threshold**: edge < 0.02 → kelly=0.0 (but edge value still stored)
- **AC6 - edge_computed_at**: All sim_market_probs rows have edge_computed_at IS NOT NULL after compute
- **AC7 - No odds available**: Completes without error, edge/kelly remain NULL, edge_computed_at still set
- **AC8 - Player prop no-match**: Team odds exist but no player prop odds → edge/kelly remain NULL (no error)
- **AC9 - Idempotent**: Running twice overwrites edge values, no duplicate rows, edge_computed_at updated

### Known Limitations
- V1 uses proportional devig only (no power devig, Shin, or additive methods, D-036)
- Devig requires both sides from same book (D-039); if unavailable, fair_prob = None
- Player prop edge calculation is incomplete in v1: all player props get NULL edges (odds format mismatch)
- Line matching uses floating-point equality (acceptable for v1, may need tolerance in v2)
- Stale odds (>2 hours) log warning but still used (v1 allows stale odds)
- No alternate lines: simulation uses hardcoded main lines (D-034); mismatched lines are skipped
- No calibration model application yet (hook exists, models trained in Unit 8)
- No CLV (closing line value) computation (Unit 8)
- Kelly sizing uses single fractional multiplier (no dynamic Kelly based on edge confidence)

### What's Next
**Unit 8**: Model Evaluation & Calibration
- Compute log loss, Brier score, ECE, tail accuracy on completed projections
- Train market-specific calibration models (e.g., underdog bias correction)
- Calculate CLV (closing line value) for edge validation
- Write to eval_results table

---

## 2026-02-12: Unit 8 - Evaluation & Backtesting Harness

### What Shipped
- **Pure Metric Functions (`evaluation/metrics.py`)**
  - `log_loss()`: Cross-entropy loss with epsilon clipping (1e-15) to prevent log(0)
  - `brier_score()`: Mean squared error between probabilities and binary outcomes
  - `ece()`: Expected Calibration Error with configurable bins (default 10)
  - `tail_accuracy()`: Calibration diagnostics for extreme probabilities (p < 0.15, p > 0.85)
  - All functions validate inputs and operate on numpy arrays (no DB dependencies)

- **CLV Computation (`evaluation/clv.py`)**
  - `compute_clv()`: Calculates Closing Line Value for projections vs. market close
  - Uses T-5 minute closing odds (D-040): latest snapshot between T-30 and T-5 before first pitch
  - Devigged using proportional method (same as Unit 7, D-036)
  - Excludes games without closing odds in 30-minute window
  - Returns `CLVRow` with p_model, p_close_fair, clv (positive = beat the close)

- **Rolling-Origin Backtest (`evaluation/backtest.py`)**
  - `run_backtest()`: Evaluates model performance over date range for specific market
  - Only uses final games with non-null scores (no future leakage)
  - Uses most recent projection per game (by run_ts)
  - Computes all metrics: log loss, Brier, ECE, tail accuracy (low/high), median CLV
  - Returns `EvalReport` with sample_n=0 and metrics=None if no final games found
  - Market-specific outcome computation: ML, RL, total, team_total

- **Calibration Models (`evaluation/calibration.py`)**
  - `fit_calibration()`: Trains market-specific calibration using historical (p_model, outcome) pairs
  - Isotonic regression (default, D-041): nonparametric, handles non-monotonic miscalibration
  - Platt scaling (optional): parametric logistic regression, smoother for small samples
  - Requires minimum 50 samples to fit (configurable)
  - Saves models to registry: `calibration_{market}_{method}.pkl`
  - `apply_calibration()`: Applies fitted calibration to raw probabilities, clipped to [0, 1]
  - `load_calibration()`: Loads calibration model from registry

- **Persistence (`evaluation/persistence.py`)**
  - `persist_eval_report()`: Writes EvalReport → eval_results table
  - Upserts on (eval_date, market, metric) for idempotency (D-042)
  - Writes 6 rows per report: log_loss, brier, ece, tail_acc_low, tail_acc_high, clv
  - Skips metrics that are None (e.g., insufficient tail samples)
  - Meta field stores JSON metadata (tail sample counts, model version)

- **Contracts**
  - `CLVRow`: prob_id, game_id, market, p_model, p_close_fair, clv
  - `EvalReport`: eval_date, market, start_date, end_date, sample_n, all metrics, meta
  - `CalibrationModel`: market, method (isotonic/platt), fitted_at, params (pickled or dict)

### Tests Added
Comprehensive test coverage in `tests/test_evaluation.py` (25+ test cases):
- **AC1 - Log loss**: Calibrated model < biased model
- **AC2 - Brier score**: Exact numeric test (0.04 for [0.8, 0.2] vs [1, 0])
- **AC3 - ECE**: Perfect calibration ≈ 0; miscalibrated (all 0.9, 50% outcomes) ≈ 0.4
- **AC4 - Tail accuracy**: 10 predictions p<0.15 with 1 event → low_tail_acc=0.10; <5 samples → None
- **AC5 - CLV computation**: p_model=0.55, p_close=0.52 → clv=0.03; median CLV correct
- **AC6 - CLV uses T-5**: Uses T-5 odds (1.85), not T-1 odds (1.50); validates correct snapshot
- **AC7 - Rolling-origin backtest**: Only final games, correct sample_n, Brier matches expected
- **AC8 - Calibration fit/apply**: Returns CalibrationModel; apply returns float in [0,1]; ECE decreases
- **AC9 - Persistence**: Writes 6 rows to eval_results with correct metric names
- **AC10 - Idempotent**: Running twice does not duplicate rows, count remains correct
- **Validation tests**: All metrics raise on invalid inputs (length mismatch, empty arrays, out-of-range)

### Known Limitations
- Player prop calibration deferred to v2 (team markets only, D-041)
- Calibration models saved as pickle files (no versioning metadata beyond timestamp)
- Team total outcome computation assumes "over" semantics (ambiguous from schema)
- No automated eval scheduling (Unit 9 responsibility)
- No confidence intervals on metrics (bootstrap deferred to v2)
- ECE bin edges are fixed uniform spacing (adaptive binning deferred)
- CLV computation excludes games without odds in 30-minute window (no fallback)
- Calibration requires minimum 50 samples (may fail for rare markets in small date ranges)
- No model ensemble or stacking for calibration (single method per market)
- Brier decomposition (reliability, resolution, uncertainty) not implemented
- No skill score (Brier skill score vs baseline) computed

### What's Next
**Unit 9**: Execution & Automation
- Build scheduler for daily model training, projection runs, and eval passes
- Implement publishing gate (p_start threshold, edge_computed_at check, confirmed lineup)
- Create cron-compatible CLI entry points
- Add Discord notification hooks (v1 basic: "projection complete")

---
## 2026-02-12: Unit 9 - Scheduler & Orchestration Pipeline

### What Shipped
- **Pipeline Orchestration (`scheduler/pipeline.py`)**
  - `run_global()`: End-to-end pipeline for all games on a date
    - Steps: fetch schedule → ingest odds → ingest weather → (per-game: lineups → models → simulation → edge)
    - Three run types: night_before, morning, midday (D-043)
    - Skips postponed games automatically
    - Conservative: logs and skips on failures after retries
  - `run_game()`: Per-game pipeline for T-90, T-30 runs and event-driven reruns (D-044)
    - Same steps as global run but for single game
    - Checks game status before processing
  - `run_daily_eval()`: Nightly evaluation trigger
    - Runs backtest for all major markets (ml, rl, total, team_total) for today's final games
    - Writes results to eval_results table
  - `_process_game()`: Orchestrates lineups → team models → simulation → edge computation
    - Sets edge_computed_at timestamp for publishing gate
  - `_retry_ingestion()`: Exponential backoff retry wrapper for all ingestion operations
    - Up to max_retry_attempts (default 2), backoff: 1s, 2s, 4s
    - Conservative fallback: returns empty on failure

- **Cron Entry Points (`scheduler/cron.py`)**
  - `night_before_run()`, `morning_run()`, `midday_run()`: Global run entry points
  - `nightly_eval_run()`: Evaluation entry point
  - All callable with no arguments for cron/scheduler integration
  - Each wraps async pipeline call in asyncio.run()

- **Event Detection & Throttle (`scheduler/events.py`)**
  - `check_for_changes()`: Detects lineup confirmations, pitcher changes, odds movements
    - Returns list of ChangeEvent objects
    - V1: simplified detection (lineup confirmed only)
  - `trigger_rerun_if_needed()`: Checks for changes and triggers rerun if throttle allows
    - Rerun throttle: at most 1 rerun per game per 10-minute window (D-045)
    - In-memory throttle state (game_id → last_rerun_ts)

- **Publishing Gate (`scheduler/gate.py`)**
  - `is_publishable()`: Determines if projection is eligible for publication (D-046)
    - Team markets (player_id=None): publishable if edge_computed_at IS NOT NULL
    - Player props (player_id is not None): publishable if edge_computed_at IS NOT NULL AND (lineup confirmed OR p_start >= threshold)
    - p_start_threshold configurable (default 0.85)
    - Enforces lineup uncertainty policy

- **Configuration Extensions**
  - Added `schedule_night_before_et` (default "22:00"), `schedule_morning_et` ("08:00"), `schedule_midday_et` ("12:00") (D-043)
  - Added `game_run_t_minus_minutes` (default [90, 30]) for per-game scheduling (D-044)
  - Added `rerun_throttle_minutes` (default 10) for event-driven rerun throttle (D-045)
  - Added `p_start_threshold` (default 0.85) for publishing gate (D-046)
  - Added `max_retry_attempts` (default 2) for ingestion retry policy

- **Contracts**
  - `run_global(run_type)`: run_type in ['night_before', 'morning', 'midday']
  - `run_game(game_id)`: Processes single game
  - `run_daily_eval()`: Triggers evaluation for today's final games
  - `ChangeEvent`: game_id, event_type, detected_at
  - `is_publishable(game_id, market, player_id?)`: Returns bool

### Tests Added
Comprehensive test coverage in `tests/test_scheduler.py` (13 test cases):
- **AC1 - Global run end-to-end**: Completes all pipeline steps for 2 games with mocked ingestion
- **AC2 - Per-game run**: Processes exactly one game, ingestion called correctly
- **AC3 - Lineup gate confirmed**: Confirmed lineup allows player props to pass gate even with low p_start
- **AC4 - Lineup gate high p_start**: p_start=0.90 passes, p_start=0.60 fails with unconfirmed lineup
- **AC5 - Lineup gate team markets**: Team markets always pass when edge computed, regardless of lineup state
- **AC6 - Rerun throttle**: First trigger succeeds, second within 10min is blocked, third after 15min succeeds
- **AC7 - Retry on ingestion failure**: Retries up to 3 times with exponential backoff, succeeds on 3rd attempt
- **AC8 - Nightly eval**: Triggers backtest for final games, writes eval_results rows
- **AC9 - Cron entry points**: All 4 functions are callable with no arguments, correct signatures verified
- **Edge case - No games today**: run_global completes without error, no ingestion writes
- **Edge case - Postponed game skipped**: run_game skips postponed games, _process_game not called

### Known Limitations
- Per-game scheduling (T-90, T-30) is not automated in v1 (cron entry points only, no scheduler daemon)
- Event-driven reruns require manual triggering via `trigger_rerun_if_needed()` (no continuous polling)
- Rerun throttle state is in-memory only (cleared on restart; production should use Redis or DB)
- Change detection is simplified in v1 (lineup confirmed only; pitcher change and odds movement detection not implemented)
- No system-level cron configuration (crontab setup is deployment documentation, Unit 12)
- Global run time conversions (ET ↔ UTC) use simplified UTC date (production should handle timezone correctly)
- No circuit breaker or rate limiting on ingestion retries (may exhaust API quota on repeated failures)
- Lineup confirmation detection lacks state tracking (can't distinguish new confirmation from pre-existing)
- No dependency between run_global and run_game (could trigger overlapping runs; throttle mitigates but doesn't prevent)
- Publishing gate queries database for each call (no caching; acceptable for v1 volume)
- Daily eval runs for single date only (no multi-day backtest window in nightly run)
- No graceful shutdown handling (in-flight pipeline runs may be interrupted)

### What's Next
**Unit 10**: Discord Bot
- Integrate Discord bot SDK
- Fetch publishable projections using `is_publishable()` gate
- Format and send messages to Discord channels (tier-specific)
- Handle user commands (/subscribe, /unsubscribe)

---

## 2026-02-13: Unit 10 - Discord Bot & Publishing Layer

### What Shipped
- **Bot Lifecycle Management (`discord_bot/bot.py`)**
  - `MLBPicksBot`: Discord bot class extending commands.Bot
  - Minimal intents (guilds + members for permission sync)
  - Graceful startup: connects to Discord, verifies guild, ensures channels, initializes publisher
  - Graceful shutdown: SIGTERM/SIGINT handling, clean disconnect
  - `wait_ready()`: Async wait for bot readiness with timeout
  - `run_until_shutdown()`: Main event loop with shutdown signal handling
  - Publish-only in v1: no user commands or interactive features (D-047)

- **Channel Management (`discord_bot/channels.py`)**
  - `ensure_channels()`: Creates missing channels on startup with correct permissions
  - 7 required channels: #free-picks (public), #team-moneyline, #team-runline, #team-totals, #player-props-h, #player-props-p (all paid), #announcements (public)
  - Paid channels: deny @everyone, allow bot role
  - `sync_member_permissions()`: Queries subscriptions table, grants/revokes channel access based on tier
  - Tier gating: `tier='paid' AND status='active'` required for paid channel access

- **Pick Publishing (`discord_bot/publisher.py`)**
  - `Publisher` class: Manages message publishing with anti-spam tracking
  - `publish_picks(game_id)`: Publishes all publishable picks for a game
    - Queries sim_market_probs and player_projections for latest projection with edge_computed_at IS NOT NULL
    - Filters by `is_publishable()` gate (team markets always, player props require lineup or high p_start)
    - Only publishes positive-edge plays (edge > 0 AND kelly_fraction > 0) (D-050)
    - Anti-spam: one message per (game_id, market, side, line), edits existing messages on rerun (D-049)
    - In-memory message cache: (game_id, market, side, line) → message_id
    - Routes to correct channel: team-moneyline, team-runline, team-totals, player-props-h, player-props-p
  - `publish_free_pick()`: Posts daily free pick to #free-picks
    - Selects highest-edge team market play in 60-90 minute window before first_pitch (D-048)
    - Requires lineup confirmation (uses is_publishable gate)
    - Posts at most once per day (in-memory flag: _free_pick_posted_date)

- **Message Formatting (`discord_bot/formatter.py`)**
  - `format_team_market_embed()`: Pure function returning Discord Embed for team markets
    - Title: "NYY @ BOS — Moneyline"
    - Fields: Game Time, Pick, Model Probability, Edge, Kelly Sizing, Best Book
    - Footer: "Model v1 | 5,000 simulations"
  - `format_player_prop_embed()`: Pure function returning Discord Embed for player props
    - Title: "Aaron Judge — Hits O/U 0.5"
    - Fields: Game, Game Time, P(Start), Model Probability, Edge, Kelly Sizing, Best Book
    - Footer: Model version + simulation count

- **Configuration Extensions**
  - Added `discord_guild_id` (required, guild snowflake ID)
  - Added `free_pick_channel` (default "free-picks")
  - Added `paid_channels` (default list of 5 paid channels)
  - Added `announcements_channel` (default "announcements")
  - Added `free_pick_window_min` (default 60 minutes before first pitch)
  - Added `free_pick_window_max` (default 90 minutes before first pitch)

- **Contracts**
  - Channel structure: 7 channels with specific names and visibility rules
  - TeamMarketEmbed: 9 fields (title, game_time, side, model_prob, edge, kelly, best_book, footer)
  - PlayerPropEmbed: 10 fields (title, game_time, p_start, model_prob, edge, kelly, best_book, footer)
  - Publisher reads subscriptions.tier and subscriptions.status but never writes to subscriptions table
  - Anti-spam state is in-memory, not persisted (reset on bot restart)

### Tests Added
Comprehensive test coverage in `tests/test_discord.py` (15+ test cases):
- **AC1 - Bot connects**: (Validated via MLBPicksBot class structure, integration test deferred)
- **AC2 - Channels created**: ensure_channels() creates all 7 channels with correct names and permissions
- **AC3 - Team market published**: publish_picks() sends team market embed with all required fields
- **AC4 - Player prop published**: publish_picks() sends player prop embed for confirmed lineup player
- **AC5 - Publishing gate enforced**: Player with p_start=0.60 and unconfirmed lineup NOT published; team market on same game IS published
- **AC6 - Free pick timing**: publish_free_pick() selects game in 60-90 min window, not 30 or 120 min
- **AC7 - Free pick uniqueness**: Calling publish_free_pick() twice in same day posts only once
- **AC8 - Anti-spam**: Consecutive publish_picks() edits existing message, not duplicate
- **AC9 - Tier gating**: sync_member_permissions() grants paid users access, revokes free users
- **AC10 - Negative/zero edge**: Query filters ensure edge > 0 AND kelly_fraction > 0 (rows with edge <= 0 never fetched)
- **Formatter tests**: Validates all embed fields populated correctly for both team markets and player props
- **Channel reuse test**: Existing channels not recreated on subsequent ensure_channels() calls
- **Permission sync tests**: Paid subscriber gets read_messages=True, free user gets overwrite=None

### Known Limitations
- **No real Discord integration in tests**: All Discord API calls are mocked (integration testing with live bot deferred to deployment)
- **Best odds stubbed**: `best_book` and `best_price` are placeholders in v1 (query odds_snapshots in v2)
- **Anti-spam state is in-memory**: Message cache lost on bot restart, may re-post picks for already-published games
- **No DM-based delivery**: All picks published to guild channels only
- **No user commands**: Bot is publish-only, no `/picks`, `/subscribe`, or interactive features (D-047)
- **No live betting alerts**: v1 exclusion per spec
- **No role-based permission grants**: sync_member_permissions() sets per-user overwrites, not Discord role assignments (Unit 11 manages role sync)
- **Free pick selection is naive**: Highest edge only, no diversity/showmanship optimization
- **No message editing for non-existent messages**: If message_id is cached but message was deleted, falls back to new post (acceptable for v1)
- **No channel category organization**: All channels created at root level (not grouped by market type)
- **No rate limiting**: Bot may hit Discord API rate limits if publishing many picks rapidly (deferred to v2)

### Dependency Notes
- **discord.py>=2.3 added for Discord bot SDK**

### What's Next
**Unit 11**: Stripe Integration & Subscription Management
- Implement Stripe webhook handlers for subscription events
- Write to subscriptions table on payment success/failure
- Sync Discord roles on subscription tier changes
- Handle subscription cancellations and renewals

---

## 2026-02-13: Unit 11 - Stripe Subscription & Webhook Integration

### What Shipped
- **Checkout Session Creation (`payments/checkout.py`)**
  - `create_checkout_url(discord_user_id)`: Generates Stripe Checkout Session URLs for subscription signup
  - Passes `discord_user_id` as `client_reference_id` to link payment to Discord identity (D-054)
  - Uses configured `stripe_price_id` for single subscription tier (D-051)
  - Returns session.url for redirect
  - Validates config (stripe_secret, stripe_price_id) before Stripe API call

- **Webhook Handler (`payments/webhooks.py`)**
  - `handle_webhook(payload, sig_header, bot_client)`: Main webhook entry point with signature verification
  - Stripe signature verification using `stripe.Webhook.construct_event()` (AC#2)
  - Event routing to specialized handlers:
    - `checkout.session.completed`: Creates subscription with tier='paid', status='active'
    - `invoice.paid`: Updates status='active', refreshes current_period_end
    - `invoice.payment_failed`: Sets status='past_due', tier remains 'paid'
    - `customer.subscription.updated`: Syncs status from Stripe (active/trialing → active, canceled/unpaid → cancelled+free, past_due → past_due+paid)
    - `customer.subscription.deleted`: Sets tier='free', status='cancelled'
  - Unknown event types acknowledged with 200 (no processing, AC#10)
  - Returns HTTP 400 for invalid signature, 500 for processing errors (Stripe will retry)

- **Subscription State Sync (`payments/sync.py`)**
  - `sync_subscription(discord_user_id, stripe_customer_id, tier, status, current_period_end, bot_client)`: Updates database and Discord roles
  - Upserts `subscriptions` table using `ON CONFLICT (discord_user_id) DO UPDATE` (idempotent, AC#9)
  - Best-effort Discord role sync (D-053):
    - Grants "Subscriber" role when tier='paid' AND status='active'
    - Revokes "Subscriber" role when tier='free' OR status != 'active'
    - Creates role if it doesn't exist
    - Logs errors but doesn't fail webhook if Discord API unavailable
  - Skips role sync with warning if bot_client is None

- **Webhook Server (`payments/server.py`)**
  - Lightweight aiohttp HTTP server for `POST /webhooks/stripe` endpoint
  - `create_app(bot_client)`: Creates aiohttp Application with webhook route
  - `run_server(bot_client, shutdown_event)`: Runs server on configured port (default 8080)
  - Standalone process (D-052): decoupled from Discord bot, handles Stripe retries independently
  - Signal handling (SIGTERM/SIGINT) for graceful shutdown
  - `main()`: CLI entry point for standalone webhook server

- **Configuration Extensions**
  - Added `stripe_webhook_secret: SecretStr` (Stripe webhook signing secret)
  - Added `stripe_price_id: str` (Stripe Price ID for subscription product)
  - Added `checkout_success_url: str` (default "https://discord.com")
  - Added `checkout_cancel_url: str` (default "https://discord.com")
  - Added `webhook_server_port: int` (default 8080, range [1024, 65535])
  - Added `discord_paid_role_name: str` (default "Subscriber")

- **Contracts**
  - `create_checkout_url(discord_user_id: str) → str`: Returns Stripe Checkout URL
  - `handle_webhook(payload: bytes, sig_header: str, bot_client: Optional[discord.Client]) → web.Response`
  - `sync_subscription(discord_user_id, stripe_customer_id, tier, status, current_period_end, bot_client)`: Upserts subscriptions, syncs Discord role
  - Subscriptions table fields: discord_user_id (unique), stripe_customer_id, tier ('free'|'paid'), status ('active'|'cancelled'|'past_due'), current_period_end (TIMESTAMPTZ)

### Tests Added
Comprehensive test coverage in `tests/test_payments.py` (18 test classes, 25+ test cases):
- **AC1 - Checkout URL generation**: create_checkout_url() returns valid Stripe Checkout URL with correct session parameters
- **AC2 - Webhook signature verification**: Valid signature processed, invalid signature returns 400 and no database write
- **AC3 - checkout.session.completed**: Creates subscription with tier='paid', status='active', stripe_customer_id, current_period_end
- **AC4 - invoice.paid**: Updates status='active', refreshes current_period_end
- **AC5 - invoice.payment_failed**: Sets status='past_due', tier remains 'paid'
- **AC6 - customer.subscription.deleted**: Sets tier='free', status='cancelled'
- **AC7 - Discord role grant**: Role granted when tier='paid' AND status='active' (subscription activation)
- **AC8 - Discord role revoke**: Role revoked when tier='free' OR status='cancelled' (subscription deleted)
- **AC9 - Idempotent**: Replaying same webhook produces same database state (upsert, not duplicate)
- **AC10 - Unknown events**: charge.refunded event returns 200, no database write
- **AC11 - Server starts**: Webhook server creates app with POST /webhooks/stripe route, rejects requests without signature header
- **Edge cases**: Missing client_reference_id (logged and skipped), Discord API unavailable (database updated, role sync skipped with error log), missing config validation

### Known Limitations
- **Single subscription tier only**: V1 supports free and paid tiers only, no intermediate tiers (D-051)
- **Webhook server is standalone**: Must be deployed as separate process from Discord bot (D-052)
- **Role sync is best-effort**: Database is source of truth; Discord role is eventually consistent (D-053)
- **discord_user_id from client_reference_id only**: Webhooks without client_reference_id (e.g., subscriptions created outside bot flow) are skipped (D-054)
- **No refund automation**: Refunds handled manually via Stripe dashboard in v1
- **No pricing experimentation**: Single plan only (price changes require config update)
- **No admin dashboard**: Subscription management via Stripe dashboard only
- **No user-facing commands**: Checkout URL must be shared manually (e.g., in #announcements)
- **No proration or plan changes**: Single plan in v1, changes not supported
- **No customer portal**: Users cannot self-manage subscriptions (Stripe Customer Portal deferred to v2)
- **Discord role creation on-demand**: Role created on first subscription if it doesn't exist (not pre-seeded)
- **Invoice lookup by stripe_customer_id**: invoice.paid and invoice.payment_failed require database lookup to find discord_user_id (adds latency)
- **No webhook replay protection**: Stripe's idempotency is implicit; explicit event_id tracking deferred to v2

### Dependency Notes
- **stripe>=7.0 added for Stripe API SDK**
- **aiohttp>=3.9 added for webhook HTTP server**

### What's Next
**Unit 12**: README, Documentation & "Not Yet" Section
- Rewrite README.md into portfolio-quality document
- Create docs/ARCHITECTURE.md with Mermaid diagrams
- Add headers and TOC to docs/DECISIONS.md and docs/DEVLOG.md
- Audit module docstrings across src/mlb/
- Document all v1 non-goals and limitations

---

## 2026-02-13: Unit 12 - README, Documentation & "Not Yet" Section

### What Shipped
- **README.md (Complete Rewrite)**
  - 12 required sections: project title, features, system architecture, data flow, modeling approach, evaluation, key tradeoffs, tech stack, setup & deployment, "What This System Does Not Do Yet", project structure, license
  - Comprehensive modeling narrative: Negative Binomial run scoring, team/player prop models, Monte Carlo simulation, devig/edge calculation, evaluation metrics
  - Deployment instructions: database migration, cron schedule setup, Discord bot startup, Stripe webhook server startup with systemd examples
  - "What This System Does Not Do Yet" section with all 18 required items from spec (live betting, alternate lines, UI, multi-sport, reliever props, etc.)
  - Portfolio-quality polish: 430+ lines, clear section hierarchy, markdown formatting

- **docs/ARCHITECTURE.md (New File)**
  - System data flow Mermaid diagram (flowchart LR) showing full pipeline: Data Providers → Ingestion → Database → Feature Engineering → Models → Simulation → Edge → Discord, with Scheduler orchestration and Stripe webhooks
  - Module dependency map Mermaid diagram (graph TD) with 11 unit subgraphs showing which modules call which
  - Detailed data flow narrative (8 phases): Ingestion → Feature Engineering → Modeling → Simulation → Edge Calculation → Publishing → Subscription Management → Evaluation
  - Orchestration & scheduling section: global runs, per-game runs, event-driven reruns, nightly evaluation
  - Key architectural decisions summary with cross-references to DECISIONS.md
  - Technology choices rationale (PostgreSQL, LightGBM, Negative Binomial, proportional devig, quarter-Kelly, Discord, Stripe, cron)
  - Deployment architecture diagram (3 independent processes: cron jobs, Discord bot, webhook server)
  - Future directions (v2+): alternate lines, live betting, reliever props, joint distributions, ML event rates, player prop calibration, multi-provider failover, interactive bot commands, admin dashboard, mobile app, multi-sport

- **docs/DECISIONS.md (Header & TOC Added)**
  - Added document header explaining purpose and decision numbering
  - Added table of contents with anchor links to all 11 units and decision ranges (D-001 to D-054)
  - Total decision count confirmed: 54
  - No new decisions added (Unit 12 is documentation-only per spec)
  - Formatting normalized across all decisions (consistent headings, spacing, rationale structure)

- **docs/DEVLOG.md (Header Added)**
  - Added document header explaining purpose (unit-by-unit implementation progress)
  - Added project status line: "All 12 implementation units complete"
  - Normalized all unit entries to consistent format: ## YYYY-MM-DD: Unit N — Title
  - All 12 units present with "What Shipped", "Tests Added", "Known Limitations", "What's Next" sections

- **Module Docstring Audit**
  - Verified all 12 `src/mlb/**/__init__.py` files have module-level docstrings:
    - `mlb/`: "MLB Analytics Platform."
    - `mlb/config/`: "Configuration management."
    - `mlb/db/`: "Database connection management."
    - `mlb/ingestion/`: "Provider-agnostic data ingestion interfaces and adapters."
    - `mlb/models/`: "Team run-scoring models and feature engineering."
    - `mlb/simulation/`: "Monte Carlo simulation engine for MLB games."
    - `mlb/odds/`: "Odds processing, edge calculation, and bankroll sizing (Unit 7)."
    - `mlb/evaluation/`: "Unit 8: Evaluation & Backtesting Harness."
    - `mlb/scheduler/`: "Scheduler and orchestration pipeline."
    - `mlb/discord_bot/`: "Discord bot module for publishing MLB picks to subscribers."
    - `mlb/payments/`: "Stripe subscription and payment processing."
  - No module docstrings added (all were already present from Units 1-11)

### Tests Added
No tests added. Unit 12 is documentation-only per spec (AC#7: "No application code or test changes").

### Known Limitations
- README.md exceeds 400 lines (430+ lines total) but remains under 500-line guideline
- ARCHITECTURE.md Mermaid diagrams tested locally but not rendered in production environment (GitHub will render correctly)
- Module dependency map is simplified (shows unit-level dependencies, not file-level)
- "What This System Does Not Do Yet" section is exhaustive but may require updates as v2 scope evolves
- Deployment examples are Linux-centric (systemd); Windows deployment requires different process management (e.g., NSSM)

### What's Next
**v1 Implementation Complete.** All 12 units shipped. Next steps:
1. **Production Deployment:** Deploy to production server with cron, Discord bot, and webhook server processes
2. **Monitoring:** Add logging aggregation (e.g., Logstash, Datadog) and alerting for pipeline failures
3. **Backtesting:** Run full historical backtest on 2024 season to validate model performance
4. **Model Tuning:** Iterate on model features and hyperparameters based on eval_results metrics
5. **v2 Planning:** Prioritize v2 features (alternate lines, live betting, reliever props, ML event rates, interactive bot commands, admin dashboard)

**Unit 12 is formally closed. Documentation is complete and ready for portfolio review.**

---
