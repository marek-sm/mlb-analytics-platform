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
