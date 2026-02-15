# Architecture Decision Records

This document tracks all architectural and implementation decisions made throughout the MLB Analytics Platform project. Decisions are numbered D-001 through D-054 and organized by implementation unit.

---

## Table of Contents

- [Unit 1: Repository Skeleton & Configuration](#unit-1-repository-skeleton--configuration) (D-001 to D-004)
- [Unit 2: Database Schema & Migrations](#unit-2-database-schema--migrations) (D-005 to D-016)
- [Unit 3: Data Ingestion](#unit-3-data-ingestion--provider-agnostic-interfaces) (D-017 to D-020, D-055, D-056)
- [Unit 4: Team Run-Scoring Models](#unit-4-team-run-scoring-models) (D-021 to D-026)
- [Unit 5: Player Prop Models](#unit-5-player-prop-models) (D-027 to D-031)
- [Unit 6: Monte Carlo Simulation Engine](#unit-6-monte-carlo-simulation-engine) (D-032 to D-035)
- [Unit 7: Odds Processing, Edge Calculation & Bankroll Sizing](#unit-7-odds-processing-edge-calculation--bankroll-sizing) (D-036 to D-039)
- [Unit 8: Evaluation & Backtesting Harness](#unit-8-evaluation--backtesting-harness) (D-040 to D-042)
- [Unit 9: Scheduler & Orchestration Pipeline](#unit-9-scheduler--orchestration-pipeline) (D-043 to D-046)
- [Unit 10: Discord Bot & Publishing Layer](#unit-10-discord-bot--publishing-layer) (D-047 to D-050)
- [Unit 11: Stripe Subscription & Webhook Integration](#unit-11-stripe-subscription--webhook-integration) (D-051 to D-054)

**Total Decisions:** 56

---

## Unit 1: Repository Skeleton & Configuration

### D-001: Configuration via Pydantic BaseSettings

**Decision**: Config via Pydantic BaseSettings, 12-factor env vars only. No YAML/TOML config files at runtime.

**Rationale**: Simplicity, secret hygiene, compatibility with cron and container deploys per §Monetization & Engineering.

---

### D-002: asyncpg as the Postgres driver

**Decision**: asyncpg as the Postgres driver. All database access is async.

**Rationale**: Aligns with event-driven rerun requirements (§Execution & Automation) and keeps a single concurrency model.

---

### D-003: Source layout

**Decision**: Source layout: src/mlb/ namespace package.

**Rationale**: Prevents import ambiguity; standard Python packaging practice per §Repo Requirements.

---

### D-004: Default simulation count

**Decision**: default_sim_n = 5000, hard bounds [2000, 10000].

**Rationale**: Direct from §Modeling Strategy adaptive simulation counts. Enforced at config validation.

---

## Unit 2: Database Schema & Migrations

### D-005: No ORM. Raw SQL migrations and queries.

**Decision**: No ORM. Raw SQL migrations and queries.

**Rationale**: Keeps the stack minimal, avoids abstraction overhead, matches ≤$100/mo budget constraint. §Monetization & Engineering.

---

### D-006: Odds stored as European decimal prices (≥ 1.0)

**Decision**: Odds stored as European decimal prices (≥ 1.0, where 2.000 = even money). American odds are converted to European decimal on ingestion in Unit 3. CHECK constraint enforces price >= 1.0.

**Rationale**: Simplifies devig math in Unit 7. American odds converted on ingestion. CHECK constraint prevents invalid data and makes the format contract explicit.

---

### D-007: Single player_game_logs table for both hitters and pitchers.

**Decision**: Single player_game_logs table for both hitters and pitchers.

**Rationale**: Nullable columns (ip_outs, er for hitters; pa, ab for pitchers) avoid a join-heavy split-table design while remaining simple for v1 scope.

---

### D-008: park_factor is a single scalar per park (static seasonal).

**Decision**: park_factor is a single scalar per park (static seasonal).

**Rationale**: Per spec §Parks & Weather: "static seasonal park factors." Splits by handedness or stat type deferred to v2.

---

### D-009: Forward-only migrations; no down/rollback.

**Decision**: Forward-only migrations; no down/rollback.

**Rationale**: Sufficient for a v1 solo-developer project. Rollback = fix-forward with a new migration.

---

### D-010: park_factor semantic contract (apply exactly once)

**Decision**: park_factor is a multiplicative adjustment to expected runs (μ). Applied exactly once, in the feature engineering layer (Unit 4). Value 1.000 = neutral. The simulation engine (Unit 6) receives pre-adjusted μ and does NOT re-apply park_factor.

**Rationale**: Prevents double-application bias. If both Unit 4 and Unit 6 applied park_factor, a 1.200 park would become 1.44× (squared), systematically inflating all non-neutral projections. Single-point application in Unit 4 keeps the contract explicit and testable. Handedness splits, stat-type splits, and dynamic park factors deferred to v2.

---

### D-011: Current lineup query pattern and confirmed uniqueness

**Decision**: Current lineup is defined as the row set with MAX(source_ts) WHERE is_confirmed = TRUE, per (game_id, team_id). A partial unique index on (game_id, team_id, batting_order) WHERE is_confirmed = TRUE enforces at most one confirmed lineup per slot. Superseded confirmed lineups must have is_confirmed set to FALSE before inserting a new confirmed set.

**Rationale**: The lineups table stores multiple temporal snapshots via source_ts to track lineup changes over time. Without a constraint, downstream units (especially Unit 9's publishing gate) risk joining on stale or duplicate confirmed data. The partial unique index prevents accidental double-confirmation while still allowing multiple unconfirmed snapshots for the same slot. Query pattern is explicit and deterministic. An is_current boolean or materialized view is unnecessary complexity for v1.

---

### D-012: sim_market_probs.edge_computed_at consumer contract

**Decision**: sim_market_probs.edge_computed_at is set by Unit 7 when edge calculation completes. Downstream consumers (Unit 10) must filter on edge_computed_at IS NOT NULL. Rows with edge_computed_at IS NULL are incomplete and must not be published.

**Rationale**: The edge column is nullable, making it impossible to distinguish "not yet computed" (NULL because calculation pending) from "no edge found" (NULL because fair odds matched market odds). Without edge_computed_at, Unit 10 (Discord bot) could read incomplete projections and either crash on NULL edge values or silently skip valid plays that haven't finished processing. The timestamp provides an explicit completion signal and doubles as an audit trail for pipeline latency. Row-level status enums or complex state machines are unnecessary for v1.

---

### D-013: player_game_logs unique constraint limitation

**Decision**: player_game_logs enforces one row per (player_id, game_id). In the rare case a traded player appears for two teams on the same day, only the first entry is retained. Composite key (player_id, game_id, team_id) deferred to v2.

**Rationale**: The UNIQUE constraint on (player_id, game_id) prevents duplicate stat entries but rejects the edge case where a traded player appears in both games of a doubleheader for different teams. This is exceedingly rare in MLB (requires mid-season trade + same-day doubleheader + player used by both teams). For v1, the simplicity of the two-column key outweighs the complexity of handling this edge case. If it occurs, the second game's stats are rejected on insert, which is acceptable given the <1/season occurrence rate and minimal analytical impact. V2 can add team_id to the unique constraint if comprehensive stat tracking becomes a priority.

---

### D-014: game_id is provider-canonical, single-source assumption

**Decision**: game_id is the primary data provider's canonical game key (TEXT). v1 assumes a single provider. If a second provider is added, a mapping table must be introduced. Do not assume game_id is numeric or follows any format.

**Rationale**: Using the provider's native game_id as the primary key avoids the overhead of maintaining an internal ID namespace and eliminates join/mapping logic in the common case. The TEXT type accommodates any provider's scheme (numeric, UUID, composite strings). For v1, a single data source is sufficient for the MVP scope. If multiple providers are needed in v2 (e.g., adding a secondary odds source or backup stats API), a provider_game_mappings table can be introduced to resolve cross-provider game identities. Until then, the simplicity of provider_id = db_id keeps the schema lean and queries fast.

---

### D-015: Current weather query pattern

**Decision**: Current weather for a game is defined as the row with MAX(fetched_at) for that game_id. Unit 4 feature engineering uses this pattern. No partial index needed for v1 given low row volume per game.

**Rationale**: The weather table stores multiple temporal snapshots per game as conditions update (fetched hours before first pitch, at game time, etc.). Unit 4 feature engineering requires "most recent weather at projection time" to build accurate features. The MAX(fetched_at) pattern is explicit, deterministic, and sufficient for v1 volumes (typically 2-3 weather snapshots per game). A partial index on (game_id, fetched_at) or is_current boolean flag would optimize the query but adds schema complexity for minimal performance gain given v1 scale. The UNIQUE(game_id, fetched_at) constraint prevents duplicate snapshots at the same timestamp, keeping the query unambiguous.

---

### D-016: odds_snapshots index coverage and performance threshold

**Decision**: odds_snapshots index idx_odds_game_market covers (game_id, market, snapshot_ts). The book column is not indexed. Sufficient for v1 volume. If Unit 7 query latency exceeds 100ms, add a covering index including book. Revisit in v2.

**Rationale**: Unit 7 devig logic queries for "best available line at most recent snapshot per book" across multiple sportsbooks. The existing index (game_id, market, snapshot_ts) supports filtering by game and market but requires a sequential scan across books for the latest snapshot per book. For v1 volumes (~30 books × 15 games/day × 10 snapshots = 4,500 rows/day, ~1.6M rows/year), this is acceptable. Adding a covering index on (game_id, market, book, snapshot_ts) now would be premature optimization. The 100ms latency threshold provides a concrete trigger for adding the index if needed. Until then, keeping the schema minimal reduces index maintenance overhead and simplifies the migration history.

---

## Unit 3: Data Ingestion — Provider-Agnostic Interfaces

### D-017: Odds format detection and conversion

**Decision**: Odds format detection: adapters trust provider metadata field to distinguish American vs. decimal. If no metadata, values in range [−99999, −100] ∪ [100, 99999] are treated as American; values in [1.0, 50.0] are treated as European decimal. Ambiguous values are logged and skipped.

**Known limitation (FC-14)**: European decimal odds ≥ 100.0 would be misclassified as American. Not reachable for v1 main-line MLB markets.

**Rationale**: Prevents silent mis-conversion. §Data Layer: "provider-agnostic interfaces." Explicit format detection keeps conversion logic deterministic and debuggable. Ambiguous values (e.g., 75 could be American +75 or unlikely decimal 75.0) are rare in real odds feeds but must be handled defensively.

---

### D-018: Weather not fetched for retractable-roof parks

**Decision**: Weather is not fetched for retractable-roof parks. They receive no weather row and are treated as neutral run environment.

**Rationale**: Direct from spec §Parks & Weather: "Retractable-roof parks treated as neutral. Roof inference deferred." Fetching weather for retractable parks would waste API quota and introduce ambiguity (roof open vs. closed). V1 treats all retractable parks as neutral regardless of actual roof state. V2 can add roof-state inference if needed.

---

### D-019: Ingestion adapters are stateless async functions

**Decision**: Ingestion adapters are stateless async functions. They do not retry on failure — retry policy is owned by the scheduler (Unit 9).

**Rationale**: Keeps ingestion pure and testable. §Execution & Automation assigns orchestration to the scheduler. Embedding retry logic in adapters would violate single-responsibility principle and complicate testing. Adapters return empty lists or None on failure, log warnings, and delegate retry/backoff decisions to the scheduler.

---

### D-020: Unknown players upserted with available metadata

**Decision**: Unknown players encountered during ingestion are upserted into `players` with available metadata. Missing fields (`position`, `bats`, `throws`) are NULL.

**Rationale**: Prevents FK violations on `lineups` and `player_game_logs` inserts. Metadata is backfilled on subsequent stat fetches. Alternative (rejecting inserts with unknown players) would require pre-populating a complete player roster, which is fragile and unnecessary given MLB roster fluidity (callups, trades). Upsert-on-demand keeps the system resilient to incomplete provider data.

---

### D-055: V1GameProvider uses MLB Stats API with conservative fallbacks

**Decision**: V1GameProvider.fetch_schedule() fetches from MLB Stats API endpoint `/schedule?sportId=1&date={YYYY-MM-DD}&hydrate=team,venue` with 10-second timeout. Field mappings: game_id ← gamePk (as string), game_date ← officialDate, home_team_id/away_team_id ← teams.home/away.team.id, park_id ← venue.id (with DB fallback to home team's park if missing), first_pitch ← gameDate (parsed as UTC), status ← abstractGameCode (F→final, D→postponed, else→scheduled), home_score/away_score ← teams.home/away.score (populated only when status=final). On any exception (timeout, HTTP error, parse error), log warning and return empty list. Base URL is configurable via mlb_stats_api_base_url in AppConfig.

**Rationale**: MLB Stats API is the canonical free source for MLB game schedules with team/venue hydration. Conservative fallback (return []) prevents pipeline failures from propagating downstream while logging failures for monitoring. Park fallback (SELECT park_id FROM parks WHERE team_id = home team's park if missing) handles edge cases where venue.id is missing or unknown, preventing game skips due to incomplete API data. 10-second timeout balances reliability (typical API latency <2s) vs. pipeline responsiveness. Configurable base URL allows testing against mock servers or switching providers without code changes. Field mappings align with D-014 (game_id is provider-canonical) and D-025 (scores populated only when final).

---

### D-056: V1OddsProvider uses The Odds API with cached responses and defensive mappings

**Decision**: V1OddsProvider.fetch_odds() fetches from The Odds API endpoint `/sports/baseball_mlb/odds?apiKey={key}&regions=us&markets=h2h,spreads,totals&oddsFormat=american` with 10-second timeout and 5-minute TTL cache. Market mappings: h2h→ml, spreads→rl, totals→total. American odds converted to decimal via D-006 formula. Team name to ID lookup uses fuzzy matching against teams table. Game ID resolution queries games table by (game_date, home_team_id, away_team_id) with doubleheader_seq disambiguation. On lookup failures (unknown team, no matching game, ambiguous doubleheader), log warning and skip that odds entry. On API failures (timeout, HTTP error, invalid JSON), return empty list. Base URL and API key configurable via odds_api_base_url and odds_api_key in AppConfig.

**Rationale**: The Odds API is a free-tier provider for MLB odds with h2h/spreads/totals markets covering v1 scope. TTL cache reduces API quota consumption (free tier: 500 requests/month) while keeping odds fresh enough for ingestion runs (D-043, D-044). Fuzzy team name matching handles provider inconsistencies (e.g., "LA Dodgers" vs "Los Angeles Dodgers") without requiring brittle exact-match lookups. Game ID resolution by (date, teams, doubleheader_seq) bridges provider game identities to our canonical game_id (D-014). Skipping unresolvable odds entries prevents FK violations while logging failures for monitoring. Conservative error handling (return []) aligns with D-019 (adapters are stateless, retry owned by scheduler). Market mapping to our internal nomenclature (ml/rl/total) keeps odds_snapshots.market values consistent with downstream consumers (Unit 7). Configurable base URL/key enables testing and provider switching.

---

## Unit 4: Team Run-Scoring Models

### D-021: Shrinkage constant defaults for empirical Bayes estimation

**Decision**: Shrinkage constant `k` defaults: batters k=200 PA, pitchers k=80 IP (outs/3). Configurable in `AppConfig`.

**Rationale**: Balances signal vs. noise for v1 sample sizes. Standard empirical Bayesian approach. §Modeling Strategy: "shrinkage / pooling via rolling baselines and priors."

---

### D-022: Rolling window durations for player statistics

**Decision**: Rolling windows: batting stats use 60-day window, pitching stats use 30-day window. Both trailing from game_date.

**Rationale**: Pitching performance is more volatile and recent-biased. Windows are configurable.

---

### D-023: Weather-missing fallback for outdoor parks

**Decision**: Weather-missing fallback: use 72°F, 5 mph wind, 0% precip as neutral defaults when weather data is unavailable for an outdoor park.

**Rationale**: Prevents model failure while remaining close to a league-average outdoor environment. §Data Layer: "conservative fallbacks."

---

### D-024: Dispersion model trained separately from mean model

**Decision**: Dispersion model (r) is trained separately, not jointly with μ. Input features are identical to the μ model.

**Rationale**: Avoids training complexity in v1. Joint estimation deferred to v2. §Modeling Strategy: "both mean and dispersion explicitly predicted."

---

### D-025: games.home_score and games.away_score population contract

**Decision**: games.home_score and games.away_score (added in migration 005) are populated by the games ingestion adapter when status = 'final'. They are NULL for non-final games. Primary consumer is Unit 8 (evaluation). Unit 4 reads them as training targets for the μ model.

**Rationale**: Migration 005 added these columns but no DECISION documented who writes them or when. Without a clear contract, downstream units may read incomplete data or fail to populate them correctly. The score columns are required for Unit 8 (evaluation) to compute log loss/Brier against predictions, and Unit 4 needs them as training targets. The contract is simple: scores are populated only when the game is final, NULL otherwise. This prevents partial or inconsistent data from entering the pipeline.

---

### D-026: Model features passed as named columns, never positional arrays

**Decision**: Model features are passed as named columns (DataFrame or dict), never as positional arrays. Feature names are derived from GameFeatures.feature_names() and must match between training and inference. This prevents silent feature misalignment when the feature set evolves.

**Rationale**: The original implementation converted GameFeatures to raw numpy arrays in _features_to_array(), discarding column names and creating implicit positional coupling. If Unit 5 extends GameFeatures or fields are reordered, the model would silently map features to wrong columns, producing garbage predictions with no error. Using DataFrames with explicit feature names eliminates this coupling and allows sklearn/LightGBM to validate feature names at prediction time. The test_feature_name_stability test enforces that GameFeatures.feature_names() matches the trained model's booster_.feature_name(), catching misalignment at test time rather than in production.

---

## Unit 5: Player Prop Models

### D-027: P(start) is a LightGBM binary classifier with threshold configured separately

**Decision**: P(start) is a LightGBM binary classifier. Features: platoon matchup, days_rest, starts_last_7, starts_last_14, batting_order history. Threshold for publishing gate is configured separately in Unit 9 (0.85–0.90).

**Rationale**: Spec §Player Props: "Probabilistic start model: P(start | handedness, platoon usage, rest, lineup history)." Model produces the probability; the threshold is a publishing decision.

---

### D-028: PA distribution modeled as discrete multiclass, not Poisson

**Decision**: PA distribution is modeled as a discrete probability vector (PA = 0 through 6+) via a LightGBM multiclass classifier, not a Poisson or continuous model.

**Rationale**: Captures the strong discreteness and ceiling effects of MLB PA counts (rarely >6). Simpler than a count model for v1.

---

### D-029: Event rates use shrunk rolling means, not separate ML models

**Decision**: Event rates (H/PA, HR/PA, K/BF, etc.) are modeled as shrunk rolling means, not separate ML models. Shrinkage uses the same framework as Unit 4 (D-021).

**Rationale**: Spec says "per-opportunity event rates" with "shrinkage / pooling via rolling baselines and priors." For v1, shrunk means are sufficient and avoid overfitting on small samples. Upgrading to ML-based rate models is a v2 option.

---

### D-030: Switch hitters always receive platoon_adv = True

**Decision**: Switch hitters always receive `platoon_adv = True`.

**Rationale**: They bat from the advantaged side by rule. Avoids a feature-engineering special case that adds complexity with no v1 benefit.

---

### D-031: Pitcher k_rate is K per batter faced (K/BF) using approximation

**Decision**: Pitcher k_rate is K per batter faced (K/BF). Since batters_faced is not stored in player_game_logs, BF is approximated as ip_outs × 1.35 (league-average BF-per-out ratio). This ratio is a configurable constant in AppConfig (bf_per_out_ratio). Adding a batters_faced column to player_game_logs is a v2 improvement.

**Rationale**: The spec defines k_rate as "P(K | batter faced)" and Unit 6's simulation needs to sample strikeouts from batters faced, not from outs. Using K/out instead of K/BF produces biased projections because pitchers who walk many batters record fewer outs per BF, inflating K/out relative to K/BF. The approximation BF ≈ ip_outs × 1.35 is derived from league-average baserunner rates and provides a statistically sound proxy without requiring schema changes.

---

## Unit 6: Monte Carlo Simulation Engine

### D-032: Correlated noise uses bivariate normal copula

**Decision**: Correlated noise uses a bivariate normal copula. A shared standard normal Z is drawn per trial; each team's NB quantile is shifted by ρ × Z. Default ρ = 0.15, configurable in AppConfig.

**Rationale**: Spec: "optional correlated noise introduced during simulation without requiring latent variable inference." Copula approach is lightweight and avoids joint distribution modeling.

---

### D-033: Extra-innings tie-break is simplified probabilistic resolution

**Decision**: Extra-innings tie-break: P(home_win | tie) = home_mu / (home_mu + away_mu), adjusted by ±0.02 for bullpen fatigue differential. No additional runs are scored; the tie-break only assigns a winner.

**Rationale**: Spec: "simplified probabilistic tie-break based on team strength and bullpen state." This is the minimal faithful implementation. Full run-scoring extras deferred to v2.

---

### D-034: Player prop main lines are hardcoded for v1

**Decision**: Player prop main lines are hardcoded for v1: H=0.5, TB=1.5, HR=0.5, RBI=0.5, R=0.5, BB=0.5, K=4.5, OUTS=16.5, ER=2.5. These are the most common sportsbook lines.

**Rationale**: Spec: "main line only." Dynamic line detection from odds is deferred to Unit 7 or v2. Hardcoded lines are sufficient for v1 probability derivation.

---

### D-035: Hitter stat sampling uses independent Bernoulli per PA

**Decision**: Hitter stat sampling within a trial: PA is drawn first from pa_dist. Then H, HR, BB are drawn as independent Bernoulli/Binomial per PA. TB is derived from H + extra bases (not independently sampled). RBI and R use Poisson with rate × PA. Independence assumption is a v1 simplification.

**Rationale**: Spec: "event models conditional on PA." Full joint distribution of hitter outcomes is a v2 improvement.

---

## Unit 7: Odds Processing, Edge Calculation & Bankroll Sizing

### D-036: Proportional (multiplicative) devig only

**Decision**: Proportional (multiplicative) devig only. No power devig, Shin, or additive methods in v1.

**Rationale**: Spec: "proportional devig for fair probabilities." Single method keeps the system simple and auditable.

---

### D-037: Minimum edge threshold defaults to 0.02 (2%)

**Decision**: Minimum edge threshold defaults to 0.02 (2%). Configurable in AppConfig. Edges below threshold still stored but kelly_fraction is set to 0.0.

**Rationale**: Prevents noise-level edges from generating sizing recommendations. §Odds, Edge & Bankroll: "minimum thresholds."

---

### D-038: Kelly fraction = 0.25 (quarter-Kelly)

**Decision**: Kelly fraction = 0.25 (quarter-Kelly). Configurable in AppConfig. Formula: `0.25 × edge / (best_decimal_price − 1)`.

**Rationale**: Spec: "Fractional Kelly (0.25×)." Quarter-Kelly is standard for reducing variance in sports betting bankroll management.

---

### D-039: Devig requires both sides of a two-way market from the same book

**Decision**: Devig requires both sides of a two-way market from the same book. If no book provides both sides, that market receives no fair probability and no edge for that game.

**Rationale**: Proportional devig is mathematically undefined without the complementary side. Mixing books for devig introduces inconsistent vig assumptions.

---

## Unit 8: Evaluation & Backtesting Harness

### D-040: CLV closing line = latest odds snapshot at T−5 minutes before first_pitch

**Decision**: CLV closing line = latest odds snapshot at T−5 minutes before first_pitch. Devigged using the same proportional method (D-036). If no snapshot exists within 30 minutes before first pitch, the game is excluded from CLV.

**Rationale**: Spec: "CLV (median close at T−5)." The 30-minute fallback window prevents excluding too many games while keeping the "close" meaningful.

---

### D-041: Calibration uses isotonic regression as default

**Decision**: Calibration uses isotonic regression as default. Platt scaling available as config option. Calibration is per-market (ml, rl, total, team_total). Player prop calibration deferred to v2.

**Rationale**: Spec: "market-specific calibration models." Isotonic is nonparametric and handles non-monotonic miscalibration better than Platt for small samples.

---

### D-042: Eval results are upserted on (eval_date, market, metric)

**Decision**: Eval results are upserted on (eval_date, market, metric). Re-running a backtest for the same date and market overwrites prior results.

**Rationale**: Prevents accumulation of stale eval rows when models are updated and re-evaluated.

---

## Unit 9: Scheduler & Orchestration Pipeline

### D-043: Three global runs at 10 PM, 8 AM, 12 PM ET. Times configurable in AppConfig.

**Decision**: Three global runs at 10 PM, 8 AM, 12 PM ET. Times configurable in AppConfig.

**Rationale**: Spec: "night-before, morning, and midday global runs." ET aligns with MLB scheduling (most games 1 PM–10 PM ET).

---

### D-044: Per-game runs at T−90 and T−30 minutes before first_pitch. Configurable list.

**Decision**: Per-game runs at T−90 and T−30 minutes before first_pitch. Configurable list.

**Rationale**: Spec: "per-game scheduling relative to first pitch." Two runs balance freshness vs. compute cost within budget.

---

### D-045: Rerun throttle: at most 1 event-driven rerun per game per 10-minute window.

**Decision**: Rerun throttle: at most 1 event-driven rerun per game per 10-minute window.

**Rationale**: Prevents pipeline thrashing on rapid lineup/odds changes. §Execution & Automation: "event-driven reruns."

---

### D-046: Publishing gate threshold: p_start ≥ 0.85 (configurable). Team markets exempt — always publishable when edge is computed.

**Decision**: Publishing gate threshold: p_start ≥ 0.85 (configurable). Team markets exempt — always publishable when edge is computed.

**Rationale**: Spec §Lineup Uncertainty Policy: "P(start) ≥ 0.85–0.90." Lower end chosen as default to maximize coverage. Team market exemption follows from spec: "public outputs limited to informational summaries" applies only to player props pre-lineup.

---

## Unit 10: Discord Bot & Publishing Layer

### D-047: Discord bot is publish-only in v1. No user commands, no interactive features.

**Decision**: Discord bot is publish-only in v1. No user commands, no interactive features.

**Rationale**: Spec: "structured channels by market" and "emphasis on reliability and clarity." Interactive commands add complexity with no spec requirement.

---

### D-048: Free pick selection: highest absolute edge among team market plays for games with first_pitch in the [60, 90]-minute window, after lineup confirmation.

**Decision**: Free pick selection: highest absolute edge among team market plays for games with first_pitch in the [60, 90]-minute window, after lineup confirmation.

**Rationale**: Spec: "free pick posted 60–90 minutes before first pitch after lineup confirmation." Highest-edge selection maximizes showcase value.

---

### D-049: Anti-spam: one message per (game_id, market, side) per channel. Reruns edit existing messages. Message-ID tracking is in-memory, not persisted.

**Decision**: Anti-spam: one message per (game_id, market, side) per channel. Reruns edit existing messages. Message-ID tracking is in-memory, not persisted.

**Rationale**: Spec: "anti-spam publishing rules." In-memory tracking is sufficient for a single-process bot. Persistence deferred to v2.

---

### D-050: Only positive-edge plays (edge > 0 AND kelly_fraction > 0) are published.

**Decision**: Only positive-edge plays (edge > 0 AND kelly_fraction > 0) are published.

**Rationale**: Publishing zero or negative edge plays provides no value and undermines subscriber trust. Consistent with spec: this is a projections platform, not a firehose.

---

## Unit 11: Stripe Subscription & Webhook Integration

### D-051: Single subscription tier: free and paid. No intermediate tiers in v1.

**Decision**: Single subscription tier: free and paid. No intermediate tiers in v1.

**Rationale**: Spec: "Discord bot + Stripe webhooks" with free pick + paid content. Simplest monetization model. Multiple tiers deferred to v2.

---

### D-052: Webhook server is a standalone lightweight HTTP process (e.g., aiohttp or FastAPI), not embedded in the Discord bot process.

**Decision**: Webhook server is a standalone lightweight HTTP process (e.g., aiohttp or FastAPI), not embedded in the Discord bot process.

**Rationale**: Decouples payment handling from bot uptime. A bot restart doesn't drop webhook deliveries. Stripe retries failed deliveries for up to 72 hours.

---

### D-053: Discord role sync is best-effort. If Discord API is unavailable during webhook processing, the database is still updated. Role is corrected on next bot startup or manual sync.

**Decision**: Discord role sync is best-effort. If Discord API is unavailable during webhook processing, the database is still updated. Role is corrected on next bot startup or manual sync.

**Rationale**: Database is the source of truth for tier state. Discord role is a cache for channel permissions. Eventual consistency is acceptable.

---

### D-054: discord_user_id is passed via Stripe's client_reference_id at checkout. Webhooks that lack this field are logged and skipped.

**Decision**: discord_user_id is passed via Stripe's client_reference_id at checkout. Webhooks that lack this field are logged and skipped.

**Rationale**: Links Stripe customer to Discord identity without requiring a separate mapping table. Sufficient for single-plan v1.

---

### D-055: MLB Stats API schedule endpoint is v1 game provider.

**Decision**: MLB Stats API schedule endpoint is v1 game provider. Endpoint: `GET /api/v1/schedule?sportId=1&date=YYYY-MM-DD`. Maps `gamePk` to `game_id`, `venue.id` to `park_id` with fallback to home team's park (FC-32). Status mapping: `F`→`final`, `L`→`final`, `P`/`S`→`scheduled`, `D`/`I`→`postponed`. Scores populated when `abstractGameCode` in `{F, L}`. Timeout 10s, `[]` on failure per D-019. No API key required.

---

### D-056: The Odds API is v1 odds provider with defensive team name→ID mapping.

**Decision**: The Odds API is v1 odds provider. Endpoint: `GET /v4/sports/baseball_mlb/odds?apiKey=X&regions=us&markets=h2h,spreads,totals&oddsFormat=american`. Maps team names to `team_id` via DB query. Markets: `h2h`→`ml`, `spreads`→`rl`, `totals`→`total`. Side: home/away (h2h, spreads), over/under (totals). Matches games via home `team_id` + `commence_time` ±6h. Converts American→decimal per D-017. Skips events with unmappable teams or no matching game. HTTP response cache (1-minute TTL) per D-055. Timeout 10s, `[]` on failure per D-019. API key required (config: `odds_api_key`).

---

### D-057: MLB Stats API boxscore is v1 lineup provider.

**Decision**: MLB Stats API boxscore is v1 lineup provider. Endpoint: `GET /api/v1/game/{gamePk}/boxscore`. Confirmation: `is_confirmed=TRUE` if status is Live/Final (`L`/`F`) OR Preview/Scheduled (`P`/`S`) with exactly 9 players (not 9+, to avoid confirming placeholder lineups). Batting order from `battingOrder` field ("100"–"900" → 1–9); fallback to `battingOrder` array. Players with `battingOrder="0"` or outside [1,9] skipped. Unknown players upserted per D-020. Prior confirmed rows flipped per D-011. Lineups are append-only (versioned via `source_ts`); no ON CONFLICT on insert. Timeout 10s, `[]` on failure per D-019. No API key required.

---
