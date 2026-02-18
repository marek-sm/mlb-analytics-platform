# Development Log

## 2026-02-16: Step 3 - Historical Backfill (≥30 Games)

### What Shipped

**Implementation: `src/mlb/operations/backfill.py`**
- `run_backfill(start_date, end_date)` async orchestration function executing seven FK-safe phases:
  1. **Schedule** — `fetch_schedule` + `write_games` per date, 1 s sleep between dates
  2. **Lineups** — `fetch_lineup(game_id, team_id)` called **twice per game** (home + away); persistence handled internally by adapter (D-060 boxscore cache makes second call cheap)
  3. **Stats** — `fetch_game_logs` + `write_game_logs` per unique date, 1 s sleep between dates
  4. **Odds** — `fetch_odds` per final game, 0.5 s sleep (see Known Limitations)
  5. **Weather** — `fetch_weather` per final game; `None` returns skipped per D-066, 0.5 s sleep
  6. **Verify** — six SQL scalar queries logged; RuntimeError if games < 30 or game_logs = 0
  7. **Train** — `team_runs.train(conn)` + `player_props.train(conn)`, artifact presence confirmed
- CLI entry point: `python -m mlb.operations.backfill --start YYYY-MM-DD --end YYYY-MM-DD`
- All adapter failures (exception, `[]`, `None`) are caught, logged as WARNING, and skipped — the backfill never aborts mid-run due to a single adapter failure
- Idempotent: UPSERT adapters (games, stats) handle re-runs gracefully; append-only tables (lineups, odds, weather) add new rows on re-run (per D-015)

**New package: `src/mlb/operations/__init__.py`** (empty, required for `python -m` entry)

**Test Coverage: `tests/test_backfill.py`**
- 6 unit tests, all using mocked adapters and pool — zero real DB or API calls

### Tests Added

1. `test_backfill_ingests_games_in_date_order` — `fetch_schedule` called once per date in ascending date order
2. `test_backfill_calls_lineup_for_both_teams` — `fetch_lineup` called exactly twice per final game (62 calls for 31 games); both home and away team IDs present
3. `test_backfill_ingests_odds_for_all_games` — `fetch_odds` called once for every final game (31 calls)
4. `test_backfill_weather_skips_none` — `write_weather` not called when `fetch_weather` returns `None`
5. `test_backfill_fails_under_30_games` — `RuntimeError` with count raised when DB reports < 30 final games
6. `test_backfill_continues_on_adapter_failure` — adapter exception on one date is swallowed; lineup phase still executes for games from successful dates

### Known Limitations

1. **Odds adapter uses live endpoint only** — `V1OddsProvider.fetch_odds()` calls `/v4/sports/baseball_mlb/odds` (live). For historical games no longer in the live feed it returns `[]` (per D-019). Production backfill requires extending the adapter with an optional `event_date` parameter routed to `/v4/historical/sports/baseball_mlb/odds?date=YYYY-MM-DD`. Flagged via `logger.warning` at runtime. No adapter modification made (out of scope per spec).
2. **Lineup write method discrepancy** — The spec assumed a separate `write_lineups()` method, but `V1LineupProvider.fetch_lineup()` handles DB persistence internally via `_persist_lineups()`. The backfill calls only `fetch_lineup()` per team, which is the correct usage.
3. **Model training needs DB connection** — Both `team_runs.train()` and `player_props.train()` require `asyncpg.Connection`, not a zero-argument call. The backfill acquires a pool connection and passes it directly. Spec assumed no-arg calls.
4. **Weather archive lag** — Open-Meteo archive has a 1–5 day data availability lag (FC-43); games within that window may return `None`. Unit 4 applies D-023 neutral defaults.
5. **Sequential ingestion** — No parallelization; total runtime ~60–90 minutes for a 21-day window with sleeps. Acceptable for a one-time operation per spec.

### What's Next

- Extend `V1OddsProvider` with a `fetch_odds(game_id, event_date=None)` historical-endpoint path before running production backfill
- Run `python -m mlb.operations.backfill --start 2025-06-01 --end 2025-06-21` against a populated database to populate training data
- Proceed to Unit 4 (Features) and Unit 5 (Player Props) model evaluation once backfill data is in place

### Files Modified

```
src/mlb/operations/
    __init__.py        # NEW — empty package marker
    backfill.py        # NEW — run_backfill() + CLI entry point
tests/
    test_backfill.py   # NEW — 6 unit tests (mock adapters)
docs/
    DEVLOG.md          # APPEND — this entry
```

---

## 2026-02-14: Step 1C - Lineups Ingestion (MLB Stats API)

### What Shipped

**Implementation: `src/mlb/ingestion/lineups.py`**
- Replaced `V1LineupProvider.fetch_lineup()` stub with full MLB Stats API boxscore integration
- Implemented MLB API boxscore endpoint (`GET /api/v1/game/{gamePk}/boxscore`)
- Parsing logic for home/away lineups with batting order extraction (1-9)
- Confirmation logic: Live/Final games always confirmed; Preview/Scheduled confirmed only with exactly 9 players
- Batting order derivation: Primary from `battingOrder` field ("100"→1, "900"→9), fallback to `battingOrder` array index
- D-011 flip logic: Prior confirmed rows set to `is_confirmed=FALSE` before inserting new confirmed lineup
- D-020 player upsert: Unknown players automatically upserted before lineup insert
- Conservative fallback: Returns `[]` on timeout, HTTP error, or parse failure (with appropriate logging)
- Transaction safety: Database errors caught, logged at ERROR level, with automatic rollback

**Test Coverage: `tests/test_ingestion.py`**
- 10 new acceptance criteria tests (AC1-AC10)
- Test fixtures: 4 JSON files covering standard lineup, live game, partial lineup, and bench players
- Tests validate: parsing, confirmation logic, D-011 flip, D-020 upsert, edge cases (bench players, missing fields, timeouts, invalid data, transaction rollback)

**Documentation:**
- D-057 appended to `docs/DECISIONS.md`: Documents MLB Stats API boxscore as v1 lineup provider with full contract details

### Tests Added

1. `test_lineup_parse_standard_lineup` - Parses 9 players with battingOrder "100"-"900", status "P", confirms is_confirmed=TRUE
2. `test_lineup_confirm_on_game_start` - Status "L" (Live) with 9 players confirms all rows
3. `test_lineup_partial_lineup_unconfirmed` - 6 players with status "P" results in is_confirmed=FALSE
4. `test_lineup_d011_flip_logic` - New confirmed lineup flips prior confirmed rows to FALSE
5. `test_lineup_d020_player_upsert` - Unknown players auto-upserted before lineup insert (no FK errors)
6. `test_lineup_skip_bench_players` - Players with battingOrder="0" skipped (9 starters, 2 bench → 9 rows)
7. `test_lineup_fallback_to_array` - Missing battingOrder field falls back to battingOrder array index
8. `test_lineup_api_timeout` - API timeout returns `[]` with warning log
9. `test_lineup_invalid_batting_order` - battingOrder="1000" (out of range) skipped with warning
10. `test_lineup_transaction_rollback` - Database error during insert triggers rollback (no partial writes)

**Test Fixtures Created:**
- `tests/fixtures/boxscore_full_lineup.json` - Standard 9-player lineup for both teams
- `tests/fixtures/boxscore_game_live.json` - Live game (status "L") with 9 players
- `tests/fixtures/boxscore_partial.json` - Partial lineup (6 players)
- `tests/fixtures/boxscore_with_bench.json` - 11 players (9 starters + 2 bench with battingOrder="0")

### Known Limitations

1. **No HTTP response caching**: Unlike odds ingestion (D-055), lineup fetches are not cached. Can be wired later if needed for cost/performance optimization.
2. **v1 only fetches at scheduled times**: Live in-game lineup changes (pinch hitters, substitutions) not tracked. Fetches occur at scheduler-defined intervals (T-90, T-30, global runs).
3. **No multi-provider failover**: Single MLB Stats API provider (D-014). If API is unavailable, fallback is `[]` with ERROR log.
4. **Pitcher-specific logic deferred**: Pitchers treated as position players in v1. Starting pitcher identification and rotation tracking deferred to future units.
5. **Historical backfill not implemented**: Uses same `fetch_lineup()` adapter but requires separate orchestration logic (out of scope for Step 1C).

### What's Next

**Immediate Dependencies (Unlocked by Step 1C):**
- **Unit 4 (Features)**: Can now compute `p_start` (probability player starts) from `lineups` table with `is_confirmed` filtering
- **Unit 6 (Simulation)**: Monte Carlo engine can conditionally include players based on `p_start >= 0.85` threshold (D-046)
- **Unit 9 (Scheduler)**: Event-driven reruns can trigger on lineup confirmation changes (D-045)

**Next Steps:**
- **Step 1D (Weather)**: Wire weather provider to external API (currently stub)
- **Step 1E (Stats/Game Logs)**: Wire stats provider for historical game logs (backfill + daily updates)
- **Unit 4 Feature Engineering**: Implement game-level and player-level feature pipelines using populated `lineups`, `games`, and `player_game_logs` tables

### Files Modified

```
src/mlb/
  ingestion/
    lineups.py          # Replaced fetch_lineup() stub with full implementation
tests/
  test_ingestion.py     # Added 10 test functions (AC1-AC10)
  fixtures/             # Created 4 JSON fixtures
    boxscore_full_lineup.json
    boxscore_game_live.json
    boxscore_partial.json
    boxscore_with_bench.json
docs/
  DECISIONS.md          # Appended D-057
  DEVLOG.md             # This file (created)
```

### Contract Summary

**Input:** `fetch_lineup(game_id: str, team_id: int | None = None)`
**Output:** `list[LineupRow]` with fields: `game_id`, `team_id`, `player_id`, `batting_order` (1-9), `is_confirmed`, `source_ts` (UTC)
**API:** `GET /api/v1/game/{gamePk}/boxscore` (timeout: 10s)
**Confirmation Rule:** Live/Final OR (Preview/Scheduled AND exactly 9 players)
**Batting Order:** "100"-"900" → 1-9; fallback to array; skip "0" and out-of-range
**D-011 Flip:** Prior confirmed rows → `is_confirmed=FALSE` before inserting new confirmed lineup
**D-020 Upsert:** Unknown players upserted with `name`, `position`, `team_id` before lineup insert
**Fallback:** `[]` on timeout/HTTP error/parse failure/DB error (with ERROR log)
**Persistence:** Append-only (versioned via `source_ts`); no `ON CONFLICT` on insert

---

*End of Step 1C Implementation Log*

---

## Step 1D: Player Stats / Game Logs Ingestion

**Date:** 2026-02-15
**Status:** ✅ Implemented
**Branch:** main

### What Shipped

Implemented Step 1D per mini-spec: player game logs ingestion via MLB Stats API. Replaced `V1StatsProvider.fetch_game_logs()` stub with full implementation.

**Core Features:**
1. **Fetch hitting + pitching logs** from MLB Stats API `/people/{playerId}/stats` endpoints
2. **Innings conversion**: Parse "5.2" format to 17 outs (5 × 3 + 2)
3. **Starter detection**: v1 heuristic `ip_outs >= 9` (3+ innings)
4. **Two-way player merge**: Single `GameLogRow` per player per game with both hitting and pitching stats
5. **UPSERT on conflict**: `(player_id, game_id)` with DO UPDATE for stats corrections
6. **D-020 player upsert**: Unknown players auto-created before inserting logs
7. **Orchestration**: Query completed games → query lineups for players → fetch season logs → filter to target games
8. **Conservative fallback**: Return `[]` on API timeout/error with WARNING log (D-019)

### Tests Added

Added 10 test functions in `test_ingestion.py` covering all acceptance criteria:

1. `test_parse_hitting_stats`: Parse hitting API response, verify all hitting fields populated
2. `test_parse_pitching_stats`: Parse pitching API response, verify all pitching fields and innings conversion
3. `test_innings_conversion`: Test `parse_ip_to_outs()` with valid/invalid formats
4. `test_starter_detection`: Test `detect_starter()` heuristic (>= 9 outs = starter)
5. `test_twoway_player_merge`: Merge hitting + pitching splits into single row
6. `test_gamelog_upsert_conflict`: UPSERT updates existing row on conflict
7. `test_gamelog_d020_player_upsert`: Unknown player auto-created before log insert
8. `test_gamelog_missing_fields`: Handle partial API responses (missing fields → None)
9. `test_gamelog_api_timeout`: Timeout returns `[]` with WARNING
10. `test_gamelog_empty_lineups`: No lineups for date returns `[]` with INFO (not error)

All tests use mock API responses with fixtures:
- `gamelog_hitting.json`: Hitting stats for Ohtani
- `gamelog_pitching.json`: Pitching stats for Ohtani
- `gamelog_twoway.json`: Combined hitting + pitching for same game

### Known Limitations

1. **v1 starter heuristic is approximate**: `ip_outs >= 9` misclassifies:
   - 2-inning openers as non-starters
   - Long relievers (4+ IP) as starters
   - Injury/rain-shortened starts (< 3 IP) as relievers
   - **v2 improvement**: Use boxscore pitching appearance order or `gameType` field

2. **Season-wide API calls**: Fetches entire season game logs per player, filters to target games client-side. MLB API doesn't support date range filtering for game logs.

3. **No historical backfill orchestration**: Daily batch execution only. High-volume multi-day backfill requires separate script.

4. **Prerequisite assumption**: Step 1C (lineups) must run before Step 1D. If lineups table is empty for date, returns `[]` + INFO log.

5. **No response caching**: Each player requires 2 HTTP calls (hitting + pitching). ~540 calls for typical 15-game slate. Acceptable for v1; consider caching in v2.

### What's Next

**Immediate Dependencies (Unlocked by Step 1D):**
- **Unit 4 (Features)**: Can now compute rolling player stats (wOBA, K/BF, etc.) from `player_game_logs` table
- **Unit 5 (Models)**: Batter and pitcher models can use historical game logs for shrinkage and projections
- **Unit 6 (Simulation)**: Monte Carlo engine can sample from player stat distributions

**Next Steps:**
- **Step 1E (Weather)**: Wire weather provider to external API (currently stub)
- **Unit 4 Feature Engineering**: Implement player-level feature pipelines using `player_game_logs`
- **Unit 5 Player Models**: Build batter/pitcher models with shrinkage

### Files Modified

```
src/mlb/
  ingestion/
    stats.py            # Replaced fetch_game_logs() stub; added parse_ip_to_outs(), detect_starter()
tests/
  test_ingestion.py     # Added 10 test functions (AC1-AC10)
  fixtures/             # Created 3 JSON fixtures
    gamelog_hitting.json
    gamelog_pitching.json
    gamelog_twoway.json
docs/
  DECISIONS.md          # Appended D-058
  DEVLOG.md             # This entry
```

### Contract Summary

**Input:** `fetch_game_logs(game_date: date)`
**Output:** `list[GameLogRow]` with fields: `player_id`, `game_id`, hitting stats (pa, ab, h, tb, hr, rbi, r, bb, k), pitching stats (ip_outs, er, pitch_count, is_starter)
**API:** 
- `GET /api/v1/people/{playerId}/stats?stats=gameLog&group=hitting&season={YYYY}` (timeout: 10s)
- `GET /api/v1/people/{playerId}/stats?stats=gameLog&group=pitching&season={YYYY}` (timeout: 10s)
**Innings Format:** "5.2" → 17 outs; invalid formats → None + WARNING
**Starter Heuristic:** `ip_outs >= 9` → `is_starter=True`
**Two-Way Merge:** Single row per `(player_id, game_id)` with both stat sets
**D-020 Upsert:** Unknown players auto-created before log insert
**UPSERT Conflict:** `(player_id, game_id)` DO UPDATE; `created_at` not updated
**Fallback:** `[]` on timeout/HTTP error/parse failure (with WARNING log per D-019)
**Orchestration:** Query completed games → query lineups → fetch player logs → filter to target games
**Persistence:** `write_game_logs()` handles batch upsert with executemany

---

*End of Step 1D Implementation Log*

---

## Step 1E: Weather Ingestion - Known Limitation (Open-Meteo Archive Lag)

**Date:** 2026-02-16
**Issue:** FC-43 - Archive endpoint data availability lag

### Open-Meteo Archive Data Lag

The Open-Meteo archive endpoint (`archive-api.open-meteo.com/v1/archive`) has a **data availability lag of 1–5 days** after the game date, depending on the weather station. Historical weather data may not be immediately available.

**Impact on Historical Backfill:**
- If Step 1E is called for yesterday's game during the morning pipeline run (Operational Step 3 backfill), the archive endpoint may return empty or incomplete hourly data
- The weather adapter correctly handles this by returning `None` (per D-019 conservative fallback)
- This produces gaps in the weather table that are **not automatically retried**

**Mitigation:**
- Unit 4 feature engineering applies D-023 neutral defaults (72°F, 5mph wind, 0% precip) when weather data is missing for outdoor parks
- Historical backfill can be **manually re-run 3–5 days later** to fill gaps once archive data becomes available
- Indoor/retractable parks are unaffected (no weather row expected per D-018)

**Out of Scope for v1:**
- Automatic retry logic for historical weather gaps
- Switching to a different historical weather provider with faster availability

**Recommendation:**
- For accurate historical backtesting, wait 5+ days after game date before running weather backfill
- Alternatively, accept D-023 neutral defaults as a reasonable approximation for missing historical weather

---

*Note: This limitation does not affect real-time weather fetches (forecast endpoint) for upcoming games.*
