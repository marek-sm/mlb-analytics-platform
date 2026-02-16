# Development Log

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
