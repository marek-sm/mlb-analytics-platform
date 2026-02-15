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
