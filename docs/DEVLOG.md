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
