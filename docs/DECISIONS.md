# Architecture Decision Records

This document tracks key architectural and implementation decisions made throughout the project.

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
