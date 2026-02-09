-- 001_initial.sql: Create all tables for v1 system

-- Migration tracking table (must exist first)
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMPTZ DEFAULT now()
);

-- Reference tables
CREATE TABLE teams (
    team_id SMALLINT PRIMARY KEY,
    abbr VARCHAR(3) NOT NULL UNIQUE,
    name TEXT NOT NULL,
    league VARCHAR(2) NOT NULL,
    division VARCHAR(7) NOT NULL
);

CREATE TABLE parks (
    park_id SMALLINT PRIMARY KEY,
    name TEXT NOT NULL,
    team_id SMALLINT REFERENCES teams,
    is_outdoor BOOLEAN NOT NULL,
    is_retractable BOOLEAN NOT NULL,
    park_factor NUMERIC(5,3) NOT NULL DEFAULT 1.000
);

-- Game & lineup tables
CREATE TABLE games (
    game_id TEXT PRIMARY KEY,
    game_date DATE NOT NULL,
    home_team_id SMALLINT REFERENCES teams,
    away_team_id SMALLINT REFERENCES teams,
    park_id SMALLINT REFERENCES parks,
    first_pitch TIMESTAMPTZ,
    status VARCHAR(16) NOT NULL DEFAULT 'scheduled',
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE players (
    player_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    team_id SMALLINT REFERENCES teams,
    position VARCHAR(4),
    bats VARCHAR(1),
    throws VARCHAR(1)
);

CREATE TABLE lineups (
    lineup_id BIGSERIAL PRIMARY KEY,
    game_id TEXT REFERENCES games,
    team_id SMALLINT REFERENCES teams,
    player_id INTEGER REFERENCES players,
    batting_order SMALLINT NOT NULL,
    is_confirmed BOOLEAN NOT NULL DEFAULT FALSE,
    source_ts TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(game_id, team_id, batting_order, source_ts)
);

-- Stats table
CREATE TABLE player_game_logs (
    log_id BIGSERIAL PRIMARY KEY,
    player_id INTEGER REFERENCES players,
    game_id TEXT REFERENCES games,
    pa SMALLINT,
    ab SMALLINT,
    h SMALLINT,
    tb SMALLINT,
    hr SMALLINT,
    rbi SMALLINT,
    r SMALLINT,
    bb SMALLINT,
    k SMALLINT,
    ip_outs SMALLINT,
    er SMALLINT,
    pitch_count SMALLINT,
    is_starter BOOLEAN,
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(player_id, game_id)
);

-- Odds table
CREATE TABLE odds_snapshots (
    snapshot_id BIGSERIAL PRIMARY KEY,
    game_id TEXT REFERENCES games,
    book VARCHAR(32) NOT NULL,
    market VARCHAR(16) NOT NULL,
    side VARCHAR(8),
    line NUMERIC(5,2),
    price NUMERIC(7,3) NOT NULL,
    snapshot_ts TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_odds_game_market ON odds_snapshots(game_id, market, snapshot_ts);

-- Projection & simulation tables
CREATE TABLE projections (
    projection_id BIGSERIAL PRIMARY KEY,
    game_id TEXT REFERENCES games,
    run_ts TIMESTAMPTZ NOT NULL,
    home_mu NUMERIC(6,3),
    away_mu NUMERIC(6,3),
    home_disp NUMERIC(6,3),
    away_disp NUMERIC(6,3),
    sim_n INTEGER NOT NULL,
    meta JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE sim_market_probs (
    prob_id BIGSERIAL PRIMARY KEY,
    projection_id BIGINT REFERENCES projections,
    market VARCHAR(16) NOT NULL,
    side VARCHAR(8),
    line NUMERIC(5,2),
    prob NUMERIC(7,5) NOT NULL,
    edge NUMERIC(7,5),
    kelly_fraction NUMERIC(7,5),
    UNIQUE(projection_id, market, side, line)
);

CREATE TABLE player_projections (
    pp_id BIGSERIAL PRIMARY KEY,
    projection_id BIGINT REFERENCES projections,
    player_id INTEGER REFERENCES players,
    game_id TEXT REFERENCES games,
    p_start NUMERIC(5,4),
    stat VARCHAR(4) NOT NULL,
    line NUMERIC(5,2),
    prob_over NUMERIC(7,5),
    edge NUMERIC(7,5),
    kelly_fraction NUMERIC(7,5),
    UNIQUE(projection_id, player_id, stat, line)
);

-- Evaluation table
CREATE TABLE eval_results (
    eval_id BIGSERIAL PRIMARY KEY,
    eval_date DATE NOT NULL,
    market VARCHAR(16) NOT NULL,
    metric VARCHAR(16) NOT NULL,
    value NUMERIC(10,6) NOT NULL,
    sample_n INTEGER,
    meta JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Subscription table
CREATE TABLE subscriptions (
    sub_id BIGSERIAL PRIMARY KEY,
    discord_user_id TEXT NOT NULL UNIQUE,
    stripe_customer_id TEXT,
    tier VARCHAR(8) NOT NULL DEFAULT 'free',
    status VARCHAR(16) NOT NULL DEFAULT 'active',
    current_period_end TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Weather table
CREATE TABLE weather (
    weather_id BIGSERIAL PRIMARY KEY,
    game_id TEXT REFERENCES games,
    temp_f SMALLINT,
    wind_speed_mph SMALLINT,
    wind_dir VARCHAR(4),
    precip_pct SMALLINT,
    fetched_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(game_id, fetched_at)
);
