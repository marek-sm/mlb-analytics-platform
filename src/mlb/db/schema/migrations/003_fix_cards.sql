-- 003_fix_cards.sql: Apply fix cards identified in design review

-- FC-01: Add CHECK constraint to ensure odds_snapshots.price is European decimal (>= 1.0)
ALTER TABLE odds_snapshots
ADD CONSTRAINT odds_price_min CHECK (price >= 1.0);

-- FC-03: Add partial unique index to enforce at most one confirmed lineup per slot
CREATE UNIQUE INDEX lineups_confirmed_unique
ON lineups (game_id, team_id, batting_order)
WHERE is_confirmed = TRUE;

-- FC-04: Create reusable trigger function to automatically update updated_at on row updates
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Attach trigger to mutable tables that have updated_at column (games, subscriptions)
CREATE TRIGGER games_updated_at
BEFORE UPDATE ON games
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();

CREATE TRIGGER subscriptions_updated_at
BEFORE UPDATE ON subscriptions
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();

-- FC-05: Add edge_computed_at to distinguish incomplete edge calculations
ALTER TABLE sim_market_probs
ADD COLUMN edge_computed_at TIMESTAMPTZ;
