-- 004_players_trigger.sql: Add updated_at timestamp tracking to players table

-- Add timestamp columns to players (missing from 001_initial.sql)
ALTER TABLE players
ADD COLUMN created_at TIMESTAMPTZ DEFAULT now(),
ADD COLUMN updated_at TIMESTAMPTZ DEFAULT now();

-- Attach trigger to automatically update updated_at on row updates
CREATE TRIGGER players_updated_at
BEFORE UPDATE ON players
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();
