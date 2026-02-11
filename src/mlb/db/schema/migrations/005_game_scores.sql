-- 005_game_scores.sql: Add score columns to games table for model training

ALTER TABLE games
ADD COLUMN home_score SMALLINT,
ADD COLUMN away_score SMALLINT;

COMMENT ON COLUMN games.home_score IS 'Final home team score (NULL for games not yet final)';
COMMENT ON COLUMN games.away_score IS 'Final away team score (NULL for games not yet final)';
