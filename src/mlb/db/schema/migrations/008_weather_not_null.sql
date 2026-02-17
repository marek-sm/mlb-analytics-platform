-- 008_weather_not_null.sql: Add NOT NULL constraints to weather table columns (FC-39)
-- Prevents ambiguous signals to Unit 4 (NULL fields vs. no row for indoor parks)

-- Add NOT NULL constraints to weather data columns
-- Safe to add because adapter returns None (no row) for incomplete data rather than inserting NULLs
ALTER TABLE weather ALTER COLUMN temp_f SET NOT NULL;
ALTER TABLE weather ALTER COLUMN wind_speed_mph SET NOT NULL;
ALTER TABLE weather ALTER COLUMN wind_dir SET NOT NULL;
ALTER TABLE weather ALTER COLUMN precip_pct SET NOT NULL;
