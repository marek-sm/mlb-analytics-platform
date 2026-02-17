-- 007_add_park_coords.sql: Add latitude/longitude columns to parks table for weather API

-- Add coordinate columns
ALTER TABLE parks ADD COLUMN latitude NUMERIC(9,6);
ALTER TABLE parks ADD COLUMN longitude NUMERIC(10,6);

-- Populate coordinates for all 30 MLB parks
UPDATE parks SET latitude = 40.829643, longitude = -73.926175 WHERE park_id = 3313;  -- Yankee Stadium
UPDATE parks SET latitude = 27.768267, longitude = -82.653497 WHERE park_id = 3394;  -- Tropicana Field
UPDATE parks SET latitude = 39.283921, longitude = -76.621512 WHERE park_id = 2;     -- Oriole Park at Camden Yards
UPDATE parks SET latitude = 42.346676, longitude = -71.097218 WHERE park_id = 3;     -- Fenway Park
UPDATE parks SET latitude = 43.641438, longitude = -79.389353 WHERE park_id = 14;    -- Rogers Centre
UPDATE parks SET latitude = 41.830000, longitude = -87.633812 WHERE park_id = 4;     -- Guaranteed Rate Field
UPDATE parks SET latitude = 41.495861, longitude = -81.685255 WHERE park_id = 5;     -- Progressive Field
UPDATE parks SET latitude = 44.981697, longitude = -93.277737 WHERE park_id = 3312;  -- Target Field
UPDATE parks SET latitude = 42.339080, longitude = -83.048530 WHERE park_id = 2394;  -- Comerica Park
UPDATE parks SET latitude = 39.051678, longitude = -94.480450 WHERE park_id = 7;     -- Kauffman Stadium
UPDATE parks SET latitude = 29.756967, longitude = -95.355186 WHERE park_id = 2392;  -- Minute Maid Park
UPDATE parks SET latitude = 37.751511, longitude = -122.200698 WHERE park_id = 10;   -- Oakland Coliseum
UPDATE parks SET latitude = 47.591333, longitude = -122.332222 WHERE park_id = 680;  -- T-Mobile Park
UPDATE parks SET latitude = 33.800308, longitude = -117.882732 WHERE park_id = 1;    -- Angel Stadium
UPDATE parks SET latitude = 32.747299, longitude = -97.082559 WHERE park_id = 5325;  -- Globe Life Field
UPDATE parks SET latitude = 40.757089, longitude = -73.845765 WHERE park_id = 3289;  -- Citi Field
UPDATE parks SET latitude = 39.906042, longitude = -75.166584 WHERE park_id = 2681;  -- Citizens Bank Park
UPDATE parks SET latitude = 33.890672, longitude = -84.467641 WHERE park_id = 4705;  -- Truist Park
UPDATE parks SET latitude = 25.778136, longitude = -80.219832 WHERE park_id = 4169;  -- loanDepot park
UPDATE parks SET latitude = 38.872861, longitude = -77.007501 WHERE park_id = 3309;  -- Nationals Park
UPDATE parks SET latitude = 41.948376, longitude = -87.655400 WHERE park_id = 17;    -- Wrigley Field
UPDATE parks SET latitude = 39.097389, longitude = -84.506611 WHERE park_id = 2602;  -- Great American Ball Park
UPDATE parks SET latitude = 40.446904, longitude = -80.005753 WHERE park_id = 31;    -- PNC Park
UPDATE parks SET latitude = 43.028111, longitude = -87.971167 WHERE park_id = 32;    -- American Family Field
UPDATE parks SET latitude = 38.622619, longitude = -90.192886 WHERE park_id = 2889;  -- Busch Stadium
UPDATE parks SET latitude = 33.445302, longitude = -112.066687 WHERE park_id = 15;   -- Chase Field
UPDATE parks SET latitude = 39.755882, longitude = -104.994178 WHERE park_id = 19;   -- Coors Field
UPDATE parks SET latitude = 34.073851, longitude = -118.240449 WHERE park_id = 22;   -- Dodger Stadium
UPDATE parks SET latitude = 32.707530, longitude = -117.156830 WHERE park_id = 2680; -- Petco Park
UPDATE parks SET latitude = 37.778383, longitude = -122.389448 WHERE park_id = 24;   -- Oracle Park

-- Add NOT NULL constraints after data is populated
ALTER TABLE parks ALTER COLUMN latitude SET NOT NULL;
ALTER TABLE parks ALTER COLUMN longitude SET NOT NULL;
