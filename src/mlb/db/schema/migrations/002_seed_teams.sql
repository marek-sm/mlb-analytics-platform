-- 002_seed_teams.sql: Seed 30 MLB teams and their home parks

-- Insert teams (all 30 MLB teams)
INSERT INTO teams (team_id, abbr, name, league, division) VALUES
    -- AL East
    (147, 'NYY', 'New York Yankees', 'AL', 'East'),
    (139, 'TBR', 'Tampa Bay Rays', 'AL', 'East'),
    (110, 'BAL', 'Baltimore Orioles', 'AL', 'East'),
    (111, 'BOS', 'Boston Red Sox', 'AL', 'East'),
    (141, 'TOR', 'Toronto Blue Jays', 'AL', 'East'),
    -- AL Central
    (145, 'CWS', 'Chicago White Sox', 'AL', 'Central'),
    (114, 'CLE', 'Cleveland Guardians', 'AL', 'Central'),
    (142, 'MIN', 'Minnesota Twins', 'AL', 'Central'),
    (116, 'DET', 'Detroit Tigers', 'AL', 'Central'),
    (118, 'KCR', 'Kansas City Royals', 'AL', 'Central'),
    -- AL West
    (117, 'HOU', 'Houston Astros', 'AL', 'West'),
    (133, 'OAK', 'Oakland Athletics', 'AL', 'West'),
    (136, 'SEA', 'Seattle Mariners', 'AL', 'West'),
    (108, 'LAA', 'Los Angeles Angels', 'AL', 'West'),
    (140, 'TEX', 'Texas Rangers', 'AL', 'West'),
    -- NL East
    (121, 'NYM', 'New York Mets', 'NL', 'East'),
    (143, 'PHI', 'Philadelphia Phillies', 'NL', 'East'),
    (144, 'ATL', 'Atlanta Braves', 'NL', 'East'),
    (146, 'MIA', 'Miami Marlins', 'NL', 'East'),
    (120, 'WSH', 'Washington Nationals', 'NL', 'East'),
    -- NL Central
    (112, 'CHC', 'Chicago Cubs', 'NL', 'Central'),
    (113, 'CIN', 'Cincinnati Reds', 'NL', 'Central'),
    (134, 'PIT', 'Pittsburgh Pirates', 'NL', 'Central'),
    (158, 'MIL', 'Milwaukee Brewers', 'NL', 'Central'),
    (138, 'STL', 'St. Louis Cardinals', 'NL', 'Central'),
    -- NL West
    (109, 'ARI', 'Arizona Diamondbacks', 'NL', 'West'),
    (115, 'COL', 'Colorado Rockies', 'NL', 'West'),
    (119, 'LAD', 'Los Angeles Dodgers', 'NL', 'West'),
    (135, 'SDP', 'San Diego Padres', 'NL', 'West'),
    (137, 'SFG', 'San Francisco Giants', 'NL', 'West');

-- Insert parks with seasonal park factors
-- Park factors are approximate seasonal averages (1.000 = neutral)
-- is_retractable treated as neutral per spec
INSERT INTO parks (park_id, name, team_id, is_outdoor, is_retractable, park_factor) VALUES
    -- AL East
    (3313, 'Yankee Stadium', 147, TRUE, FALSE, 1.050),
    (3394, 'Tropicana Field', 139, FALSE, FALSE, 0.950),
    (2, 'Oriole Park at Camden Yards', 110, TRUE, FALSE, 1.020),
    (3, 'Fenway Park', 111, TRUE, FALSE, 1.040),
    (14, 'Rogers Centre', 141, FALSE, TRUE, 1.000),
    -- AL Central
    (4, 'Guaranteed Rate Field', 145, TRUE, FALSE, 1.010),
    (5, 'Progressive Field', 114, TRUE, FALSE, 0.980),
    (3312, 'Target Field', 142, TRUE, FALSE, 0.990),
    (2394, 'Comerica Park', 116, TRUE, FALSE, 0.970),
    (7, 'Kauffman Stadium', 118, TRUE, FALSE, 1.000),
    -- AL West
    (2392, 'Minute Maid Park', 117, FALSE, TRUE, 1.000),
    (10, 'Oakland Coliseum', 133, TRUE, FALSE, 0.960),
    (680, 'T-Mobile Park', 136, FALSE, TRUE, 1.000),
    (1, 'Angel Stadium', 108, TRUE, FALSE, 0.990),
    (5325, 'Globe Life Field', 140, FALSE, TRUE, 1.000),
    -- NL East
    (3289, 'Citi Field', 121, TRUE, FALSE, 0.970),
    (2681, 'Citizens Bank Park', 143, TRUE, FALSE, 1.030),
    (4705, 'Truist Park', 144, TRUE, FALSE, 1.000),
    (4169, 'loanDepot park', 146, FALSE, TRUE, 1.000),
    (3309, 'Nationals Park', 120, TRUE, FALSE, 1.000),
    -- NL Central
    (17, 'Wrigley Field', 112, TRUE, FALSE, 1.040),
    (2602, 'Great American Ball Park', 113, TRUE, FALSE, 1.070),
    (31, 'PNC Park', 134, TRUE, FALSE, 0.980),
    (32, 'American Family Field', 158, FALSE, TRUE, 1.000),
    (2889, 'Busch Stadium', 138, TRUE, FALSE, 0.990),
    -- NL West
    (15, 'Chase Field', 109, FALSE, TRUE, 1.000),
    (19, 'Coors Field', 115, TRUE, FALSE, 1.200),
    (22, 'Dodger Stadium', 119, TRUE, FALSE, 0.980),
    (2680, 'Petco Park', 135, TRUE, FALSE, 0.930),
    (2395, 'Oracle Park', 137, TRUE, FALSE, 0.920);
