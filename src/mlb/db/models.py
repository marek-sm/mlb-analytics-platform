"""Lightweight table-name constants and column-name enums."""

from enum import Enum


# Table name constants
class Table:
    """Database table names."""

    TEAMS = "teams"
    PARKS = "parks"
    GAMES = "games"
    PLAYERS = "players"
    LINEUPS = "lineups"
    PLAYER_GAME_LOGS = "player_game_logs"
    ODDS_SNAPSHOTS = "odds_snapshots"
    PROJECTIONS = "projections"
    SIM_MARKET_PROBS = "sim_market_probs"
    PLAYER_PROJECTIONS = "player_projections"
    EVAL_RESULTS = "eval_results"
    SUBSCRIPTIONS = "subscriptions"
    WEATHER = "weather"
    SCHEMA_MIGRATIONS = "schema_migrations"


# Column name enums
class League(str, Enum):
    """MLB league."""

    AL = "AL"
    NL = "NL"


class Division(str, Enum):
    """MLB division."""

    EAST = "East"
    CENTRAL = "Central"
    WEST = "West"


class GameStatus(str, Enum):
    """Game status."""

    SCHEDULED = "scheduled"
    CONFIRMED = "confirmed"
    FINAL = "final"
    POSTPONED = "postponed"


class Position(str, Enum):
    """Player position."""

    C = "C"
    _1B = "1B"
    _2B = "2B"
    _3B = "3B"
    SS = "SS"
    LF = "LF"
    CF = "CF"
    RF = "RF"
    DH = "DH"
    P = "P"


class BatHand(str, Enum):
    """Batting hand."""

    L = "L"
    R = "R"
    S = "S"


class ThrowHand(str, Enum):
    """Throwing hand."""

    L = "L"
    R = "R"


class Market(str, Enum):
    """Betting market."""

    ML = "ml"
    RL = "rl"
    TOTAL = "total"
    TEAM_TOTAL = "team_total"


class Side(str, Enum):
    """Market side."""

    HOME = "home"
    AWAY = "away"
    OVER = "over"
    UNDER = "under"


class PlayerStat(str, Enum):
    """Player projection stat."""

    H = "H"
    TB = "TB"
    HR = "HR"
    RBI = "RBI"
    R = "R"
    BB = "BB"
    K = "K"
    OUTS = "OUTS"
    ER = "ER"


class EvalMetric(str, Enum):
    """Evaluation metric."""

    LOG_LOSS = "log_loss"
    BRIER = "brier"
    ECE = "ece"
    TAIL_ACC = "tail_acc"
    CLV = "clv"


class SubscriptionTier(str, Enum):
    """Subscription tier."""

    FREE = "free"
    PAID = "paid"


class SubscriptionStatus(str, Enum):
    """Subscription status."""

    ACTIVE = "active"
    CANCELLED = "cancelled"
    PAST_DUE = "past_due"
