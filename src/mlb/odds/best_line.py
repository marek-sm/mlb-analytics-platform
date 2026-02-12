"""Best-line selection across sportsbooks.

Implements best available odds per market position for devig and edge calculation.
"""

from dataclasses import dataclass
from datetime import datetime

import asyncpg

from mlb.db.models import Table


@dataclass
class BestLine:
    """Best available odds per market position.

    Represents the highest-price (most favorable) odds available across all books
    for a specific market position at the most recent snapshot.
    """

    game_id: str
    market: str  # 'ml' | 'rl' | 'total' | 'team_total'
    side: str | None  # 'home' | 'away' | 'over' | 'under'
    line: float | None  # e.g. -1.5, 8.5
    best_price: float  # European decimal, ≥ 1.0 (D-006)
    book: str  # Book offering the best price
    snapshot_ts: datetime  # UTC timestamp
    fair_prob: float | None = None  # Populated after devig; None before


async def get_best_lines(
    conn: asyncpg.Connection,
    game_id: str,
) -> list[BestLine]:
    """Get best available lines for all markets for a given game.

    For each (market, side, line) combination, returns the book offering the
    highest price (most favorable odds) at the most recent snapshot.

    Args:
        conn: Database connection
        game_id: Game identifier

    Returns:
        List of BestLine objects, one per unique (market, side, line)

    Notes:
        - Only returns lines from the most recent snapshot_ts per book
        - If multiple books have the same market/side/line at the latest snapshot,
          returns the one with the highest price
        - Returns empty list if no odds exist for the game
    """
    # Query to get the most recent snapshot per book
    rows = await conn.fetch(
        f"""
        WITH latest_snapshots AS (
            SELECT
                book,
                MAX(snapshot_ts) AS max_ts
            FROM {Table.ODDS_SNAPSHOTS}
            WHERE game_id = $1
            GROUP BY book
        ),
        latest_odds AS (
            SELECT
                o.game_id,
                o.book,
                o.market,
                o.side,
                o.line,
                o.price,
                o.snapshot_ts
            FROM {Table.ODDS_SNAPSHOTS} o
            INNER JOIN latest_snapshots ls
                ON o.book = ls.book
                AND o.snapshot_ts = ls.max_ts
            WHERE o.game_id = $1
        )
        SELECT
            game_id,
            market,
            side,
            line,
            MAX(price) AS best_price,
            (ARRAY_AGG(book ORDER BY price DESC))[1] AS book,
            MAX(snapshot_ts) AS snapshot_ts
        FROM latest_odds
        GROUP BY game_id, market, side, line
        ORDER BY market, side, line
        """,
        game_id,
    )

    return [
        BestLine(
            game_id=row["game_id"],
            market=row["market"],
            side=row["side"],
            line=float(row["line"]) if row["line"] is not None else None,
            best_price=float(row["best_price"]),
            book=row["book"],
            snapshot_ts=row["snapshot_ts"],
            fair_prob=None,
        )
        for row in rows
    ]
