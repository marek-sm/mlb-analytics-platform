"""Discord embed formatting for team market and player prop picks.

Pure functions that return Discord Embed objects. No database or state dependencies.
"""

from datetime import datetime

import discord


def format_team_market_embed(
    game_id: str,
    home_team: str,
    away_team: str,
    market: str,
    side: str,
    line: float | None,
    model_prob: float,
    edge: float,
    kelly_fraction: float,
    best_book: str,
    best_price: float,
    first_pitch: datetime,
    sim_n: int,
    model_version: str = "v1",
) -> discord.Embed:
    """Format a team market pick as a Discord embed.

    Args:
        game_id: Game identifier
        home_team: Home team abbreviation (e.g., "NYY")
        away_team: Away team abbreviation (e.g., "BOS")
        market: Market type ("ml", "rl", "total", "team_total")
        side: Side of the bet ("home", "away", "over", "under")
        line: Line value (None for moneyline, numeric for RL/totals)
        model_prob: Model-implied probability (0.0-1.0)
        edge: Edge over best available odds (0.0-1.0)
        kelly_fraction: Recommended Kelly bet fraction (0.0-1.0)
        best_book: Name of the sportsbook with best odds
        best_price: Best available decimal odds
        first_pitch: Game start time
        sim_n: Number of Monte Carlo simulations
        model_version: Model version string

    Returns:
        Discord Embed object ready to send
    """
    # Format title based on market type
    market_display = {
        "ml": "Moneyline",
        "rl": "Run Line",
        "total": "Total",
        "team_total": "Team Total",
    }.get(market, market.upper())

    title = f"{away_team} @ {home_team} — {market_display}"

    # Format side display
    if side == "home":
        side_display = f"{home_team} (Home)"
    elif side == "away":
        side_display = f"{away_team} (Away)"
    elif side == "over":
        side_display = f"Over {line}"
    elif side == "under":
        side_display = f"Under {line}"
    else:
        side_display = side

    # Add line to side display for RL
    if market == "rl" and line is not None:
        if side == "home":
            side_display = f"{home_team} {line:+.1f}"
        elif side == "away":
            side_display = f"{away_team} {line:+.1f}"

    # Format game time in ET
    game_time = first_pitch.strftime("%I:%M %p ET").lstrip("0")

    # Create embed
    embed = discord.Embed(
        title=title,
        color=discord.Color.green(),
        timestamp=datetime.utcnow(),
    )

    embed.add_field(name="Game Time", value=game_time, inline=True)
    embed.add_field(name="Pick", value=side_display, inline=True)
    embed.add_field(name="\u200b", value="\u200b", inline=True)  # Spacer

    embed.add_field(
        name="Model Probability",
        value=f"{model_prob * 100:.1f}%",
        inline=True,
    )
    embed.add_field(
        name="Edge",
        value=f"+{edge * 100:.1f}%",
        inline=True,
    )
    embed.add_field(
        name="Kelly Sizing",
        value=f"{kelly_fraction * 100:.1f}% bankroll",
        inline=True,
    )

    embed.add_field(
        name="Best Book",
        value=f"{best_book} @ {best_price:.2f}",
        inline=False,
    )

    embed.set_footer(text=f"Model {model_version} | {sim_n:,} simulations")

    return embed


def format_player_prop_embed(
    player_name: str,
    stat: str,
    line: float,
    side: str,
    p_start: float,
    model_prob: float,
    edge: float,
    kelly_fraction: float,
    best_book: str,
    best_price: float,
    home_team: str,
    away_team: str,
    first_pitch: datetime,
    sim_n: int,
    model_version: str = "v1",
) -> discord.Embed:
    """Format a player prop pick as a Discord embed.

    Args:
        player_name: Player full name
        stat: Stat type ("H", "TB", "HR", "RBI", "R", "BB", "K", "OUTS", "ER")
        line: Line value (e.g., 0.5 for hits)
        side: "over" or "under"
        p_start: Probability player starts (0.0-1.0)
        model_prob: Model-implied probability (0.0-1.0)
        edge: Edge over best available odds (0.0-1.0)
        kelly_fraction: Recommended Kelly bet fraction (0.0-1.0)
        best_book: Name of the sportsbook with best odds
        best_price: Best available decimal odds
        home_team: Home team abbreviation
        away_team: Away team abbreviation
        first_pitch: Game start time
        sim_n: Number of Monte Carlo simulations
        model_version: Model version string

    Returns:
        Discord Embed object ready to send
    """
    # Format stat display
    stat_display = {
        "H": "Hits",
        "TB": "Total Bases",
        "HR": "Home Runs",
        "RBI": "RBIs",
        "R": "Runs",
        "BB": "Walks",
        "K": "Strikeouts",
        "OUTS": "Outs Recorded",
        "ER": "Earned Runs",
    }.get(stat, stat)

    # Title
    title = f"{player_name} — {stat_display} O/U {line}"

    # Format game time in ET
    game_time = first_pitch.strftime("%I:%M %p ET").lstrip("0")

    # Format side
    side_display = f"{side.capitalize()} {line}"

    # Create embed
    embed = discord.Embed(
        title=title,
        color=discord.Color.blue(),
        timestamp=datetime.utcnow(),
    )

    embed.add_field(name="Game", value=f"{away_team} @ {home_team}", inline=True)
    embed.add_field(name="Game Time", value=game_time, inline=True)
    embed.add_field(name="\u200b", value="\u200b", inline=True)  # Spacer

    embed.add_field(
        name="P(Start)",
        value=f"{p_start * 100:.1f}%",
        inline=True,
    )
    embed.add_field(
        name="Model Probability",
        value=f"{side.capitalize()}: {model_prob * 100:.1f}%",
        inline=True,
    )
    embed.add_field(
        name="Edge",
        value=f"+{edge * 100:.1f}%",
        inline=True,
    )

    embed.add_field(
        name="Kelly Sizing",
        value=f"{kelly_fraction * 100:.1f}% bankroll",
        inline=True,
    )
    embed.add_field(
        name="Best Book",
        value=f"{best_book} @ {best_price:.2f}",
        inline=True,
    )
    embed.add_field(name="\u200b", value="\u200b", inline=True)  # Spacer

    embed.set_footer(text=f"Model {model_version} | {sim_n:,} simulations")

    return embed
