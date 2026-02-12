"""Proportional (multiplicative) devig for fair probability calculation.

Implements D-036: Proportional devig only, no power/Shin/additive methods in v1.
"""


def proportional_devig(prices: list[float]) -> list[float]:
    """Convert European decimal prices to fair probabilities using proportional method.

    Args:
        prices: List of European decimal prices for all sides of a market (≥ 1.0)

    Returns:
        List of fair probabilities summing to 1.0

    Raises:
        ValueError: If prices list is empty or contains invalid values

    Example:
        >>> proportional_devig([1.91, 1.91])
        [0.5, 0.5]
        >>> proportional_devig([1.50, 2.80])
        [0.6667, 0.3333]  # approximate
    """
    if not prices:
        raise ValueError("prices list cannot be empty")

    if any(p < 1.0 for p in prices):
        raise ValueError(f"All prices must be >= 1.0 (European decimal), got: {prices}")

    # Convert prices to implied probabilities
    implied = [1.0 / price for price in prices]

    # Calculate total (overround)
    total = sum(implied)

    if total == 0:
        raise ValueError(f"Sum of implied probabilities is zero for prices: {prices}")

    # Normalize to fair probabilities
    fair = [imp / total for imp in implied]

    return fair
