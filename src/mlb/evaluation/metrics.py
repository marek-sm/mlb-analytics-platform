"""Pure metric functions for probabilistic forecast evaluation.

All functions operate on numpy arrays and have no database dependencies.
Implements standard scoring rules: log loss, Brier score, ECE, and tail accuracy.
"""

import numpy as np
from numpy.typing import NDArray


def log_loss(p_model: NDArray[np.float64], outcomes: NDArray[np.float64]) -> float:
    """Compute log loss (cross-entropy) for probabilistic forecasts.

    Args:
        p_model: Model probabilities in [0, 1]
        outcomes: Binary outcomes (1.0 if event occurred, 0.0 otherwise)

    Returns:
        Log loss (lower is better, 0 is perfect)

    Raises:
        ValueError: If arrays have different lengths or contain invalid values

    Notes:
        - Probabilities are clipped to [1e-15, 1-1e-15] to prevent log(0)
        - Formula: -mean(y*log(p) + (1-y)*log(1-p))
        - Lower log loss indicates better calibration
    """
    p_model = np.asarray(p_model, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.float64)

    if len(p_model) != len(outcomes):
        raise ValueError(
            f"p_model and outcomes must have same length: {len(p_model)} != {len(outcomes)}"
        )

    if len(p_model) == 0:
        raise ValueError("Cannot compute log loss on empty arrays")

    if not np.all((p_model >= 0) & (p_model <= 1)):
        raise ValueError("p_model must be in [0, 1]")

    if not np.all((outcomes == 0) | (outcomes == 1)):
        raise ValueError("outcomes must be 0 or 1")

    # Clip probabilities to prevent log(0)
    epsilon = 1e-15
    p_clipped = np.clip(p_model, epsilon, 1 - epsilon)

    # Compute log loss
    loss = -(outcomes * np.log(p_clipped) + (1 - outcomes) * np.log(1 - p_clipped))
    return float(np.mean(loss))


def brier_score(p_model: NDArray[np.float64], outcomes: NDArray[np.float64]) -> float:
    """Compute Brier score for probabilistic forecasts.

    Args:
        p_model: Model probabilities in [0, 1]
        outcomes: Binary outcomes (1.0 if event occurred, 0.0 otherwise)

    Returns:
        Brier score (lower is better, 0 is perfect)

    Raises:
        ValueError: If arrays have different lengths or contain invalid values

    Notes:
        - Formula: mean((p_model - outcome)²)
        - Brier score is strictly proper (rewards honest forecasts)
        - Range is [0, 1] with 0 being perfect
    """
    p_model = np.asarray(p_model, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.float64)

    if len(p_model) != len(outcomes):
        raise ValueError(
            f"p_model and outcomes must have same length: {len(p_model)} != {len(outcomes)}"
        )

    if len(p_model) == 0:
        raise ValueError("Cannot compute Brier score on empty arrays")

    if not np.all((p_model >= 0) & (p_model <= 1)):
        raise ValueError("p_model must be in [0, 1]")

    if not np.all((outcomes == 0) | (outcomes == 1)):
        raise ValueError("outcomes must be 0 or 1")

    # Compute Brier score
    return float(np.mean((p_model - outcomes) ** 2))


def ece(
    p_model: NDArray[np.float64],
    outcomes: NDArray[np.float64],
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Measures calibration by binning predictions and comparing bin-level
    accuracy to average confidence in each bin.

    Args:
        p_model: Model probabilities in [0, 1]
        outcomes: Binary outcomes (1.0 if event occurred, 0.0 otherwise)
        n_bins: Number of probability bins (default 10)

    Returns:
        ECE (lower is better, 0 is perfect calibration)

    Raises:
        ValueError: If arrays have different lengths or contain invalid values

    Notes:
        - Bins are equally spaced in probability space: [0, 0.1), [0.1, 0.2), ...
        - ECE = weighted average of |bin_accuracy - bin_confidence|
        - Weight is proportion of samples in each bin
        - Empty bins contribute 0 to ECE
    """
    p_model = np.asarray(p_model, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.float64)

    if len(p_model) != len(outcomes):
        raise ValueError(
            f"p_model and outcomes must have same length: {len(p_model)} != {len(outcomes)}"
        )

    if len(p_model) == 0:
        raise ValueError("Cannot compute ECE on empty arrays")

    if not np.all((p_model >= 0) & (p_model <= 1)):
        raise ValueError("p_model must be in [0, 1]")

    if not np.all((outcomes == 0) | (outcomes == 1)):
        raise ValueError("outcomes must be 0 or 1")

    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(p_model, bin_edges[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # Compute ECE
    ece_sum = 0.0
    n_total = len(p_model)

    for i in range(n_bins):
        bin_mask = bin_indices == i
        bin_size = np.sum(bin_mask)

        if bin_size == 0:
            continue

        bin_confidence = float(np.mean(p_model[bin_mask]))
        bin_accuracy = float(np.mean(outcomes[bin_mask]))
        bin_weight = bin_size / n_total

        ece_sum += bin_weight * abs(bin_accuracy - bin_confidence)

    return float(ece_sum)


def tail_accuracy(
    p_model: NDArray[np.float64],
    outcomes: NDArray[np.float64],
    threshold: float = 0.15,
) -> dict[str, float | None]:
    """Compute tail accuracy diagnostics for extreme probabilities.

    Measures calibration specifically in the tails (low and high confidence
    predictions) where miscalibration causes the largest edge errors.

    Args:
        p_model: Model probabilities in [0, 1]
        outcomes: Binary outcomes (1.0 if event occurred, 0.0 otherwise)
        threshold: Tail threshold (default 0.15)

    Returns:
        Dictionary with keys:
            - low_tail_n: Number of predictions with p < threshold
            - low_tail_acc: Observed rate in low tail (None if n < 5)
            - high_tail_n: Number of predictions with p > (1-threshold)
            - high_tail_acc: Observed rate in high tail (None if n < 5)

    Raises:
        ValueError: If arrays have different lengths or contain invalid values

    Notes:
        - Low tail: p < threshold (e.g., p < 0.15)
        - High tail: p > (1 - threshold) (e.g., p > 0.85)
        - Accuracy is None if fewer than 5 samples in tail (insufficient data)
        - Accuracy is the empirical rate: mean(outcomes[in_tail])
    """
    p_model = np.asarray(p_model, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.float64)

    if len(p_model) != len(outcomes):
        raise ValueError(
            f"p_model and outcomes must have same length: {len(p_model)} != {len(outcomes)}"
        )

    if len(p_model) == 0:
        raise ValueError("Cannot compute tail accuracy on empty arrays")

    if not np.all((p_model >= 0) & (p_model <= 1)):
        raise ValueError("p_model must be in [0, 1]")

    if not np.all((outcomes == 0) | (outcomes == 1)):
        raise ValueError("outcomes must be 0 or 1")

    if not (0 < threshold < 0.5):
        raise ValueError(f"threshold must be in (0, 0.5), got {threshold}")

    # Low tail: p < threshold
    low_tail_mask = p_model < threshold
    low_tail_n = int(np.sum(low_tail_mask))
    low_tail_acc = None
    if low_tail_n >= 5:
        low_tail_acc = float(np.mean(outcomes[low_tail_mask]))

    # High tail: p > (1 - threshold)
    high_tail_mask = p_model > (1 - threshold)
    high_tail_n = int(np.sum(high_tail_mask))
    high_tail_acc = None
    if high_tail_n >= 5:
        high_tail_acc = float(np.mean(outcomes[high_tail_mask]))

    return {
        "low_tail_n": low_tail_n,
        "low_tail_acc": low_tail_acc,
        "high_tail_n": high_tail_n,
        "high_tail_acc": high_tail_acc,
    }
