"""Unit 8: Evaluation & Backtesting Harness.

Provides metric computation, rolling-origin backtesting, CLV analysis,
and market-specific calibration for probabilistic forecasts.
"""

from mlb.evaluation.backtest import EvalReport, run_backtest
from mlb.evaluation.calibration import (
    CalibrationModel,
    apply_calibration,
    fit_calibration,
)
from mlb.evaluation.clv import CLVRow, compute_clv
from mlb.evaluation.metrics import brier_score, ece, log_loss, tail_accuracy
from mlb.evaluation.persistence import persist_eval_report

__all__ = [
    "brier_score",
    "ece",
    "log_loss",
    "tail_accuracy",
    "CLVRow",
    "compute_clv",
    "run_backtest",
    "EvalReport",
    "fit_calibration",
    "apply_calibration",
    "CalibrationModel",
    "persist_eval_report",
]
