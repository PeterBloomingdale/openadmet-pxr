"""
Competition metrics for the OpenADMET PXR challenge.

Primary metric: RAE = MAE / dynamic_range
  - dynamic_range = max(y_true) - min(y_true) on the test set
  - During CV we approximate using the validation fold's range
  - Lower RAE = better; competition target is approximately <= 0.15
"""

from typing import Optional
import numpy as np


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_dynamic_range(y_true: np.ndarray) -> float:
    """max - min of the label array. Competition uses the full test set range."""
    return float(np.max(y_true) - np.min(y_true))


def compute_rae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dynamic_range: Optional[float] = None,
) -> float:
    """
    Relative Absolute Error = MAE / dynamic_range.

    When dynamic_range is None, it is computed from y_true.
    During CV, pass the training-set dynamic_range as an approximation
    (the true test-set range is unknown until leaderboard evaluation).
    """
    mae = compute_mae(y_true, y_pred)
    dr = dynamic_range if dynamic_range is not None else compute_dynamic_range(y_true)
    if dr == 0.0:
        raise ValueError("dynamic_range is 0 — all labels are identical")
    return mae / dr


def compute_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from scipy.stats import spearmanr
    rho, _ = spearmanr(y_true, y_pred)
    return float(rho)


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def bootstrap_rae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Returns (mean_rae, std_rae) over bootstrap resamples.
    Matches the competition's evaluation methodology.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    raes = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        raes.append(compute_rae(y_true[idx], y_pred[idx]))
    return float(np.mean(raes)), float(np.std(raes))


def full_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dynamic_range: Optional[float] = None,
) -> dict[str, float]:
    """Compute all reported metrics in one call."""
    return {
        "mae": compute_mae(y_true, y_pred),
        "rae": compute_rae(y_true, y_pred, dynamic_range),
        "r2": compute_r2(y_true, y_pred),
        "spearman": compute_spearman(y_true, y_pred),
        "dynamic_range": dynamic_range or compute_dynamic_range(y_true),
    }
