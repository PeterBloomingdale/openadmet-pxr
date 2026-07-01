"""
Phase 2 linear residual calibration using Analog Set 1 labels.

Run this on May 26 when Analog Set 1 labels are released.

The linear calibration:  y_corrected = a * y_predicted + b

Models often exhibit systematic biases:
- Scale compression (a < 1): predictions are "pulled toward the mean" — common
  when Huber loss is used and the model hasn't seen potent compounds at training.
  Manifests as: very potent compounds (pEC50 > 7) are underpredicted.
- Offset bias (b ≠ 0): systematic under- or over-prediction across the board.

Fitting on Analog Set 1 residuals corrects these biases before predicting Set 2.

IMPORTANT SAFETY CHECKS:
- If |a - 1.0| > 0.3: the model has severe variance compression, investigate
  before applying. This suggests the ensemble is poorly calibrated.
- If |b| > 0.5: large offset bias — also warrants investigation.
- If n_set1 < 30: don't calibrate — too few points to fit reliably. Use identity.

Always run diagnostic_calibration_plot() before the Phase 2 submission.
"""

from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger


def fit_residual_calibration(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    min_samples: int = 30,
) -> tuple[float, float]:
    """
    Fits y_true = a * y_pred + b by ordinary least squares.

    Returns (a, b) coefficients.
    If len(y_true) < min_samples, returns (1.0, 0.0) (identity — no correction).
    """
    n = len(y_true)
    if n < min_samples:
        logger.warning(
            f"Only {n} Set 1 compounds — too few to fit calibration reliably. "
            f"Using identity calibration (a=1.0, b=0.0)."
        )
        return 1.0, 0.0

    # OLS: minimize ||y_true - (a * y_pred + b)||²
    A = np.column_stack([y_pred, np.ones(n)])
    result = np.linalg.lstsq(A, y_true, rcond=None)
    a, b = float(result[0][0]), float(result[0][1])

    logger.info(f"Calibration fit: a={a:.4f}, b={b:.4f} (n={n})")

    # Safety checks
    if abs(a - 1.0) > 0.3:
        logger.warning(
            f"Calibration slope a={a:.3f} deviates from 1.0 by {abs(a-1.0):.3f} "
            f"(threshold 0.3). This suggests variance compression in the ensemble. "
            f"Review model predictions before applying this calibration."
        )
    if abs(b) > 0.5:
        logger.warning(
            f"Calibration offset b={b:.3f} > 0.5. "
            f"Large offset bias detected — investigate systematic errors."
        )

    return a, b


def apply_residual_calibration(
    y_pred: np.ndarray,
    a: float,
    b: float,
) -> np.ndarray:
    """Applies y_corrected = a * y_pred + b."""
    return a * y_pred + b


def diagnostic_calibration_plot(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    a: float,
    b: float,
    output_path: str = "submissions/phase2/calibration_diagnostic.png",
) -> None:
    """
    Saves a 2-panel calibration diagnostic figure.

    Panel 1: Scatter of y_pred vs y_true (pre-calibration) with regression line
    Panel 2: Residual distribution (pre and post calibration histograms)

    ALWAYS inspect this before submitting Phase 2 predictions.
    """
    import matplotlib.pyplot as plt

    y_corr = apply_residual_calibration(y_pred, a, b)
    residuals_pre = y_true - y_pred
    residuals_post = y_true - y_corr

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter
    lo, hi = min(y_true.min(), y_pred.min()) - 0.2, max(y_true.max(), y_pred.max()) + 0.2
    ax1.scatter(y_pred, y_true, alpha=0.6, s=30, label="Pre-calibration")
    x_line = np.linspace(lo, hi, 100)
    ax1.plot(x_line, a * x_line + b, "r-", lw=2, label=f"Fit: y = {a:.3f}x + {b:.3f}")
    ax1.plot([lo, hi], [lo, hi], "k--", alpha=0.4, label="Identity")
    ax1.set_xlabel("Predicted pEC50")
    ax1.set_ylabel("True pEC50 (Analog Set 1)")
    ax1.legend()
    ax1.set_title("Calibration: Predicted vs True")

    # Residual histogram
    ax2.hist(residuals_pre, bins=20, alpha=0.5, label=f"Pre-cal (MAE={np.mean(np.abs(residuals_pre)):.3f})", color="steelblue")
    ax2.hist(residuals_post, bins=20, alpha=0.5, label=f"Post-cal (MAE={np.mean(np.abs(residuals_post)):.3f})", color="tomato")
    ax2.axvline(0, color="k", linestyle="--", alpha=0.7)
    ax2.set_xlabel("Residual (True - Predicted)")
    ax2.set_ylabel("Count")
    ax2.legend()
    ax2.set_title("Residual Distribution")

    plt.suptitle(f"Phase 2 Calibration Diagnostic (n={len(y_true)} Set 1 compounds)", y=1.02)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Calibration diagnostic saved to {output_path}")
