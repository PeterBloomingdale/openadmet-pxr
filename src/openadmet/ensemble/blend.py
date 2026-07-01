"""
Prediction blending for the multi-model ensemble.

Ensemble philosophy:
The three base models (LightGBM, Chemprop MPNN, MMP-delta Siamese) are
architecturally diverse — tabular features, graph neural network, and
structural perturbation — so their errors are partially uncorrelated.
Blending uncorrelated errors reduces variance. A simple weighted mean is
almost always better than more complex combiners at this data scale.

Why NOT rank-averaging:
Rank-averaging is a classification trick (minimizes AUC loss, not MAE).
The competition metric is RAE (MAE-based), so rank-averaging would hurt.
Always blend in prediction space, not rank space.

Blend weight optimization:
Weights are optimized with scipy.optimize.minimize (SLSQP) on OOF predictions,
minimizing MAE subject to non-negative weights summing to 1. SLSQP handles
linear constraints natively, unlike Nelder-Mead which requires penalty terms.
"""

from typing import Optional
import numpy as np
from scipy.optimize import minimize
from loguru import logger


def dynamic_recal(
    predictions: np.ndarray,
    target_std: float,
    min_factor: float = 1.0,
    max_factor: float = 4.0,
    skip_threshold: float = 1.1,
) -> tuple[np.ndarray, float]:
    """
    Variance recalibration around the prediction mean.

    Models trained with MAE/L1 loss on a hard regression problem tend to
    compress their output range toward the prediction mean (mean-collapse).
    This recalibration expands the spread to target the *training* pEC50
    standard deviation while leaving the mean and rank order unchanged.

    factor = clip(target_std / raw_std, min_factor, max_factor)

    Returns (calibrated_predictions, factor). When factor < skip_threshold
    no recalibration is applied (factor=1.0) — cheap insurance against
    over-correction when the model already preserves variance.

    Replaces the previous hard-coded 2.5× constant. The constant was
    calibrated against LightGBM's specific raw std (~0.087) and did not
    transfer to other models or to the n=4,135 retrain.
    """
    raw_std = float(np.std(predictions))
    if raw_std <= 0.0:
        logger.warning("dynamic_recal: predictions are constant — skipping calibration.")
        return predictions.copy(), 1.0

    factor = float(np.clip(target_std / raw_std, min_factor, max_factor))
    if factor < skip_threshold:
        logger.info(
            f"dynamic_recal: factor={factor:.2f} < skip_threshold={skip_threshold} "
            f"(raw_std={raw_std:.3f}, target_std={target_std:.3f}) — no calibration."
        )
        return predictions.copy(), 1.0

    mean = float(np.mean(predictions))
    cal = mean + factor * (predictions - mean)
    logger.info(
        f"dynamic_recal: factor={factor:.2f} (raw_std={raw_std:.3f}, "
        f"target_std={target_std:.3f}, cal_std={cal.std():.3f})"
    )
    return cal, factor


def mean_blend(
    predictions: dict[str, np.ndarray],
    weights: Optional[dict[str, float]] = None,
) -> np.ndarray:
    """
    Weighted mean of predictions from multiple models.

    weights=None → equal weights (1/n_models each).
    Weights do NOT need to sum to 1 — they are normalized internally.

    Returns blended prediction array of shape (n_compounds,).
    """
    model_names = list(predictions.keys())
    pred_array = np.stack([predictions[name] for name in model_names])  # (n_models, n)

    if weights is None:
        w = np.ones(len(model_names))
    else:
        w = np.array([weights.get(name, 1.0) for name in model_names])

    w = w / w.sum()  # Normalize
    blended = (pred_array * w[:, None]).sum(axis=0)

    for name, wi in zip(model_names, w):
        logger.info(f"  {name}: weight={wi:.3f}")

    return blended


def optimize_blend_weights(
    oof_predictions: dict[str, np.ndarray],
    y_true: np.ndarray,
    metric: str = "mae",
) -> dict[str, float]:
    """
    Optimizes blend weights to minimize MAE on OOF predictions.

    Optimization problem:
    minimize  MAE(Σ w_i * oof_i, y_true)
    subject to: Σ w_i = 1, w_i >= 0  (simplex constraint)

    Uses SLSQP (Sequential Least Squares Quadratic Programming) which handles
    both equality and inequality constraints natively.

    CAUTION: With n_models ≈ 3-4 and n_OOF ≈ 4000, this optimization has
    relatively few degrees of freedom and is unlikely to overfit badly.
    Still, verify that optimized weights are in [0.05, 0.90] — weights near
    0 or 1 suggest the optimizer is essentially selecting a single model,
    which is suspicious and warrants investigation.

    Returns dict {model_name: optimal_weight}.
    """
    model_names = list(oof_predictions.keys())
    n_models = len(model_names)
    pred_array = np.stack([oof_predictions[name] for name in model_names])  # (n_models, n)

    # Filter rows where y_true or any OOF is NaN — NaN propagates into SLSQP objective
    valid_mask = ~np.isnan(y_true) & ~np.isnan(pred_array).any(axis=0)
    y_opt = y_true[valid_mask]
    pred_opt = pred_array[:, valid_mask]

    def objective(w: np.ndarray) -> float:
        blended = (pred_opt * w[:, None]).sum(axis=0)
        if metric == "mae":
            return float(np.mean(np.abs(y_opt - blended)))
        elif metric == "rae":
            dr = float(np.max(y_opt) - np.min(y_opt))
            return float(np.mean(np.abs(y_opt - blended))) / dr
        else:
            raise ValueError(f"Unknown metric: {metric}")

    x0 = np.ones(n_models) / n_models  # Start from equal weights
    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
    bounds = [(0.0, 1.0)] * n_models

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9},
    )

    if not result.success:
        logger.warning(f"Blend weight optimization did not converge: {result.message}")

    optimal_weights = {name: float(w) for name, w in zip(model_names, result.x)}
    baseline_mae = objective(x0)
    optimized_mae = result.fun

    logger.info(f"Blend optimization: equal-weight MAE={baseline_mae:.4f} → optimized MAE={optimized_mae:.4f}")
    for name, w in optimal_weights.items():
        if w < 0.05:
            logger.warning(f"  {name}: weight={w:.3f} (near zero — consider dropping from ensemble)")
        else:
            logger.info(f"  {name}: weight={w:.3f}")

    return optimal_weights
