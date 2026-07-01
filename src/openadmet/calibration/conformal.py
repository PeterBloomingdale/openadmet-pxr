"""
MAPIE conformal prediction intervals for uncertainty quantification.

Conformal prediction provides coverage-guaranteed prediction intervals:
given alpha=0.1, the interval contains the true pEC50 >= 90% of the time
(under the exchangeability assumption).

IMPORTANT CAVEAT for this competition:
MAPIE's exchangeability assumption states that calibration points and test
points are exchangeable (roughly: i.i.d.). For Analog Set 2 predictions,
Set 1 is used as the calibration set — but Set 1 and Set 2 are drawn from
the same analog clusters, so exchangeability approximately holds.

However, if Set 1 and Set 2 have different potency distributions (which is
possible if the organizers balanced them by activity class), the coverage
guarantee may not be exact. Use intervals for ranking uncertainty and
identifying high-risk predictions, not as hard coverage claims.

The intervals are useful for:
1. Identifying which Analog Set 2 predictions to flag as uncertain in the write-up
2. Understanding where the model extrapolates vs interpolates
3. Getting credit for UQ in the methodology section (co-authorship criteria)
"""

from typing import Optional
import numpy as np
from loguru import logger

try:
    from mapie.regression import MapieRegressor
    from mapie.conformity_scores import AbsoluteConformityScore
    MAPIE_AVAILABLE = True
except ImportError:
    MAPIE_AVAILABLE = False
    logger.warning("MAPIE not available — conformal prediction will not function")


def fit_conformal_predictor(
    y_pred_cal: np.ndarray,
    y_true_cal: np.ndarray,
    alpha: float = 0.1,
) -> object:
    """
    Fits a MAPIE conformal predictor using calibration residuals.

    This is the "split conformal" method: residuals |y_true - y_pred| on the
    calibration set define the interval width. For a new prediction ŷ, the
    interval is [ŷ - q, ŷ + q] where q is the (1-alpha) quantile of residuals.

    alpha=0.1 → 90% coverage.
    alpha=0.05 → 95% coverage (wider intervals).

    Returns fitted MapieRegressor.
    """
    if not MAPIE_AVAILABLE:
        raise ImportError("MAPIE required for conformal prediction")

    from sklearn.linear_model import LinearRegression

    # MAPIE wraps a base estimator; for post-hoc calibration on pre-computed
    # predictions, we use a simple passthrough estimator
    class PassthroughEstimator:
        def fit(self, X, y):
            return self
        def predict(self, X):
            return X.ravel()

    mapie = MapieRegressor(
        estimator=PassthroughEstimator(),
        cv="prefit",
        conformity_score=AbsoluteConformityScore(),
    )

    # y_pred_cal as features, y_true_cal as target
    mapie.fit(y_pred_cal.reshape(-1, 1), y_true_cal)

    # Compute empirical coverage
    _, intervals = mapie.predict(y_pred_cal.reshape(-1, 1), alpha=alpha)
    in_interval = (
        (y_true_cal >= intervals[:, 0, 0]) & (y_true_cal <= intervals[:, 1, 0])
    )
    coverage = float(in_interval.mean())
    logger.info(f"Conformal predictor: target coverage={1-alpha:.0%}, actual={coverage:.1%}")

    return mapie


def predict_with_intervals(
    mapie_model,
    y_pred: np.ndarray,
    alpha: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (point_predictions, lower_bounds, upper_bounds).

    point_predictions: same as y_pred (conformal does not change point estimates)
    lower/upper: coverage-guaranteed interval bounds

    Interval width = upper - lower. Wide intervals flag uncertain predictions.
    """
    if not MAPIE_AVAILABLE:
        raise ImportError("MAPIE required")

    _, intervals = mapie_model.predict(y_pred.reshape(-1, 1), alpha=alpha)
    lower = intervals[:, 0, 0]
    upper = intervals[:, 1, 0]
    return y_pred, lower, upper
