"""
Out-of-fold (OOF) prediction infrastructure.

OOF predictions are the backbone of the ensemble: each model is trained on
n-1 folds and predicts the held-out fold. The result is a prediction for
every training compound that is roughly comparable to "how well would this
model do on new, structurally distinct data?"

OOF predictions are used for:
1. Cross-validation performance estimation (our proxy for leaderboard RAE)
2. Training the Ridge meta-learner (ensemble stacking)
3. Fitting conformal calibration sets
"""

import numpy as np
import pandas as pd
from loguru import logger

from openadmet.data.splits import get_train_val_indices
from openadmet.utils.metrics import full_metrics


# Test-set dynamic range, derived from Sub 1 leaderboard result
# (LGBM 3.7×, MAE=0.7625, RAE=0.9582 → dynamic_range = 0.7625/0.9582 ≈ 0.7958).
# The Phase 1 test set spans ~0.80 pEC50 units, ~7× narrower than the full
# training range. RAE computed against the training range is therefore very
# optimistic and was misleading our submission strategy. We report both.
TEST_DYNAMIC_RANGE_ESTIMATE = 0.80


def evaluate_oof(
    y_true: np.ndarray,
    y_oof: np.ndarray,
    fold_assignments: np.ndarray,
    n_folds: int = 5,
    target_dynamic_range: float = TEST_DYNAMIC_RANGE_ESTIMATE,
) -> dict[str, float | dict]:
    """
    Compute OOF metrics overall and per fold.

    Reports two RAE values:
      * ``rae`` — using the *training* dynamic range (legacy behaviour).
      * ``rae_test`` — using ``target_dynamic_range`` (defaults to the test
        estimate ~0.80). This is the version that tracks the leaderboard.

    Other returned keys: mae, r2, spearman, dynamic_range, per_fold.
    """
    from openadmet.utils.metrics import compute_dynamic_range, compute_mae, compute_rae

    # Filter out NaN in either y_true (censored compounds) or y_oof (failed predictions)
    valid = ~(np.isnan(y_true) | np.isnan(y_oof))
    if not valid.all():
        n_dropped = (~valid).sum()
        logger.debug(f"evaluate_oof: dropping {n_dropped} NaN entries before metric computation")
        y_true = y_true[valid]
        y_oof = y_oof[valid]
        fold_assignments = fold_assignments[valid]

    dynamic_range = compute_dynamic_range(y_true)
    overall = full_metrics(y_true, y_oof, dynamic_range=dynamic_range)
    overall["rae_test"] = compute_mae(y_true, y_oof) / target_dynamic_range
    overall["target_dynamic_range"] = float(target_dynamic_range)

    per_fold: dict[int, dict[str, float]] = {}
    for fold in range(n_folds):
        mask = fold_assignments == fold
        if mask.sum() == 0:
            continue
        per_fold[fold] = {
            "mae": compute_mae(y_true[mask], y_oof[mask]),
            "rae": compute_rae(y_true[mask], y_oof[mask]),
            "rae_test": compute_mae(y_true[mask], y_oof[mask]) / target_dynamic_range,
            "n": int(mask.sum()),
        }
        logger.info(
            f"Fold {fold}: MAE={per_fold[fold]['mae']:.3f}, "
            f"RAE={per_fold[fold]['rae']:.3f}, RAE_test={per_fold[fold]['rae_test']:.3f} "
            f"(n={per_fold[fold]['n']})"
        )

    logger.info(
        f"OOF overall: MAE={overall['mae']:.3f}, "
        f"RAE(train_dr={dynamic_range:.2f})={overall['rae']:.3f}, "
        f"RAE(test_dr={target_dynamic_range:.2f})={overall['rae_test']:.3f}, "
        f"R²={overall['r2']:.3f}, ρ={overall['spearman']:.3f}"
    )

    return {**overall, "per_fold": per_fold}
