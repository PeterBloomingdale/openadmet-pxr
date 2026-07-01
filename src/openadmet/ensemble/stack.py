"""
Ridge meta-learner stacking on OOF predictions.

Stacking trains a Ridge regression model to find the optimal linear combination
of base model OOF predictions (meta-features) to predict the true labels.

When to use stacking vs blending:
- Blending with optimized weights: always included (low risk of overfitting)
- Ridge stacking: adds ~1-2% RAE improvement IF models have complementary errors
  AND the OOF pool is large enough (>500 compounds per fold). With ~4000 OOF
  rows and 3-4 base models, stacking is safe.

Ridge regularization:
High alpha (10-100) is preferred over default (alpha=1) because:
- We have 3-4 meta-features (one per base model) but 4000+ OOF rows
- Despite the apparent data abundance, the meta-features are highly correlated
  (all models predict similar ranges) — Ridge's L2 penalty prevents degenerate
  solutions where one model gets extreme weight

Setting negative_weights=False allows the Ridge stacker to assign negative
coefficients, which can represent implicit bias correction (e.g., if model A
consistently over-predicts when model B under-predicts for the same cluster).
"""

import numpy as np
import pandas as pd
from loguru import logger
from scipy.optimize import nnls as scipy_nnls
from sklearn.linear_model import ElasticNetCV, Ridge, RidgeCV


def train_ridge_stacker(
    oof_predictions: dict[str, np.ndarray],
    y_true: np.ndarray,
    alphas: list[float] = [1.0, 10.0, 100.0, 1000.0],
    cv: int = 5,
) -> Ridge:
    """
    Trains a Ridge regression meta-learner on OOF predictions.

    Input features = one column per base model (their OOF predictions).
    Target = true pEC50 values.

    RidgeCV selects alpha from alphas list using cross-validation on the OOF
    predictions. The default alpha list starts at 1.0 but emphasizes higher
    values (10-1000) because the meta-features are correlated.

    Returns fitted Ridge model.
    """
    model_names = list(oof_predictions.keys())
    X_meta = np.column_stack([oof_predictions[name] for name in model_names])

    logger.info(f"Ridge stacker: {len(model_names)} meta-features, {len(y_true)} OOF rows")

    ridge_cv = RidgeCV(alphas=alphas, cv=cv, scoring="neg_mean_absolute_error")
    ridge_cv.fit(X_meta, y_true)

    logger.info(f"Ridge stacker: best alpha={ridge_cv.alpha_:.1f}")
    for name, coef in zip(model_names, ridge_cv.coef_):
        logger.info(f"  {name}: coefficient={coef:.4f}")

    # Check for degenerate solutions
    if max(abs(ridge_cv.coef_)) > 2.0:
        logger.warning(
            "Ridge coefficient > 2.0 detected — meta-learner may be overfitting. "
            "Consider increasing alpha or checking OOF alignment."
        )

    # Return a fitted Ridge with the selected alpha
    ridge = Ridge(alpha=ridge_cv.alpha_)
    ridge.fit(X_meta, y_true)
    ridge.feature_names = model_names  # Store for later use
    return ridge


def predict_stack(
    ridge_model: Ridge,
    test_predictions: dict[str, np.ndarray],
) -> np.ndarray:
    """
    Applies the Ridge meta-learner to test-set predictions.

    CRITICAL: The column order must match the training order exactly.
    If any model is missing from test_predictions, this will raise KeyError.
    """
    feature_names = getattr(ridge_model, "feature_names", list(test_predictions.keys()))
    X_test = np.column_stack([test_predictions[name] for name in feature_names])
    return ridge_model.predict(X_test)


def honest_oof_stack(
    oof_predictions: dict[str, np.ndarray],
    y_true: np.ndarray,
    fold_assignments: np.ndarray,
    alphas: list[float] = [1.0, 10.0, 100.0, 1000.0],
) -> np.ndarray:
    """
    Build an honest out-of-fold stacker prediction.

    For each fold f: fit Ridge on the OOF rows where fold != f, predict on
    rows where fold == f. The result is an OOF prediction the meta-learner
    has never trained on, so comparing it to the SLSQP blend is fair.

    The previous code path fit RidgeCV on ALL OOF rows and then reported
    `ridge.predict(oof_preds)` as "OOF" — but that's training-set fit,
    not OOF, so it always looked artificially good and led the script to
    pick the stacker even when it would lose on the leaderboard.

    Returns honest_oof_pred of shape (n,).
    """
    model_names = list(oof_predictions.keys())
    X_meta = np.column_stack([oof_predictions[name] for name in model_names])
    n = len(y_true)
    honest = np.full(n, np.nan)

    unique_folds = np.unique(fold_assignments)
    for f in unique_folds:
        train_mask = fold_assignments != f
        val_mask = fold_assignments == f
        if val_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        ridge_cv = RidgeCV(alphas=alphas, scoring="neg_mean_absolute_error")
        ridge_cv.fit(X_meta[train_mask], y_true[train_mask])
        honest[val_mask] = ridge_cv.predict(X_meta[val_mask])

    n_missing = int(np.isnan(honest).sum())
    if n_missing:
        logger.warning(
            f"honest_oof_stack: {n_missing} rows had no fold assignment — "
            f"backfilled with simple mean blend."
        )
        backfill = X_meta.mean(axis=1)
        honest = np.where(np.isnan(honest), backfill, honest)

    logger.info(
        f"honest_oof_stack: built honest meta-CV predictions across "
        f"{len(unique_folds)} folds, n={n}"
    )
    return honest


def train_elasticnet_stacker(
    oof_predictions: dict[str, np.ndarray],
    y_true: np.ndarray,
    l1_ratios: list[float] | None = None,
    cv: int = 5,
    max_iter: int = 10000,
) -> ElasticNetCV:
    """
    Non-negative ElasticNet meta-learner on OOF columns (same pattern as Ridge).

    positive=True matches competition stacks that require nonnegative blend weights
    while still allowing L1 sparsity across correlated base models.
    """
    if l1_ratios is None:
        l1_ratios = [0.1, 0.5, 0.9]

    model_names = list(oof_predictions.keys())
    X_meta = np.column_stack([oof_predictions[name] for name in model_names])

    logger.info(
        f"ElasticNet stacker: {len(model_names)} meta-features, {len(y_true)} OOF rows, "
        f"l1_ratios={l1_ratios}"
    )

    enet = ElasticNetCV(
        l1_ratio=l1_ratios,
        cv=cv,
        positive=True,
        max_iter=max_iter,
        random_state=42,
        n_jobs=1,
    )
    enet.fit(X_meta, y_true)

    logger.info(f"ElasticNet stacker: best alpha={enet.alpha_:.6f}, l1_ratio={enet.l1_ratio_:.3f}")
    for name, coef in zip(model_names, enet.coef_):
        logger.info(f"  {name}: coefficient={coef:.4f}")

    enet.feature_names = model_names  # type: ignore[attr-defined]
    return enet


def predict_elasticnet_stack(
    enet_model: ElasticNetCV,
    test_predictions: dict[str, np.ndarray],
) -> np.ndarray:
    """Apply ElasticNet meta-learner to test columns (order matches training)."""
    feature_names = getattr(enet_model, "feature_names", list(test_predictions.keys()))
    X_test = np.column_stack([test_predictions[name] for name in feature_names])
    return enet_model.predict(X_test)


def train_nnls_stacker(
    oof_predictions: dict[str, np.ndarray],
    y_true: np.ndarray,
) -> dict:
    """
    Non-Negative Least Squares meta-learner: min ||Xw - y||₂ s.t. w ≥ 0.

    Unlike Ridge, NNLS has no regularization and no negative-coefficient
    solutions. Rank-18 competition teams report NNLS consistently beats
    Ridge on blind test sets — the lack of regularization bias helps when
    base models are already well-calibrated.

    Returns dict with 'weights' (raw NNLS solution, NOT normalized) and
    'feature_names'. Use raw weights for prediction via predict_nnls_stack.
    """
    model_names = sorted(oof_predictions.keys())
    X_meta = np.column_stack([oof_predictions[k] for k in model_names])

    w, residual = scipy_nnls(X_meta, y_true)

    logger.info(f"NNLS stacker: {len(model_names)} meta-features, residual={residual:.4f}")
    w_sum = w.sum()
    for name, wi in zip(model_names, w):
        logger.info(f"  {name}: weight={wi:.4f} ({100 * wi / max(w_sum, 1e-9):.1f}%)")

    return {"weights": w, "feature_names": model_names}


def predict_nnls_stack(
    nnls_model: dict,
    test_predictions: dict[str, np.ndarray],
) -> np.ndarray:
    """Apply NNLS meta-learner to test predictions (column order matches training)."""
    X = np.column_stack([test_predictions[k] for k in nnls_model["feature_names"]])
    return X @ nnls_model["weights"]


def honest_oof_nnls(
    oof_predictions: dict[str, np.ndarray],
    y_true: np.ndarray,
    fold_assignments: np.ndarray,
) -> np.ndarray:
    """
    Leave-one-fold-out NNLS: fit on folds != f, predict fold == f.
    Honest OOF estimate directly comparable to blend and honest Ridge OOF.

    Raw (unnormalized) NNLS weights are used for prediction so the scale
    of predictions matches y_true — do not normalize before applying.
    """
    model_names = sorted(oof_predictions.keys())
    X_meta = np.column_stack([oof_predictions[k] for k in model_names])
    n = len(y_true)
    honest = np.full(n, np.nan)

    unique_folds = np.unique(fold_assignments)
    for f in unique_folds:
        train_mask = fold_assignments != f
        val_mask = fold_assignments == f
        if val_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        w, _ = scipy_nnls(X_meta[train_mask], y_true[train_mask])
        honest[val_mask] = X_meta[val_mask] @ w

    n_missing = int(np.isnan(honest).sum())
    if n_missing:
        logger.warning(
            f"honest_oof_nnls: {n_missing} rows had no fold assignment — "
            f"backfilled with column mean blend."
        )
        backfill = X_meta.mean(axis=1)
        honest = np.where(np.isnan(honest), backfill, honest)

    logger.info(
        f"honest_oof_nnls: built honest NNLS meta-CV predictions across "
        f"{len(unique_folds)} folds, n={n}"
    )
    return honest


def honest_oof_elasticnet(
    oof_predictions: dict[str, np.ndarray],
    y_true: np.ndarray,
    fold_assignments: np.ndarray,
    l1_ratios: list[float] | None = None,
    max_iter: int = 10000,
) -> np.ndarray:
    """
    Leave-one-fold-out ElasticNet meta-CV (honest OOF), mirroring honest_oof_stack.
    """
    if l1_ratios is None:
        l1_ratios = [0.1, 0.5, 0.9]

    model_names = list(oof_predictions.keys())
    X_meta = np.column_stack([oof_predictions[name] for name in model_names])
    n = len(y_true)
    honest = np.full(n, np.nan)

    unique_folds = np.unique(fold_assignments)
    for f in unique_folds:
        train_mask = fold_assignments != f
        val_mask = fold_assignments == f
        if val_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        inner_cv = min(5, max(3, train_mask.sum() // 200))
        enet = ElasticNetCV(
            l1_ratio=l1_ratios,
            cv=inner_cv,
            positive=True,
            max_iter=max_iter,
            random_state=42,
            n_jobs=1,
        )
        enet.fit(X_meta[train_mask], y_true[train_mask])
        honest[val_mask] = enet.predict(X_meta[val_mask])

    n_missing = int(np.isnan(honest).sum())
    if n_missing:
        logger.warning(
            f"honest_oof_elasticnet: {n_missing} rows had no fold assignment — "
            f"backfilled with column mean blend."
        )
        backfill = X_meta.mean(axis=1)
        honest = np.where(np.isnan(honest), backfill, honest)

    logger.info(
        f"honest_oof_elasticnet: built honest meta-CV predictions across "
        f"{len(unique_folds)} folds, n={n}"
    )
    return honest
