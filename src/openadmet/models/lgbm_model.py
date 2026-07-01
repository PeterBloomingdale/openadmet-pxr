"""
LightGBM ensemble model on tabular features (fingerprints + 2D descriptors).

Why LightGBM for this task:
- Gradient boosted trees are the strongest tabular learner at data scales of
  1k-10k compounds (Praski et al. 2025 meta-benchmark confirms this).
- LightGBM trains in minutes on CPU — fast enough to run 5 folds × 5 seeds.
- Feature importance from LightGBM (mean gain across trees) is the most
  interpretable signal we have for understanding SAR drivers.

Objective choice: regression_l1 (MAE)
The competition metric is RAE = MAE / dynamic_range. Training with L2 (MSE)
penalizes distant outliers more than nearby ones, which pulls predictions
toward the mean — exactly wrong for activity cliff compounds. L1 is more
robust to the extreme values that dominate the test set (the 63 potent parents
have pEC50 ≥ 6.0, which is 10x more potent than most of the training data).
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    logger.warning("LightGBM not available")

from openadmet.data.splits import get_train_val_indices
from openadmet.cv.oof import evaluate_oof


DEFAULT_PARAMS = {
    "objective": "regression_l1",       # MAE-aligned with competition metric
    "metric": "mae",
    "num_leaves": 127,                  # More leaves than default 31 for complex SAR
    "learning_rate": 0.02,              # Low LR + more rounds = better generalization
    "feature_fraction": 0.7,            # Sample 70% of features per tree (decorrelates)
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 10,            # Prevents overfitting on small clusters
    "n_estimators": 2000,
    "early_stopping_rounds": 50,
    "verbose": -1,
    "n_jobs": -1,
    "device": "cpu",                    # Change to "cuda" to use GPU
}


def train_lgbm_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict,
    feature_names: Optional[list[str]] = None,
    fold_id: int = 0,
    run=None,
) -> tuple[lgb.Booster, np.ndarray]:
    """
    Trains one LightGBM model on one CV fold.

    Early stopping is on validation MAE. The number of trees at the best
    iteration is logged to W&B if run is provided.

    Returns (fitted_booster, val_predictions).
    """
    params = params.copy()
    n_estimators = params.pop("n_estimators", 2000)
    early_stopping_rounds = params.pop("early_stopping_rounds", 50)

    lgb_train = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

    callbacks = [
        lgb.early_stopping(early_stopping_rounds),
        lgb.log_evaluation(-1),  # Suppress per-round output
    ]

    booster = lgb.train(
        params,
        lgb_train,
        num_boost_round=n_estimators,
        valid_sets=[lgb_val],
        callbacks=callbacks,
    )

    val_preds = booster.predict(X_val)
    val_mae = float(np.mean(np.abs(y_val - val_preds)))
    logger.info(f"LightGBM fold {fold_id}: {booster.best_iteration} trees, val MAE={val_mae:.4f}")

    if run is not None:
        run.log({f"lgbm_fold{fold_id}_best_iter": booster.best_iteration,
                 f"lgbm_fold{fold_id}_val_mae": val_mae})

    return booster, val_preds


def train_lgbm_ensemble(
    feature_matrix: np.ndarray,
    targets: np.ndarray,
    fold_df: pd.DataFrame,
    params: Optional[dict] = None,
    feature_names: Optional[list[str]] = None,
    n_folds: int = 5,
    n_seeds: int = 5,
    output_dir: str = "models/lgbm",
    run=None,
) -> tuple[list[lgb.Booster], np.ndarray]:
    """
    Trains n_seeds × n_folds LightGBM models.

    Seed variation: each seed varies bagging_seed and feature_fraction_seed.
    This creates ensemble diversity without retraining the same model.

    OOF predictions are computed as the mean across seeds for each fold,
    which reduces variance in the meta-learner training data.

    Returns (all_boosters list of length n_seeds*n_folds, oof_predictions array).
    """
    params = params or DEFAULT_PARAMS.copy()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    oof_preds_by_seed = np.zeros((n_seeds, len(targets)))
    all_boosters = []

    for seed in range(n_seeds):
        seed_params = params.copy()
        seed_params["bagging_seed"] = seed + 42
        seed_params["feature_fraction_seed"] = seed + 100

        oof_this_seed = np.zeros(len(targets))

        for fold in range(n_folds):
            train_idx, val_idx = get_train_val_indices(fold_df, fold)
            X_train = feature_matrix[train_idx]
            y_train = targets[train_idx]
            X_val = feature_matrix[val_idx]
            y_val = targets[val_idx]

            booster, val_preds = train_lgbm_fold(
                X_train, y_train, X_val, y_val,
                params=seed_params,
                feature_names=feature_names,
                fold_id=fold,
                run=run,
            )
            oof_this_seed[val_idx] = val_preds
            all_boosters.append(booster)

            # Save each booster
            booster.save_model(str(out / f"booster_seed{seed}_fold{fold}.txt"))

        oof_preds_by_seed[seed] = oof_this_seed

    oof_predictions = oof_preds_by_seed.mean(axis=0)
    metrics = evaluate_oof(targets, oof_predictions, fold_df["fold"].values, n_folds)

    if run is not None:
        run.log({f"lgbm_oof_{k}": v for k, v in metrics.items() if not isinstance(v, dict)})

    np.save(str(out / "oof_predictions.npy"), oof_predictions)
    logger.info(f"LightGBM ensemble complete: {len(all_boosters)} models")
    return all_boosters, oof_predictions


def predict_lgbm_ensemble(
    boosters: list[lgb.Booster],
    X_test: np.ndarray,
) -> np.ndarray:
    """Averages predictions across all boosters."""
    preds = np.stack([b.predict(X_test) for b in boosters])
    return preds.mean(axis=0)


def get_lgbm_feature_importance(
    boosters: list[lgb.Booster],
    feature_names: list[str],
    importance_type: str = "gain",
) -> pd.DataFrame:
    """
    Averages feature importance across all boosters.

    importance_type="gain": sum of gain (information gain) across all splits
    using that feature. This is generally more informative than "split" count
    because it accounts for the magnitude of improvement, not just frequency.
    """
    importances = np.stack([b.feature_importance(importance_type=importance_type)
                            for b in boosters])
    mean_importance = importances.mean(axis=0)
    std_importance = importances.std(axis=0)

    return pd.DataFrame({
        "feature": feature_names,
        "mean_importance": mean_importance,
        "std_importance": std_importance,
    }).sort_values("mean_importance", ascending=False).reset_index(drop=True)
