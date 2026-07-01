"""
Train LightGBM on Uni-Mol 3D embeddings (512-dimensional).

Why a separate LightGBM for Uni-Mol:
Uni-Mol embeddings encode 3D shape/electronic structure and are more correlated
internally (learned representations) vs sparse fingerprints. A dedicated GBDT
with tuned hyperparameters for this feature space captures structure-activity patterns
that 2D-fingerprint LGBM misses, adding genuine 3D diversity to the ensemble.

Key differences from scripts/06_train_lgbm.py:
  - Input: Uni-Mol 512-d embeddings only (not the full 2D feature matrix)
  - num_leaves=63 (vs 127): Uni-Mol features are denser/more correlated,
    so shallower trees with more regularization generalize better.
  - lambda_l1=0.5, lambda_l2=0.1: stronger L1 regularization to handle
    correlated embedding dimensions.

Prerequisites:
  - scripts/17_extract_unimol_embeddings.py → data/features/train_unimol_emb.npy, test_unimol_emb.npy
  - scripts/05_build_cv_splits.py → data/splits/butina_folds.parquet

Outputs:
  - models/unimol_lgbm/oof_predictions.npy   (4135,)
  - models/unimol_lgbm/test_predictions.npy  (513,)

Next: python scripts/11_ensemble.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from loguru import logger

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

from openadmet.cv.oof import evaluate_oof


LGBM_PARAMS = {
    "objective": "regression_l1",      # MAE — matches RAE competition metric
    "num_leaves": 63,                  # smaller than 2D LGBM (127): denser features → less depth
    "learning_rate": 0.02,
    "feature_fraction": 0.6,           # slightly higher than 2D (0.5): fewer dims, less sparse
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 15,
    "lambda_l1": 0.5,                  # stronger L1 to prune correlated embedding dims
    "lambda_l2": 0.1,
    "path_smooth": 0.2,
    "n_estimators": 2000,
    "early_stopping_rounds": 100,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}

N_FOLDS = 5
N_SEEDS = 5


def main() -> None:
    if not LGB_AVAILABLE:
        logger.error("lightgbm not installed.")
        sys.exit(1)

    unimol_train_path = Path("data/features/train_unimol_emb.npy")
    unimol_test_path = Path("data/features/test_unimol_emb.npy")

    if not unimol_train_path.exists() or not unimol_test_path.exists():
        logger.error(
            "Uni-Mol embeddings not found. Run scripts/17_extract_unimol_embeddings.py first."
        )
        sys.exit(1)

    X_train = np.load(unimol_train_path).astype(np.float32)
    X_test = np.load(unimol_test_path).astype(np.float32)
    logger.info(f"Uni-Mol features: train={X_train.shape}, test={X_test.shape}")

    train_df = pd.read_parquet("data/splits/butina_folds.parquet")
    if len(X_train) != len(train_df):
        raise ValueError(
            f"train_unimol_emb.npy rows ({len(X_train)}) != butina_folds rows ({len(train_df)}). "
            "Re-run 17_extract_unimol_embeddings.py after confirming master_train.parquet matches."
        )

    pec50_col = "pec50_median" if "pec50_median" in train_df.columns else "pec50"
    y = train_df[pec50_col].values.astype(np.float64)
    folds = train_df["fold"].values

    feature_names = [f"unimol_{i}" for i in range(X_train.shape[1])]

    oof_all = np.full((N_SEEDS, len(train_df)), np.nan)
    test_all = []

    out = Path("models/unimol_lgbm")
    out.mkdir(parents=True, exist_ok=True)

    for seed in range(N_SEEDS):
        params = {**LGBM_PARAMS, "random_state": 42 + seed}
        seed_test = []

        for fold in range(N_FOLDS):
            train_mask = folds != fold
            val_mask = folds == fold

            X_tr, X_va = X_train[train_mask], X_train[val_mask]
            y_tr, y_va = y[train_mask], y[val_mask]

            dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_names)
            dval = lgb.Dataset(X_va, label=y_va, feature_name=feature_names, reference=dtrain)

            callbacks = [
                lgb.early_stopping(stopping_rounds=params.pop("early_stopping_rounds", 100), verbose=False),
                lgb.log_evaluation(period=-1),
            ]
            n_est = params.pop("n_estimators", 2000)

            booster = lgb.train(
                params,
                dtrain,
                num_boost_round=n_est,
                valid_sets=[dval],
                callbacks=callbacks,
            )
            # Restore popped keys for next iteration
            params["early_stopping_rounds"] = 100
            params["n_estimators"] = n_est

            oof_all[seed, val_mask] = booster.predict(X_va)
            seed_test.append(booster.predict(X_test))

            fold_mae = float(np.mean(np.abs(oof_all[seed, val_mask] - y_va)))
            logger.info(f"  Seed {seed}, fold {fold}: val MAE = {fold_mae:.4f} (best iter {booster.best_iteration})")

        test_all.append(np.mean(seed_test, axis=0))

    oof_preds = np.nanmean(oof_all, axis=0)
    test_preds = np.mean(test_all, axis=0)

    np.save(out / "oof_predictions.npy", oof_preds)
    np.save(out / "test_predictions.npy", test_preds)

    valid = ~np.isnan(oof_preds)
    metrics = evaluate_oof(y[valid], oof_preds[valid], folds[valid])
    logger.info(
        f"\nUni-Mol LGBM OOF: MAE={metrics['mae']:.4f}, RAE={metrics['rae']:.4f}, "
        f"R²={metrics['r2']:.4f}"
    )
    logger.info(f"Test predictions: mean={test_preds.mean():.3f}, std={test_preds.std():.3f}")
    logger.info("Saved models/unimol_lgbm/oof_predictions.npy and test_predictions.npy")
    logger.info("Next: python scripts/11_ensemble.py")


if __name__ == "__main__":
    main()
