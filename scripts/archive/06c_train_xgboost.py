"""
Train an XGBoost baseline for comparison against LightGBM and Random Forest.

Background: A colleague's analysis of the same challenge found that XGBoost and
Random Forest are statistically indistinguishable despite XGBoost consistently
looking better in raw numbers — a useful reminder that small numeric differences
can be noise. This script lets us verify whether that holds on our data and
adds a third tree model to the ensemble diversity pool.

XGBoost vs LightGBM:
- Both are GBDT frameworks with similar hyperparameter spaces.
- XGBoost grows trees level-wise (breadth-first); LightGBM leaf-wise (depth-first).
- On small/medium datasets, XGBoost's level-wise strategy often generalizes better
  because it grows more balanced trees. LightGBM's leaf-wise is faster but can
  overfit individual leaves — relevant for our n=1,334 training set.
- The `reg_alpha` (L1) and `reg_lambda` (L2) parameters mirror LightGBM's lambda_l1/l2.

Prerequisite: scripts/04_build_features.py, scripts/05_build_cv_splits.py
Runtime: ~5-15 minutes CPU (1000 trees × 5 folds)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import pearsonr
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

from openadmet.utils.metrics import compute_rae, compute_dynamic_range
from openadmet.utils.submission import format_submission


def main():
    train_df = pd.read_parquet("data/splits/butina_folds.parquet")
    test_df = pd.read_parquet("data/raw/openadmet_test.parquet")

    X_train_full = np.load("data/features/train_features_all.npy")
    X_test_full = np.load("data/features/test_features_all.npy")

    pec50_col = "pec50_median" if "pec50_median" in train_df.columns else "pec50"
    y_train = train_df[pec50_col].values

    # Same variance filter as LightGBM for fair comparison
    selector = VarianceThreshold(threshold=0.01)
    X_train = selector.fit_transform(X_train_full)
    X_test = selector.transform(X_test_full)
    logger.info(f"Features after VarianceThreshold: {X_train.shape[1]}")

    # XGBoost with parameters analogous to LightGBM v2 for apples-to-apples comparison.
    # Key differences from LightGBM: tree_method='hist' for speed, max_depth instead of
    # num_leaves, reg_alpha/reg_lambda instead of lambda_l1/lambda_l2.
    model = xgb.XGBRegressor(
        objective="reg:absoluteerror",  # MAE objective — matches competition metric
        n_estimators=2000,
        learning_rate=0.02,
        max_depth=6,                    # ~log2(num_leaves=63) ≈ 6
        min_child_weight=25,            # analogous to min_child_samples=25
        subsample=0.8,                  # bagging_fraction equivalent
        colsample_bytree=0.5,           # feature_fraction equivalent
        reg_alpha=0.1,                  # L1 — same as lambda_l1
        reg_lambda=0.05,                # L2 — same as lambda_l2
        tree_method="hist",             # faster histogram-based splitting
        early_stopping_rounds=100,
        eval_metric="mae",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    n_folds = train_df["fold"].nunique()
    oof_preds = np.zeros(len(y_train))

    for fold_id in range(n_folds):
        val_mask = train_df["fold"] == fold_id
        train_idx = np.where(~val_mask)[0]
        val_idx = np.where(val_mask)[0]

        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_val, y_val = X_train[val_idx], y_train[val_idx]

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        oof_preds[val_idx] = model.predict(X_val)

        fold_mae = mean_absolute_error(y_val, oof_preds[val_idx])
        logger.info(f"  Fold {fold_id}: val_MAE={fold_mae:.4f}  ({model.best_iteration} trees)")

    # OOF metrics
    oof_mae = mean_absolute_error(y_train, oof_preds)
    dr = compute_dynamic_range(y_train)
    oof_rae = compute_rae(y_train, oof_preds)
    r, _ = pearsonr(y_train, oof_preds)

    logger.info(f"\nXGBoost OOF Results:")
    logger.info(f"  MAE = {oof_mae:.4f}")
    logger.info(f"  RAE = {oof_rae:.4f}  (dynamic range = {dr:.3f})")
    logger.info(f"  Pearson r = {r:.4f}")
    logger.info(f"  OOF std = {oof_preds.std():.4f}  (true std = {y_train.std():.4f})")
    logger.info(f"  Variance factor = {y_train.std() / oof_preds.std():.2f}×")

    np.save("models/lgbm/xgb_oof_predictions.npy", oof_preds)

    # Retrain on all data for test predictions
    model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)
    test_preds = model.predict(X_test)
    np.save("models/lgbm/xgb_test_predictions.npy", test_preds)

    logger.info(f"\nTest predictions: mean={test_preds.mean():.3f}, std={test_preds.std():.3f}")

    compound_id_col = "compound_id" if "compound_id" in test_df.columns else test_df.columns[0]
    format_submission(
        test_df=test_df,
        predictions=test_preds,
        compound_id_col=compound_id_col,
        output_path="submissions/phase1/xgb_baseline.csv",
    )
    logger.info("Submission saved to submissions/phase1/xgb_baseline.csv")


if __name__ == "__main__":
    main()
