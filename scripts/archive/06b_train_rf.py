"""
Train a Random Forest baseline as a variance sanity check against LightGBM.

Random Forest aggregates many shallow trees with random feature subsets;
it typically produces less compressed output distributions than GBDT on
high-dimensional molecular feature matrices. If RF shows better rank ordering
than LightGBM (higher OOF Pearson r), it suggests LightGBM needs stronger
regularization or a simpler architecture.

Prerequisite: scripts/04_build_features.py, scripts/05_build_cv_splits.py
Runtime: ~5-15 minutes CPU (500 trees × n_folds iterations)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_absolute_error

from openadmet.utils.metrics import compute_rae, compute_dynamic_range
from openadmet.utils.submission import format_submission


def main():
    train_df = pd.read_parquet("data/splits/butina_folds.parquet")
    test_df = pd.read_parquet("data/raw/openadmet_test.parquet")

    X_train_full = np.load("data/features/train_features_all.npy")
    X_test_full = np.load("data/features/test_features_all.npy")

    pec50_col = "pec50_median" if "pec50_median" in train_df.columns else "pec50"
    y_train = train_df[pec50_col].values

    # Apply same variance filter as LightGBM script for consistency
    selector = VarianceThreshold(threshold=0.01)
    X_train = selector.fit_transform(X_train_full)
    X_test = selector.transform(X_test_full)
    logger.info(f"Features after VarianceThreshold: {X_train.shape[1]}")

    rf = RandomForestRegressor(
        n_estimators=500,
        max_features=0.3,       # ~1200 features per split; more than sqrt but less than full
        min_samples_leaf=5,     # each leaf represents a broader chemical neighborhood
        random_state=42,
        n_jobs=-1,
    )

    # OOF predictions using the same fold structure as LightGBM
    n_folds = train_df["fold"].nunique()
    oof_preds = np.zeros(len(y_train))

    for fold_id in range(n_folds):
        val_mask = train_df["fold"] == fold_id
        train_idx = np.where(~val_mask)[0]
        val_idx = np.where(val_mask)[0]

        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_val = X_train[val_idx]

        rf.fit(X_tr, y_tr)
        oof_preds[val_idx] = rf.predict(X_val)

        fold_mae = mean_absolute_error(y_train[val_idx], oof_preds[val_idx])
        logger.info(f"  Fold {fold_id}: val_MAE={fold_mae:.4f}")

    # OOF metrics
    oof_mae = mean_absolute_error(y_train, oof_preds)
    dr = compute_dynamic_range(y_train)
    oof_rae = compute_rae(y_train, oof_preds)
    r, _ = pearsonr(y_train, oof_preds)

    logger.info(f"\nRandom Forest OOF Results:")
    logger.info(f"  MAE = {oof_mae:.4f}")
    logger.info(f"  RAE = {oof_rae:.4f}  (dynamic range = {dr:.3f})")
    logger.info(f"  Pearson r = {r:.4f}")
    logger.info(f"  OOF std = {oof_preds.std():.4f}  (true std = {y_train.std():.4f})")

    np.save("models/lgbm/rf_oof_predictions.npy", oof_preds)

    # Fit final model on all training data for test predictions
    rf.fit(X_train, y_train)
    test_preds = rf.predict(X_test)
    np.save("models/lgbm/rf_test_predictions.npy", test_preds)

    logger.info(f"\nTest predictions: mean={test_preds.mean():.3f}, std={test_preds.std():.3f}")
    logger.info(f"Test range: [{test_preds.min():.3f}, {test_preds.max():.3f}]")

    compound_id_col = "compound_id" if "compound_id" in test_df.columns else test_df.columns[0]
    format_submission(
        test_df=test_df,
        predictions=test_preds,
        compound_id_col=compound_id_col,
        output_path="submissions/phase1/rf_baseline.csv",
    )
    logger.info("Submission saved to submissions/phase1/rf_baseline.csv")


if __name__ == "__main__":
    main()
