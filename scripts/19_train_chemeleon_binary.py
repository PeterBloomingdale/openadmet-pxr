"""
Train a binary activity classifier on CheMeleon 2048-d embeddings.

Optimizes for pEC50 >= 5.0 classification (not magnitude) — provides
orthogonal signal to continuous regressors. Mimics the rank-18 team's
"TabICL binary" component of their meta-stacker.

Output predictions are linearly calibrated to pEC50 scale so they slot
into the ensemble blend and stacker without special handling.

Outputs:
  - models/chemeleon_binary/oof_predictions.npy   (4135,) calibrated pEC50
  - models/chemeleon_binary/test_predictions.npy  (513,) calibrated pEC50
  - models/chemeleon_binary/calibration.json      {a, b} for a + b*prob
"""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from loguru import logger

from openadmet.cv.oof import evaluate_oof


def main():
    train_df = pd.read_parquet("data/splits/butina_folds.parquet")
    test_df = pd.read_parquet("data/raw/openadmet_test.parquet")

    pec50_col = "pec50_median" if "pec50_median" in train_df.columns else "pec50"
    y_reg = train_df[pec50_col].values
    y_bin = (y_reg >= 5.0).astype(int)
    folds = train_df["fold"].values

    train_emb_path = Path("data/features/train_chemeleon_emb.npy")
    test_emb_path = Path("data/features/test_chemeleon_emb.npy")
    if not train_emb_path.exists():
        logger.error("CheMeleon embeddings not found. Run scripts/16_extract_chemeleon_embeddings.py first.")
        sys.exit(1)

    X_train = np.load(train_emb_path).astype(np.float32)
    X_test = np.load(test_emb_path).astype(np.float32)
    logger.info(f"CheMeleon embeddings: train={X_train.shape}, test={X_test.shape}")
    logger.info(f"Binary label: {y_bin.sum()}/{len(y_bin)} active (pEC50 >= 5.0)")

    # LightGBM binary classifier — logloss objective gives well-calibrated probabilities
    # scale_pos_weight balances the 32%/68% active/inactive split
    n_pos = y_bin.sum()
    n_neg = len(y_bin) - n_pos
    clf_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "n_estimators": 2000,
        "scale_pos_weight": n_neg / n_pos,
        "subsample": 0.8,
        "colsample_bytree": 0.3,
        "lambda_l1": 1.0,
        "lambda_l2": 1.0,
        "min_child_samples": 20,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    out = Path("models/chemeleon_binary")
    out.mkdir(parents=True, exist_ok=True)

    # 5-fold Butina CV for honest OOF probabilities
    oof_probs = np.zeros(len(y_bin))
    unique_folds = np.unique(folds)

    for f in unique_folds:
        train_mask = folds != f
        val_mask = folds == f
        logger.info(f"Fold {f}: train={train_mask.sum()}, val={val_mask.sum()}")

        clf = LGBMClassifier(**clf_params)
        clf.fit(
            X_train[train_mask], y_bin[train_mask],
            eval_set=[(X_train[val_mask], y_bin[val_mask])],
            callbacks=[
                __import__("lightgbm").early_stopping(stopping_rounds=50, verbose=False),
                __import__("lightgbm").log_evaluation(period=0),
            ],
        )
        oof_probs[val_mask] = clf.predict_proba(X_train[val_mask])[:, 1]
        logger.info(f"  Fold {f} OOF AUC-ROC: computing...")

    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_bin, oof_probs)
    logger.info(f"OOF AUC-ROC: {auc:.4f} (expected >0.85 for well-calibrated CheMeleon)")

    # Linear calibration: map probability [0,1] → pEC50 scale so the output
    # can participate in the pEC50-space blend/stacker without scale mismatch.
    # Fit on OOF to avoid train-set bias.
    b, a = np.polyfit(oof_probs, y_reg, 1)  # y_pec50 ≈ a + b*prob
    logger.info(f"Calibration: pEC50 = {a:.3f} + {b:.3f} × prob")
    logger.info(f"  At prob=0.0: pEC50={a:.3f}, at prob=1.0: pEC50={a+b:.3f}")

    oof_cal = a + b * oof_probs
    cal_metrics = evaluate_oof(y_reg, oof_cal, folds)
    logger.info(
        f"Calibrated OOF: MAE={cal_metrics['mae']:.4f}, "
        f"RAE={cal_metrics['rae']:.4f}, R²={cal_metrics['r2']:.4f}"
    )

    # Full model on all training data for test predictions
    clf_full = LGBMClassifier(**clf_params)
    clf_full.fit(X_train, y_bin)
    test_probs = clf_full.predict_proba(X_test)[:, 1]
    test_cal = a + b * test_probs

    np.save(out / "oof_predictions.npy", oof_cal)
    np.save(out / "test_predictions.npy", test_cal)
    np.save(out / "oof_probs_raw.npy", oof_probs)
    np.save(out / "test_probs_raw.npy", test_probs)

    with open(out / "calibration.json", "w") as fh:
        json.dump({"intercept": float(a), "slope": float(b), "auc_roc": float(auc)}, fh, indent=2)

    logger.info(f"\nCheMeleon binary classifier complete!")
    logger.info(f"OOF shape: {oof_cal.shape}, mean={oof_cal.mean():.3f}, std={oof_cal.std():.3f}")
    logger.info(f"Test shape: {test_cal.shape}, mean={test_cal.mean():.3f}, std={test_cal.std():.3f}")
    logger.info(f"Outputs saved to models/chemeleon_binary/")
    logger.info("Next: python scripts/11_ensemble.py")


if __name__ == "__main__":
    main()
