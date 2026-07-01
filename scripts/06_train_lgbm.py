"""
Train LightGBM ensemble (5 folds × 5 seeds = 25 models).

Prerequisites: scripts/04_build_features.py, scripts/05_build_cv_splits.py

Runtime: ~10-30 minutes on CPU for all 25 models.

First submission target: submit LGBM-only predictions by May 8
to validate the full pipeline and anchor leaderboard scale.
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.feature_selection import VarianceThreshold

from openadmet.models.lgbm_model import train_lgbm_ensemble, predict_lgbm_ensemble, get_lgbm_feature_importance
from openadmet.utils.tracking import init_wandb_run, log_oof_summary
from openadmet.utils.submission import format_submission


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chemeleon", action="store_true",
                        help="Append 2048-d CheMeleon embeddings to features. "
                             "Outputs to models/lgbm_chemeleon/ instead of models/lgbm/.")
    args = parser.parse_args()

    # Load config
    with open("configs/lgbm.yaml") as f:
        config = yaml.safe_load(f)

    # Load data
    train_df = pd.read_parquet("data/splits/butina_folds.parquet")
    test_df = pd.read_parquet("data/raw/openadmet_test.parquet")

    X_train = np.load("data/features/train_features_all.npy")
    X_test = np.load("data/features/test_features_all.npy")
    with open("data/features/all_feature_names.json") as f:
        feature_names = json.load(f)

    if args.chemeleon:
        train_emb_path = Path("data/features/train_chemeleon_emb.npy")
        test_emb_path  = Path("data/features/test_chemeleon_emb.npy")
        if not train_emb_path.exists():
            logger.error("CheMeleon embeddings not found. Run scripts/16_extract_chemeleon_embeddings.py first.")
            sys.exit(1)
        train_emb = np.load(train_emb_path).astype(np.float32)
        test_emb  = np.load(test_emb_path).astype(np.float32)
        X_train = np.concatenate([X_train, train_emb], axis=1)
        X_test  = np.concatenate([X_test,  test_emb],  axis=1)
        emb_names = [f"chemeleon_{i}" for i in range(train_emb.shape[1])]
        feature_names = feature_names + emb_names
        logger.info(f"CheMeleon embeddings appended: {train_emb.shape[1]}d → total {X_train.shape[1]} features")

    pec50_col = "pec50_median" if "pec50_median" in train_df.columns else "pec50"

    # Filter to primary Octant-assay sources only for tabular models.
    # External PubChem data (different assay setup) is used for Chemprop pretraining only.
    # dargason (rank 14): "ChEMBL/BindingDB always hurt — assay differences confound data."
    PRIMARY_SOURCES = {"openadmet", "analog_set1", "htchem", "htchem_semi_pure"}
    if "source" in train_df.columns:
        primary_mask = train_df["source"].isin(PRIMARY_SOURCES).values
        train_df = train_df[primary_mask].reset_index(drop=True)
        X_train  = X_train[primary_mask]
        logger.info(
            f"Filtered to primary sources {PRIMARY_SOURCES}: "
            f"{primary_mask.sum()}/{len(primary_mask)} compounds"
        )

    y_train = train_df[pec50_col].values

    out_dir = "models/lgbm_chemeleon" if args.chemeleon else "models/lgbm"
    run_name = "lgbm_chemeleon" if args.chemeleon else "lgbm_baseline"
    mask_path = f"data/features/lgbm{'_chemeleon' if args.chemeleon else ''}_selected_feature_mask.npy"
    sub_path  = f"submissions/phase1/lgbm{'_chemeleon' if args.chemeleon else '_baseline'}.csv"

    # Remove zero-variance and near-zero-variance features before training.
    # Fit selector on training data only; apply the same mask to test.
    # This does NOT modify the parquet files — just filters at train time.
    # Replace NaN and use .var() directly — sklearn's VarianceThreshold uses nanvar
    # which has a numerical precision bug on large float32 arrays in numpy 2.2.x.
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test  = np.nan_to_num(X_test,  nan=0.0, posinf=0.0, neginf=0.0)
    selected_mask = X_train.var(axis=0) > 0
    X_train = X_train[:, selected_mask]
    X_test  = X_test[:, selected_mask]
    np.save(mask_path, selected_mask)
    feature_names = [feature_names[i] for i in range(len(selected_mask)) if selected_mask[i]]
    logger.info(
        f"Features after variance filter: {selected_mask.sum()} / {len(selected_mask)} kept"
    )

    logger.info(f"LightGBM training: {X_train.shape[0]} compounds, {X_train.shape[1]} features")

    run = init_wandb_run(
        project="openadmet-pxr",
        name=run_name,
        config=config,
        tags=["lgbm", "phase1"],
    )

    boosters, oof_preds = train_lgbm_ensemble(
        feature_matrix=X_train,
        targets=y_train,
        fold_df=train_df,
        params=config["params"],
        feature_names=feature_names,
        n_folds=config["training"]["n_folds"],
        n_seeds=config["training"]["n_seeds"],
        output_dir=out_dir,
        run=run,
    )

    # Feature importance
    importance_df = get_lgbm_feature_importance(boosters, feature_names)
    importance_df.to_parquet(f"{out_dir}/feature_importance.parquet", index=False)
    logger.info(f"Top 10 features:\n{importance_df.head(10)[['feature', 'mean_importance']].to_string()}")

    # Test predictions
    test_preds = predict_lgbm_ensemble(boosters, X_test)
    np.save(f"{out_dir}/test_predictions.npy", test_preds)

    # Variance recalibration: if the model has compressed its output range
    # (pred_std << train_std), rescale predictions to match training distribution.
    # This is a post-hoc correction; ideally path_smooth+regularization reduce the
    # need for this. Check the scale_factor — if > 2.0, regularization needs revisiting.
    train_mean = float(y_train.mean())
    train_std  = float(y_train.std())
    pred_mean  = float(test_preds.mean())
    pred_std   = float(test_preds.std())
    logger.info(f"Test prediction distribution: mean={pred_mean:.3f}, std={pred_std:.3f}")
    logger.info(f"Training label distribution:  mean={train_mean:.3f}, std={train_std:.3f}")

    if pred_std > 0.05 and pred_std < train_std * 0.7:
        scale_factor = train_std / pred_std
        test_preds_cal = pred_mean + (test_preds - pred_mean) * scale_factor
        logger.info(
            f"Variance recalibration applied: scale_factor={scale_factor:.2f}, "
            f"new std={test_preds_cal.std():.3f}"
        )
        if scale_factor > 2.5:
            logger.warning(
                "scale_factor > 2.5 — variance compression is severe. "
                "Consider increasing lambda_l1/lambda_l2 or path_smooth further."
            )
        np.save(f"{out_dir}/test_predictions_calibrated.npy", test_preds_cal)
    else:
        test_preds_cal = test_preds
        logger.info("Variance recalibration skipped (pred_std within acceptable range of train_std)")

    # Format for submission — use calibrated predictions
    compound_id_col = "compound_id" if "compound_id" in test_df.columns else test_df.columns[0]
    sub = format_submission(
        test_df=test_df,
        predictions=test_preds_cal,
        compound_id_col=compound_id_col,
        output_path=sub_path,
    )

    if run:
        run.finish()

    logger.info("\nLightGBM training complete!")
    logger.info("Submission ready at: submissions/phase1/lgbm_baseline.csv")
    logger.info("SUBMISSION SCHEDULE: First submit by May 8 to validate pipeline")
    logger.info("Next: python scripts/08_build_mmps.py")


if __name__ == "__main__":
    main()
