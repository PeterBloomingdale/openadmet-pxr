"""
LightGBM on the optimal descriptor combo identified by Jeremy (rank-19 peer).

Why this feature set beats lgbm_chemeleon:
  lgbm_chemeleon uses all 8262 features = Morgan + Avalon + ErG + FCFP4 + MACCS +
  RDKit-2D + Mordred(536) + CheMeleon(2048). The Mordred features add noise — Jeremy's
  systematic study showed kitchen-sink combos (RMSE 0.577) are worse than the clean
  CheMeleon+ECFP4+rdkit2d combo (RMSE 0.531, his best LGB result from 53_descriptor_combo_study.py).

Feature set: CheMeleon(2048) + ECFP4/Morgan-r2(2048) + rdkit2d(~208) = ~4304 features.
  - CheMeleon: pretrained graph embeddings capturing PXR-relevant bioactivity landscape
  - ECFP4 (morgan_* columns, radius 2): local substructure patterns; ECFP4 > ECFP6 for PXR
  - rdkit2d (rdkit_* columns): physicochemical descriptors (logP, TPSA, HBA/HBD, MolWt, etc.)
  - Excluded: Mordred, Avalon, FCFP4, ErG, MACCS — add noise, increase collinearity

Prerequisites:
  scripts/04_build_features.py → data/features/train_features_all.npy
  scripts/05_build_cv_splits.py → data/splits/butina_folds.parquet
  scripts/16_extract_chemeleon_embeddings.py → data/features/train_chemeleon_emb.npy

Outputs:
  models/lgbm_optimal/oof_predictions.npy   (4135,) honest OOF
  models/lgbm_optimal/test_predictions.npy  (513,)
  models/lgbm_optimal/feature_importance.parquet

Next: add ("lgbm_optimal", "models/lgbm_optimal", ...) to scripts/11_ensemble.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.feature_selection import VarianceThreshold

from openadmet.models.lgbm_model import (
    get_lgbm_feature_importance,
    predict_lgbm_ensemble,
    train_lgbm_ensemble,
)
from openadmet.utils.tracking import init_wandb_run


def main() -> None:
    logger.info("=== LightGBM Optimal Features (CheMeleon + ECFP4 + rdkit2d) ===")

    out_dir = Path("models/lgbm_optimal")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load config (same hyperparameters as lgbm_chemeleon)
    with open("configs/lgbm.yaml") as f:
        config = yaml.safe_load(f)

    train_df = pd.read_parquet("data/splits/butina_folds.parquet")
    test_df = pd.read_parquet("data/raw/openadmet_test.parquet")

    pec50_col = "pec50_median" if "pec50_median" in train_df.columns else "pec50"
    PRIMARY_SOURCES = {"openadmet", "analog_set1", "htchem", "htchem_semi_pure"}
    primary_row_mask = None
    if "source" in train_df.columns:
        primary_row_mask = train_df["source"].isin(PRIMARY_SOURCES).values
        train_df = train_df[primary_row_mask].reset_index(drop=True)
        logger.info(f"Filtered to primary sources: {primary_row_mask.sum()} compounds")
    y_train = train_df[pec50_col].values

    # ─── Build optimal feature matrix ─────────────────────────────────────────
    with open("data/features/all_feature_names.json") as f:
        all_names = json.load(f)

    X_all = np.load("data/features/train_features_all.npy").astype(np.float32)
    X_all_test = np.load("data/features/test_features_all.npy").astype(np.float32)

    # Morgan radius-2 (ECFP4) bits: feature names start with "morgan_" but NOT "morgan_r3_"
    ecfp4_mask = np.array([n.startswith("morgan_") and "_r3_" not in n for n in all_names])
    # RDKit physicochemical descriptors
    rdkit_mask = np.array([n.startswith("rdkit_") for n in all_names])

    X_ecfp4 = X_all[:, ecfp4_mask] if primary_row_mask is None else X_all[primary_row_mask][:, ecfp4_mask]
    X_rdkit = X_all[:, rdkit_mask]  if primary_row_mask is None else X_all[primary_row_mask][:, rdkit_mask]
    X_ecfp4_test = X_all_test[:, ecfp4_mask]
    X_rdkit_test = X_all_test[:, rdkit_mask]

    logger.info(f"ECFP4 (morgan_r2) features: {ecfp4_mask.sum()}")
    logger.info(f"RDKit-2D features:           {rdkit_mask.sum()}")

    # CheMeleon embeddings
    chemeleon_path = Path("data/features/train_chemeleon_emb.npy")
    chemeleon_test_path = Path("data/features/test_chemeleon_emb.npy")
    if not chemeleon_path.exists():
        logger.error("CheMeleon embeddings not found. Run scripts/16_extract_chemeleon_embeddings.py first.")
        sys.exit(1)

    X_chem = np.load(chemeleon_path).astype(np.float32)
    X_chem_test = np.load(chemeleon_test_path).astype(np.float32)
    if primary_row_mask is not None:
        X_chem = X_chem[primary_row_mask]
    logger.info(f"CheMeleon embeddings:         {X_chem.shape[1]}")

    # Concatenate in Jeremy's order: CheMeleon + ECFP4 + rdkit2d
    X_train = np.concatenate([X_chem, X_ecfp4, X_rdkit], axis=1)
    X_test = np.concatenate([X_chem_test, X_ecfp4_test, X_rdkit_test], axis=1)

    chem_names = [f"chemeleon_{i}" for i in range(X_chem.shape[1])]
    ecfp4_names = [n for n, m in zip(all_names, ecfp4_mask) if m]
    rdkit_names = [n for n, m in zip(all_names, rdkit_mask) if m]
    feature_names = chem_names + ecfp4_names + rdkit_names

    logger.info(
        f"Total features before VarianceThreshold: {X_train.shape[1]} "
        f"(CheMeleon={X_chem.shape[1]}, ECFP4={ecfp4_mask.sum()}, rdkit2d={rdkit_mask.sum()})"
    )

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test  = np.nan_to_num(X_test,  nan=0.0, posinf=0.0, neginf=0.0)
    selected_mask = X_train.var(axis=0) > 0
    X_train = X_train[:, selected_mask]
    X_test  = X_test[:, selected_mask]
    np.save(out_dir / "feature_mask.npy", selected_mask)
    feature_names = [feature_names[i] for i in range(len(selected_mask)) if selected_mask[i]]
    logger.info(f"After variance filter: {selected_mask.sum()} / {len(selected_mask)} features kept")
    logger.info(f"Training matrix: {X_train.shape}")

    run = init_wandb_run(
        project="openadmet-pxr",
        name="lgbm_optimal",
        config=config,
        tags=["lgbm", "optimal", "phase1"],
    )

    boosters, oof_preds = train_lgbm_ensemble(
        feature_matrix=X_train,
        targets=y_train,
        fold_df=train_df,
        params=config["params"],
        feature_names=feature_names,
        n_folds=config["training"]["n_folds"],
        n_seeds=config["training"]["n_seeds"],
        output_dir=str(out_dir),
        run=run,
    )

    importance_df = get_lgbm_feature_importance(boosters, feature_names)
    importance_df.to_parquet(out_dir / "feature_importance.parquet", index=False)
    logger.info(f"Top 10 features:\n{importance_df.head(10)[['feature', 'mean_importance']].to_string()}")

    test_preds = predict_lgbm_ensemble(boosters, X_test)
    np.save(out_dir / "test_predictions.npy", test_preds)

    if run:
        run.finish()

    logger.info(f"\n=== lgbm_optimal complete ===")
    logger.info(f"  OOF predictions: {out_dir}/oof_predictions.npy")
    logger.info(f"  Test predictions: {out_dir}/test_predictions.npy")
    logger.info(
        "\nNext: add to scripts/11_ensemble.py:\n"
        '  ("lgbm_optimal", "models/lgbm_optimal", "oof_predictions.npy", "test_predictions.npy"),'
    )


if __name__ == "__main__":
    main()
