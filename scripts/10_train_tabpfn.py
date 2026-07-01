"""
Generate TabPFN v2 OOF and test predictions for ensemble stacking.

Feature strategy:
  When CheMeleon embeddings are available (scripts/16_extract_chemeleon_embeddings.py),
  concatenate them with RDKit-2D descriptors before PCA reduction. This matches the
  rank-42 team's approach: CheMeleon + RDKit-2D → PCA(100) → TabPFN in-context learning.
  Falls back to the full tabular feature matrix (train_features_all.npy) when embeddings
  are not yet generated.

Prerequisites:
  - scripts/04_build_features.py → data/features/train_features_all.npy, test_features_all.npy
  - scripts/05_build_cv_splits.py → data/splits/butina_folds.parquet
  - scripts/16_extract_chemeleon_embeddings.py (optional but strongly recommended)
    → data/features/train_chemeleon_emb.npy, test_chemeleon_emb.npy

No gradient training — TabPFN is in-context only; GPU helps inference.

Outputs:
  - models/tabpfn/oof_predictions.npy
  - models/tabpfn/test_predictions.npy

Next: python scripts/11_ensemble.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import numpy as np
import pandas as pd
from loguru import logger

from openadmet.cv.oof import evaluate_oof
from openadmet.models.tabpfn_model import (
    TABPFN_AVAILABLE,
    TABPFN_MAX_TRAIN,
    TabpfnFeatureReducer,
    predict_tabpfn,
    subsample_diverse,
    train_tabpfn,
)


def _load_tabpfn_features(
    train_df: pd.DataFrame,
    row_mask=None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Load and assemble features for TabPFN.

    Priority:
    1. CheMeleon embeddings + RDKit-2D descriptors (when available).
       CheMeleon captures pretrained chemical context; RDKit-2D captures physicochemical
       properties relevant to PXR (logP, TPSA, HBD/HBA). This combination is what
       the rank-42 team used for their TabPFN base learner.
    2. Full tabular feature matrix (fallback when CheMeleon not yet extracted).

    Returns (X_train, X_test, feature_source_description).
    """
    chemeleon_train = Path("data/features/train_chemeleon_emb.npy")
    chemeleon_test = Path("data/features/test_chemeleon_emb.npy")
    desc_names_path = Path("data/features/desc_names.json")
    all_names_path = Path("data/features/all_feature_names.json")

    if chemeleon_train.exists() and chemeleon_test.exists():
        logger.info("CheMeleon embeddings found — using CheMeleon + ECFP4 + RDKit-2D for TabPFN")
        train_emb = np.load(chemeleon_train).astype(np.float32)
        if row_mask is not None:
            train_emb = train_emb[row_mask]
        test_emb = np.load(chemeleon_test).astype(np.float32)

        # Extract ECFP4 (morgan_*) + RDKit-2D (rdkit_*) from the all-features matrix.
        # Feature names: morgan_0..2047 = ECFP4 (Morgan r=2, 2048 bits), then Avalon, etc.,
        # then rdkit_* at the end. This mirrors Jeremy (rank-19)'s optimal LGB combo:
        # CheMeleon(2048) + ECFP4(2048) + rdkit2d(217) = 4313-d → PCA-256.
        if all_names_path.exists():
            with open(all_names_path) as f:
                all_names = json.load(f)
            ecfp4_mask = np.array([n.startswith("morgan_") and "_r3_" not in n for n in all_names])
            rdkit_mask = np.array([n.startswith("rdkit_") for n in all_names])
            X_all = np.load("data/features/train_features_all.npy").astype(np.float32)
            if row_mask is not None:
                X_all = X_all[row_mask]
            X_all_test = np.load("data/features/test_features_all.npy").astype(np.float32)
            train_ecfp4 = X_all[:, ecfp4_mask]
            test_ecfp4 = X_all_test[:, ecfp4_mask]
            train_rdkit = X_all[:, rdkit_mask]
            test_rdkit = X_all_test[:, rdkit_mask]
            logger.info(
                f"ECFP4 (morgan_*): {ecfp4_mask.sum()}d, RDKit-2D: {rdkit_mask.sum()}d"
            )
        elif desc_names_path.exists():
            # Fallback: descriptor matrix only (no morgan bits)
            with open(desc_names_path) as f:
                desc_names = json.load(f)
            rdkit_mask = np.array([n.startswith("rdkit_") for n in desc_names])
            train_desc_full = np.load("data/features/train_descriptors.npy").astype(np.float32)
            test_desc_full = np.load("data/features/test_descriptors.npy").astype(np.float32)
            train_ecfp4 = np.zeros((len(train_emb), 0), dtype=np.float32)
            test_ecfp4 = np.zeros((len(test_emb), 0), dtype=np.float32)
            train_rdkit = train_desc_full[:, rdkit_mask]
            test_rdkit = test_desc_full[:, rdkit_mask]
            logger.info(f"RDKit-2D only (fallback): {rdkit_mask.sum()}d (no ECFP4)")
        else:
            logger.warning("No feature name files found — using CheMeleon embeddings alone")
            train_ecfp4 = np.zeros((len(train_emb), 0), dtype=np.float32)
            test_ecfp4 = np.zeros((len(test_emb), 0), dtype=np.float32)
            train_rdkit = np.zeros((len(train_emb), 0), dtype=np.float32)
            test_rdkit = np.zeros((len(test_emb), 0), dtype=np.float32)

        parts_train = [p for p in [train_emb, train_ecfp4, train_rdkit] if p.shape[1] > 0]
        parts_test = [p for p in [test_emb, test_ecfp4, test_rdkit] if p.shape[1] > 0]
        X_train = np.concatenate(parts_train, axis=1)
        X_test = np.concatenate(parts_test, axis=1)

        src = (
            f"CheMeleon({train_emb.shape[1]}d)"
            + (f" + ECFP4({train_ecfp4.shape[1]}d)" if train_ecfp4.shape[1] > 0 else "")
            + (f" + RDKit-2D({train_rdkit.shape[1]}d)" if train_rdkit.shape[1] > 0 else "")
        )
    else:
        logger.info(
            "CheMeleon embeddings not found — falling back to full tabular features. "
            "Run scripts/16_extract_chemeleon_embeddings.py first for better performance."
        )
        X_train = np.load("data/features/train_features_all.npy").astype(np.float32)
        X_test = np.load("data/features/test_features_all.npy").astype(np.float32)
        src = "tabular features (fallback)"

    if len(X_train) != len(train_df):
        raise ValueError(
            f"Feature matrix rows ({len(X_train)}) != training rows ({len(train_df)}). "
            "Re-run scripts/04_build_features.py."
        )

    return X_train, X_test, src


def main() -> None:
    if not TABPFN_AVAILABLE:
        logger.error("TabPFN not installed. pip install tabpfn>=7.0.0")
        sys.exit(1)

    train_df_full = pd.read_parquet("data/splits/butina_folds.parquet")
    pec50_col = "pec50_median" if "pec50_median" in train_df_full.columns else "pec50"
    PRIMARY_SOURCES = {"openadmet", "analog_set1", "htchem", "htchem_semi_pure"}
    if "source" in train_df_full.columns:
        src_mask = train_df_full["source"].isin(PRIMARY_SOURCES).values
    else:
        src_mask = np.ones(len(train_df_full), dtype=bool)
    train_df_prim = train_df_full[src_mask]
    active_rel = train_df_prim[pec50_col].notna().values
    combined_mask = src_mask.copy(); combined_mask[src_mask] = active_rel
    train_df = train_df_full[combined_mask].reset_index(drop=True)
    logger.info(f"Primary + active compounds: {len(train_df)}")

    X_train, X_test, feature_src = _load_tabpfn_features(train_df, row_mask=combined_mask)
    logger.info(f"TabPFN feature source: {feature_src}  shape={X_train.shape}")

    y = train_df[pec50_col].values.astype(np.float64)
    smiles_col = "smiles_std" if "smiles_std" in train_df.columns else "smiles"
    folds = train_df["fold"].values

    oof = np.full(len(train_df), np.nan, dtype=np.float64)
    unique_folds = sorted(np.unique(folds))

    for f in unique_folds:
        train_mask = folds != f
        val_mask = folds == f
        if val_mask.sum() == 0:
            continue

        red = TabpfnFeatureReducer().fit(X_train[train_mask])
        X_tr = red.transform(X_train[train_mask])
        X_va = red.transform(X_train[val_mask])

        y_tr = y[train_mask]
        smiles_tr = train_df.loc[train_mask, smiles_col].astype(str).tolist()

        try:
            # Use all training data up to TABPFN_MAX_TRAIN — TabPFN v2 handles large contexts.
            # subsample_diverse returns all indices when n >= len(smiles_tr), so the MaxMin
            # O(n²) loop is skipped entirely when the fold fits within the limit.
            div_ix = subsample_diverse(smiles_tr, n=min(TABPFN_MAX_TRAIN, len(smiles_tr)))
            model = train_tabpfn(X_tr[div_ix], y_tr[div_ix])
            oof[val_mask] = predict_tabpfn(model, X_va)
        except Exception as e:
            logger.error(f"TabPFN fold {f} failed: {e}")
            oof[val_mask] = np.nanmean(y_tr)

    # Final test predictions: reducer fit on full train; diverse context for TabPFN
    red_full = TabpfnFeatureReducer().fit(X_train)
    X_tr_full = red_full.transform(X_train)
    X_te = red_full.transform(X_test)

    smiles_all = train_df[smiles_col].astype(str).tolist()
    div_ix = subsample_diverse(smiles_all, n=min(TABPFN_MAX_TRAIN, len(smiles_all)))
    try:
        model_full = train_tabpfn(X_tr_full[div_ix], y[div_ix])
        test_preds = predict_tabpfn(model_full, X_te).astype(np.float64)
    except Exception as e:
        logger.error(f"TabPFN full-train fit failed: {e}")
        test_preds = np.full(len(X_test), np.nanmean(y), dtype=np.float64)

    out = Path("models/tabpfn")
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "oof_predictions.npy", oof)
    np.save(out / "test_predictions.npy", test_preds)

    valid = ~np.isnan(oof)
    metrics = evaluate_oof(y[valid], oof[valid], folds[valid])
    logger.info(
        f"\nTabPFN OOF: MAE={metrics['mae']:.4f}, RAE={metrics['rae']:.4f}, R²={metrics['r2']:.4f}"
    )
    logger.info(f"Test predictions: mean={test_preds.mean():.3f}, std={test_preds.std():.3f}")
    logger.info("Saved models/tabpfn/oof_predictions.npy and test_predictions.npy")
    logger.info("Next: python scripts/11_ensemble.py")


if __name__ == "__main__":
    main()
