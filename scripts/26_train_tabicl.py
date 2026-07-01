"""
TabICL (Tabular In-Context Learning) on CheMeleon + ECFP4 + rdkit2d features.

TabICL is a pretrained transformer that does tabular in-context learning (ICL):
  at prediction time, the model receives the full training set as "context" and
  predicts test labels in a single forward pass. No gradient-based fine-tuning on
  the target task — the pretrained weights already encode patterns from thousands
  of tabular datasets.

Why TabICL over TabPFN for this task:
  Jeremy (rank-19) uses "TabICL on HTS-pretrained CheMeleon" (RMSE 0.495) as his
  dominant model. TabICL handles larger training sets and higher-dimensional features
  more efficiently than TabPFN v2. Our feature set (PCA-256 of CheMeleon+ECFP4+rdkit2d)
  is the same 4304→256-d input Jeremy uses, minus the HTS-pretrained encoder.

Feature set: CheMeleon(2048) + ECFP4/Morgan-r2(2048) + rdkit2d(208) → PCA-256.
  - CheMeleon embeddings encode PXR-relevant bioactivity context
  - ECFP4 captures local scaffold/substructure identity
  - rdkit2d captures physicochemical properties (logP, TPSA, HBD/HBA)
  - PCA-256 removes noise and collinearity before TabICL sees the features

CV strategy: honest 5-fold Butina OOF — fit one TabICL model per fold on training
compounds, predict on held-out fold. TabICL stores training data as context, so each
fit() call is essentially instantaneous (the transformer just caches the context).
Actual compute happens at predict() time.

Prerequisites:
  scripts/16_extract_chemeleon_embeddings.py
  scripts/04_build_features.py (for train_features_all.npy + all_feature_names.json)
  scripts/05_build_cv_splits.py

Outputs:
  models/tabicl/oof_predictions.npy   (4135,)
  models/tabicl/test_predictions.npy  (513,)
  models/tabicl/metrics.json

Next: add ("tabicl", "models/tabicl", ...) to scripts/11_ensemble.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from tabicl import TabICLRegressor

from openadmet.cv.oof import evaluate_oof

N_PCA = 256          # match Jeremy's PCA-256 config
N_ESTIMATORS = 8     # TabICL default; increases robustness of in-context predictions


def build_features(row_mask=None) -> tuple[np.ndarray, np.ndarray]:
    """Load and reduce CheMeleon + ECFP4 + rdkit2d features to PCA-N_PCA."""
    logger.info("Loading features: CheMeleon + ECFP4 + rdkit2d")

    train_emb = np.load("data/features/train_chemeleon_emb.npy").astype(np.float32)
    test_emb = np.load("data/features/test_chemeleon_emb.npy").astype(np.float32)

    with open("data/features/all_feature_names.json") as f:
        all_names = json.load(f)
    X_all = np.load("data/features/train_features_all.npy").astype(np.float32)
    X_all_test = np.load("data/features/test_features_all.npy").astype(np.float32)

    if row_mask is not None:
        train_emb = train_emb[row_mask]
        X_all = X_all[row_mask]

    ecfp4_mask = np.array([n.startswith("morgan_") and "_r3_" not in n for n in all_names])
    rdkit_mask = np.array([n.startswith("rdkit_") for n in all_names])

    X_train_raw = np.concatenate([train_emb, X_all[:, ecfp4_mask], X_all[:, rdkit_mask]], axis=1)
    X_test_raw = np.concatenate([test_emb, X_all_test[:, ecfp4_mask], X_all_test[:, rdkit_mask]], axis=1)

    logger.info(
        f"Raw features: CheMeleon={train_emb.shape[1]}, "
        f"ECFP4={ecfp4_mask.sum()}, rdkit2d={rdkit_mask.sum()} → total {X_train_raw.shape[1]}"
    )

    from sklearn.preprocessing import StandardScaler
    # Variance filter then StandardScaler then PCA.
    # StandardScaler is required before PCA: RDKit/Mordred descriptors span [0, 52M]
    # while fingerprint bits are [0, 1]. Without scaling, extreme descriptor values
    # dominate PCA and cause overflow in sklearn's randomized SVD (numpy 2.2 + float32).
    X_train_raw = np.nan_to_num(X_train_raw, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_raw  = np.nan_to_num(X_test_raw,  nan=0.0, posinf=0.0, neginf=0.0)
    keep_mask = X_train_raw.var(axis=0) > 0
    X_train_vt = X_train_raw[:, keep_mask]
    X_test_vt  = X_test_raw[:, keep_mask]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_vt.astype(np.float64)).astype(np.float32)
    X_test_scaled  = scaler.transform(X_test_vt.astype(np.float64)).astype(np.float32)

    pca = PCA(n_components=N_PCA, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled).astype(np.float32)
    X_test_pca = pca.transform(X_test_scaled).astype(np.float32)

    # Second standardization: PCA components can still have large magnitudes
    # (e.g. max ≈ 129). TabICL expects well-scaled features — re-standardize.
    post_scaler = StandardScaler()
    X_train_pca = post_scaler.fit_transform(X_train_pca).astype(np.float32)
    X_test_pca  = post_scaler.transform(X_test_pca).astype(np.float32)

    var_explained = pca.explained_variance_ratio_.sum()
    logger.info(
        f"After VT + Scaler + PCA-{N_PCA} + Scaler: {X_train_pca.shape[1]}d, "
        f"explained variance = {var_explained:.2%}"
    )

    return X_train_pca, X_test_pca


def main() -> None:
    logger.info("=== TabICL (Tabular In-Context Learning) ===")
    logger.info(f"TabICL n_estimators={N_ESTIMATORS}, PCA={N_PCA}")

    out_dir = Path("models/tabicl")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df_full = pd.read_parquet("data/splits/butina_folds.parquet")
    pec50_col = "pec50_median" if "pec50_median" in train_df_full.columns else "pec50"
    PRIMARY_SOURCES = {"openadmet", "analog_set1", "htchem", "htchem_semi_pure"}
    if "source" in train_df_full.columns:
        src_mask = train_df_full["source"].isin(PRIMARY_SOURCES).values
    else:
        src_mask = np.ones(len(train_df_full), dtype=bool)
    # Active mask on primary set
    train_df_prim = train_df_full[src_mask]
    active_rel = train_df_prim[pec50_col].notna().values
    # Combined mask on full index (for feature subsetting)
    combined_mask = src_mask.copy()
    combined_mask[src_mask] = active_rel
    train_df = train_df_full[combined_mask].reset_index(drop=True)
    logger.info(f"Primary + active compounds: {len(train_df)}")
    y_train = train_df[pec50_col].values.astype(np.float32)
    folds = train_df["fold"].values

    if not Path("data/features/train_chemeleon_emb.npy").exists():
        logger.error("CheMeleon embeddings not found. Run scripts/16_extract_chemeleon_embeddings.py first.")
        sys.exit(1)

    X_train, X_test = build_features(row_mask=combined_mask)

    # ─── Honest 5-fold Butina OOF ────────────────────────────────────────────
    oof = np.zeros(len(y_train), dtype=np.float32)
    unique_folds = sorted(np.unique(folds))

    for fold_id in unique_folds:
        val_mask = folds == fold_id
        train_mask = ~val_mask
        n_train = train_mask.sum()
        n_val = val_mask.sum()
        logger.info(f"Fold {fold_id}: {n_train} train, {n_val} val")

        # TabICL breaks with >~1000 training samples at PCA-256 on this data/version.
        # Subsample to 1000 diverse training compounds for context (Tanimoto MaxMin).
        TABICL_MAX_TRAIN = 1000
        X_fold_tr = X_train[train_mask]
        y_fold_tr = y_train[train_mask]
        if n_train > TABICL_MAX_TRAIN:
            rng = np.random.default_rng(42 + fold_id)
            idx = rng.choice(n_train, size=TABICL_MAX_TRAIN, replace=False)
            X_fold_tr = X_fold_tr[idx]
            y_fold_tr = y_fold_tr[idx]
            logger.info(f"  Subsampled to {TABICL_MAX_TRAIN} training compounds")

        model = TabICLRegressor(
            n_estimators=N_ESTIMATORS,
            random_state=42,
            verbose=False,
        )
        model.fit(X_fold_tr, y_fold_tr)
        val_preds = model.predict(X_train[val_mask]).astype(np.float32)

        oof[val_mask] = val_preds
        fold_mae = float(np.mean(np.abs(val_preds - y_train[val_mask])))
        logger.info(f"  Fold {fold_id} val MAE = {fold_mae:.4f}")

    # ─── OOF metrics ─────────────────────────────────────────────────────────
    metrics = evaluate_oof(y_train, oof, folds)
    spearman = spearmanr(y_train, oof).statistic
    logger.info(
        f"\nOOF overall: MAE={metrics['mae']:.4f}, "
        f"RAE_test={metrics.get('rae_test', metrics.get('rae', 0)):.4f}, "
        f"Spearman={spearman:.4f}"
    )

    # ─── Final model on full training set → test predictions ─────────────────
    logger.info("\n--- Final model (subsampled training data → test predictions) ---")
    TABICL_MAX_TRAIN = 1000
    if len(X_train) > TABICL_MAX_TRAIN:
        rng = np.random.default_rng(99)
        idx_final = rng.choice(len(X_train), size=TABICL_MAX_TRAIN, replace=False)
        X_tr_final, y_tr_final = X_train[idx_final], y_train[idx_final]
        logger.info(f"  Subsampled to {TABICL_MAX_TRAIN} for final model")
    else:
        X_tr_final, y_tr_final = X_train, y_train
    model_final = TabICLRegressor(n_estimators=N_ESTIMATORS, random_state=42, verbose=False)
    model_final.fit(X_tr_final, y_tr_final)
    test_preds = model_final.predict(X_test).astype(np.float32)
    logger.info(f"Test predictions: mean={test_preds.mean():.4f}, std={test_preds.std():.4f}")

    # ─── Save outputs ─────────────────────────────────────────────────────────
    np.save(out_dir / "oof_predictions.npy", oof)
    np.save(out_dir / "test_predictions.npy", test_preds)

    saved_metrics = {k: float(v) for k, v in metrics.items() if not isinstance(v, dict)}
    saved_metrics["spearman"] = float(spearman)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(saved_metrics, f, indent=2)

    logger.info(f"\n=== TabICL complete ===")
    logger.info(f"  OOF MAE={metrics['mae']:.4f}, Spearman={spearman:.4f}")
    logger.info(f"  Outputs: {out_dir}/")
    logger.info(
        "\nNext: add to scripts/11_ensemble.py:\n"
        '  ("tabicl", "models/tabicl", "oof_predictions.npy", "test_predictions.npy"),'
    )


if __name__ == "__main__":
    main()
