"""
TabICL on CheMeleon + frozen UniMol(3D) + ECFP4 + rdkit2d → PCA-256.

Why add frozen UniMol embeddings to TabICL:
  Our existing TabICL (script 26) uses CheMeleon + ECFP4 + rdkit2d — purely 2D features.
  Frozen UniMol embeddings (512-d) encode 3D conformer geometry and atomic distances from
  a model pretrained on 209M drug-like conformers. Adding them to the feature set gives
  TabICL's attention mechanism access to 3D structural patterns that 2D fingerprints miss.

  Importantly, this model is *different* from our fine-tuned UniMol2 (script 24), which
  adapts UniMol's weights to PXR pEC50. Here the UniMol weights are frozen and the 512-d
  mean-pooled 3D embeddings are used as additional tabular input features.

Feature pipeline: CheMeleon(2048) + UniMol(512) + ECFP4(2048) + rdkit2d(208) = 4820-d
  → VarianceThreshold(0.01) → PCA-256 (99.9%+ variance retained).

CV strategy: honest 5-fold Butina OOF — same as all other models.

Prerequisites:
  scripts/16_extract_chemeleon_embeddings.py  → data/features/train_chemeleon_emb.npy
  scripts/17_extract_unimol_embeddings.py     → data/features/train_unimol_emb.npy
  scripts/04_build_features.py               → data/features/train_features_all.npy

Outputs:
  models/tabicl_3d/oof_predictions.npy   (4135,)
  models/tabicl_3d/test_predictions.npy  (513,)
  models/tabicl_3d/metrics.json

Next: add ("tabicl_3d", "models/tabicl_3d", ...) to scripts/11_ensemble.py
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


N_PCA = 256
N_ESTIMATORS = 8


def build_features() -> tuple[np.ndarray, np.ndarray]:
    """Load CheMeleon + frozen UniMol + ECFP4 + rdkit2d, reduce to PCA-N_PCA."""
    logger.info("Loading features: CheMeleon + UniMol(3D) + ECFP4 + rdkit2d")

    train_emb = np.load("data/features/train_chemeleon_emb.npy").astype(np.float32)
    test_emb = np.load("data/features/test_chemeleon_emb.npy").astype(np.float32)

    train_unimol = np.load("data/features/train_unimol_emb.npy").astype(np.float32)
    test_unimol = np.load("data/features/test_unimol_emb.npy").astype(np.float32)

    with open("data/features/all_feature_names.json") as f:
        all_names = json.load(f)
    X_all = np.load("data/features/train_features_all.npy").astype(np.float32)
    X_all_test = np.load("data/features/test_features_all.npy").astype(np.float32)

    ecfp4_mask = np.array([n.startswith("morgan_") and "_r3_" not in n for n in all_names])
    rdkit_mask = np.array([n.startswith("rdkit_") for n in all_names])

    # Concatenate: CheMeleon + frozen UniMol 3D + ECFP4 + rdkit2d
    X_train_raw = np.concatenate(
        [train_emb, train_unimol, X_all[:, ecfp4_mask], X_all[:, rdkit_mask]], axis=1
    )
    X_test_raw = np.concatenate(
        [test_emb, test_unimol, X_all_test[:, ecfp4_mask], X_all_test[:, rdkit_mask]], axis=1
    )

    logger.info(
        f"Raw features: CheMeleon={train_emb.shape[1]}, UniMol={train_unimol.shape[1]}, "
        f"ECFP4={ecfp4_mask.sum()}, rdkit2d={rdkit_mask.sum()} → {X_train_raw.shape[1]}d total"
    )

    # Fit VT and PCA on train only to prevent data leakage into test
    X_train_raw = np.nan_to_num(X_train_raw, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_raw  = np.nan_to_num(X_test_raw,  nan=0.0, posinf=0.0, neginf=0.0)
    keep_mask = X_train_raw.var(axis=0) > 0
    X_train_vt = X_train_raw[:, keep_mask]
    X_test_vt  = X_test_raw[:, keep_mask]

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_vt = scaler.fit_transform(X_train_vt.astype(np.float64)).astype(np.float32)
    X_test_vt  = scaler.transform(X_test_vt.astype(np.float64)).astype(np.float32)

    pca = PCA(n_components=N_PCA, random_state=42)
    X_train_pca = pca.fit_transform(X_train_vt).astype(np.float32)
    X_test_pca = pca.transform(X_test_vt).astype(np.float32)

    var_explained = pca.explained_variance_ratio_.sum()
    logger.info(
        f"After VT + PCA-{N_PCA}: {X_train_pca.shape[1]}d, "
        f"explained variance = {var_explained:.2%}"
    )

    return X_train_pca, X_test_pca


def main() -> None:
    logger.info("=== TabICL-3D (CheMeleon + UniMol + ECFP4 + rdkit2d → PCA-256) ===")
    logger.info(f"TabICL n_estimators={N_ESTIMATORS}, PCA={N_PCA}")

    out_dir = Path("models/tabicl_3d")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet("data/splits/butina_folds.parquet")
    pec50_col = "pec50_median" if "pec50_median" in train_df.columns else "pec50"
    y_train = train_df[pec50_col].values.astype(np.float32)
    folds = train_df["fold"].values

    for path in ["data/features/train_chemeleon_emb.npy", "data/features/train_unimol_emb.npy"]:
        if not Path(path).exists():
            logger.error(f"Missing: {path}")
            sys.exit(1)

    X_train, X_test = build_features()

    # ─── Honest 5-fold Butina OOF ────────────────────────────────────────────
    oof = np.zeros(len(y_train), dtype=np.float32)
    unique_folds = sorted(np.unique(folds))

    for fold_id in unique_folds:
        val_mask = folds == fold_id
        train_mask = ~val_mask
        logger.info(f"Fold {fold_id}: {train_mask.sum()} train, {val_mask.sum()} val")

        model = TabICLRegressor(
            n_estimators=N_ESTIMATORS,
            random_state=42,
            verbose=False,
        )
        model.fit(X_train[train_mask], y_train[train_mask])
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
    logger.info("\n--- Final model (all training data → test predictions) ---")
    model_final = TabICLRegressor(n_estimators=N_ESTIMATORS, random_state=42, verbose=False)
    model_final.fit(X_train, y_train)
    test_preds = model_final.predict(X_test).astype(np.float32)
    logger.info(f"Test predictions: mean={test_preds.mean():.4f}, std={test_preds.std():.4f}")

    # ─── Save outputs ─────────────────────────────────────────────────────────
    np.save(out_dir / "oof_predictions.npy", oof)
    np.save(out_dir / "test_predictions.npy", test_preds)

    saved_metrics = {k: float(v) for k, v in metrics.items() if not isinstance(v, dict)}
    saved_metrics["spearman"] = float(spearman)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(saved_metrics, f, indent=2)

    logger.info(f"\n=== TabICL-3D complete ===")
    logger.info(f"  OOF MAE={metrics['mae']:.4f}, Spearman={spearman:.4f}")
    logger.info(f"  Outputs: {out_dir}/")
    logger.info(
        "\nNext: add to scripts/11_ensemble.py:\n"
        '  ("tabicl_3d", "models/tabicl_3d", "oof_predictions.npy", "test_predictions.npy"),'
    )


if __name__ == "__main__":
    main()
