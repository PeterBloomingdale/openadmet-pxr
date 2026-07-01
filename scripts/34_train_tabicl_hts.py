"""
TabICL on HTS-pretrained Chemprop embeddings + CheMeleon + ECFP4 + rdkit2d → PCA-256.

Why this may outperform tabicl_chemprop (script 31):
  Script 31 used PXR-fine-tuned Chemprop embeddings (task-specific, but trained on only
  4,135 compounds). Those embeddings added zero diversity over base tabicl — the PXR
  fine-tuned backbone learned very similar features to CheMeleon+ECFP4.

  This script uses the HTS-pretrained Chemprop backbone (trained on ~10,000 Tox21 +
  NCATS PXR HTS compounds, never fine-tuned on OpenADMET). The HTS backbone has
  seen broader chemical diversity and may encode PXR "bioactivity fingerprints" that
  differ meaningfully from CheMeleon's ChEMBL-based pretraining.

  This directly replicates the approach of the rank-19 team (HTS-pretrained encoder →
  TabICL features), using our Chemprop backbone as the encoder.

Feature pipeline:
  CheMeleon(2048) + HTS_emb(300) + ECFP4(2048) + rdkit2d(208) = 4604-d
  → VarianceThreshold(0.01) → PCA-256

Diversity gate: After training, compute pearsonr(oof_tabicl_hts, oof_tabicl).
  If r < 0.95 → add to ensemble. If r ≥ 0.95 → same information, skip.

Outputs:
  models/tabicl_hts/oof_predictions.npy  (4135,)
  models/tabicl_hts/test_predictions.npy (513,)
  models/tabicl_hts/metrics.json

Prerequisites: scripts/33_extract_chemprop_hts_pretrained_emb.py must have completed.
"""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from tabicl import TabICLRegressor

from openadmet.cv.oof import evaluate_oof


N_PCA = 256
N_ESTIMATORS = 8


def build_features() -> tuple[np.ndarray, np.ndarray]:
    """Load CheMeleon + HTS_emb + ECFP4 + rdkit2d, reduce to PCA-N_PCA."""
    logger.info("Loading features: CheMeleon + HTS_emb + ECFP4 + rdkit2d")

    train_emb = np.load("data/features/train_chemeleon_emb.npy").astype(np.float32)
    test_emb = np.load("data/features/test_chemeleon_emb.npy").astype(np.float32)

    # HTS-pretrained Chemprop embeddings (no PXR fine-tuning, no leakage)
    train_hts = np.load("data/features/train_chemprop_hts_emb.npy").astype(np.float32)
    test_hts = np.load("data/features/test_chemprop_hts_emb.npy").astype(np.float32)

    with open("data/features/all_feature_names.json") as f:
        all_names = json.load(f)
    X_all = np.load("data/features/train_features_all.npy").astype(np.float32)
    X_all_test = np.load("data/features/test_features_all.npy").astype(np.float32)

    ecfp4_mask = np.array([n.startswith("morgan_") and "_r3_" not in n for n in all_names])
    rdkit_mask = np.array([n.startswith("rdkit_") for n in all_names])

    X_train_raw = np.concatenate(
        [train_emb, train_hts, X_all[:, ecfp4_mask], X_all[:, rdkit_mask]], axis=1
    )
    X_test_raw = np.concatenate(
        [test_emb, test_hts, X_all_test[:, ecfp4_mask], X_all_test[:, rdkit_mask]], axis=1
    )

    logger.info(
        f"Raw features: CheMeleon={train_emb.shape[1]}, HTS_emb={train_hts.shape[1]}, "
        f"ECFP4={ecfp4_mask.sum()}, rdkit2d={rdkit_mask.sum()} → {X_train_raw.shape[1]}d total"
    )

    vt = VarianceThreshold(threshold=0.01)
    X_train_vt = vt.fit_transform(X_train_raw)
    X_test_vt = vt.transform(X_test_raw)

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
    logger.info("=== TabICL-HTS (CheMeleon + HTS_emb + ECFP4 + rdkit2d → PCA-256) ===")

    out_dir = Path("models/tabicl_hts")
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in ["data/features/train_chemeleon_emb.npy", "data/features/train_chemprop_hts_emb.npy"]:
        if not Path(path).exists():
            logger.error(f"Missing: {path}")
            raise FileNotFoundError(path)

    train_df = pd.read_parquet("data/splits/butina_folds.parquet")
    pec50_col = "pec50_median" if "pec50_median" in train_df.columns else "pec50"
    y_train = train_df[pec50_col].values.astype(np.float32)
    folds = train_df["fold"].values

    X_train, X_test = build_features()

    oof = np.zeros(len(y_train), dtype=np.float32)
    unique_folds = sorted(np.unique(folds))

    for fold_id in unique_folds:
        val_mask = folds == fold_id
        train_mask = ~val_mask
        logger.info(f"Fold {fold_id}: {train_mask.sum()} train, {val_mask.sum()} val")

        model = TabICLRegressor(n_estimators=N_ESTIMATORS, random_state=42, verbose=False)
        model.fit(X_train[train_mask], y_train[train_mask])
        val_preds = model.predict(X_train[val_mask]).astype(np.float32)

        oof[val_mask] = val_preds
        fold_mae = float(np.mean(np.abs(val_preds - y_train[val_mask])))
        logger.info(f"  Fold {fold_id} val MAE = {fold_mae:.4f}")

    metrics = evaluate_oof(y_train, oof, folds)
    spearman = spearmanr(y_train, oof).statistic
    logger.info(
        f"\nOOF overall: MAE={metrics['mae']:.4f}, "
        f"RAE_test={metrics.get('rae_test', metrics.get('rae', 0)):.4f}, "
        f"Spearman={spearman:.4f}"
    )

    # Diversity check vs base tabicl
    for tag, path in [
        ("tabicl",         "models/tabicl/oof_predictions.npy"),
        ("tabicl_chemprop","models/tabicl_chemprop/oof_predictions.npy"),
    ]:
        p = Path(path)
        if p.exists():
            r = pearsonr(oof, np.load(p)).statistic
            logger.info(f"Pearson r(tabicl_hts, {tag}) = {r:.4f}  (want < 0.95 for genuine diversity)")

    logger.info("\n--- Final model (all training data → test predictions) ---")
    model_final = TabICLRegressor(n_estimators=N_ESTIMATORS, random_state=42, verbose=False)
    model_final.fit(X_train, y_train)
    test_preds = model_final.predict(X_test).astype(np.float32)
    logger.info(f"Test predictions: mean={test_preds.mean():.4f}, std={test_preds.std():.4f}")

    np.save(out_dir / "oof_predictions.npy", oof)
    np.save(out_dir / "test_predictions.npy", test_preds)

    saved_metrics = {k: float(v) for k, v in metrics.items() if not isinstance(v, dict)}
    saved_metrics["spearman"] = float(spearman)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(saved_metrics, f, indent=2)

    logger.info(f"\n=== tabicl_hts complete ===")
    logger.info(f"  OOF MAE={metrics['mae']:.4f}, Spearman={spearman:.4f}")
    logger.info(f"  Outputs: {out_dir}/")
    logger.info(
        "\nIf r(tabicl_hts, tabicl) < 0.95, add to scripts/11_ensemble.py:\n"
        '  ("tabicl_hts", "models/tabicl_hts", "oof_predictions.npy", "test_predictions.npy"),'
    )


if __name__ == "__main__":
    main()
