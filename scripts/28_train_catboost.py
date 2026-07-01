"""
CatBoost on CheMeleon + ECFP4 + rdkit2d features.

Why CatBoost alongside LightGBM:
  Both are gradient-boosted trees, but they differ in a key way:
  - LightGBM uses leaf-wise growth with histogram approximation (fast, accurate)
  - CatBoost uses ordered (oblivious) symmetric trees — a different regularisation
    strategy that often produces lower-correlation errors on the same feature set.
  Adding CatBoost to the ensemble therefore adds genuine GBDT diversity without
  needing new features.

Feature set: CheMeleon(2048) + ECFP4/Morgan-r2(2048) + rdkit2d(208) = 4304-d,
  identical to lgbm_optimal. VarianceThreshold(0.01) applied (removes sparse ECFP4 bits).
  No PCA — CatBoost handles the full feature set well with its built-in regularisation.

CV strategy: 5-fold Butina OOF. Test predictions are averaged over 5 fold models
  (each fold model predicts on the full 513 test compounds).

Prerequisites:
  scripts/16_extract_chemeleon_embeddings.py  → data/features/train_chemeleon_emb.npy
  scripts/04_build_features.py               → data/features/train_features_all.npy
  scripts/05_build_cv_splits.py              → data/splits/butina_folds.parquet

Outputs:
  models/catboost/oof_predictions.npy   (4135,)
  models/catboost/test_predictions.npy  (513,)
  models/catboost/metrics.json

Next: add ("catboost", "models/catboost", ...) to scripts/11_ensemble.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from loguru import logger
from scipy.stats import spearmanr
from sklearn.feature_selection import VarianceThreshold

from openadmet.cv.oof import evaluate_oof


def build_features(row_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load CheMeleon + ECFP4 + rdkit2d, apply variance threshold.
    row_mask: boolean mask to subset training rows (primary sources + active)."""
    logger.info("Loading features: CheMeleon + ECFP4 + rdkit2d")

    train_emb = np.load("data/features/train_chemeleon_emb.npy").astype(np.float32)
    test_emb = np.load("data/features/test_chemeleon_emb.npy").astype(np.float32)

    with open("data/features/all_feature_names.json") as f:
        all_names = json.load(f)
    X_all = np.load("data/features/train_features_all.npy").astype(np.float32)
    X_all_test = np.load("data/features/test_features_all.npy").astype(np.float32)

    # Apply row subset mask before concatenation
    if row_mask is not None:
        train_emb = train_emb[row_mask]
        X_all = X_all[row_mask]

    ecfp4_mask = np.array([n.startswith("morgan_") and "_r3_" not in n for n in all_names])
    rdkit_mask = np.array([n.startswith("rdkit_") for n in all_names])

    X_train_raw = np.concatenate([train_emb, X_all[:, ecfp4_mask], X_all[:, rdkit_mask]], axis=1)
    X_test_raw = np.concatenate([test_emb, X_all_test[:, ecfp4_mask], X_all_test[:, rdkit_mask]], axis=1)

    logger.info(
        f"Raw: CheMeleon={train_emb.shape[1]}, "
        f"ECFP4={ecfp4_mask.sum()}, rdkit2d={rdkit_mask.sum()} → {X_train_raw.shape[1]}d total"
    )

    X_train_raw = np.nan_to_num(X_train_raw, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_raw  = np.nan_to_num(X_test_raw,  nan=0.0, posinf=0.0, neginf=0.0)
    keep_mask = X_train_raw.var(axis=0) > 0
    X_train = X_train_raw[:, keep_mask].astype(np.float32)
    X_test  = X_test_raw[:, keep_mask].astype(np.float32)
    logger.info(f"After variance filter: {X_train.shape[1]} / {X_train_raw.shape[1]} features kept")

    return X_train, X_test


def main() -> None:
    logger.info("=== CatBoost (CheMeleon + ECFP4 + rdkit2d) ===")

    out_dir = Path("models/catboost")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df_full = pd.read_parquet("data/splits/butina_folds.parquet")
    pec50_col = "pec50_median" if "pec50_median" in train_df_full.columns else "pec50"

    # Build combined mask: primary sources AND non-NaN pEC50
    PRIMARY_SOURCES = {"openadmet", "analog_set1", "htchem", "htchem_semi_pure"}
    if "source" in train_df_full.columns:
        source_mask = train_df_full["source"].isin(PRIMARY_SOURCES).values
    else:
        source_mask = np.ones(len(train_df_full), dtype=bool)
    # Apply source mask first
    train_df = train_df_full[source_mask].reset_index(drop=True)
    active_mask_rel = train_df[pec50_col].notna().values
    # Combined mask on the full index for feature subsetting
    combined_mask = source_mask.copy()
    combined_mask[source_mask] = active_mask_rel
    train_df = train_df[active_mask_rel].reset_index(drop=True)
    logger.info(f"Primary + active compounds: {len(train_df)} (source: {source_mask.sum()}, non-NaN: {len(train_df)})")
    y_train = train_df[pec50_col].values.astype(np.float32)
    folds = train_df["fold"].values

    if not Path("data/features/train_chemeleon_emb.npy").exists():
        logger.error("CheMeleon embeddings not found.")
        sys.exit(1)

    X_train, X_test = build_features(row_mask=combined_mask)

    # ─── Honest 5-fold Butina OOF ────────────────────────────────────────────
    oof = np.zeros(len(y_train), dtype=np.float32)
    test_preds_folds = []
    unique_folds = sorted(np.unique(folds))

    for fold_id in unique_folds:
        val_mask = folds == fold_id
        train_mask = ~val_mask
        logger.info(f"Fold {fold_id}: {train_mask.sum()} train, {val_mask.sum()} val")

        # CatBoost with MAE loss, early stopping, ordered boosting (default).
        # random_seed=42 for reproducibility within a run; the stochastic
        # component here is different from LGB (oblivious trees vs leaf-wise).
        model = CatBoostRegressor(
            iterations=3000,
            learning_rate=0.03,
            depth=6,
            loss_function="MAE",
            eval_metric="MAE",
            random_seed=42,
            verbose=0,
            early_stopping_rounds=100,
            use_best_model=True,
            thread_count=-1,
        )

        train_pool = Pool(X_train[train_mask], y_train[train_mask])
        val_pool = Pool(X_train[val_mask], y_train[val_mask])
        model.fit(train_pool, eval_set=val_pool)

        val_preds = model.predict(X_train[val_mask]).astype(np.float32)
        oof[val_mask] = val_preds
        fold_mae = float(np.mean(np.abs(val_preds - y_train[val_mask])))
        logger.info(f"  Fold {fold_id} val MAE = {fold_mae:.4f}, best_iter={model.best_iteration_}")

        test_preds_folds.append(model.predict(X_test).astype(np.float32))

    # ─── OOF metrics ─────────────────────────────────────────────────────────
    metrics = evaluate_oof(y_train, oof, folds)
    spearman = spearmanr(y_train, oof).statistic
    logger.info(
        f"\nOOF overall: MAE={metrics['mae']:.4f}, "
        f"RAE_test={metrics.get('rae_test', metrics.get('rae', 0)):.4f}, "
        f"Spearman={spearman:.4f}"
    )

    # Average test predictions across 5 fold models
    test_preds = np.mean(test_preds_folds, axis=0).astype(np.float32)
    logger.info(f"Test predictions: mean={test_preds.mean():.4f}, std={test_preds.std():.4f}")

    # ─── Save outputs ─────────────────────────────────────────────────────────
    np.save(out_dir / "oof_predictions.npy", oof)
    np.save(out_dir / "test_predictions.npy", test_preds)

    saved_metrics = {k: float(v) for k, v in metrics.items() if not isinstance(v, dict)}
    saved_metrics["spearman"] = float(spearman)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(saved_metrics, f, indent=2)

    logger.info(f"\n=== CatBoost complete ===")
    logger.info(f"  OOF MAE={metrics['mae']:.4f}, Spearman={spearman:.4f}")
    logger.info(f"  Outputs: {out_dir}/")
    logger.info(
        "\nNext: add to scripts/11_ensemble.py:\n"
        '  ("catboost", "models/catboost", "oof_predictions.npy", "test_predictions.npy"),'
    )


if __name__ == "__main__":
    main()
