"""
Train LightGBM on docking features to create a physics-informed base model.

Two variants are trained and compared on Butina OOF:
  Variant A: Docking features only (6-7 scalar features from smina/gnina)
  Variant B: Docking features APPENDED to the 8262 classical features
              (fingerprints + Mordred + RDKit 2D)

Variant B is expected to be stronger: docking provides a physics-based prior
on which compounds fit the PXR pocket, while classical features capture
substructure/pharmacophore patterns. Together they cover different error modes.

Why a separate model rather than adding docking features to lgbm_chemeleon?
Adding docking features to lgbm_chemeleon (8262 FP+Mordred+CheMeleon features)
would create a 8269-feature model where the 7 docking features are swamped by
the 8262 existing ones. Training a separate model (with only docking features or
docking + classical without CheMeleon) preserves the docking signal and gives the
ensemble a truly new prediction to blend.

The key question answered by this script:
  Does explicit 3D receptor-ligand shape-fitness (docking score) predict PXR
  pEC50 beyond what 2D fingerprints + graph embeddings already capture?

Success criterion: Variant B OOF MAE < lgbm_chemeleon OOF MAE - 0.02
(if docking adds ≥0.02 MAE improvement on Butina OOF, add to ensemble)

Prerequisites:
  - data/features/train_docking_feats.parquet (from script 22)
  - data/features/test_docking_feats.parquet  (from script 22)
  - data/features/train_features.parquet      (classical features from script 04)
  - data/splits/butina_folds.parquet

Outputs:
  models/lgbm_docking/oof_predictions.npy    — Variant B OOF predictions (4135,)
  models/lgbm_docking/test_predictions.npy   — Variant B test predictions (513,)
  models/lgbm_docking/metrics.json           — OOF MAE, RAE, Spearman per variant
  models/lgbm_docking/feature_importance.csv — feature importances

Next: python scripts/11_ensemble.py (add lgbm_docking to model list)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from loguru import logger
from scipy.stats import spearmanr
from sklearn.feature_selection import VarianceThreshold

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from openadmet.cv.oof import evaluate_oof


# LightGBM hyperparameters — same as lgbm_chemeleon for fair comparison
# (same regularization, same tree structure)
LGBM_PARAMS = {
    "objective": "mae",
    "metric": "mae",
    "num_leaves": 63,
    "learning_rate": 0.02,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "min_child_samples": 25,
    "n_estimators": 3000,
    "lambda_l1": 0.1,
    "lambda_l2": 0.05,
    "path_smooth": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

N_SEEDS = 3  # seed ensemble for variance reduction (fewer than lgbm_chemeleon's 5 for speed)


def train_lgbm_oof(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    folds: np.ndarray,
    feature_names: list[str],
    n_seeds: int = N_SEEDS,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Trains LightGBM with Butina 5-fold CV + seed ensemble.

    Returns (oof_predictions, test_predictions, feature_importance_df).
    """
    n = len(y_train)
    oof = np.zeros(n)
    test_preds = np.zeros(len(X_test))
    importances = pd.DataFrame({"feature": feature_names})
    unique_folds = sorted(np.unique(folds))

    for seed_offset in range(n_seeds):
        oof_seed = np.zeros(n)
        test_seed = np.zeros(len(X_test))

        for fold_id in unique_folds:
            train_mask = folds != fold_id
            val_mask = folds == fold_id

            params = {**LGBM_PARAMS, "random_state": 42 + seed_offset * 100}
            model = LGBMRegressor(**params)
            model.fit(
                X_train[train_mask], y_train[train_mask],
                eval_set=[(X_train[val_mask], y_train[val_mask])],
                callbacks=[
                    early_stopping(stopping_rounds=100, verbose=False),
                    log_evaluation(period=0),
                ],
            )
            oof_seed[val_mask] = model.predict(X_train[val_mask])
            test_seed += model.predict(X_test) / len(unique_folds)

            col_name = f"seed{seed_offset}_fold{fold_id}"
            importances[col_name] = model.feature_importances_

        oof += oof_seed / n_seeds
        test_preds += test_seed / n_seeds

    # Average importance across seeds/folds
    seed_cols = [c for c in importances.columns if c.startswith("seed")]
    importances["importance_mean"] = importances[seed_cols].mean(axis=1)
    importances = importances[["feature", "importance_mean"]].sort_values("importance_mean", ascending=False)

    return oof, test_preds, importances


def load_classical_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Loads the same 2D classical features used by lgbm_chemeleon (no CheMeleon embeddings).

    Features are stored as numpy arrays in data/features/ (written by scripts/04_build_features.py).
    train_features_all.npy: (4135, 8262) = fingerprints + RDKit 2D + Mordred
    Rows are in the same order as butina_folds.parquet / master_train.parquet.
    """
    train_npy = Path("data/features/train_features_all.npy")
    test_npy = Path("data/features/test_features_all.npy")
    names_json = Path("data/features/all_feature_names.json")

    if not train_npy.exists():
        logger.warning("Classical feature numpy not found — using docking features only (Variant A)")
        return None, None, []

    import json
    X_tr = np.load(train_npy).astype(np.float32)
    X_te = np.load(test_npy).astype(np.float32)
    feat_cols = json.loads(names_json.read_text()) if names_json.exists() else [f"feat_{i}" for i in range(X_tr.shape[1])]

    X_tr = np.nan_to_num(X_tr, nan=0.0)
    X_te = np.nan_to_num(X_te, nan=0.0)

    logger.info(f"Classical features: train={X_tr.shape}, test={X_te.shape}")
    return X_tr, X_te, feat_cols


def main() -> None:
    out_dir = Path("models/lgbm_docking")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_df = pd.read_parquet("data/splits/butina_folds.parquet")
    test_df = pd.read_parquet("data/raw/openadmet_test.parquet")

    pec50_col = "pec50_median" if "pec50_median" in train_df.columns else "pec50"
    y_train = train_df[pec50_col].values
    folds = train_df["fold"].values
    id_col = "compound_id" if "compound_id" in train_df.columns else train_df.columns[0]
    test_id_col = "compound_id" if "compound_id" in test_df.columns else test_df.columns[0]

    # Load docking features
    train_dock_path = Path("data/features/train_docking_feats.parquet")
    test_dock_path = Path("data/features/test_docking_feats.parquet")

    if not train_dock_path.exists() or not test_dock_path.exists():
        logger.error("Docking features not found — run scripts/22_extract_docking_features.py first.")
        sys.exit(1)

    train_dock = pd.read_parquet(train_dock_path)
    test_dock = pd.read_parquet(test_dock_path)

    # Align docking features with training row order
    train_dock = train_df[[id_col]].merge(train_dock, left_on=id_col, right_on="compound_id", how="left")
    test_dock = test_df[[test_id_col]].merge(test_dock, left_on=test_id_col, right_on="compound_id", how="left")

    dock_feat_cols = [c for c in train_dock.columns if c not in ("compound_id", id_col)]
    X_dock_train = train_dock[dock_feat_cols].values.astype(np.float32)
    X_dock_test = test_dock[dock_feat_cols].values.astype(np.float32)
    X_dock_train = np.nan_to_num(X_dock_train, nan=0.0)
    X_dock_test = np.nan_to_num(X_dock_test, nan=0.0)

    logger.info(f"Docking features: train={X_dock_train.shape}, test={X_dock_test.shape}")
    logger.info(f"Docking features: {dock_feat_cols}")

    all_metrics = {}

    # ─── Variant A: Docking features only ────────────────────────────────────
    logger.info("\n=== Variant A: Docking features only ===")
    vt_a = VarianceThreshold(threshold=1e-6)
    X_a_train = vt_a.fit_transform(X_dock_train)
    X_a_test = vt_a.transform(X_dock_test)
    feat_names_a = [dock_feat_cols[i] for i in range(len(dock_feat_cols)) if vt_a.get_support()[i]]
    logger.info(f"After variance filter: {X_a_train.shape[1]} features (removed {X_dock_train.shape[1] - X_a_train.shape[1]})")

    oof_a, test_a, imp_a = train_lgbm_oof(X_a_train, y_train, X_a_test, folds, feat_names_a)
    metrics_a = evaluate_oof(y_train, oof_a, folds)
    logger.info(f"Variant A OOF: MAE={metrics_a['mae']:.4f}, RAE={metrics_a['rae']:.4f}, ρ={metrics_a.get('spearman', spearmanr(y_train, oof_a).statistic):.4f}")
    all_metrics["variant_a_docking_only"] = {k: float(v) for k, v in metrics_a.items() if not isinstance(v, dict)}

    # ─── Variant B: Docking + classical features ──────────────────────────────
    logger.info("\n=== Variant B: Docking + classical features ===")
    X_classical_train, X_classical_test, classical_feat_cols = load_classical_features(train_df, test_df)

    if X_classical_train is not None:
        X_b_train = np.hstack([X_classical_train, X_dock_train])
        X_b_test = np.hstack([X_classical_test, X_dock_test])
        feat_names_b = classical_feat_cols + dock_feat_cols

        vt_b = VarianceThreshold(threshold=0.01)
        X_b_train = vt_b.fit_transform(X_b_train)
        X_b_test = vt_b.transform(X_b_test)
        feat_names_b = [feat_names_b[i] for i in range(len(feat_names_b)) if vt_b.get_support()[i]]
        logger.info(f"Variant B: {X_b_train.shape[1]} features after variance filter")

        oof_b, test_b, imp_b = train_lgbm_oof(X_b_train, y_train, X_b_test, folds, feat_names_b)
        metrics_b = evaluate_oof(y_train, oof_b, folds)
        logger.info(f"Variant B OOF: MAE={metrics_b['mae']:.4f}, RAE={metrics_b['rae']:.4f}, ρ={metrics_b.get('spearman', spearmanr(y_train, oof_b).statistic):.4f}")
        all_metrics["variant_b_docking_classical"] = {k: float(v) for k, v in metrics_b.items() if not isinstance(v, dict)}

        # Select best variant for canonical output
        if metrics_b["mae"] < metrics_a["mae"]:
            logger.info("Variant B (docking + classical) wins — using for canonical output")
            oof_final, test_final = oof_b, test_b
            imp_final = imp_b
        else:
            logger.info("Variant A (docking only) wins — using for canonical output")
            oof_final, test_final = oof_a, test_a
            imp_final = imp_a
    else:
        logger.info("Classical features not found — using Variant A (docking only) as canonical")
        oof_final, test_final = oof_a, test_a
        imp_final = imp_a

    # Save outputs
    np.save(out_dir / "oof_predictions.npy", oof_final)
    np.save(out_dir / "test_predictions.npy", test_final)
    imp_final.to_csv(out_dir / "feature_importance.csv", index=False)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    logger.info(f"\n=== lgbm_docking complete ===")
    logger.info(f"  OOF: {oof_final.shape}, mean={oof_final.mean():.3f}")
    logger.info(f"  Test: {test_final.shape}, mean={test_final.mean():.3f}")
    logger.info(f"  Outputs: {out_dir}/")
    logger.info(f"  Top docking features by importance:")
    for _, row in imp_final.head(10).iterrows():
        logger.info(f"    {row['feature']}: {row['importance_mean']:.1f}")

    # Decision gate — should we add to ensemble?
    best_mae = min(metrics_a["mae"], all_metrics.get("variant_b_docking_classical", {}).get("mae", 9999))
    # lgbm_chemeleon OOF MAE is approximately 0.45-0.48 (from past runs)
    lgbm_chemeleon_mae = 0.47  # approximate reference — update from actual run
    if best_mae < lgbm_chemeleon_mae - 0.02:
        logger.info(
            f"\n✅ Docking model OOF MAE={best_mae:.4f} < lgbm_chemeleon MAE={lgbm_chemeleon_mae:.4f} - 0.02\n"
            f"ADD lgbm_docking to scripts/11_ensemble.py model list:\n"
            f"  ('lgbm_docking', 'models/lgbm_docking', 'oof_predictions.npy', 'test_predictions.npy'),"
        )
    else:
        logger.warning(
            f"\n⚠ Docking model OOF MAE={best_mae:.4f} does NOT improve on lgbm_chemeleon by ≥0.02.\n"
            f"Docking features may not add orthogonal signal on this dataset.\n"
            f"Options:\n"
            f"  1. Add anyway and let SLSQP assign it 0% weight (safe — ensemble won't regress)\n"
            f"  2. Try ensemble docking (multiple PXR receptor structures) for better pose sampling\n"
            f"  3. Skip docking and focus on other improvements"
        )

    logger.info("\nNext: python scripts/11_ensemble.py")


if __name__ == "__main__":
    main()
