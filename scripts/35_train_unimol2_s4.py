"""
UniMol v1 fine-tuning — fourth run at LR=5e-4.

Diversity rationale:
  unimol_tools is fully deterministic — only changing a hyperparameter produces a genuinely
  different model. s1 (LR=5e-5) and s3 (LR=2e-4) gave Pearson r=0.941 and earned the
  highest single-model blend weight in Sub 20 (24.8%).

  This run uses LR=5e-4 (10× higher than s1, 2.5× higher than s3) to explore a more
  aggressive region of the fine-tuning landscape. Early stopping (patience=5) guards
  against divergence — the model will halt if validation MAE fails to improve.

  After training: compute pearsonr(oof_s4, oof_s1) and pearsonr(oof_s4, oof_s3).
  If both < 0.95 → genuinely diverse, add to ensemble in 11_ensemble.py.

Outputs:
  models/unimol2_s4/oof_predictions.npy  (4135,)
  models/unimol2_s4/test_predictions.npy (513,)
  models/unimol2_s4/metrics.json
"""

import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from scipy.stats import pearsonr, spearmanr
from unimol_tools import MolPredict, MolTrain

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from openadmet.cv.oof import evaluate_oof
from openadmet.utils.device import (
    log_device_info, clear_device_cache, get_unimol_device_params, augment_smiles,
)


_DEVICE_PARAMS = get_unimol_device_params(base_batch_size=16)

UNIMOL_BASE_PARAMS = {
    "task":           "regression",
    "data_type":      "molecule",
    "model_name":     "unimolv1",
    "epochs":         25,
    "learning_rate":  5e-4,   # 10× higher than s1 (5e-5), 2.5× higher than s3 (2e-4)
    "batch_size":     _DEVICE_PARAMS["batch_size"],
    "early_stopping": 5,
    "use_amp":        _DEVICE_PARAMS["use_amp"],
    "use_gpu":        _DEVICE_PARAMS["use_gpu"],
    "remove_hs":      False,
    "kfold":          1,
    "split":          "random",
    "metrics":        "mae",
    "conf_cache_level": 0,
}


def make_train_data(smiles: list[str], y: np.ndarray) -> dict:
    return {"SMILES": smiles, "target": y.tolist()}


def main() -> None:
    logger.info("=== UniMol Fine-tuning — Fourth Run (LR=5e-4, unimol2_s4) ===")
    log_device_info()

    out_dir = Path("models/unimol2_s4")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet("data/splits/butina_folds.parquet")
    test_std = pd.read_parquet("data/curated/openadmet_test_std.parquet")

    smiles_col = "smiles_std" if "smiles_std" in train_df.columns else "smiles"
    pec50_col = "pec50_median" if "pec50_median" in train_df.columns else "pec50"
    test_smiles_col = "smiles_std" if "smiles_std" in test_std.columns else "smiles"

    PRIMARY_SOURCES = {"openadmet", "analog_set1", "htchem", "htchem_semi_pure"}
    if "source" in train_df.columns:
        primary_mask = train_df["source"].isin(PRIMARY_SOURCES).values
        train_df = train_df[primary_mask].reset_index(drop=True)
        logger.info(f"Filtered to primary sources: {len(train_df)} compounds")

    active_mask = train_df[pec50_col].notna().values
    train_df = train_df[active_mask].reset_index(drop=True)
    logger.info(f"Active (non-NaN pEC50): {len(train_df)} compounds")

    folds = train_df["fold"].values
    smiles_train = train_df[smiles_col].tolist()
    y_train = train_df[pec50_col].values.astype(np.float32)
    smiles_test = test_std[test_smiles_col].tolist()

    logger.info(f"Train: {len(smiles_train)} compounds, Test: {len(smiles_test)} compounds")

    oof = np.zeros(len(y_train), dtype=np.float32)
    unique_folds = sorted(np.unique(folds))

    for fold_id in unique_folds:
        val_mask = folds == fold_id
        train_mask = ~val_mask
        logger.info(f"\n--- Fold {fold_id}: {train_mask.sum()} train, {val_mask.sum()} val ---")

        fold_save = str(out_dir / f"fold{fold_id}")
        fold_done_marker = out_dir / f"fold{fold_id}" / "metric.result"
        smiles_fold_val = [smiles_train[i] for i in np.where(val_mask)[0]]

        if fold_done_marker.exists():
            logger.info(f"  Fold {fold_id} model found on disk — skipping training, re-predicting.")
        else:
            smiles_fold_train = [smiles_train[i] for i in np.where(train_mask)[0]]
            y_fold_train = y_train[train_mask]
            clf = MolTrain(save_path=fold_save, **UNIMOL_BASE_PARAMS)
            clf.fit(make_train_data(smiles_fold_train, y_fold_train))
            del clf
            gc.collect()
            clear_device_cache()

        predictor = MolPredict(load_model=fold_save)
        val_preds = predictor.predict(smiles_fold_val).flatten()
        del predictor
        gc.collect()
        clear_device_cache()

        oof[val_mask] = val_preds.astype(np.float32)
        fold_mae = float(np.mean(np.abs(val_preds - y_train[val_mask])))
        logger.info(f"  Fold {fold_id} val MAE = {fold_mae:.4f}")

    metrics = evaluate_oof(y_train, oof, folds)
    spearman = spearmanr(y_train, oof).statistic
    logger.info(
        f"\nOOF overall: MAE={metrics['mae']:.4f}, "
        f"RAE_test={metrics.get('rae_test', metrics.get('rae', 0)):.4f}, "
        f"Spearman={spearman:.4f}"
    )

    # Diversity check vs existing UniMol runs
    s1_path = Path("models/unimol2/oof_predictions.npy")
    s3_path = Path("models/unimol2_s3/oof_predictions.npy")
    if s1_path.exists():
        r_s1 = pearsonr(oof, np.load(s1_path)).statistic
        logger.info(f"Pearson r(s4, s1) = {r_s1:.4f}  (want < 0.95 for genuine diversity)")
    if s3_path.exists():
        r_s3 = pearsonr(oof, np.load(s3_path)).statistic
        logger.info(f"Pearson r(s4, s3) = {r_s3:.4f}  (want < 0.95 for genuine diversity)")

    logger.info("\n--- Final model (all training data → test predictions) ---")
    final_save = str(out_dir / "final")
    clf_final = MolTrain(save_path=final_save, **UNIMOL_BASE_PARAMS)
    clf_final.fit(make_train_data(smiles_train, y_train))

    predictor_final = MolPredict(load_model=final_save)
    N_AUG = 10
    aug_smiles, aug_idx = augment_smiles(smiles_test, n_aug=N_AUG)
    aug_preds = predictor_final.predict(aug_smiles).flatten().astype(np.float32)
    test_preds = np.array([
        aug_preds[np.array(aug_idx) == i].mean()
        for i in range(len(smiles_test))
    ], dtype=np.float32)
    logger.info(f"Test predictions: mean={test_preds.mean():.4f}, std={test_preds.std():.4f}")

    np.save(out_dir / "oof_predictions.npy", oof)
    np.save(out_dir / "test_predictions.npy", test_preds)

    saved_metrics = {k: float(v) for k, v in metrics.items() if not isinstance(v, dict)}
    saved_metrics["spearman"] = float(spearman)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(saved_metrics, f, indent=2)

    logger.info(f"\n=== unimol2_s4 complete ===")
    logger.info(f"  OOF MAE={metrics['mae']:.4f}, Spearman={spearman:.4f}")
    logger.info(f"  Outputs: {out_dir}/")
    logger.info(
        "\nIf both r(s4,s1) < 0.95 and r(s4,s3) < 0.95, add to scripts/11_ensemble.py:\n"
        '  ("unimol2_s4", "models/unimol2_s4", "oof_predictions.npy", "test_predictions.npy"),'
    )


if __name__ == "__main__":
    main()
