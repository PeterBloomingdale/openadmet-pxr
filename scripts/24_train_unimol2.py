"""
Fine-tune UniMol v1 end-to-end on PXR pEC50 (GPU).

Why fine-tuning beats frozen embeddings:
  Currently we extract frozen 512-d UniMol embeddings and feed them to LGBM.
  The rank-18 team fine-tuned UniMol directly on PXR pEC50 — making the
  transformer weights task-specific. Fine-tuning learns which 3D structural
  features matter for PXR binding; frozen embeddings are generic across all
  bioassays and cannot adapt to PXR's unusual large plastic LBD.

Model:
  UniMol v1 (84M params), pretrained on 209M 3D conformers of drug-like molecules.
  Inputs: SMILES → RDKit ETKDGv3 conformer (generated internally by unimol_tools).
  Loss: MAE (via regression task). GPU-accelerated on Apple Silicon MPS.

CV strategy:
  5 separate fine-tuned models, one per Butina fold (held-out fold = OOF).
  This matches the fold structure of all other models in the ensemble.
  Final full-training model generates test predictions.

Hardware: Apple Silicon M5 Pro (MPS backend). Batch size 16 (unified memory).
  FP16 AMP disabled — not supported on MPS.

Prerequisites:
  pip install unimol_tools  (already satisfied)
  torch with MPS (python -c "import torch; print(torch.backends.mps.is_available())" → True)
  data/splits/butina_folds.parquet

Outputs:
  models/unimol2/oof_predictions.npy  — (4135,) honest OOF predictions
  models/unimol2/test_predictions.npy — (513,)  test predictions
  models/unimol2/metrics.json         — OOF MAE, RAE, Spearman
  models/unimol2/fold{k}/             — saved model per fold (for reproducibility)
  models/unimol2/final/               — model trained on full train set

Next: add ("unimol2", "models/unimol2", ...) to scripts/11_ensemble.py
"""

import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from scipy.stats import spearmanr
from unimol_tools import MolPredict, MolTrain

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from openadmet.cv.oof import evaluate_oof
from openadmet.utils.device import (
    log_device_info, clear_device_cache, get_unimol_device_params, augment_smiles,
)


# Device params resolved at import time — MPS for Apple Silicon, CUDA for NVIDIA, else CPU
_DEVICE_PARAMS = get_unimol_device_params(base_batch_size=16)

UNIMOL_BASE_PARAMS = {
    "task":          "regression",
    "data_type":     "molecule",
    "model_name":    "unimolv1",   # 84M-param model, weights bundled with package
    "epochs":        25,           # UniMol converges fast on ~3300 training examples
    "learning_rate": 5e-5,         # low LR to avoid catastrophic forgetting of pretrained weights
    "batch_size":    _DEVICE_PARAMS["batch_size"],  # 16 for M5 Pro unified memory
    "early_stopping": 5,           # stop if val loss doesn't improve for 5 epochs
    "use_amp":       _DEVICE_PARAMS["use_amp"],     # False on MPS; True on CUDA
    "use_gpu":       _DEVICE_PARAMS["use_gpu"],     # MPS: True; CUDA: "all"; CPU: False
    "remove_hs":     False,        # keep H — UniMol uses full 3D geometry including H positions
    "kfold":         1,            # no internal CV; we handle folds manually
    "split":         "random",     # irrelevant when kfold=1 (all data = train)
    "metrics":       "mae",
    "conf_cache_level": 0,
}


def make_train_data(smiles: list[str], y: np.ndarray) -> dict:
    """Format for unimol_tools: dict with SMILES list + target array."""
    return {"SMILES": smiles, "target": y.tolist()}


def main() -> None:
    logger.info("=== UniMol Fine-tuning (GPU) ===")
    log_device_info()

    out_dir = Path("models/unimol2")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_df = pd.read_parquet("data/splits/butina_folds.parquet")
    test_std = pd.read_parquet("data/curated/openadmet_test_std.parquet")

    smiles_col = "smiles_std" if "smiles_std" in train_df.columns else "smiles"
    pec50_col = "pec50_median" if "pec50_median" in train_df.columns else "pec50"
    test_smiles_col = "smiles_std" if "smiles_std" in test_std.columns else "smiles"

    # Filter to primary Octant-assay sources only
    PRIMARY_SOURCES = {"openadmet", "analog_set1", "htchem", "htchem_semi_pure"}
    if "source" in train_df.columns:
        primary_mask = train_df["source"].isin(PRIMARY_SOURCES).values
        train_df = train_df[primary_mask].reset_index(drop=True)
        logger.info(f"Filtered to primary sources: {len(train_df)} compounds")

    # Filter to non-NaN pEC50 only — UniMol regression requires numeric targets
    active_mask = train_df[pec50_col].notna().values
    train_df = train_df[active_mask].reset_index(drop=True)
    logger.info(f"Active (non-NaN pEC50): {len(train_df)} compounds")

    folds = train_df["fold"].values
    smiles_train = train_df[smiles_col].tolist()
    y_train = train_df[pec50_col].values.astype(np.float32)
    smiles_test = test_std[test_smiles_col].tolist()

    logger.info(f"Train: {len(smiles_train)} compounds, Test: {len(smiles_test)} compounds")
    logger.info(f"pEC50 range: {y_train.min():.2f}–{y_train.max():.2f}")

    # ─── Honest 5-fold Butina OOF ────────────────────────────────────────────
    oof = np.zeros(len(y_train), dtype=np.float32)
    unique_folds = sorted(np.unique(folds))

    for fold_id in unique_folds:
        val_mask = folds == fold_id
        train_mask = ~val_mask
        n_train = train_mask.sum()
        n_val = val_mask.sum()

        logger.info(f"\n--- Fold {fold_id}: {n_train} train, {n_val} val ---")

        fold_save = str(out_dir / f"fold{fold_id}")
        fold_done_marker = out_dir / f"fold{fold_id}" / "metric.result"

        smiles_fold_val = [smiles_train[i] for i in np.where(val_mask)[0]]

        # Resume: if this fold's model already exists, skip training and just re-predict.
        # This lets us restart after a crash without losing completed folds.
        if fold_done_marker.exists():
            logger.info(f"  Fold {fold_id} model found on disk — skipping training, re-predicting val.")
        else:
            # Subset SMILES and targets for this fold
            smiles_fold_train = [smiles_train[i] for i in np.where(train_mask)[0]]
            y_fold_train = y_train[train_mask]

            # Fine-tune on training folds
            clf = MolTrain(save_path=fold_save, **UNIMOL_BASE_PARAMS)
            clf.fit(make_train_data(smiles_fold_train, y_fold_train))
            del clf  # free GPU memory before inference
            gc.collect()
            clear_device_cache()

        # Predict on validation fold (whether trained now or loaded from disk)
        predictor = MolPredict(load_model=fold_save)
        val_preds = predictor.predict(smiles_fold_val).flatten()
        del predictor
        gc.collect()
        clear_device_cache()

        oof[val_mask] = val_preds.astype(np.float32)
        fold_mae = float(np.mean(np.abs(val_preds - y_train[val_mask])))
        logger.info(f"  Fold {fold_id} val MAE = {fold_mae:.4f}")

    # OOF metrics
    metrics = evaluate_oof(y_train, oof, folds)
    spearman = spearmanr(y_train, oof).statistic
    logger.info(
        f"\nOOF overall: MAE={metrics['mae']:.4f}, "
        f"RAE_test={metrics.get('rae_test', metrics.get('rae', 0)):.4f}, "
        f"Spearman={spearman:.4f}"
    )

    # ─── Final model on full training set → test predictions ─────────────────
    logger.info("\n--- Final model (all training data → test predictions) ---")
    final_save = str(out_dir / "final")
    clf_final = MolTrain(save_path=final_save, **UNIMOL_BASE_PARAMS)
    clf_final.fit(make_train_data(smiles_train, y_train))

    predictor_final = MolPredict(load_model=final_save)

    # 10-conformer inference augmentation: average predictions across 10 randomised SMILES
    # per compound. Different atom orderings → different ETKDGv3 conformers → reduced variance.
    # discoverybytes (top performer) found this reduced test prediction noise.
    N_AUG = 10
    aug_smiles, aug_idx = augment_smiles(smiles_test, n_aug=N_AUG)
    aug_preds = predictor_final.predict(aug_smiles).flatten().astype(np.float32)
    # Average across augmentations per compound
    test_preds = np.array([
        aug_preds[np.array(aug_idx) == i].mean()
        for i in range(len(smiles_test))
    ], dtype=np.float32)
    logger.info(
        f"Test predictions (aug×{N_AUG}): shape={test_preds.shape}, "
        f"mean={test_preds.mean():.4f}, std={test_preds.std():.4f}"
    )

    # Save outputs
    np.save(out_dir / "oof_predictions.npy", oof)
    np.save(out_dir / "test_predictions.npy", test_preds)

    saved_metrics = {
        k: float(v) for k, v in metrics.items() if not isinstance(v, dict)
    }
    saved_metrics["spearman"] = float(spearman)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(saved_metrics, f, indent=2)

    logger.info(f"\n=== unimol2 complete ===")
    logger.info(f"  OOF MAE={metrics['mae']:.4f}, Spearman={spearman:.4f}")
    logger.info(f"  Outputs: {out_dir}/")
    logger.info(
        "\nNext: add to scripts/11_ensemble.py:\n"
        '  ("unimol2", "models/unimol2", "oof_predictions.npy", "test_predictions.npy"),'
    )


if __name__ == "__main__":
    main()
