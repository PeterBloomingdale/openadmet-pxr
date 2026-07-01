"""
UniMol v1 fine-tuning — second independent run (different random initialisation).

Why a second run adds value:
  Neural network training is stochastic (random weight init, SGD noise, dropout).
  Two independently-trained models make correlated but non-identical errors.
  Averaging predictions reduces variance, typically improving MAE by 0.003–0.008 units.
  unimol2 already gets 35% ensemble weight, so variance reduction on it propagates strongly.

This script is identical to 24_train_unimol2.py except the output directory is
`models/unimol2_s2/`. No explicit seed is set — PyTorch draws a fresh random seed,
ensuring true independence from the first run.

Outputs:
  models/unimol2_s2/oof_predictions.npy  (4135,)
  models/unimol2_s2/test_predictions.npy (513,)
  models/unimol2_s2/metrics.json

Next: add ("unimol2_s2", "models/unimol2_s2", ...) to scripts/11_ensemble.py
      and optionally create an averaged model from unimol2 + unimol2_s2.
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


UNIMOL_BASE_PARAMS = {
    "task":          "regression",
    "data_type":     "molecule",
    "model_name":    "unimolv1",
    "epochs":        25,
    "learning_rate": 5e-5,
    "batch_size":    8,
    "early_stopping": 5,
    "use_amp":       True,
    "use_gpu":       "all",
    "remove_hs":     False,
    "kfold":         1,
    "split":         "random",
    "metrics":       "mae",
    "conf_cache_level": 0,
}


def make_train_data(smiles: list[str], y: np.ndarray) -> dict:
    return {"SMILES": smiles, "target": y.tolist()}


def main() -> None:
    logger.info("=== UniMol Fine-tuning — Second Independent Run (unimol2_s2) ===")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"Device: {torch.cuda.get_device_name(0)}")

    out_dir = Path("models/unimol2_s2")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet("data/splits/butina_folds.parquet")
    test_std = pd.read_parquet("data/curated/openadmet_test_std.parquet")

    smiles_col = "smiles_std" if "smiles_std" in train_df.columns else "smiles"
    pec50_col = "pec50_median" if "pec50_median" in train_df.columns else "pec50"
    test_smiles_col = "smiles_std" if "smiles_std" in test_std.columns else "smiles"

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
            torch.cuda.empty_cache()

        predictor = MolPredict(load_model=fold_save)
        val_preds = predictor.predict(smiles_fold_val).flatten()
        del predictor
        gc.collect()
        torch.cuda.empty_cache()

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

    logger.info("\n--- Final model (all training data → test predictions) ---")
    final_save = str(out_dir / "final")
    clf_final = MolTrain(save_path=final_save, **UNIMOL_BASE_PARAMS)
    clf_final.fit(make_train_data(smiles_train, y_train))

    predictor_final = MolPredict(load_model=final_save)
    test_preds = predictor_final.predict(smiles_test).flatten().astype(np.float32)
    logger.info(f"Test predictions: mean={test_preds.mean():.4f}, std={test_preds.std():.4f}")

    np.save(out_dir / "oof_predictions.npy", oof)
    np.save(out_dir / "test_predictions.npy", test_preds)

    saved_metrics = {k: float(v) for k, v in metrics.items() if not isinstance(v, dict)}
    saved_metrics["spearman"] = float(spearman)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(saved_metrics, f, indent=2)

    logger.info(f"\n=== unimol2_s2 complete ===")
    logger.info(f"  OOF MAE={metrics['mae']:.4f}, Spearman={spearman:.4f}")
    logger.info(f"  Outputs: {out_dir}/")


if __name__ == "__main__":
    main()
