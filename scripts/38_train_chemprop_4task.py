"""
Train Chemprop v2 with CheMeleon pretrained init + 4-task multitask head.

Key differences from 07_train_chemprop.py:
  - CheMeleon pretrained weights as initialization (biggest single gain per discoverybytes)
  - 4 tasks: pEC50 + counter_pec50 + primary_activation_10um + primary_activation_33um
  - discoverybytes: +0.070 RAE vs 2-task; CheMeleon init alone: RAE 0.62 → 0.59

The counter-screen (PXR-null cell line) and single-concentration tasks provide
regularization signals that help the backbone distinguish PXR-specific from
cytotoxic/non-specific activity.

Prerequisites:
  1. Download CheMeleon weights:
       python -c "
       from huggingface_hub import hf_hub_download
       hf_hub_download('openadmet/pxr-chemeleon-baseline', 'model.pth',
                       local_dir='models/chemeleon')
       "
  2. data/splits/butina_folds.parquet (scripts/05_build_cv_splits.py)

Outputs:
  models/chemprop_4task/oof_predictions.npy  — (n_train,) pec50 predictions
  models/chemprop_4task/test_predictions.npy — (513,) pec50 predictions
  models/chemprop_4task/metrics.json

Usage:
  python scripts/38_train_chemprop_4task.py
  python scripts/38_train_chemprop_4task.py --quick    # CPU smoke test
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import numpy as np
import pandas as pd
import torch
import yaml
from loguru import logger
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.models import MPNN
from chemprop.nn import BondMessagePassing, MeanAggregation, RegressionFFN
from chemprop.nn.metrics import MAE
from chemprop.nn.transforms import UnscaleTransform

from openadmet.cv.oof import evaluate_oof
from openadmet.utils.submission import format_submission


def _make_datapoints(df: pd.DataFrame, smiles_col: str, target_cols: list[str]) -> list:
    dps = []
    for _, row in df.iterrows():
        y = np.array([row.get(c, np.nan) for c in target_cols], dtype=np.float32)
        dps.append(MoleculeDatapoint.from_smi(row[smiles_col], y))
    return dps


def _make_test_datapoints(smiles: list[str], n_tasks: int) -> list:
    dummy = np.zeros(n_tasks, dtype=np.float32)
    return [MoleculeDatapoint.from_smi(smi, dummy) for smi in smiles]


def _load_chemeleon_mp_weights(checkpoint: str | None, d_h: int) -> dict | None:
    """Load CheMeleon message-passing weights (backbone only, not head)."""
    if not checkpoint:
        return None
    p = Path(checkpoint)
    if not p.exists():
        logger.warning(
            f"CheMeleon checkpoint not found: {checkpoint}\n"
            "Download via: python -c \"from huggingface_hub import hf_hub_download; "
            "hf_hub_download('openadmet/pxr-chemeleon-baseline', 'model.pth', "
            "local_dir='models/chemeleon')\""
        )
        return None

    data = torch.load(p, weights_only=False, map_location="cpu")

    # CheMeleon Anvil format: flat state dict with message_passing.* keys at top level.
    # Lightning checkpoint format: wrapped in {'state_dict': {...}}.
    if isinstance(data, dict) and "state_dict" in data:
        flat = data["state_dict"]
    else:
        flat = data  # Anvil format: already the flat state dict

    # Extract message_passing.* keys and strip the prefix
    mp_state = {
        k[len("message_passing."):]: v
        for k, v in flat.items()
        if k.startswith("message_passing.")
    }

    if not mp_state:
        logger.warning("No message_passing.* keys found in CheMeleon checkpoint — skipping init")
        return None

    logger.info(f"Loaded CheMeleon MP weights: {len(mp_state)} keys from {checkpoint}")
    return mp_state


def train_one_fold(
    fold_train: pd.DataFrame,
    fold_val: pd.DataFrame,
    test_smiles: list[str],
    target_cols: list[str],
    task_weights: list[float],
    arch: dict,
    train_cfg: dict,
    out_dir: str,
    seed: int,
    smiles_col: str,
    chemeleon_mp_state: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    n_tasks = len(target_cols)
    featurizer = SimpleMoleculeMolGraphFeaturizer()

    train_dps = _make_datapoints(fold_train, smiles_col, target_cols)
    val_dps   = _make_datapoints(fold_val,   smiles_col, target_cols)
    test_dps  = _make_test_datapoints(test_smiles, n_tasks)

    train_dset = MoleculeDataset(train_dps, featurizer)
    val_dset   = MoleculeDataset(val_dps,   featurizer)
    test_dset  = MoleculeDataset(test_dps,  featurizer)

    scaler = train_dset.normalize_targets()
    val_dset.normalize_targets(scaler)

    bs = train_cfg["batch_size"]
    train_loader = build_dataloader(train_dset, batch_size=bs, num_workers=0, shuffle=True)
    val_loader   = build_dataloader(val_dset,   batch_size=bs, num_workers=0, shuffle=False)
    test_loader  = build_dataloader(test_dset,  batch_size=bs, num_workers=0, shuffle=False)

    d_h = arch["hidden_size"]
    mp  = BondMessagePassing(d_h=d_h, depth=arch["depth"], dropout=arch["dropout"])
    agg = MeanAggregation()

    tw        = torch.tensor(task_weights, dtype=torch.float)
    criterion = MAE(task_weights=tw)
    output_tf = UnscaleTransform.from_standard_scaler(scaler)

    ffn = RegressionFFN(
        n_tasks=n_tasks,
        input_dim=d_h,
        hidden_dim=d_h,
        n_layers=1,
        dropout=arch["dropout"],
        criterion=criterion,
        output_transform=output_tf,
    )

    lr = train_cfg["learning_rate"]
    model = MPNN(
        message_passing=mp,
        agg=agg,
        predictor=ffn,
        init_lr=lr / 100,
        max_lr=lr,
        final_lr=lr / 1000,
    )

    # Load CheMeleon pretrained backbone (head is always randomly initialized)
    if chemeleon_mp_state is not None:
        missing, unexpected = model.message_passing.load_state_dict(
            chemeleon_mp_state, strict=False
        )
        if missing:
            logger.debug(f"  CheMeleon: {len(missing)} missing keys (normal if d_h differs)")
        logger.info("  CheMeleon pretrained backbone loaded.")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=out_dir, filename="best", monitor="val_loss", mode="min", save_top_k=1,
    )
    early_cb = EarlyStopping(monitor="val_loss", patience=train_cfg["patience"], mode="min")

    trainer = pl.Trainer(
        max_epochs=train_cfg["max_epochs"],
        accelerator="auto",
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        callbacks=[ckpt_cb, early_cb],
    )
    trainer.fit(model, train_loader, val_loader)

    best_model = MPNN.load_from_checkpoint(ckpt_cb.best_model_path)
    predict_trainer = pl.Trainer(
        accelerator="auto", devices=1, logger=False, enable_progress_bar=False,
    )

    val_preds  = torch.cat(predict_trainer.predict(best_model, val_loader),  dim=0).cpu().numpy()
    test_preds = torch.cat(predict_trainer.predict(best_model, test_loader), dim=0).cpu().numpy()
    return val_preds, test_preds


def main():
    parser = argparse.ArgumentParser(description="Train Chemprop 4-task + CheMeleon init.")
    parser.add_argument("--quick", action="store_true",
                        help="Fast CPU run: 1 seed, fewer epochs (smoke test).")
    args = parser.parse_args()

    with open("configs/chemprop_4task.yaml") as f:
        config = yaml.safe_load(f)

    train_df = pd.read_parquet("data/splits/butina_folds.parquet")
    test_df  = pd.read_parquet("data/raw/openadmet_test.parquet")

    smiles_col      = "smiles_std" if "smiles_std" in train_df.columns else "smiles"
    test_smiles_col = "smiles"

    # Build target list from config, keeping only tasks present in training data
    all_targets = [t["name"]   for t in config["tasks"]]
    all_weights = [t["weight"] for t in config["tasks"]]

    # Single-concentration data: check for either _10um or _33um column naming
    col_aliases = {
        "primary_activation_33um": "primary_activation_30um",  # fallback alias
    }
    target_cols, task_weights = [], []
    for t, w in zip(all_targets, all_weights):
        if t in train_df.columns:
            target_cols.append(t)
            task_weights.append(w)
        elif t in col_aliases and col_aliases[t] in train_df.columns:
            target_cols.append(col_aliases[t])
            task_weights.append(w)

    if not target_cols or target_cols[0] != "pec50_median":
        logger.error("Primary task 'pec50_median' not found in training data — aborting.")
        return

    # Filter to primary Octant-assay sources (same as LightGBM)
    PRIMARY_SOURCES = {"openadmet", "analog_set1", "htchem", "htchem_semi_pure"}
    if "source" in train_df.columns:
        primary_mask = train_df["source"].isin(PRIMARY_SOURCES).values
        train_df = train_df[primary_mask].reset_index(drop=True)
        logger.info(f"Filtered to primary sources: {len(train_df)} compounds")

    logger.info(f"4-task targets: {target_cols}  weights: {task_weights}")
    n_dropped = len(all_targets) - len(target_cols)
    if n_dropped:
        logger.warning(
            f"{n_dropped} configured tasks not found in training data: "
            f"{[t for t in all_targets if t not in target_cols]}"
        )

    arch      = dict(config["architecture"])
    train_cfg = dict(config["training"])
    n_seeds   = int(train_cfg["n_seeds"])
    n_folds   = 5

    # Load CheMeleon pretrained backbone
    chemeleon_checkpoint = config.get("pretrain", {}).get("chemeleon_checkpoint")
    chemeleon_mp_state = _load_chemeleon_mp_weights(chemeleon_checkpoint, arch["hidden_size"])

    if args.quick:
        train_cfg["max_epochs"] = min(8, int(train_cfg.get("max_epochs", 25)))
        train_cfg["patience"]   = min(3, int(train_cfg.get("patience", 8)))
        n_seeds = 1
        arch["hidden_size"] = min(int(arch.get("hidden_size", 300)), 64)
        arch["depth"]       = min(int(arch.get("depth", 3)), 2)
        train_cfg["batch_size"] = 64
        logger.warning("--quick: reduced schedule + architecture for CPU smoke test.")

    test_smiles = test_df[test_smiles_col].tolist()
    out_dir     = Path("models/chemprop_4task")
    out_dir.mkdir(parents=True, exist_ok=True)

    oof_preds_all  = np.full((n_seeds, len(train_df)), np.nan)
    test_preds_all = []

    for seed in range(n_seeds):
        seed_test_preds = []
        for fold in range(n_folds):
            val_mask   = train_df["fold"] == fold
            fold_train = train_df[~val_mask].reset_index(drop=True)
            fold_val   = train_df[val_mask].reset_index(drop=True)
            fold_out   = str(out_dir / f"seed{seed}_fold{fold}")
            logger.info(f"Training: seed={seed}, fold={fold} ...")

            try:
                val_preds, test_preds = train_one_fold(
                    fold_train=fold_train,
                    fold_val=fold_val,
                    test_smiles=test_smiles,
                    target_cols=target_cols,
                    task_weights=task_weights,
                    arch=arch,
                    train_cfg=train_cfg,
                    out_dir=fold_out,
                    seed=seed,
                    smiles_col=smiles_col,
                    chemeleon_mp_state=chemeleon_mp_state,
                )
                pec50_val  = val_preds[:, 0]
                pec50_test = test_preds[:, 0]
                oof_preds_all[seed, val_mask.values] = pec50_val
                seed_test_preds.append(pec50_test)
                fold_mae = np.mean(np.abs(pec50_val - fold_val[target_cols[0]].values))
                logger.info(f"  Seed {seed}, fold {fold}: val MAE = {fold_mae:.4f}")
            except Exception as e:
                import traceback
                logger.error(f"Seed {seed}, fold {fold} failed: {e}")
                logger.error(traceback.format_exc())

        if seed_test_preds:
            test_preds_all.append(np.mean(seed_test_preds, axis=0))

    oof_preds  = np.nanmean(oof_preds_all, axis=0)
    test_preds = np.mean(test_preds_all, axis=0) if test_preds_all else np.zeros(len(test_df))

    np.save(out_dir / "oof_predictions.npy", oof_preds)
    np.save(out_dir / "test_predictions.npy", test_preds)

    pec50_col = target_cols[0]
    valid     = ~np.isnan(oof_preds)
    metrics   = evaluate_oof(
        train_df[pec50_col].values[valid],
        oof_preds[valid],
        train_df["fold"].values[valid],
    )
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

    logger.info(
        f"\nChemprop 4-task OOF: MAE={metrics['mae']:.4f}, "
        f"RAE={metrics['rae']:.4f}, R²={metrics['r2']:.4f}"
    )
    logger.info(f"Test predictions: mean={test_preds.mean():.3f}, std={test_preds.std():.3f}")

    Path("submissions/phase2").mkdir(parents=True, exist_ok=True)
    format_submission(
        test_df=test_df,
        predictions=test_preds,
        compound_id_col="compound_id",
        smiles_col="smiles",
        output_path="submissions/phase2/chemprop_4task.csv",
    )
    logger.info("Written: submissions/phase2/chemprop_4task.csv")
    logger.info(
        "\nNext: add to scripts/11_ensemble.py:\n"
        '  ("chemprop_4task", "models/chemprop_4task", "oof_predictions.npy", "test_predictions.npy"),'
    )


if __name__ == "__main__":
    main()
