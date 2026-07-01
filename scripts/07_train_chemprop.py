"""
Train Chemprop v2 multitask MPNN using the Python API.

Why Python API instead of CLI subprocess:
  The chemprop v2 CLI crashes on Windows with STATUS_ACCESS_VIOLATION (0xC0000005)
  inside pd.read_csv() when invoked via subprocess. The Python API works on all platforms.

Architecture (Run 2 — see configs/chemprop.yaml):
  - Message passing: BondMessagePassing (d_h=300, depth=3, dropout=0.3)
  - Aggregation: MeanAggregation
  - Predictor: RegressionFFN (2 tasks: pec50_median weight=1.0, emax weight=0.4)
  - Run 1 (d_h=1200) had 13.5M params for n=1,334 → collapsed to mean (OOF Spearman=0.205)
  - Loss: MAE (matches RAE competition metric; per-task weighted)

HTS warm start (--pretrain-hts):
  When models/chemprop_pretrained/hts_pretrain_mp.pt exists, loads the MP backbone
  weights before challenge fine-tuning (same as rank-42 team Sub 5+). The backbone
  was pretrained on ~10k PXR HTS compounds (Tox21 AID 1347033 + NCATS AID 720659).
  The FFN head is always randomly initialized — only backbone weights transfer.
  Run scripts/07b_pretrain_chemprop_hts.py first to generate the checkpoint.
  HTS-pretrained outputs go to models/chemprop_hts/ (separate from scratch-trained).

Target normalization:
  Per-fold StandardScaler on both tasks before training.
  pec50_median std=0.324, emax std=0.573 — without normalization emax would dominate
  the MAE loss even with task_weights=[1.0, 0.4]. UnscaleTransform is baked into the
  predictor head and reverts predictions to original scale at inference time.

Output:
  models/chemprop/oof_predictions.npy  — shape (n_train,), pec50 scale (scratch-trained)
  models/chemprop/test_predictions.npy — shape (513,),  pec50 scale (scratch-trained)
  models/chemprop_hts/oof_predictions.npy  — (if --pretrain-hts)
  models/chemprop_hts/test_predictions.npy — (if --pretrain-hts)

Quick mode (CPU / smoke parity with splits):
  python scripts/07_train_chemprop.py --quick
  Uses fewer epochs and one seed so OOF aligns with butina_folds without a multi-hour GPU run.
  For leaderboard quality, re-run full config on CUDA without --quick.
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

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
    """
    Build MoleculeDatapoint list from DataFrame rows.

    NaN target values stay as NaN; chemprop's MAE/MSE losses skip NaN targets
    per-task automatically. This is what allows multi-task training when
    only a subset of compounds has emax measured (most do, but some legacy
    sources are missing it).
    """
    dps = []
    for _, row in df.iterrows():
        y = np.array([row[c] for c in target_cols], dtype=np.float32)
        dps.append(MoleculeDatapoint.from_smi(row[smiles_col], y))
    return dps


def _make_test_datapoints(smiles: list[str], n_tasks: int) -> list:
    """Build test MoleculeDatapoints with dummy zero targets."""
    dummy = np.zeros(n_tasks, dtype=np.float32)
    return [MoleculeDatapoint.from_smi(smi, dummy) for smi in smiles]


def _load_pretrain_mp(pretrain_path: str | None, arch: dict) -> dict | None:
    """
    Load pretrained message-passing state dict from a .pt file.

    The .pt file format (same as CheMeleon): {'hyper_parameters': {...}, 'state_dict': OrderedDict}
    Returns the state dict if the architecture matches (same d_h, depth), else None.
    """
    if pretrain_path is None:
        return None
    p = Path(pretrain_path)
    if not p.exists():
        logger.warning(f"Pretrained MP weights not found: {pretrain_path}. Training from scratch.")
        return None
    data = torch.load(p, weights_only=True, map_location="cpu")
    hyper = data.get("hyper_parameters", {})
    pretrain_dh = hyper.get("d_h", hyper.get("hidden_size", None))
    if pretrain_dh is not None and pretrain_dh != arch["hidden_size"]:
        logger.warning(
            f"Pretrained d_h={pretrain_dh} != config d_h={arch['hidden_size']} — "
            "skipping warm start (dimension mismatch). Re-run pretraining with matching d_h."
        )
        return None
    logger.info(f"Loaded pretrained MP weights from {pretrain_path}")
    return data["state_dict"]


def train_one_fold(
    fold_train: pd.DataFrame,
    fold_val: pd.DataFrame,
    test_smiles: list[str],
    available_targets: list[str],
    available_weights: list[float],
    arch: dict,
    train_cfg: dict,
    out_dir: str,
    seed: int,
    smiles_col: str,
    pretrain_mp_state: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Train one fold using the chemprop Python API.

    Returns
    -------
    val_preds : np.ndarray, shape (n_val, n_tasks)
        Predictions on the hold-out fold in original (unscaled) pec50 space.
    test_preds : np.ndarray, shape (n_test, n_tasks)
        Predictions on the 513 test compounds.
    """
    torch.manual_seed(seed)
    n_tasks = len(available_targets)
    featurizer = SimpleMoleculeMolGraphFeaturizer()

    train_dps = _make_datapoints(fold_train, smiles_col, available_targets)
    val_dps   = _make_datapoints(fold_val,   smiles_col, available_targets)
    test_dps  = _make_test_datapoints(test_smiles, n_tasks)

    train_dset = MoleculeDataset(train_dps, featurizer)
    val_dset   = MoleculeDataset(val_dps,   featurizer)
    test_dset  = MoleculeDataset(test_dps,  featurizer)

    # Per-fold normalization — fit on train, apply to val so validation loss is on
    # normalized scale during training. UnscaleTransform baked into predictor reverts
    # predictions to original scale at inference.
    scaler = train_dset.normalize_targets()
    val_dset.normalize_targets(scaler)

    bs = train_cfg["batch_size"]
    train_loader = build_dataloader(train_dset, batch_size=bs, num_workers=0, shuffle=True)
    val_loader   = build_dataloader(val_dset,   batch_size=bs, num_workers=0, shuffle=False)
    test_loader  = build_dataloader(test_dset,  batch_size=bs, num_workers=0, shuffle=False)

    d_h = arch["hidden_size"]

    mp  = BondMessagePassing(d_h=d_h, depth=arch["depth"], dropout=arch["dropout"])
    agg = MeanAggregation()

    tw        = torch.tensor(available_weights, dtype=torch.float)
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

    # Load HTS pretrained backbone weights (FFN always randomly initialized).
    # Only loads when dimensions match (checked in _load_pretrain_mp).
    if pretrain_mp_state is not None:
        model.message_passing.load_state_dict(pretrain_mp_state, strict=False)
        logger.info("    Loaded HTS pretrained MP weights into backbone.")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=out_dir,
        filename="best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        patience=train_cfg["patience"],
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=train_cfg["max_epochs"],
        accelerator="auto",
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        callbacks=[ckpt_cb, early_stop_cb],
    )
    trainer.fit(model, train_loader, val_loader)

    best_ckpt = ckpt_cb.best_model_path
    logger.info(f"    Best checkpoint: {best_ckpt}")

    best_model = MPNN.load_from_checkpoint(best_ckpt)

    # Fresh Trainer for inference — ensures correct device placement
    predict_trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        logger=False,
        enable_progress_bar=False,
    )

    val_batches  = predict_trainer.predict(best_model, val_loader)
    test_batches = predict_trainer.predict(best_model, test_loader)

    # Predictions are in original (unscaled) space due to UnscaleTransform in eval mode
    val_preds  = torch.cat(val_batches,  dim=0).cpu().numpy()   # (n_val,  n_tasks)
    test_preds = torch.cat(test_batches, dim=0).cpu().numpy()   # (n_test, n_tasks)

    return val_preds, test_preds


def main():
    parser = argparse.ArgumentParser(description="Train Chemprop v2 multitask MPNN (OOF + test).")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast CPU-friendly run: 1 seed, fewer epochs (aligned OOF for ensemble — not full quality).",
    )
    parser.add_argument(
        "--pretrain-hts",
        action="store_true",
        help=(
            "Warm-start backbone from HTS pretrained weights "
            "(models/chemprop_pretrained/hts_pretrain_mp.pt). "
            "Run scripts/07b_pretrain_chemprop_hts.py first. "
            "Outputs go to models/chemprop_hts/ (separate from scratch-trained)."
        ),
    )
    args = parser.parse_args()

    with open("configs/chemprop.yaml") as f:
        config = yaml.safe_load(f)

    train_df = pd.read_parquet("data/splits/butina_folds.parquet")
    test_df  = pd.read_parquet("data/raw/openadmet_test.parquet")

    smiles_col      = "smiles_std" if "smiles_std" in train_df.columns else "smiles"
    test_smiles_col = "smiles"

    target_cols  = [t["name"]   for t in config["tasks"]]
    task_weights = [t["weight"] for t in config["tasks"]]

    # Filter to targets that exist in the training data
    available_targets = [c for c in target_cols if c in train_df.columns]
    available_weights = [w for c, w in zip(target_cols, task_weights) if c in train_df.columns]
    logger.info(f"Targets: {available_targets}  Weights: {available_weights}")

    # Guard against silent task drift — if the configured task list doesn't
    # match what the training parquet actually contains, fail loudly so the
    # discrepancy is surfaced rather than absorbed into a degraded model.
    n_dropped = len(target_cols) - len(available_targets)
    if n_dropped > 0:
        logger.warning(
            f"Chemprop task drift: {n_dropped} configured task(s) "
            f"{[c for c in target_cols if c not in train_df.columns]} are missing from "
            f"the training parquet — they were silently dropped. Update "
            f"configs/chemprop.yaml or rebuild data/curated/master_train.parquet."
        )
    if not available_targets:
        raise RuntimeError(
            "No configured Chemprop tasks are present in the training data — refusing to train."
        )

    arch      = dict(config["architecture"])
    train_cfg = dict(config["training"])
    n_seeds   = int(train_cfg["n_seeds"])
    n_folds   = int(train_cfg["n_folds"])

    # Load HTS pretrained MP weights if --pretrain-hts requested
    pretrain_mp_state = None
    hts_mode = args.pretrain_hts
    if hts_mode:
        pretrain_mp_state = _load_pretrain_mp(
            "models/chemprop_pretrained/hts_pretrain_mp.pt", arch
        )
        if pretrain_mp_state is None:
            logger.warning(
                "--pretrain-hts requested but weights not loaded. "
                "Run scripts/07b_pretrain_chemprop_hts.py first."
            )
            hts_mode = False

    if args.quick:
        train_cfg["max_epochs"] = min(12, int(train_cfg.get("max_epochs", 100)))
        train_cfg["patience"] = min(4, int(train_cfg.get("patience", 20)))
        train_cfg["n_seeds"] = 1
        n_seeds = 1
        # Smaller MPNN for CPU wall time (still fills full OOF length).
        arch["hidden_size"] = min(int(arch.get("hidden_size", 300)), 128)
        arch["depth"] = min(int(arch.get("depth", 3)), 2)
        arch["dropout"] = 0.2
        train_cfg["batch_size"] = min(128, int(train_cfg.get("batch_size", 32)) * 2)
        logger.warning(
            "CHEMPROP --quick: short schedule + small MPNN for pipeline completion on CPU. "
            "Re-run without --quick on GPU for competition-quality weights."
        )

    test_smiles = test_df[test_smiles_col].tolist()

    oof_preds_all  = np.full((n_seeds, len(train_df)), np.nan)
    test_preds_all = []

    # HTS pretrained outputs go to a separate directory so scratch-trained OOF is preserved
    chemprop_out_dir = "models/chemprop_hts" if hts_mode else "models/chemprop"
    Path(chemprop_out_dir).mkdir(parents=True, exist_ok=True)
    if hts_mode:
        logger.info(f"HTS warm-start mode: outputs → {chemprop_out_dir}")

    for seed in range(n_seeds):
        seed_test_preds = []

        for fold in range(n_folds):
            val_mask   = train_df["fold"] == fold
            fold_train = train_df[~val_mask].reset_index(drop=True)
            fold_val   = train_df[val_mask].reset_index(drop=True)

            out_dir = f"{chemprop_out_dir}/seed{seed}_fold{fold}"
            logger.info(f"Training: seed={seed}, fold={fold} ...")

            try:
                val_preds, test_preds = train_one_fold(
                    fold_train=fold_train,
                    fold_val=fold_val,
                    test_smiles=test_smiles,
                    available_targets=available_targets,
                    available_weights=available_weights,
                    arch=arch,
                    train_cfg=train_cfg,
                    out_dir=out_dir,
                    seed=seed,
                    smiles_col=smiles_col,
                    pretrain_mp_state=pretrain_mp_state,
                )

                # Column 0 = pec50_median (first / primary task)
                pec50_val  = val_preds[:, 0]
                pec50_test = test_preds[:, 0]

                oof_preds_all[seed, val_mask.values] = pec50_val
                seed_test_preds.append(pec50_test)

                true_val = fold_val[available_targets[0]].values
                fold_mae = np.mean(np.abs(pec50_val - true_val))
                logger.info(f"  Seed {seed}, fold {fold}: val MAE = {fold_mae:.4f}")

            except Exception as e:
                import traceback
                logger.error(f"Seed {seed}, fold {fold} failed: {e}")
                logger.error(traceback.format_exc())

        if seed_test_preds:
            test_preds_all.append(np.mean(seed_test_preds, axis=0))

    oof_preds  = np.nanmean(oof_preds_all, axis=0)
    test_preds = np.mean(test_preds_all, axis=0) if test_preds_all else np.zeros(len(test_df))

    np.save(f"{chemprop_out_dir}/oof_predictions.npy", oof_preds)
    np.save(f"{chemprop_out_dir}/test_predictions.npy", test_preds)
    logger.info("Saved OOF and test predictions.")

    pec50_col = available_targets[0]
    valid     = ~np.isnan(oof_preds)
    metrics   = evaluate_oof(
        train_df[pec50_col].values[valid],
        oof_preds[valid],
        train_df["fold"].values[valid],
    )
    tag = "HTS-pretrained" if hts_mode else "scratch"
    logger.info(
        f"\nChemprop OOF ({tag}): MAE={metrics['mae']:.4f}, "
        f"RAE={metrics['rae']:.4f}, R²={metrics['r2']:.4f}"
    )
    logger.info(f"Saved OOF and test predictions.")
    logger.info(f"Test predictions: mean={test_preds.mean():.3f}, std={test_preds.std():.3f}")

    Path("submissions/phase1").mkdir(parents=True, exist_ok=True)
    sub_name = "chemprop_hts.csv" if hts_mode else "chemprop_multitask.csv"
    format_submission(
        test_df=test_df,
        predictions=test_preds,
        compound_id_col="compound_id",
        smiles_col="smiles",
        output_path=f"submissions/phase1/{sub_name}",
    )
    logger.info(f"Written: submissions/phase1/{sub_name}")
    logger.info("Next: python scripts/11_ensemble.py")


if __name__ == "__main__":
    main()
