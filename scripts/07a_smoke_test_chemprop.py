"""
Smoke test for scripts/07_train_chemprop.py — uses the Python API.

Runs one fold on 50 compounds for 3 epochs (CPU, no GPU required).
Expected runtime: <60 seconds.

Pass/fail criteria:
  PASS — training completes, checkpoint saved, 10 predictions produced, shapes correct
  FAIL — any exception or assertion error
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import tempfile
import numpy as np
import pandas as pd
import torch
from loguru import logger
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.models import MPNN
from chemprop.nn import BondMessagePassing, MeanAggregation, RegressionFFN
from chemprop.nn.metrics import MAE
from chemprop.nn.transforms import UnscaleTransform


def main():
    train_df = pd.read_parquet("data/splits/butina_folds.parquet")
    smiles_col = "smiles_std" if "smiles_std" in train_df.columns else "smiles"
    pec50_col  = "pec50_median" if "pec50_median" in train_df.columns else "pec50"

    mini_train = train_df[train_df["fold"] != 0].head(50).reset_index(drop=True)
    mini_val   = train_df[train_df["fold"] == 0].head(10).reset_index(drop=True)
    logger.info(f"Mini train: {len(mini_train)}, mini val: {len(mini_val)}")

    # Two tasks: pec50_median (primary) + emax (auxiliary)
    target_cols = [c for c in ["pec50_median", "emax"] if c in train_df.columns]
    task_weights = [1.0, 0.4][:len(target_cols)]
    logger.info(f"Tasks: {target_cols}, weights: {task_weights}")

    featurizer = SimpleMoleculeMolGraphFeaturizer()

    def make_dps(df):
        return [
            MoleculeDatapoint.from_smi(
                row[smiles_col],
                np.array([row[c] for c in target_cols], dtype=np.float32),
            )
            for _, row in df.iterrows()
        ]

    train_dps = make_dps(mini_train)
    val_dps   = make_dps(mini_val)
    test_dps  = [
        MoleculeDatapoint.from_smi(smi, np.zeros(len(target_cols), dtype=np.float32))
        for smi in mini_val[smiles_col]  # reuse val SMILES as "test" for smoke test
    ]

    train_dset = MoleculeDataset(train_dps, featurizer)
    val_dset   = MoleculeDataset(val_dps,   featurizer)
    test_dset  = MoleculeDataset(test_dps,  featurizer)

    scaler = train_dset.normalize_targets()
    val_dset.normalize_targets(scaler)
    logger.info("Normalization OK")

    train_loader = build_dataloader(train_dset, batch_size=16, num_workers=0, shuffle=True)
    val_loader   = build_dataloader(val_dset,   batch_size=16, num_workers=0, shuffle=False)
    test_loader  = build_dataloader(test_dset,  batch_size=16, num_workers=0, shuffle=False)
    logger.info("Dataloaders OK")

    d_h = 50  # tiny for speed
    mp  = BondMessagePassing(d_h=d_h, depth=2, dropout=0.0)
    agg = MeanAggregation()
    ffn = RegressionFFN(
        n_tasks=len(target_cols),
        input_dim=d_h,
        hidden_dim=d_h,
        n_layers=1,
        dropout=0.0,
        criterion=MAE(task_weights=torch.tensor(task_weights, dtype=torch.float)),
        output_transform=UnscaleTransform.from_standard_scaler(scaler),
    )
    model = MPNN(
        message_passing=mp,
        agg=agg,
        predictor=ffn,
        init_lr=1e-5,
        max_lr=1e-3,
        final_lr=1e-6,
    )
    logger.info(f"Model built: {type(model).__name__}, n_tasks={ffn.n_tasks}")

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_cb = ModelCheckpoint(
            dirpath=tmpdir,
            filename="best",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        )
        trainer = pl.Trainer(
            max_epochs=3,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            callbacks=[ckpt_cb],
        )
        trainer.fit(model, train_loader, val_loader)
        logger.info("Training: OK (3 epochs)")

        best_ckpt = ckpt_cb.best_model_path
        assert best_ckpt, "No checkpoint was saved — training may have failed silently"
        logger.info(f"Checkpoint: {best_ckpt}")

        best_model = MPNN.load_from_checkpoint(best_ckpt)
        logger.info("Checkpoint load: OK")

        predict_trainer = pl.Trainer(
            accelerator="cpu",
            logger=False,
            enable_progress_bar=False,
        )
        val_preds  = predict_trainer.predict(best_model, val_loader)
        test_preds = predict_trainer.predict(best_model, test_loader)

        val_arr  = torch.cat(val_preds,  dim=0).numpy()
        test_arr = torch.cat(test_preds, dim=0).numpy()

    # Shape checks
    assert val_arr.shape  == (len(mini_val),          len(target_cols)), f"val shape wrong:  {val_arr.shape}"
    assert test_arr.shape == (len(mini_val),           len(target_cols)), f"test shape wrong: {test_arr.shape}"
    assert not np.isnan(val_arr).any(),  "NaN in val predictions"
    assert not np.isnan(test_arr).any(), "NaN in test predictions"
    logger.info(f"Prediction shapes: val={val_arr.shape}, test={test_arr.shape}")

    # Sanity: predictions in pec50 range (roughly 5–8)
    pec50_preds = val_arr[:, 0]
    true_vals   = mini_val[pec50_col].values
    mae = np.mean(np.abs(pec50_preds - true_vals))
    logger.info(f"Val MAE (3 epochs, 50 compounds): {mae:.4f}")
    logger.info(f"Sample predictions (pec50): {pec50_preds[:3].tolist()}")
    logger.info(f"Sample truth:               {true_vals[:3].tolist()}")

    logger.info("\n=== SMOKE TEST PASSED — chemprop v2 Python API is ready for GPU run ===")
    logger.info("Next: python scripts/07_train_chemprop.py")


if __name__ == "__main__":
    main()
