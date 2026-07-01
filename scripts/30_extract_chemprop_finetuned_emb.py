"""
Extract PXR-fine-tuned Chemprop MPNN fingerprints (300-d) for train and test.

Why this matters:
  Our frozen CheMeleon embeddings (2048-d) are pretrained on generic ChEMBL bioactivity.
  The fine-tuned Chemprop models in models/chemprop/ have been explicitly trained on PXR
  pEC50 data, so their backbone representations encode PXR-specific structural features.
  Using these task-specific embeddings in TabICL/TabPFN should give stronger signal than
  frozen CheMeleon — the same principle behind Jeremy's HTS-pretrained CheMeleon approach.

Honest OOF strategy (no data leakage):
  For fold k's validation set: use ONLY models/chemprop/seed{s}_fold{k}/best.ckpt
  (these were trained on all folds EXCEPT fold k, so fold k data was unseen at train time).
  Average across 5 seeds for variance reduction.

Test embeddings:
  Average all 25 model checkpoints (5 seeds × 5 folds) for maximum ensemble diversity.

Output:
  data/features/train_chemprop_finetuned_emb.npy  (4135, 300) — honest OOF embeddings
  data/features/test_chemprop_finetuned_emb.npy   (513,  300) — averaged test embeddings

Prerequisites: scripts/07_train_chemprop.py must have completed (models/chemprop/ exists).

Next: scripts/31_train_tabicl_chemprop.py  (TabICL on PXR-fine-tuned + ECFP4 + rdkit2d)
"""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import torch
from loguru import logger

from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.models import MPNN


CHEMPROP_DIR = Path("models/chemprop")
N_SEEDS = 5
N_FOLDS = 5
BATCH_SIZE = 256
EMBED_DIM = 300


def make_datapoints(smiles_list: list[str]) -> list:
    dummy = np.zeros(2, dtype=np.float32)
    return [MoleculeDatapoint.from_smi(s, dummy) for s in smiles_list]


def extract_fingerprints(model: MPNN, smiles_list: list[str]) -> np.ndarray:
    """Run fingerprint extraction for a list of SMILES. Returns (n, EMBED_DIM) array."""
    featurizer = SimpleMoleculeMolGraphFeaturizer()
    dps = make_datapoints(smiles_list)
    dset = MoleculeDataset(dps, featurizer)
    loader = build_dataloader(dset, batch_size=BATCH_SIZE, num_workers=0, shuffle=False, drop_last=False)

    model.cpu().eval()
    all_fps = []
    with torch.no_grad():
        for batch in loader:
            fp = model.fingerprint(batch.bmg, batch.V_d, batch.X_d)
            all_fps.append(fp.numpy())
    return np.concatenate(all_fps, axis=0)


def main() -> None:
    logger.info("=== Extract PXR-fine-tuned Chemprop fingerprints ===")

    train_df = pd.read_parquet("data/splits/butina_folds.parquet")
    test_std = pd.read_parquet("data/curated/openadmet_test_std.parquet")

    smiles_col = "smiles_std" if "smiles_std" in train_df.columns else "smiles"
    test_smiles_col = "smiles_std" if "smiles_std" in test_std.columns else "smiles"

    train_smiles = train_df[smiles_col].tolist()
    test_smiles = test_std[test_smiles_col].tolist()
    folds = train_df["fold"].values
    n_train = len(train_smiles)
    n_test = len(test_smiles)

    logger.info(f"Train: {n_train}, Test: {n_test}")

    # OOF: honest embeddings for training set
    oof_emb = np.zeros((n_train, EMBED_DIM), dtype=np.float32)
    oof_counts = np.zeros(n_train, dtype=np.int32)

    # Test: accumulate across all models, average at end
    test_emb_sum = np.zeros((n_test, EMBED_DIM), dtype=np.float32)
    test_count = 0

    for fold_id in range(N_FOLDS):
        val_mask = folds == fold_id
        val_indices = np.where(val_mask)[0]
        val_smiles = [train_smiles[i] for i in val_indices]

        logger.info(f"\n--- Fold {fold_id}: {val_mask.sum()} val compounds ---")
        fold_val_emb_sum = np.zeros((len(val_smiles), EMBED_DIM), dtype=np.float32)
        fold_test_emb_sum = np.zeros((n_test, EMBED_DIM), dtype=np.float32)

        seed_count = 0
        for seed in range(N_SEEDS):
            ckpt_path = CHEMPROP_DIR / f"seed{seed}_fold{fold_id}" / "best.ckpt"
            if not ckpt_path.exists():
                logger.warning(f"  Missing: {ckpt_path} — skipping")
                continue

            logger.info(f"  Loading seed{seed}_fold{fold_id}...")
            model = MPNN.load_from_checkpoint(str(ckpt_path))

            # Extract for fold k's validation molecules (honest OOF)
            val_fp = extract_fingerprints(model, val_smiles)
            fold_val_emb_sum += val_fp

            # Extract for test
            test_fp = extract_fingerprints(model, test_smiles)
            fold_test_emb_sum += test_fp

            seed_count += 1
            del model

        if seed_count == 0:
            logger.error(f"No models found for fold {fold_id} — OOF will be zero")
            continue

        # Average over seeds
        fold_val_emb = fold_val_emb_sum / seed_count
        fold_test_emb = fold_test_emb_sum / seed_count

        # Store OOF
        oof_emb[val_indices] = fold_val_emb
        oof_counts[val_indices] = seed_count

        # Accumulate test
        test_emb_sum += fold_test_emb
        test_count += 1

    test_emb = test_emb_sum / test_count

    out_dir = Path("data/features")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "train_chemprop_finetuned_emb.npy", oof_emb)
    np.save(out_dir / "test_chemprop_finetuned_emb.npy", test_emb)

    logger.info(f"\n=== Done ===")
    logger.info(f"Train embedding: {oof_emb.shape}, mean={oof_emb.mean():.4f}, std={oof_emb.std():.4f}")
    logger.info(f"Test embedding:  {test_emb.shape},  mean={test_emb.mean():.4f}, std={test_emb.std():.4f}")
    logger.info(f"Saved to data/features/train_chemprop_finetuned_emb.npy")
    logger.info(f"Saved to data/features/test_chemprop_finetuned_emb.npy")
    logger.info("\nNext: python scripts/31_train_tabicl_chemprop.py")


if __name__ == "__main__":
    main()
