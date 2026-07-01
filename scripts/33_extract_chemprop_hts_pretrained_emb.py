"""
Extract HTS-pretrained Chemprop MPNN fingerprints (300-d) for train and test.

Why this is different from scripts/30_extract_chemprop_finetuned_emb.py:
  Script 30 extracts embeddings from models fine-tuned on PXR pEC50 (task-specific).
  This script uses the HTS-pretrained backbone ONLY — trained on Tox21 (7,238) and
  NCATS (2,800) PXR HTS compounds, but NEVER fine-tuned on the OpenADMET pEC50 data.

  Key advantage: the HTS backbone has seen ~10,000 structurally diverse PXR-active
  compounds from independent labs, encoding broad "PXR bioactivity" patterns without
  any bias toward the specific Octant pEC50 assay. This mirrors the rank-19 team's
  HTS-pretrained CheMeleon encoder approach, using our Chemprop backbone instead.

No OOF complexity needed — no leakage:
  The HTS pretraining data (Tox21 + NCATS) contains NO OpenADMET compounds.
  The OpenADMET test and train sets are entirely absent from the HTS pretraining.
  Therefore a single forward pass through hts_pretrain_full.ckpt for all 4,648
  compounds is clean — the model has never seen any of these SMILES.

Outputs:
  data/features/train_chemprop_hts_emb.npy  (4135, 300)
  data/features/test_chemprop_hts_emb.npy   (513,  300)

Prerequisites: scripts/07b_pretrain_chemprop_hts.py must have completed.
Next: scripts/34_train_tabicl_hts.py
"""

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


CKPT_PATH = Path("models/chemprop_pretrained/hts_pretrain_full.ckpt")
BATCH_SIZE = 256
EMBED_DIM = 300


def extract_fingerprints(model: MPNN, smiles_list: list[str]) -> np.ndarray:
    """Run fingerprint extraction. Returns (n, EMBED_DIM) float32 array."""
    dummy = np.zeros(2, dtype=np.float32)
    featurizer = SimpleMoleculeMolGraphFeaturizer()
    dps = [MoleculeDatapoint.from_smi(s, dummy) for s in smiles_list]
    dset = MoleculeDataset(dps, featurizer)
    loader = build_dataloader(
        dset, batch_size=BATCH_SIZE, num_workers=0, shuffle=False, drop_last=False
    )

    model.cpu().eval()
    all_fps = []
    with torch.no_grad():
        for batch in loader:
            fp = model.fingerprint(batch.bmg, batch.V_d, batch.X_d)
            all_fps.append(fp.numpy())
    return np.concatenate(all_fps, axis=0).astype(np.float32)


def main() -> None:
    logger.info("=== Extract HTS-pretrained Chemprop fingerprints ===")
    logger.info(f"Checkpoint: {CKPT_PATH}")

    if not CKPT_PATH.exists():
        logger.error(f"Missing checkpoint: {CKPT_PATH}")
        logger.error("Run scripts/07b_pretrain_chemprop_hts.py first.")
        raise FileNotFoundError(CKPT_PATH)

    train_df = pd.read_parquet("data/splits/butina_folds.parquet")
    test_std = pd.read_parquet("data/curated/openadmet_test_std.parquet")

    smiles_col = "smiles_std" if "smiles_std" in train_df.columns else "smiles"
    test_smiles_col = "smiles_std" if "smiles_std" in test_std.columns else "smiles"

    train_smiles = train_df[smiles_col].tolist()
    test_smiles = test_std[test_smiles_col].tolist()

    logger.info(f"Train: {len(train_smiles)}, Test: {len(test_smiles)}")

    logger.info("Loading HTS-pretrained checkpoint...")
    model = MPNN.load_from_checkpoint(str(CKPT_PATH))

    logger.info("Extracting train embeddings...")
    train_emb = extract_fingerprints(model, train_smiles)

    logger.info("Extracting test embeddings...")
    test_emb = extract_fingerprints(model, test_smiles)

    del model

    out_dir = Path("data/features")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "train_chemprop_hts_emb.npy", train_emb)
    np.save(out_dir / "test_chemprop_hts_emb.npy", test_emb)

    logger.info(f"\n=== Done ===")
    logger.info(f"Train: {train_emb.shape}, mean={train_emb.mean():.4f}, std={train_emb.std():.4f}")
    logger.info(f"Test:  {test_emb.shape},  mean={test_emb.mean():.4f}, std={test_emb.std():.4f}")
    logger.info("Saved to data/features/train_chemprop_hts_emb.npy")
    logger.info("Saved to data/features/test_chemprop_hts_emb.npy")
    logger.info("\nNext: python scripts/34_train_tabicl_hts.py")


if __name__ == "__main__":
    main()
