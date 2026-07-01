"""
Extract CheMeleon foundation model graph embeddings for train and test compounds.

CheMeleon is an OpenADMET Chemprop v2 model pretrained on large-scale ChEMBL data.
The graph-level embedding (after message-passing + aggregation + batch-norm, before the
FFN head) captures learned chemical representations that complement classical fingerprints.

These embeddings feed into:
  1. TabPFN (scripts/10_train_tabpfn.py) — concatenated with RDKit-2D descriptors
  2. Chemprop warm-start (scripts/07_train_chemprop.py) — not used here; different d_h
  3. Optional: LightGBM with embeddings as extra features

Note on warm-start: CheMeleon's message_hidden_dim differs from our Chemprop (d_h=300),
so we use the foundation model only for embeddings, not for weight initialization.

Prerequisites:
  - scripts/02_curate_data.py (for smiles_std in master_train.parquet)
  - scripts/04_build_features.py (for openadmet_test_std.parquet with smiles_std)

Outputs:
  - data/features/train_chemeleon_emb.npy   (4135 × d_emb)
  - data/features/test_chemeleon_emb.npy    (513  × d_emb)
  - data/features/chemeleon_emb_dim.json

Next: python scripts/10_train_tabpfn.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import numpy as np
import pandas as pd
from loguru import logger

from openadmet.features.chemeleon import extract_chemeleon_embeddings


def main() -> None:
    out = Path("data/features")
    out.mkdir(parents=True, exist_ok=True)

    # Use standardized SMILES — same column as all other feature scripts.
    train_df = pd.read_parquet("data/curated/master_train.parquet")
    smiles_col = "smiles_std" if "smiles_std" in train_df.columns else "smiles"
    train_smiles = train_df[smiles_col].tolist()

    # Test uses the standardized version written by 04_build_features.py
    test_std_path = Path("data/curated/openadmet_test_std.parquet")
    if test_std_path.exists():
        test_df = pd.read_parquet(test_std_path)
        test_smiles_col = "smiles_std" if "smiles_std" in test_df.columns else "smiles"
    else:
        logger.warning(
            "data/curated/openadmet_test_std.parquet not found — "
            "run scripts/04_build_features.py first to generate standardized test SMILES."
        )
        test_df = pd.read_parquet("data/raw/openadmet_test.parquet")
        test_smiles_col = "smiles"
    test_smiles = test_df[test_smiles_col].tolist()

    logger.info(f"Training set: {len(train_smiles)} compounds ({smiles_col})")
    logger.info(f"Test set: {len(test_smiles)} compounds ({test_smiles_col})")

    # Extract embeddings — downloads CheMeleon weights from Zenodo on first run (~100 MB)
    train_emb = extract_chemeleon_embeddings(train_smiles)
    test_emb = extract_chemeleon_embeddings(test_smiles)

    np.save(out / "train_chemeleon_emb.npy", train_emb)
    np.save(out / "test_chemeleon_emb.npy", test_emb)

    d_emb = int(train_emb.shape[1])
    with open(out / "chemeleon_emb_dim.json", "w") as f:
        json.dump({"d_emb": d_emb, "n_train": len(train_smiles), "n_test": len(test_smiles)}, f, indent=2)

    logger.info(
        f"\nCheMeleon embeddings saved:"
        f"\n  train: {train_emb.shape} → data/features/train_chemeleon_emb.npy"
        f"\n  test:  {test_emb.shape}  → data/features/test_chemeleon_emb.npy"
        f"\n  d_emb: {d_emb}"
    )

    # Quick sanity check: embedding variance (should not be all-zeros)
    var = float(np.var(train_emb))
    logger.info(f"Embedding variance (train): {var:.6f} (expect > 0)")
    if var < 1e-8:
        logger.error(
            "Zero-variance embeddings — CheMeleon weights may not have loaded correctly. "
            "Check that the Zenodo download completed and the checkpoint is valid."
        )
    else:
        logger.info("Embeddings look healthy. Next: python scripts/10_train_tabpfn.py")


if __name__ == "__main__":
    main()
