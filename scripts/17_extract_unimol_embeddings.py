"""
Extract Uni-Mol 3D molecular embeddings for train and test compounds.

Uni-Mol is a transformer pretrained on 3D molecular conformers (209M+ molecules).
It generates a 512-dimensional CLS token embedding per molecule that encodes learned
3D shape and electronic environment — complementary to 2D fingerprints for PXR's
large, flexible ligand-binding domain (~11 Å span, shape-sensitive).

Why 3D for PXR:
PXR has an unusually large and plastic LBD. Binding is dominated by hydrophobic
packing and shape complementarity rather than specific H-bonds. 3D-aware representations
capture pocket-filling shape information that 2D fingerprints fundamentally cannot encode.
The rank-42 team added Uni-Mol in their Sub 9 and saw the single largest RAE improvement.

Install: pip install unimol_tools
On Windows: ase dependency may have path issues — if import fails, run this script via WSL
or on a Linux machine and copy the .npy files back.

Prerequisites:
  - scripts/04_build_features.py → data/curated/openadmet_test_std.parquet

Outputs:
  - data/features/train_unimol_emb.npy   (4135 × 512)
  - data/features/test_unimol_emb.npy    (513  × 512)

Next: python scripts/18_train_unimol_lgbm.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from loguru import logger

try:
    from unimol_tools import UniMolRepr
    UNIMOL_AVAILABLE = True
except ImportError:
    UNIMOL_AVAILABLE = False
    logger.error(
        "unimol_tools not installed. Install with: pip install unimol_tools\n"
        "On Windows, ase may fail — run this script on Linux/WSL and copy .npy output back."
    )


def extract_unimol(smiles_list: list[str], batch_size: int = 16) -> np.ndarray:
    """
    Extract 512-d Uni-Mol CLS embeddings.

    UniMolRepr handles 3D conformer generation internally (MMFF94/ETKDGv3).
    batch_size=16 is safe on 8 GB GPU; reduce to 8 on CPU.

    Returns (n, 512) float32 array. Failed SMILES rows are filled with zeros.
    """
    logger.info(f"Extracting Uni-Mol embeddings for {len(smiles_list)} compounds (batch_size={batch_size}) ...")
    clf = UniMolRepr(data_type="molecule", remove_hs=False)

    # UniMolRepr.get_repr() accepts a list of SMILES and returns a dict:
    # {'cls_repr': (n, 512), 'atomic_reprs': list of (n_atoms, 512)}
    try:
        result = clf.get_repr(smiles_list, return_atomic_reprs=False)
        # unimol_tools 0.1.x returns a list of (512,) arrays, not a dict
        emb = np.array(result, dtype=np.float32)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        logger.info(f"Uni-Mol embeddings: shape={emb.shape}")
        return emb
    except Exception as e:
        logger.error(f"Uni-Mol extraction failed: {e}")
        logger.error(
            "Possible fix: ensure unimol_tools is installed and ase works on your platform. "
            "On Windows, use WSL or a Linux environment."
        )
        raise


def main() -> None:
    if not UNIMOL_AVAILABLE:
        sys.exit(1)

    out = Path("data/features")
    out.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet("data/curated/master_train.parquet")
    smiles_col = "smiles_std" if "smiles_std" in train_df.columns else "smiles"
    train_smiles = train_df[smiles_col].tolist()

    test_std_path = Path("data/curated/openadmet_test_std.parquet")
    if test_std_path.exists():
        test_df = pd.read_parquet(test_std_path)
        test_smiles_col = "smiles_std" if "smiles_std" in test_df.columns else "smiles"
    else:
        logger.warning("Standardized test SMILES not found — run 04_build_features.py first.")
        test_df = pd.read_parquet("data/raw/openadmet_test.parquet")
        test_smiles_col = "smiles"
    test_smiles = test_df[test_smiles_col].tolist()

    logger.info(f"Train: {len(train_smiles)} | Test: {len(test_smiles)}")

    train_emb = extract_unimol(train_smiles)
    test_emb = extract_unimol(test_smiles)

    np.save(out / "train_unimol_emb.npy", train_emb)
    np.save(out / "test_unimol_emb.npy", test_emb)

    logger.info(
        f"\nUni-Mol embeddings saved:"
        f"\n  train: {train_emb.shape} → data/features/train_unimol_emb.npy"
        f"\n  test:  {test_emb.shape}  → data/features/test_unimol_emb.npy"
    )

    var = float(np.var(train_emb))
    if var < 1e-8:
        logger.error("Zero-variance embeddings — Uni-Mol extraction may have failed silently.")
    else:
        logger.info(f"Embedding variance (train): {var:.4f} — looks healthy.")
        logger.info("Next: python scripts/18_train_unimol_lgbm.py")


if __name__ == "__main__":
    main()
