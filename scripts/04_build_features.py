"""
Compute all molecular features for training and test compounds.

Prerequisite: scripts/02_curate_data.py

Outputs:
- data/features/train_fingerprints.npy + train_fp_names.json
- data/features/train_descriptors.npy + train_desc_names.json
- data/features/test_fingerprints.npy
- data/features/test_descriptors.npy
- data/features/mordred_selected_columns.json (saved during train, applied to test)

Runtime: ~15-30 min on 8 cores for 5000 training compounds (Mordred is slow)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import numpy as np
import pandas as pd
from loguru import logger

from openadmet.features.fingerprints import build_fingerprint_matrix
from openadmet.features.descriptors import build_descriptor_matrix
from openadmet.data.curation import standardize_smiles


def _ensure_test_standardized(test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Train was standardized in scripts/02_curate_data.py (neutralize, largest
    fragment, canonical). The raw test parquet ships only `smiles`. Building
    train fingerprints on standardized SMILES while building test fingerprints
    on raw SMILES creates a silent feature distribution mismatch — every
    leaderboard submission to date has been affected. This function adds a
    `smiles_std` column to the test frame so both sides go through the same
    standardization.
    """
    if "smiles_std" in test_df.columns and test_df["smiles_std"].notna().all():
        return test_df

    test_df = test_df.copy()
    raw = test_df["smiles"].astype(str).tolist()
    std = [standardize_smiles(s) for s in raw]
    n_failed = sum(1 for s in std if s is None)
    if n_failed > 0:
        logger.warning(
            f"{n_failed} test SMILES failed standardization — falling back to raw SMILES "
            f"for those rows so the row count stays at {len(test_df)}."
        )
    test_df["smiles_std"] = [std[i] if std[i] is not None else raw[i] for i in range(len(raw))]
    return test_df


def main():
    out = Path("data/features")
    out.mkdir(parents=True, exist_ok=True)

    # Load ALL curated training data (active + censored/weakly-active, n=4135).
    # The top team (rank 42) uses the full pEC50 range 1.61–7.55 including below-threshold
    # compounds. Excluding them (old: master_train_active.parquet, n=1334) means the model
    # has never seen low-activity compounds, which likely exist in the test analog set.
    train_path = Path("data/curated/master_train.parquet")
    if not train_path.exists():
        logger.error("Curated training data not found. Run scripts/02_curate_data.py first.")
        return

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet("data/raw/openadmet_test.parquet")

    test_df = _ensure_test_standardized(test_df)
    test_df.to_parquet("data/curated/openadmet_test_std.parquet", index=False)

    if "smiles_std" not in train_df.columns:
        raise RuntimeError(
            "Training parquet missing `smiles_std`. Re-run scripts/02_curate_data.py "
            "before building features so train and test are guaranteed to share the "
            "same SMILES standardization pipeline."
        )

    smiles_col = "smiles_std"
    train_smiles = train_df[smiles_col].tolist()
    test_smiles = test_df[smiles_col].tolist()

    logger.info(
        f"Building features for {len(train_smiles)} training + {len(test_smiles)} test "
        f"compounds (both using `{smiles_col}` — standardized SMILES)"
    )

    # Train pass must (re)compute medians and the Mordred column filter from scratch.
    # If a prior run wrote these files (e.g., during the deprecated n=1,334 era), we
    # must not reuse them — they encode the wrong distribution.
    for stale in [
        "rdkit_medians.json",
        "mordred_medians.json",
        "mordred_selected_columns.json",
        "pmi_medians.json",
    ]:
        p = out / stale
        if p.exists():
            logger.info(f"Removing stale {p} so train pass refits the imputer/filter.")
            p.unlink()

    # Fingerprints — ECFP6 (radius-3) and FCFP4 added for PXR's large flexible LBD:
    # longer-range topological environments + pharmacophore-mapped paths complement ECFP4.
    fp_types = ["morgan_count", "avalon", "erg", "morgan_count_r3", "fcfp4_count"]
    logger.info(f"Computing fingerprints {fp_types}...")
    train_fp, fp_names = build_fingerprint_matrix(train_smiles, fp_types=fp_types)
    test_fp, _ = build_fingerprint_matrix(test_smiles, fp_types=fp_types)

    np.save(out / "train_fingerprints.npy", train_fp)
    np.save(out / "test_fingerprints.npy", test_fp)
    with open(out / "fp_names.json", "w") as f:
        json.dump(fp_names, f)
    logger.info(f"Fingerprints: {train_fp.shape}")

    # 2D Descriptors + PMI 3D shape (Mordred filter and PMI medians saved on train)
    logger.info("Computing 2D descriptors + PMI 3D shape (this takes ~25 min for 5000 compounds)...")
    train_desc, desc_names = build_descriptor_matrix(
        train_smiles,
        include_rdkit=True,
        include_mordred=True,
        include_pmi=True,
        mordred_save_path=str(out / "mordred_selected_columns.json"),
        pmi_medians_path=str(out / "pmi_medians.json"),
    )
    test_desc, _ = build_descriptor_matrix(
        test_smiles,
        include_rdkit=True,
        include_mordred=True,
        include_pmi=True,
        mordred_save_path=str(out / "mordred_selected_columns.json"),  # Loads saved filter
        pmi_medians_path=str(out / "pmi_medians.json"),                # Loads train medians
    )

    np.save(out / "train_descriptors.npy", train_desc)
    np.save(out / "test_descriptors.npy", test_desc)
    with open(out / "desc_names.json", "w") as f:
        json.dump(desc_names, f)
    logger.info(f"Descriptors: {train_desc.shape}")

    # Concatenate for LightGBM
    train_all = np.concatenate([train_fp, train_desc], axis=1)
    test_all = np.concatenate([test_fp, test_desc], axis=1)
    all_names = fp_names + desc_names

    np.save(out / "train_features_all.npy", train_all)
    np.save(out / "test_features_all.npy", test_all)
    with open(out / "all_feature_names.json", "w") as f:
        json.dump(all_names, f)

    logger.info(f"\nFeature matrix summary:")
    logger.info(f"  Train: {train_all.shape} (fingerprints + descriptors)")
    logger.info(f"  Test:  {test_all.shape}")
    logger.info(f"\nNext: python scripts/05_build_cv_splits.py")


if __name__ == "__main__":
    main()
