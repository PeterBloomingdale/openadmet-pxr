"""
Matched Molecular Pair (MMP) analysis and featurization.

An MMP is a pair of molecules (A, B) that differ in exactly one structural
fragment at a single substitution point, while sharing a common molecular
scaffold (the "context"). Example: A = scaffold-CH3, B = scaffold-CF3.

For the PXR challenge, MMPs are uniquely powerful because the 513 test compounds
are analogs of 63 training parents. For each test compound, we can often find
a training compound that differs by a single transformation — and the pEC50
difference (Δ) should be predictable from the transformation.

Two models use MMP features:
1. mmp_delta_baseline.py: Vanilla LightGBM predicting Δ(pEC50)
2. delta_siamese.py: Antisymmetric Siamese with Δ(A→B) = g(B) - g(A)

This module handles:
- Building the MMP database via mmpdb CLI
- Nearest-neighbor lookup (finding the closest training analog for each query)
- Physicochemical delta features (ΔMW, ΔlogP, ΔHBD, ΔHBA, ΔTPSA, ΔRotBonds)
"""

import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


def build_mmp_database(
    smiles_list: list[str],
    ids: list[str],
    output_dir: str = "data/mmps",
    mmpdb_executable: str = "mmpdb",
) -> str:
    """
    Builds an mmpdb SQLite database for a set of molecules.

    mmpdb uses the SMILES fragmentation algorithm (Hussain-Rea) to identify
    all matched molecular pairs: molecules that differ in exactly one fragment
    at one cut point.

    Pipeline: fragment → index → SQLite database

    Returns path to the generated .mmpdb SQLite file.
    Raises RuntimeError if mmpdb is not on PATH.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    db_path = out / "pxr_training.mmpdb"

    if db_path.exists():
        logger.info(f"MMP database already exists at {db_path}")
        return str(db_path)

    # Write SMILES file for mmpdb (format: SMILES <tab> ID)
    smiles_file = out / "training_smiles.smi"
    with open(smiles_file, "w") as f:
        for smi, id_ in zip(smiles_list, ids):
            f.write(f"{smi}\t{id_}\n")

    # Step 1: Fragment
    frag_file = out / "training.fragments"
    logger.info("Running mmpdb fragment...")
    result = subprocess.run(
        [mmpdb_executable, "fragment", str(smiles_file), "-o", str(frag_file)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"mmpdb fragment failed:\n{result.stderr}")

    # Step 2: Index (builds the MMP database)
    logger.info("Running mmpdb index...")
    result = subprocess.run(
        [mmpdb_executable, "index", str(frag_file), "-o", str(db_path)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"mmpdb index failed:\n{result.stderr}")

    logger.info(f"MMP database built at {db_path}")
    return str(db_path)


def find_nearest_training_neighbor(
    query_smiles: str,
    train_smiles: list[str],
    train_pec50: list[float],
    n_bits: int = 1024,
    radius: int = 2,
) -> tuple[str, float, float]:
    """
    Finds the most structurally similar training compound to the query.

    Uses ECFP4 Tanimoto similarity — the same metric used to construct
    the test set (Enamine analogs selected at Tanimoto > 0.4).

    Returns (neighbor_smiles, neighbor_pec50, tanimoto_similarity).
    If no valid training SMILES exist, returns ("", nan, 0.0).
    """
    try:
        from rdkit.Chem import rdFingerprintGenerator as _rfg
        _gen = _rfg.GetMorganGenerator(radius=radius, fpSize=n_bits)
        query_fp = _gen.GetFingerprint(Chem.MolFromSmiles(query_smiles))
    except Exception:
        return "", float("nan"), 0.0

    train_fps = []
    valid_indices = []
    for i, smi in enumerate(train_smiles):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                train_fps.append(_gen.GetFingerprint(mol))
                valid_indices.append(i)
        except Exception:
            pass

    if not train_fps:
        return "", float("nan"), 0.0

    sims = DataStructs.BulkTanimotoSimilarity(query_fp, train_fps)
    best_idx = int(np.argmax(sims))
    original_idx = valid_indices[best_idx]

    return (
        train_smiles[original_idx],
        float(train_pec50[original_idx]),
        float(sims[best_idx]),
    )


def physchem_delta_features(
    smiles_a: str,
    smiles_b: str,
) -> Optional[np.ndarray]:
    """
    Computes physicochemical property deltas between two molecules.

    Features: [ΔMW, ΔlogP, ΔHBD, ΔHBA, ΔTPSA, ΔRotBonds, ΔRingCount, ΔAromaticRings]
    Delta = B - A (signed, so direction matters).

    These features capture the most SAR-relevant physicochemical changes for PXR:
    - ΔlogP: PXR pocket is hydrophobic; increasing logP usually increases potency
    - ΔTPSA: Surface polarity changes affect membrane penetration and binding
    - ΔMW: Larger ligands fill more of PXR's large flexible pocket
    """
    def mol_props(smi: str) -> Optional[np.ndarray]:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return None
            return np.array([
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                rdMolDescriptors.CalcNumHBD(mol),
                rdMolDescriptors.CalcNumHBA(mol),
                Descriptors.TPSA(mol),
                rdMolDescriptors.CalcNumRotatableBonds(mol),
                rdMolDescriptors.CalcNumRings(mol),
                rdMolDescriptors.CalcNumAromaticRings(mol),
            ], dtype=np.float32)
        except Exception:
            return None

    props_a = mol_props(smiles_a)
    props_b = mol_props(smiles_b)

    if props_a is None or props_b is None:
        return None

    return props_b - props_a


def build_mmp_feature_matrix(
    query_smiles_list: list[str],
    train_df: pd.DataFrame,
    smiles_col: str = "smiles_std",
    pec50_col: str = "pec50_median",
    n_bits: int = 1024,
) -> pd.DataFrame:
    """
    For each query compound, finds its nearest training neighbor and computes
    the physicochemical delta feature vector.

    Output DataFrame has columns:
    - query_smiles
    - neighbor_smiles
    - neighbor_pec50
    - tanimoto (similarity to neighbor)
    - delta_mw, delta_logp, delta_hbd, delta_hba, delta_tpsa, delta_rotbonds,
      delta_rings, delta_arom_rings (ΔPhysChem features)

    Rows where no valid neighbor is found have NaN in all delta columns.
    These should be filled with 0.0 before training (zero delta = predict
    same as neighbor pEC50 — a reasonable fallback).
    """
    train_smiles = train_df[smiles_col].tolist()
    train_pec50 = train_df[pec50_col].tolist()

    records = []
    for query_smi in query_smiles_list:
        neighbor_smi, neighbor_pec50, tanimoto = find_nearest_training_neighbor(
            query_smi, train_smiles, train_pec50, n_bits=n_bits
        )

        delta_feats = None
        if neighbor_smi and not np.isnan(neighbor_pec50):
            delta_feats = physchem_delta_features(neighbor_smi, query_smi)

        col_names = ["delta_mw", "delta_logp", "delta_hbd", "delta_hba",
                     "delta_tpsa", "delta_rotbonds", "delta_rings", "delta_arom_rings"]

        rec = {
            "query_smiles": query_smi,
            "neighbor_smiles": neighbor_smi,
            "neighbor_pec50": neighbor_pec50,
            "tanimoto": tanimoto,
        }
        if delta_feats is not None:
            for name, val in zip(col_names, delta_feats):
                rec[name] = float(val)
        else:
            for name in col_names:
                rec[name] = np.nan

        records.append(rec)

    df = pd.DataFrame(records)
    n_no_neighbor = df["neighbor_pec50"].isna().sum()
    if n_no_neighbor > 0:
        logger.warning(
            f"{n_no_neighbor} query compounds have no valid training neighbor. "
            f"Delta features set to NaN (fill with 0.0 before training)."
        )
    return df
