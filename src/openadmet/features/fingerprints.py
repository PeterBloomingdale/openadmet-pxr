"""
Molecular fingerprint computation for the PXR activity prediction ensemble.

Each fingerprint captures a different aspect of molecular structure:
- Morgan count FP: atom environment frequencies (radius-2 ECFP4 equivalent)
- Avalon FP: substructure-aware with pharmacophore features
- ErG FP: extended reduced graph — scaffold-level shape + pharmacophore
- ECFP4 bit FP: used for Tanimoto similarity only (not as ML features)

Why count fingerprints over bit fingerprints for ML:
Bit fingerprints only indicate whether a substructure is present. Count
fingerprints encode how many times it appears. For PXR, where scaffold
multiplicity (e.g., number of pendant aromatic rings) correlates with
binding affinity, count vectors capture richer SAR information.
"""

from typing import Optional
import numpy as np
from loguru import logger
from joblib import Parallel, delayed

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdFingerprintGenerator
    from rdkit.Avalon import pyAvalonTools
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logger.warning("RDKit not available — fingerprint functions will fail")


def morgan_count_fp(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
) -> Optional[np.ndarray]:
    """
    Morgan count fingerprint (ECFP4-equivalent with counts).

    radius=2: captures up to 4 bonds from each atom center, equivalent to ECFP4.
    n_bits=2048: larger bit vector reduces hash collisions vs the 1024-bit default.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        fp = gen.GetCountFingerprintAsNumPy(mol)
        return fp.astype(np.float32)
    except Exception:
        return None


def avalon_fp(
    smiles: str,
    n_bits: int = 1024,
) -> Optional[np.ndarray]:
    """
    Avalon fingerprint — substructure + pharmacophore hybrid.

    Avalon captures ring systems and pharmacophore features that Morgan
    misses because Morgan is purely topological. On ADMET datasets,
    Avalon+Morgan ensembles consistently outperform Morgan alone.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = pyAvalonTools.GetAvalonFP(mol, nBits=n_bits)
        return np.array(fp, dtype=np.float32)
    except Exception:
        return None


def erg_fp(
    smiles: str,
) -> Optional[np.ndarray]:
    """
    Extended Reduced Graph (ErG) fingerprint — 315-dimensional.

    ErG abstracts the molecule to a pharmacophore graph (H-bond donors/acceptors,
    positive/negative charges, hydrophobic atoms, aromatic rings) and computes
    path lengths between features. This captures scaffold-level shape that is
    largely invisible to atom-pair fingerprints.

    Fixed 315 dimensions (not tunable). Returns None if ErG computation fails
    (happens for some heavily functionalized compounds).
    """
    try:
        from rdkit.Chem.rdReducedGraphs import GetErGFingerprint
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = GetErGFingerprint(mol)
        return np.array(fp, dtype=np.float32)
    except Exception:
        return None


def morgan_count_fp_r3(
    smiles: str,
    n_bits: int = 2048,
) -> Optional[np.ndarray]:
    """
    Morgan count fingerprint at radius 3 (ECFP6-equivalent with counts).

    radius=3: captures up to 6 bonds from each atom center — important for PXR's
    large flexible LBD (~11 Å span) where longer-range scaffold topology matters.
    Complements the radius-2 (ECFP4) Morgan FP already in the feature set.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=n_bits)
        fp = gen.GetCountFingerprintAsNumPy(mol)
        return fp.astype(np.float32)
    except Exception:
        return None


def fcfp4_count_fp(
    smiles: str,
    n_bits: int = 2048,
) -> Optional[np.ndarray]:
    """
    Feature-count fingerprint at radius 2 (FCFP4-equivalent with counts).

    Maps atoms to pharmacophore categories (H-bond donor/acceptor, aromatic, hydrophobic,
    positive/negative charge) before computing Morgan paths. This captures pharmacophoric
    environments that atom-identity-based ECFP misses — complementary for PXR where
    H-bond donors to SER247/HIS407 and hydrophobic contacts dominate binding.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=2, fpSize=n_bits, atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
        )
        fp = gen.GetCountFingerprintAsNumPy(mol)
        return fp.astype(np.float32)
    except Exception:
        return None


def ecfp4_bitvect(
    smiles: str,
    n_bits: int = 1024,
) -> Optional[object]:
    """
    Returns an RDKit ExplicitBitVect (NOT a numpy array) for Tanimoto computations.

    Use this for Butina clustering and nearest-neighbor search.
    Use morgan_count_fp for machine learning features.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
        return gen.GetFingerprint(mol)
    except Exception:
        return None


def _compute_fp_row(smiles: str, fp_types: list[str]) -> Optional[np.ndarray]:
    """Compute concatenated fingerprints for one SMILES. Returns None on complete failure."""
    parts = []
    if "morgan_count" in fp_types:
        fp = morgan_count_fp(smiles)
        parts.append(fp if fp is not None else np.zeros(2048, dtype=np.float32))

    if "avalon" in fp_types:
        fp = avalon_fp(smiles)
        parts.append(fp if fp is not None else np.zeros(1024, dtype=np.float32))

    if "erg" in fp_types:
        fp = erg_fp(smiles)
        parts.append(fp if fp is not None else np.zeros(315, dtype=np.float32))

    if "morgan_count_r3" in fp_types:
        fp = morgan_count_fp_r3(smiles)
        parts.append(fp if fp is not None else np.zeros(2048, dtype=np.float32))

    if "fcfp4_count" in fp_types:
        fp = fcfp4_count_fp(smiles)
        parts.append(fp if fp is not None else np.zeros(2048, dtype=np.float32))

    if not parts:
        return None
    return np.concatenate(parts)


def build_fingerprint_matrix(
    smiles_list: list[str],
    fp_types: list[str] = ["morgan_count", "avalon", "erg"],
    n_jobs: int = -1,
) -> tuple[np.ndarray, list[str]]:
    """
    Computes all requested fingerprints and concatenates them column-wise.

    Failed rows (completely invalid SMILES) are filled with column-median
    values of the valid rows — computed after the full matrix is assembled.
    The fill values must be saved alongside the matrix so identical imputation
    can be applied to test data.

    Returns (feature_matrix of shape [n, total_bits], feature_names).
    """
    logger.info(f"Computing fingerprints {fp_types} for {len(smiles_list)} compounds...")
    rows = Parallel(n_jobs=n_jobs)(
        delayed(_compute_fp_row)(smi, fp_types) for smi in smiles_list
    )

    # Determine expected width from first valid row
    expected_width = next((r.shape[0] for r in rows if r is not None), None)
    if expected_width is None:
        raise ValueError("All SMILES failed fingerprint computation")

    n_failed = sum(1 for r in rows if r is None)
    if n_failed > 0:
        logger.warning(f"{n_failed} SMILES produced no fingerprint — will impute with zeros")

    matrix = np.stack(
        [r if r is not None else np.zeros(expected_width, dtype=np.float32) for r in rows]
    )

    # Build feature names
    feature_names = []
    if "morgan_count" in fp_types:
        feature_names += [f"morgan_{i}" for i in range(2048)]
    if "avalon" in fp_types:
        feature_names += [f"avalon_{i}" for i in range(1024)]
    if "erg" in fp_types:
        feature_names += [f"erg_{i}" for i in range(315)]
    if "morgan_count_r3" in fp_types:
        feature_names += [f"morgan_r3_{i}" for i in range(2048)]
    if "fcfp4_count" in fp_types:
        feature_names += [f"fcfp4_{i}" for i in range(2048)]

    logger.info(f"Fingerprint matrix: {matrix.shape}")
    return matrix, feature_names
