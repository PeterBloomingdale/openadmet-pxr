"""
2D molecular descriptor computation for the LightGBM tabular model.

Two descriptor sets are computed and concatenated:
1. RDKit 2D descriptors (~200 physicochemical properties)
2. Mordred 2D descriptors (~1600 raw, filtered to ~200-400 after variance/correlation pruning)

Why 2D descriptors alongside fingerprints:
Fingerprints encode substructure presence but are silent on global properties.
Descriptors like logP, molecular weight, TPSA, HBD count, ring counts, and
rotatable bonds are the strongest univariate predictors of PXR activity (the
LBD is hydrophobic and large — lipophilic, MW-heavy compounds predominate).
Including them explicitly prevents the model from "re-learning" these from
fingerprints, which is inefficient and inconsistent.

CRITICAL: The Mordred column filter (which descriptors pass) must be saved
to data/features/mordred_selected_columns.json so test data uses exactly the
same features. Running with a different filter on test data creates a silent
feature-dimension mismatch.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from joblib import Parallel, delayed

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# RDKit descriptors that produce NaN/inf on some molecules and break training
KNOWN_BROKEN_DESCRIPTORS = {
    "Ipc",          # Very slow and can overflow
    "BCUT2D_MWHI",  # Can be NaN for disconnected fragments
    "BCUT2D_MWLOW",
    "BCUT2D_CHGHI",
    "BCUT2D_CHGLO",
    "BCUT2D_LOGPHI",
    "BCUT2D_LOGPLOW",
    "BCUT2D_MRHI",
    "BCUT2D_MRLOW",
}


def rdkit_2d_descriptors(smiles: str) -> Optional[dict[str, float]]:
    """
    Computes all RDKit 2D molecular descriptors (~207 values).

    Returns a dict of {descriptor_name: value} or None if the molecule fails.
    Excludes descriptors in KNOWN_BROKEN_DESCRIPTORS.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        result = {}
        for name, fn in Descriptors.descList:
            if name in KNOWN_BROKEN_DESCRIPTORS:
                continue
            try:
                val = fn(mol)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    result[name] = float(val)
                else:
                    result[name] = np.nan
            except Exception:
                result[name] = np.nan

        return result
    except Exception:
        return None


def mordred_descriptors(smiles: str, ignore_3d: bool = True) -> Optional[dict[str, float]]:
    """
    Computes Mordred 2D molecular descriptors (~1600 raw values).

    ignore_3d=True skips descriptors that require a 3D conformer (saves time
    and avoids conformer generation variance). With ignore_3d=True, mordred
    still computes ~1600 2D descriptors.

    Returns dict of {descriptor_name: value} or None if molecule fails.
    Mordred returns a mordred.Result object; we convert to plain dict.
    """
    try:
        from mordred import Calculator, descriptors as mordred_descs
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        calc = Calculator(mordred_descs, ignore_3D=ignore_3d)
        result = calc(mol)

        out = {}
        for key, val in result.items():
            try:
                f = float(val)
                out[str(key)] = f if not np.isnan(f) else np.nan
            except (TypeError, ValueError):
                out[str(key)] = np.nan

        return out
    except Exception as e:
        logger.debug(f"Mordred failed for {smiles[:30]}: {e}")
        return None


def pmi_3d_descriptors(smiles: str) -> Optional[dict[str, float]]:
    """
    Computes 3D PMI-derived shape descriptors from a single ETKDGv3 conformer.

    19 descriptors capturing rod-like, disc-like, and spherical shape — orthogonal
    to 2D message-passing graph representations. discoverybytes found these additive
    to CheMeleon + ECFP4 features.

    Conformer generation: ETKDGv3 (distance geometry) with MMFF94 optimization.
    Falls back to ETKDGv2 if ETKDGv3 fails. Returns None if both fail.
    """
    try:
        from rdkit.Chem import AllChem, rdMolDescriptors

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)

        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        if AllChem.EmbedMolecule(mol, params) != 0:
            params2 = AllChem.ETKDG()
            params2.randomSeed = 42
            if AllChem.EmbedMolecule(mol, params2) != 0:
                return None

        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)

        result: dict[str, float] = {}

        # Each descriptor wrapped individually — RDKit API varies by version
        def _safe(name: str, fn):
            try:
                result[name] = float(fn())
            except Exception:
                result[name] = np.nan

        _safe("pmi1", lambda: rdMolDescriptors.CalcPMI1(mol))
        _safe("pmi2", lambda: rdMolDescriptors.CalcPMI2(mol))
        _safe("pmi3", lambda: rdMolDescriptors.CalcPMI3(mol))
        _safe("npr1", lambda: rdMolDescriptors.CalcNPR1(mol))
        _safe("npr2", lambda: rdMolDescriptors.CalcNPR2(mol))
        _safe("radius_of_gyration",   lambda: rdMolDescriptors.CalcRadiusOfGyration(mol))
        _safe("asphericity",          lambda: rdMolDescriptors.CalcAsphericity(mol))
        _safe("eccentricity",         lambda: rdMolDescriptors.CalcEccentricity(mol))
        _safe("inertial_shape_factor",lambda: rdMolDescriptors.CalcInertialShapeFactor(mol))
        _safe("spherocity_index",     lambda: rdMolDescriptors.CalcSpherocityIndex(mol))
        _safe("plane_of_best_fit",    lambda: rdMolDescriptors.CalcPBF(mol))

        # Derived shape ratios
        pmi1 = result.get("pmi1", np.nan)
        pmi2 = result.get("pmi2", np.nan)
        pmi3 = result.get("pmi3", np.nan)
        npr1 = result.get("npr1", np.nan)
        npr2 = result.get("npr2", np.nan)
        safe_pmi3 = pmi3 if (not np.isnan(pmi3) and pmi3 > 1e-10) else 1e-10
        result["pmi_ratio_12"]   = pmi1 / (pmi2 + 1e-10)
        result["pmi_ratio_13"]   = pmi1 / safe_pmi3
        result["pmi_ratio_23"]   = pmi2 / safe_pmi3
        result["npr_sum"]        = npr1 + npr2
        result["npr_diff"]       = npr2 - npr1
        result["pmi_anisotropy"] = (pmi3 - pmi1) / (safe_pmi3 + 1e-10)

        return result

    except Exception as e:
        logger.debug(f"PMI 3D failed for {smiles[:30]}: {e}")
        return None


def filter_mordred_descriptors(
    desc_df: pd.DataFrame,
    max_missing_frac: float = 0.20,
    variance_threshold: float = 0.001,
    max_corr: float = 0.95,
    save_path: Optional[str] = "data/features/mordred_selected_columns.json",
) -> pd.DataFrame:
    """
    Filters Mordred descriptors to remove uninformative and redundant columns.

    Filter steps (applied in order):
    1. Drop columns with > max_missing_frac NaN values (too sparse to train on)
    2. Drop near-zero-variance columns (constant or near-constant across compounds)
    3. Drop one of each highly correlated pair (|Pearson r| > max_corr)

    CRITICAL: Saves selected column names to save_path so test data uses the
    EXACT SAME columns. If save_path exists, loading is skipped and the saved
    columns are used instead (for test data inference).

    Returns filtered DataFrame.
    """
    if save_path and Path(save_path).exists():
        with open(save_path) as f:
            selected_cols = json.load(f)
        available = [c for c in selected_cols if c in desc_df.columns]
        if len(available) < len(selected_cols):
            logger.warning(
                f"Mordred filter: {len(selected_cols) - len(available)} saved columns "
                f"not found in current DataFrame. Using {len(available)} available."
            )
        return desc_df[available].copy()

    logger.info(f"Filtering Mordred descriptors: starting with {desc_df.shape[1]} columns")

    # Step 1: Drop high-missing columns — use adaptive threshold for large diverse datasets
    missing_frac = desc_df.isna().mean()
    keep = missing_frac[missing_frac <= max_missing_frac].index.tolist()
    if not keep:
        # Fallback: keep columns with < 50% missing rather than return empty
        keep = missing_frac[missing_frac < 0.50].index.tolist()
        logger.warning(
            f"No features survived max_missing_frac={max_missing_frac:.0%} — "
            f"relaxing to <50% missing ({len(keep)} features)"
        )
    desc_df = desc_df[keep]
    logger.info(f"After missing-fraction filter: {len(keep)} columns")

    # Fill remaining NaN for variance/correlation computation
    desc_df = desc_df.fillna(desc_df.median())

    # Step 2: Drop near-zero variance
    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold(threshold=variance_threshold)
    try:
        selector.fit(desc_df)
        keep = [col for col, support in zip(desc_df.columns, selector.get_support()) if support]
    except ValueError:
        # All features below threshold — relax to 0 (keep anything non-constant)
        logger.warning("All features below variance threshold — relaxing to threshold=0")
        selector = VarianceThreshold(threshold=0.0)
        try:
            selector.fit(desc_df)
            keep = [col for col, support in zip(desc_df.columns, selector.get_support()) if support]
        except ValueError:
            keep = list(desc_df.columns)
    desc_df = desc_df[keep]
    logger.info(f"After variance filter: {len(keep)} columns")

    # Step 3: Drop highly correlated columns (keep first of each correlated pair)
    corr_matrix = desc_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > max_corr)]
    desc_df = desc_df.drop(columns=to_drop)
    logger.info(f"After correlation filter ({max_corr}): {desc_df.shape[1]} columns remaining")

    # Save selected columns for test consistency
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(desc_df.columns.tolist(), f, indent=2)
        logger.info(f"Saved {len(desc_df.columns)} Mordred column names to {save_path}")

    return desc_df


def _apply_or_save_medians(df: pd.DataFrame, save_path: Optional[str]) -> pd.DataFrame:
    """
    Train/test consistent NaN imputation.

    On train (save_path does not exist): compute medians, save to disk, fill.
    On test (save_path exists): load saved train medians, fill. Any column
    missing from the saved medians falls back to the test median (rare).

    This prevents the silent train/test mismatch where the test frame is
    imputed with TEST medians (computed on the 513 analog set) while the
    train frame is imputed with TRAIN medians.
    """
    if save_path and Path(save_path).exists():
        with open(save_path) as f:
            saved = json.load(f)
        medians = pd.Series({c: saved.get(c, float(df[c].median())) for c in df.columns})
        return df.fillna(medians)

    medians = df.median(numeric_only=True)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump({k: float(v) for k, v in medians.to_dict().items()}, f)
    return df.fillna(medians)


def build_descriptor_matrix(
    smiles_list: list[str],
    include_rdkit: bool = True,
    include_mordred: bool = True,
    include_pmi: bool = True,
    mordred_save_path: Optional[str] = "data/features/mordred_selected_columns.json",
    rdkit_medians_path: Optional[str] = "data/features/rdkit_medians.json",
    mordred_medians_path: Optional[str] = "data/features/mordred_medians.json",
    pmi_medians_path: Optional[str] = "data/features/pmi_medians.json",
    n_jobs: int = -1,
) -> tuple[np.ndarray, list[str]]:
    """
    Computes 2D descriptor matrix for a list of SMILES.

    After computation:
    - Missing values imputed with training-set column medians (saved to disk
      on the first call so the second call — test — uses the SAME medians)
    - Mordred columns filtered (or loaded from saved filter)

    Returns (feature_matrix [n, d], feature_names [d]).
    """
    parts = []
    feature_names: list[str] = []

    if include_rdkit:
        logger.info("Computing RDKit 2D descriptors...")
        rdkit_rows = Parallel(n_jobs=n_jobs)(
            delayed(rdkit_2d_descriptors)(smi) for smi in smiles_list
        )
        rdkit_df = pd.DataFrame(rdkit_rows)
        rdkit_df = _apply_or_save_medians(rdkit_df, rdkit_medians_path)
        parts.append(rdkit_df.values.astype(np.float32))
        feature_names += [f"rdkit_{c}" for c in rdkit_df.columns]
        logger.info(f"RDKit 2D: {rdkit_df.shape[1]} descriptors")

    if include_mordred:
        logger.info("Computing Mordred descriptors (this takes ~5 min for 5000 compounds)...")
        mordred_rows = Parallel(n_jobs=n_jobs)(
            delayed(mordred_descriptors)(smi) for smi in smiles_list
        )
        mordred_df = pd.DataFrame(mordred_rows)
        mordred_df = filter_mordred_descriptors(mordred_df, save_path=mordred_save_path)
        mordred_df = _apply_or_save_medians(mordred_df, mordred_medians_path)
        parts.append(mordred_df.values.astype(np.float32))
        feature_names += [f"mordred_{c}" for c in mordred_df.columns]
        logger.info(f"Mordred (filtered): {mordred_df.shape[1]} descriptors")

    if include_pmi:
        logger.info("Computing PMI 3D shape descriptors (conformer generation per compound)...")
        pmi_rows = Parallel(n_jobs=n_jobs)(
            delayed(pmi_3d_descriptors)(smi) for smi in smiles_list
        )
        pmi_df = pd.DataFrame(pmi_rows)
        pmi_df = _apply_or_save_medians(pmi_df, pmi_medians_path)
        parts.append(pmi_df.values.astype(np.float32))
        feature_names += [f"pmi_{c}" for c in pmi_df.columns]
        logger.info(f"PMI 3D: {pmi_df.shape[1]} descriptors")

    if not parts:
        raise ValueError("No descriptor types requested")

    matrix = np.concatenate(parts, axis=1)
    logger.info(f"Descriptor matrix: {matrix.shape}")
    return matrix, feature_names
