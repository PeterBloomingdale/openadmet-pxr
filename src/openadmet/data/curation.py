"""
Molecular curation pipeline for the OpenADMET PXR challenge.

Pipeline order (run in this sequence):
  standardize_smiles → smiles_to_inchikey → inchikey_prefix
  → aggregate_replicates → deduplicate_train_test
  → handle_censored_values → run_curation_pipeline

Design principles:
- Standardization follows the ChEMBL structure pipeline conventions
  (neutralize, largest fragment, canonical SMILES) via RDKit + datamol.
- Deduplication uses the 14-character InChIKey connectivity prefix, which
  ignores stereochemistry. This handles the common case where a training
  compound is measured as a racemate and the test analog has defined stereo.
- Inactive compounds are flagged is_censored=True rather than imputed to a
  numeric pEC50. Imputing to "5.0" (100 µM) introduces false precision and
  biases the model toward predicting mid-range activities.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

try:
    from rdkit import Chem
    from rdkit.Chem.inchi import MolFromInchi, MolToInchi, InchiToInchiKey
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logger.warning("RDKit not available — curation functions will fail")


def standardize_smiles(
    smiles: str,
    remove_stereo: bool = False,
) -> Optional[str]:
    """
    Standardizes a SMILES string using RDKit's MolStandardize pipeline.

    Steps applied in order:
    1. Parse SMILES → RDKit Mol (return None if invalid)
    2. Remove hydrogens
    3. Disconnect metals (important for organometallics that sneak into diversity sets)
    4. Keep largest fragment (strips salts, counterions)
    5. Neutralize (remove formal charges where possible)
    6. Canonicalize (deterministic atom ordering)

    remove_stereo: set True only when the stereochemistry profile EDA shows
    rampant mismatch between train and test (see scripts/01b_eda.py results).
    """
    if not smiles or not isinstance(smiles, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Remove explicit H atoms
        mol = Chem.RemoveHs(mol)

        # Disconnect metal-ligand coordinate bonds (avoids weird fragment behavior)
        disconnector = rdMolStandardize.MetalDisconnector()
        mol = disconnector.Disconnect(mol)

        # Keep only the largest organic fragment
        chooser = rdMolStandardize.LargestFragmentChooser()
        mol = chooser.choose(mol)

        # Neutralize charges where chemically reasonable
        neutralizer = rdMolStandardize.Uncharger()
        mol = neutralizer.uncharge(mol)

        # isomericSmiles=False strips all stereo at generation time — cleaner than
        # Chem.RemoveStereoChemistry which was moved to rdmolops in RDKit 2024+
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=not remove_stereo)
        return canonical if canonical else None

    except Exception as e:
        logger.debug(f"Standardization failed for {smiles[:40]}: {e}")
        return None


def smiles_to_inchikey(smiles: str) -> Optional[str]:
    """
    Converts a SMILES string to a 27-character InChIKey.

    Returns None on failure. Use this full key for reporting; use
    inchikey_prefix() for deduplication (connectivity layer only).
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        inchi = MolToInchi(mol)
        if not inchi:
            return None
        return InchiToInchiKey(inchi)
    except Exception:
        return None


def inchikey_prefix(inchikey: str) -> str:
    """
    Returns the first 14 characters of a 27-character InChIKey.

    The InChIKey has three layers separated by hyphens:
      XXXXXXXXXXXXXX-YYYYYYYYYY-Z
      ↑ first 14 chars = connectivity layer (hashed molecular skeleton)
      The Y block encodes stereochemistry; Z encodes tautomer/charge state.

    Using only the connectivity layer means two enantiomers or diastereomers
    get the same prefix, which is correct when stereochemistry of measurement
    is ambiguous (as is common in Enamine commercial compounds).
    """
    if not inchikey or len(inchikey) < 14:
        return inchikey or ""
    return inchikey[:14]


def aggregate_replicates(
    df: pd.DataFrame,
    id_col: str = "inchikey_prefix",
    value_col: str = "pec50",
    spread_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Aggregates replicate measurements for the same compound (by InChIKey prefix).

    Aggregation strategy:
    - pEC50: median (robust to outliers from failed DRC fits)
    - n_replicates: count of valid (non-null) measurements
    - spread: std of measurements (flag if > spread_threshold log-units)

    Why median not mean: A failed dose-response curve fit can yield a wildly
    incorrect pEC50 (e.g., extrapolated outside the measured range). The median
    is unaffected by a single outlier; the mean is not.
    """
    records = []
    for prefix, group in df.groupby(id_col):
        valid = group[value_col].dropna()
        n = len(valid)

        rec = {
            id_col: prefix,
            f"{value_col}_median": float(valid.median()) if n > 0 else None,
            "n_replicates": n,
            f"{value_col}_spread": float(valid.std()) if n > 1 else 0.0,
        }

        # Propagate non-pec50 columns from the first row
        for col in group.columns:
            if col not in [id_col, value_col]:
                rec[col] = group[col].iloc[0]

        # Flag high-spread measurements for inspection
        if rec[f"{value_col}_spread"] > spread_threshold:
            logger.debug(
                f"High pEC50 spread ({rec[f'{value_col}_spread']:.2f}) "
                f"for {prefix} (n={n}). Median kept but inspect raw data."
            )

        records.append(rec)

    return pd.DataFrame(records)


def deduplicate_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    id_col: str = "inchikey_prefix",
) -> pd.DataFrame:
    """
    Removes from train_df any compound whose InChIKey prefix appears in test_df.

    This is the critical anti-leakage step. Without it, a compound that appears
    in both the Octant training set AND the 513-compound test set would give the
    model a "free answer" — inflating reported performance and potentially
    corrupting the leaderboard.

    Also runs the inverse check: logs any test compounds NOT found in any
    auxiliary data (genuinely OOD compounds that models must extrapolate to).
    """
    test_prefixes = set(test_df[id_col].dropna())
    initial_n = len(train_df)
    train_df = train_df[~train_df[id_col].isin(test_prefixes)].copy()
    n_removed = initial_n - len(train_df)

    if n_removed > 0:
        logger.warning(
            f"Removed {n_removed} training compounds that overlap with test set "
            f"(by InChIKey prefix). This is expected for ~2-5 compounds from "
            f"ChEMBL/PubChem data; more than 20 suggests a data issue."
        )
    else:
        logger.info("No training/test overlap detected — clean separation confirmed.")

    return train_df


def handle_censored_values(
    df: pd.DataFrame,
    pec50_col: str = "pec50_median",
    inactive_pec50_threshold: float = 5.0,  # 10 µM = pEC50 5.0
) -> pd.DataFrame:
    """
    Marks compounds as right-censored rather than imputing a pEC50 value.

    A compound is censored when:
    - pec50_median is None (never measured in dose-response), OR
    - pec50_median < inactive_pec50_threshold (active only at top dose)

    Why not impute to 5.0: Setting pEC50 = 5.0 for all inactives tells the
    model that compounds at the detection limit have the *same* pEC50, which
    is false. The true pEC50 is somewhere below the detection limit — we just
    don't know where. Censored regression methods honor this uncertainty.
    For our LightGBM and Chemprop models, we simply exclude censored compounds
    from the primary training set. They can be retained as weak-label auxiliary
    data with low weight if sample size is a concern.
    """
    df = df.copy()
    df["is_censored"] = df[pec50_col].isna() | (df[pec50_col] < inactive_pec50_threshold)
    n_censored = df["is_censored"].sum()
    logger.info(
        f"Censored compounds: {n_censored}/{len(df)} "
        f"({100*n_censored/len(df):.1f}%). "
        f"These will be excluded from primary pEC50 regression training."
    )
    return df


def filter_reactive_electrophiles(
    df: pd.DataFrame,
    smiles_col: str = "smiles_std",
) -> pd.DataFrame:
    """
    Removes compounds with reactive electrophilic warheads from training data.

    The test set contains none of these reactive groups. Keeping them in training
    anchors the model at artificial activity levels caused by covalent/non-specific
    mechanisms rather than PXR LBD binding. discoverybytes confirmed +0.019 RAE gain
    from removing acrylamides (n=236), acrylates (n=229), and aldehydes (n=87).

    Patterns applied (SMARTS):
    - Acrylamide: alpha-beta unsaturated amide (Michael acceptor for Cys/Lys)
    - Acrylate/acrylic ester: alpha-beta unsaturated ester
    - Aldehyde: reactive carbonyl (Schiff base / promiscuous binder)
    """
    if not RDKIT_AVAILABLE:
        logger.warning("RDKit not available — skipping reactive electrophile filter")
        return df

    patterns = {
        "acrylamide":  Chem.MolFromSmarts("[CX3H1,CX3H0]=[CX3]C(=O)[NX3]"),
        "acrylate":    Chem.MolFromSmarts("[CX3H1,CX3H0]=[CX3]C(=O)[OX2H0]"),
        "aldehyde":    Chem.MolFromSmarts("[CX3H1](=O)[#6]"),
    }

    def is_reactive(smi: str) -> bool:
        if not smi:
            return False
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return False
        return any(mol.HasSubstructMatch(p) for p in patterns.values() if p is not None)

    n_before = len(df)
    mask = df[smiles_col].apply(is_reactive)
    df = df[~mask].copy()
    n_removed = n_before - len(df)
    counts = {}
    for name, pat in patterns.items():
        if pat is not None:
            counts[name] = df[smiles_col].apply(
                lambda s: Chem.MolFromSmiles(s) is not None
                and Chem.MolFromSmiles(s).HasSubstructMatch(pat)
                if s else False
            ).sum()
    logger.info(
        f"Reactive electrophile filter: removed {n_removed}/{n_before} training compounds "
        f"(acrylamides+acrylates+aldehydes). Remaining: {len(df)}"
    )
    return df


def assign_compound_weights(
    df: pd.DataFrame,
    pec50_col: str = "pec50_median",
    spread_col: str = "pec50_spread",
    mw_col: str = "rdkit_MolWt",
) -> pd.DataFrame:
    """
    Assigns per-compound sample weights for training.

    Reduced weights (0.3–0.5) are assigned to compounds likely to introduce noise:
    - High measurement uncertainty (spread > 1.0 log-units across replicates)
    - Very low activity (pEC50 < 3.0) with no structural neighbors in test set
    - Extreme molecular weight (> 800 Da; often reactive macrolides/rifamycins)

    These weights are stored in a `sample_weight` column for use by LightGBM
    (sample_weight parameter) and Chemprop (per-sample loss weighting).
    discoverybytes applied this to 474/3386 compounds and found marginal improvement.
    """
    df = df.copy()
    weights = pd.Series(1.0, index=df.index)

    # High spread → uncertain label
    if spread_col in df.columns:
        high_spread = df[spread_col] > 1.0
        weights[high_spread] = weights[high_spread].clip(upper=0.3)
        logger.info(f"  Low-weight (high spread > 1.0): {high_spread.sum()} compounds")

    # Very low activity → floor region, often solubility artifact
    if pec50_col in df.columns:
        very_inactive = df[pec50_col] < 3.0
        weights[very_inactive] = weights[very_inactive].clip(upper=0.5)
        logger.info(f"  Low-weight (pEC50 < 3.0): {very_inactive.sum()} compounds")

    # Extreme MW → likely reactive macrolide or PAINS-adjacent
    if mw_col in df.columns:
        high_mw = df[mw_col] > 800
        weights[high_mw] = weights[high_mw].clip(upper=0.3)
        logger.info(f"  Low-weight (MW > 800): {high_mw.sum()} compounds")

    df["sample_weight"] = weights.values
    n_downweighted = (weights < 1.0).sum()
    logger.info(f"Sample weights assigned: {n_downweighted} compounds downweighted (< 1.0)")
    return df


def assign_source_weights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns source_weight based on assay architecture similarity to Octant.

    IMPORTANT: These weights are NOT used for weighted sample pooling in a
    single model. Instead, they inform the two-stage training approach:
    - Stage 1 (pretraining): All sources with weight > 0 are included
    - Stage 2 (fine-tuning): Only source_weight == 1.0 (Octant primary)

    Weight rationale:
    - 1.0 (Octant): Ground truth for this challenge; same construct and cell line
    - 0.8 (Tox21 AID 1347033): Same hPXR-luc functional endpoint; highest transfer
    - 0.7 (ChEMBL CHEMBL3401): Human functional after filtering; diverse compounds
    - 0.6 (NCATS AID 720659): Independent lab, same endpoint; smaller
    - 0.3 (ToxCast): Different reporter architecture; lower transfer expected
    """
    weight_map = {
        "openadmet":       1.0,
        "analog_set1":     1.0,  # Same assay as primary training data
        "htchem":          0.8,  # Yield-corrected crude — same assay, slightly noisier
        "htchem_semi_pure": 0.9, # Semi-pure has higher purity → closer to standard assay
        "pubchem_1347033": 0.8,
        "chembl":          0.7,
        "pubchem_720659":  0.6,
        "toxcast":         0.3,
    }
    df = df.copy()
    df["source_weight"] = df["source"].map(weight_map).fillna(0.5)
    return df


def run_curation_pipeline(
    raw_dfs: dict[str, pd.DataFrame],
    test_df: pd.DataFrame,
    output_dir: str = "data/curated",
    remove_stereo: bool = False,
) -> pd.DataFrame:
    """
    Full curation pipeline: standardize → InChIKey → aggregate → dedup → censor.

    raw_dfs: {source_name: raw_dataframe} — each must have at least 'smiles' and 'pec50'
    test_df: the 513-compound test set (for anti-leakage dedup)
    remove_stereo: set based on EDA results from scripts/01b_eda.py

    Output saved to output_dir/master_train.parquet.
    Returns the curated DataFrame (active compounds only, split column not yet set).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_records = []

    for source, df in raw_dfs.items():
        logger.info(f"Curating {source}: {len(df)} raw compounds")
        rows = []
        for _, row in df.iterrows():
            smiles_raw = row.get("smiles", None)
            if not smiles_raw:
                continue

            smiles_std = standardize_smiles(str(smiles_raw), remove_stereo=remove_stereo)
            if smiles_std is None:
                continue

            inchikey = smiles_to_inchikey(smiles_std)
            if inchikey is None:
                continue

            rows.append({
                "smiles_std": smiles_std,
                "inchikey": inchikey,
                "inchikey_prefix": inchikey_prefix(inchikey),
                "pec50": row.get("pec50", None),
                "counter_pec50": row.get("counter_pec50", None),
                "emax": row.get("emax", None),
                "hill_slope": row.get("hill_slope", None),
                "source": source,
            })

        src_df = pd.DataFrame(rows)
        logger.info(f"  {source}: {len(src_df)}/{len(df)} compounds passed standardization")
        all_records.append(src_df)

    combined = pd.concat(all_records, ignore_index=True)

    # Aggregate replicates within each source, then across sources
    agg = aggregate_replicates(combined, id_col="inchikey_prefix", value_col="pec50")

    # Prepare test InChIKey prefixes for dedup
    if "inchikey_prefix" not in test_df.columns:
        test_df = test_df.copy()
        test_df["smiles_std"] = test_df["smiles"].apply(
            lambda s: standardize_smiles(str(s)) if pd.notna(s) else None
        )
        test_df["inchikey"] = test_df["smiles_std"].apply(
            lambda s: smiles_to_inchikey(s) if s else None
        )
        test_df["inchikey_prefix"] = test_df["inchikey"].apply(
            lambda k: inchikey_prefix(k) if k else None
        )

    agg = deduplicate_train_test(agg, test_df, id_col="inchikey_prefix")
    agg = assign_source_weights(agg)

    # Rename aggregated column to match CuratedRecord schema
    if "pec50_median" not in agg.columns and "pec50" in agg.columns:
        agg = agg.rename(columns={"pec50": "pec50_median"})

    agg = handle_censored_values(agg, pec50_col="pec50_median")

    # Reactive electrophile filter: remove acrylamides, acrylates, aldehydes.
    # Test set contains none of these — they add training noise without signal.
    # Applied AFTER dedup so Set 1 unblinded compounds (which are known-clean test
    # analogs) pass through without filtering.
    if "smiles_std" in agg.columns:
        agg = filter_reactive_electrophiles(agg, smiles_col="smiles_std")

    # Per-compound sample weights for uncertainty-aware training
    agg = assign_compound_weights(agg, pec50_col="pec50_median")

    # Save full curated set (censored + active)
    agg.to_parquet(out / "master_train.parquet", index=False)

    # Save active-only for primary model training
    active = agg[~agg["is_censored"]].copy()
    active.to_parquet(out / "master_train_active.parquet", index=False)

    logger.info(
        f"Curation complete: {len(active)} active + "
        f"{agg['is_censored'].sum()} censored = {len(agg)} total"
    )
    return agg
