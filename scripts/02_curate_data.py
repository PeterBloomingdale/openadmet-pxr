"""
Run the full data curation pipeline.

Prerequisites: scripts/01_download_data.py and scripts/01b_eda.py

What this does:
1. Loads all raw data sources from data/raw/
2. Runs standardization, dedup, aggregation, censoring
3. Saves data/curated/master_train.parquet (all) and master_train_active.parquet (uncensored)

Decision point: set REMOVE_STEREO based on Q5 from scripts/01b_eda.py results.
If test_frac_with_stereo < 0.1 and train_frac_with_stereo > 0.2, set REMOVE_STEREO=True.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import pandas as pd
from loguru import logger

from openadmet.data.curation import run_curation_pipeline


def main():
    # Check EDA results for stereochemistry decision
    eda_path = Path("data/eda/eda_summary.json")
    remove_stereo = False
    if eda_path.exists():
        with open(eda_path) as f:
            eda = json.load(f)
        stereo_rec = eda.get("q5_stereochemistry", {}).get("recommendation", "KEEP STEREO")
        remove_stereo = "STRIP" in stereo_rec
        logger.info(f"Stereochemistry decision from EDA: {stereo_rec} → remove_stereo={remove_stereo}")
    else:
        logger.warning("EDA results not found — defaulting to remove_stereo=False. "
                       "Run scripts/01b_eda.py first!")

    # Load all raw data
    raw_dfs = {}

    train_path = Path("data/raw/openadmet_train.parquet")
    test_path = Path("data/raw/openadmet_test.parquet")
    if not train_path.exists():
        logger.error("OpenADMET training data not found. Run scripts/01_download_data.py first.")
        return

    raw_dfs["openadmet"] = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    for source, fname in [
        ("pubchem_1347033", "pubchem_aid1347033.parquet"),
        ("pubchem_720659", "pubchem_aid720659.parquet"),
        ("chembl", "chembl_pxr.parquet"),
        ("toxcast", "toxcast_ATG_PXR_TRANS_up.parquet"),
    ]:
        p = Path("data/raw") / fname
        if p.exists():
            raw_dfs[source] = pd.read_parquet(p)
            logger.info(f"Loaded {source}: {len(raw_dfs[source])} compounds")
        else:
            logger.warning(f"{source} not found at {p} — skipping")

    # Phase 2: Analog Set 1 (unblinded ground-truth labels).
    # Columns: molecule_name (id), smiles, pec50 — maps directly to curation pipeline.
    p_set1 = Path("data/raw/analog_set1_unblinded.parquet")
    if p_set1.exists():
        df_set1 = pd.read_parquet(p_set1)
        # Rename to match curation pipeline expectations
        df_set1 = df_set1.rename(columns={"molecule_name": "compound_id"})
        raw_dfs["analog_set1"] = df_set1
        logger.info(f"Loaded analog_set1: {len(df_set1)} compounds")
    else:
        logger.info("Analog Set 1 not found — run scripts/01c_download_phase2_data.py")

    # HTChem libraries (crude + semi-pure combined, yield-corrected pEC50)
    p_htchem = Path("data/raw/htchem_libraries.parquet")
    if p_htchem.exists():
        df_htchem = pd.read_parquet(p_htchem)
        # Use yield-corrected pEC50 (the most reliable column)
        pec50_col = next(
            (c for c in df_htchem.columns if "corrected" in c and "pec50" in c), None
        ) or next((c for c in df_htchem.columns if "pec50" in c), None)
        if pec50_col:
            df_htchem[pec50_col] = pd.to_numeric(df_htchem[pec50_col], errors="coerce")
            df_htchem = df_htchem.rename(columns={pec50_col: "pec50", "ocnt_id": "compound_id"})
        raw_dfs["htchem"] = df_htchem
        logger.info(f"Loaded htchem: {len(df_htchem)} compounds (pEC50 from {pec50_col})")
    else:
        logger.info("HTChem libraries not found — run scripts/01c_download_phase2_data.py")

    # HTChem semi-pure 96-compound upscale
    p_semi = Path("data/raw/htchem_semi_pure_96.parquet")
    if p_semi.exists():
        df_semi = pd.read_parquet(p_semi)
        pec50_col = next(
            (c for c in df_semi.columns if "corrected" in c and "pec50" in c), None
        ) or next((c for c in df_semi.columns if "pec50" in c), None)
        if pec50_col:
            df_semi[pec50_col] = pd.to_numeric(df_semi[pec50_col], errors="coerce")
            df_semi = df_semi.rename(columns={pec50_col: "pec50", "ocnt_id": "compound_id"})
        raw_dfs["htchem_semi_pure"] = df_semi
        logger.info(f"Loaded htchem_semi_pure: {len(df_semi)} compounds (pEC50 from {pec50_col})")
    else:
        logger.info("HTChem semi-pure 96 not found — run scripts/01c_download_phase2_data.py")

    # Analog Set 1 bypasses the anti-leakage dedup (it IS test-set compounds, but
    # now labeled). Pass it separately so it isn't removed by deduplicate_train_test.
    analog_set1_raw = raw_dfs.pop("analog_set1", None)
    raw_dfs.pop("analog_set1", None)  # ensure removed from main pipeline

    curated = run_curation_pipeline(
        raw_dfs=raw_dfs,
        test_df=test_df,
        output_dir="data/curated",
        remove_stereo=remove_stereo,
    )

    # Now append Analog Set 1 to the curated set — these are Phase 1 unblinded
    # test compounds with confirmed labels. Scoring in Phase 2 is on Set 2 only.
    if analog_set1_raw is not None:
        from openadmet.data.curation import (
            standardize_smiles, smiles_to_inchikey, inchikey_prefix, handle_censored_values
        )
        rows = []
        for _, row in analog_set1_raw.iterrows():
            smi_raw = row.get("smiles", "")
            if not smi_raw:
                continue
            smi_std = standardize_smiles(str(smi_raw), remove_stereo=remove_stereo)
            if smi_std is None:
                smi_std = str(smi_raw)
            ik = smiles_to_inchikey(smi_std)
            rows.append({
                "smiles_std": smi_std,
                "inchikey": ik,
                "inchikey_prefix": inchikey_prefix(ik) if ik else None,
                "pec50_median": pd.to_numeric(row.get("pec50"), errors="coerce"),
                "emax": pd.to_numeric(
                    row.get("emax_estimate_(log2fc_vs._baseline)"), errors="coerce"
                ),
                "source": "analog_set1",
                "source_weight": 1.0,
                "is_censored": False,
                "sample_weight": 1.0,
            })
        set1_df = pd.DataFrame(rows).dropna(subset=["pec50_median"])
        curated = pd.concat([curated, set1_df], ignore_index=True)
        logger.info(f"Appended {len(set1_df)} Analog Set 1 compounds (post-dedup override)")
        # Overwrite saved parquet with Set 1 included
        from pathlib import Path as _P
        curated.to_parquet(_P("data/curated/master_train.parquet"), index=False)
        active = curated[~curated["is_censored"]].copy()
        active.to_parquet(_P("data/curated/master_train_active.parquet"), index=False)

    logger.info(f"\nCuration complete:")
    logger.info(f"  Total curated: {len(curated)}")
    logger.info(f"  Active (uncensored): {(~curated['is_censored']).sum()}")
    logger.info(f"  Censored: {curated['is_censored'].sum()}")
    logger.info(f"  Sources: {curated['source'].value_counts().to_dict()}")
    logger.info(f"\nNext: python scripts/04_build_features.py")


if __name__ == "__main__":
    main()
