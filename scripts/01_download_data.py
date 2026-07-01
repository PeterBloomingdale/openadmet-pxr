"""
Download all data sources for the OpenADMET PXR challenge.

Run this FIRST before any other script.

What gets downloaded:
1. OpenADMET training + test sets (HuggingFace) — ~4k compounds + 513 test
2. Tox21 hPXR agonist qHTS (PubChem AID 1347033) — ~7.8k compounds
3. NCATS hPXR-luc qHTS (PubChem AID 720659) — ~2.8k compounds
4. ChEMBL CHEMBL3401 human PXR functional data — ~800 compounds
5. ToxCast ATG_PXR_TRANS_up — varies

All data is saved to data/raw/ in parquet format (never modified after download).
Re-running this script uses cached downloads — safe to run multiple times.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import os
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from openadmet.data.loaders import (
    load_openadmet_dataset,
    load_pubchem_assay,
    load_chembl_pxr,
    load_toxcast_pxr,
)


def main():
    logger.info("=== Downloading all PXR data sources ===")

    # 1. Primary data (OpenADMET HuggingFace)
    logger.info("Downloading OpenADMET dataset from HuggingFace...")
    train_df, test_df = load_openadmet_dataset(
        cache_dir="data/raw",
        hf_token=os.getenv("HF_TOKEN"),
    )
    logger.info(f"  Training: {len(train_df)} compounds")
    logger.info(f"  Test (blinded): {len(test_df)} compounds")
    logger.info(f"  Training columns: {list(train_df.columns)}")

    # 2. Tox21 hPXR (highest auxiliary transfer value)
    logger.info("Downloading Tox21 hPXR (AID 1347033)...")
    tox21_df = load_pubchem_assay(aid=1347033, cache_dir="data/raw")
    logger.info(f"  Tox21: {len(tox21_df)} compounds")

    # 3. NCATS hPXR-luc (independent lab confirmation)
    logger.info("Downloading NCATS hPXR-luc (AID 720659)...")
    ncats_df = load_pubchem_assay(aid=720659, cache_dir="data/raw")
    logger.info(f"  NCATS: {len(ncats_df)} compounds")

    # 4. ChEMBL (human functional, pChEMBL >= 5)
    logger.info("Downloading ChEMBL CHEMBL3401 human functional data...")
    chembl_df = load_chembl_pxr(target_chembl_id="CHEMBL3401", cache_dir="data/raw")
    logger.info(f"  ChEMBL: {len(chembl_df)} compounds")

    # 5. ToxCast (lower weight, different architecture)
    logger.info("Downloading ToxCast ATG_PXR_TRANS_up...")
    toxcast_df = load_toxcast_pxr(endpoint="ATG_PXR_TRANS_up", cache_dir="data/raw")
    logger.info(f"  ToxCast: {len(toxcast_df)} compounds")

    # Summary
    logger.info("\n=== Download Summary ===")
    logger.info(f"OpenADMET train:   {len(train_df):>6} compounds")
    logger.info(f"OpenADMET test:    {len(test_df):>6} compounds (blinded)")
    logger.info(f"Tox21 AID 1347033: {len(tox21_df):>6} compounds")
    logger.info(f"NCATS AID 720659:  {len(ncats_df):>6} compounds")
    logger.info(f"ChEMBL CHEMBL3401: {len(chembl_df):>6} compounds")
    logger.info(f"ToxCast ATG_PXR:   {len(toxcast_df):>6} compounds")
    logger.info(f"\nAll data saved to data/raw/")
    logger.info("Next: python scripts/01b_eda.py")


if __name__ == "__main__":
    main()
