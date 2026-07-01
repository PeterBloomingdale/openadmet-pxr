"""
Data loaders for all PXR activity sources.

Download order: OpenADMET (primary) → PubChem AID 1347033 → AID 720659 → ChEMBL → ToxCast
Each loader returns a raw DataFrame; curation.py handles all standardization.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from loguru import logger


def load_openadmet_dataset(
    cache_dir: str = "data/raw",
    hf_token: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Downloads train + test splits from the HuggingFace openadmet/pxr-challenge dataset.

    The training set contains:
    - smiles, compound_id, pec50, emax, hill_slope (from 8-pt dose-response)
    - counter_pec50, counter_emax (from PXR-null cell line counter-screen)
    - primary_activation_10um, primary_activation_30um (weak-label single-conc screen)

    The test set contains: smiles, compound_id (no labels — blinded)

    Returns (train_df, test_df).
    """
    from datasets import load_dataset

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    train_path = cache / "openadmet_train.parquet"
    test_path = cache / "openadmet_test.parquet"

    if train_path.exists() and test_path.exists():
        logger.info("Loading OpenADMET data from cache")
        return pd.read_parquet(train_path), pd.read_parquet(test_path)

    logger.info("Downloading OpenADMET dataset from HuggingFace...")
    token = hf_token or os.getenv("HF_TOKEN")
    dataset = load_dataset(
        "openadmet/pxr-challenge",
        token=token,
    )

    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    logger.info(f"OpenADMET: {len(train_df)} train, {len(test_df)} test compounds")
    return train_df, test_df


def load_pubchem_assay(
    aid: int,
    cache_dir: str = "data/raw",
) -> pd.DataFrame:
    """
    Downloads a PubChem BioAssay via the PubChem REST API.

    Supported AIDs:
    - 1347033: Tox21 hPXR agonist qHTS (~7,871 compounds) — highest transfer value
    - 720659:  NCATS hPXR-luc qHTS (~2,800 compounds) — independent lab confirmation

    Returns DataFrame with columns: [smiles, pec50, outcome, aid, cid].
    pec50 is derived from AC50 in µM: pec50 = -log10(AC50 * 1e-6).
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    out_path = cache / f"pubchem_aid{aid}.parquet"

    if out_path.exists():
        logger.info(f"Loading PubChem AID {aid} from cache")
        return pd.read_parquet(out_path)

    logger.info(f"Downloading PubChem AID {aid}...")

    # PUG REST concise CSV — replaced legacy activitycsv endpoint (deprecated ~2025)
    # Returns: SID, CID, Activity Outcome, Activity Score, Activity URL, Assay Name
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/concise/CSV"
    response = requests.get(url, timeout=120)
    response.raise_for_status()

    from io import StringIO
    raw_csv = StringIO(response.text)
    df = pd.read_csv(raw_csv)  # PUG REST CSV has a single header row
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # PUG REST concise format uses "activity_outcome" and "activity_score" (no AC50).
    # Map to ac50 equivalent if a numeric score column exists.
    # CID column is named "cid" directly.
    ac50_col = next((c for c in df.columns if any(k in c.lower() for k in ("ac50", "potency", "activity_value"))), None)

    records = []
    for _, row in df.iterrows():
        cid = row.get("cid", None)
        outcome = str(row.get("activity_outcome", "")).upper()
        ac50_um = row[ac50_col] if ac50_col else None

        pec50 = None
        if ac50_um is not None and pd.notna(ac50_um) and float(ac50_um) > 0:
            pec50 = round(6.0 - np.log10(float(ac50_um)), 4)

        records.append({"cid": cid, "pec50": pec50, "outcome": outcome, "aid": aid})

    result_df = pd.DataFrame(records)

    # Fetch SMILES for CIDs in bulk (PUG REST)
    cids = [str(int(c)) for c in result_df["cid"].dropna().unique()]
    smiles_map = _fetch_pubchem_smiles(cids)
    result_df["smiles"] = result_df["cid"].map(lambda c: smiles_map.get(str(int(c)) if pd.notna(c) else "", None))
    result_df = result_df[result_df["smiles"].notna()].copy()

    result_df.to_parquet(out_path, index=False)
    logger.info(f"PubChem AID {aid}: {len(result_df)} compounds with SMILES")
    return result_df


def _fetch_pubchem_smiles(cids: list[str], chunk_size: int = 200) -> dict[str, str]:
    """Fetches canonical SMILES for a list of PubChem CIDs via PUG REST."""
    smiles_map: dict[str, str] = {}
    # IsomericSMILES preferred; PubChem dropped CanonicalSMILES from this endpoint ~2025
    base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cids}/property/IsomericSMILES,ConnectivitySMILES/JSON"

    for i in range(0, len(cids), chunk_size):
        chunk = cids[i : i + chunk_size]
        url = base.format(cids=",".join(chunk))
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            for prop in data.get("PropertyTable", {}).get("Properties", []):
                smi = prop.get("IsomericSMILES") or prop.get("ConnectivitySMILES", "")
                smiles_map[str(prop["CID"])] = smi
        except Exception as e:
            logger.warning(f"PubChem SMILES fetch failed for chunk {i}: {e}")

    return smiles_map


def load_chembl_pxr(
    target_chembl_id: str = "CHEMBL3401",
    cache_dir: str = "data/raw",
    min_pchembl: float = 5.0,
    assay_type: str = "F",             # F = functional (cell-based); B = binding
) -> pd.DataFrame:
    """
    Queries ChEMBL for human PXR agonist data.

    Filters strictly to:
    - Human target (target_organism = 'Homo sapiens')
    - Functional assays (assay_type = 'F') — cell-based reporter, not binding
    - pChEMBL value >= min_pchembl (5.0 = 10 µM activity threshold)

    Why exclude binding assays: Our Octant assay measures transcriptional induction
    (functional endpoint), not direct binding affinity. Binding IC50s don't predict
    induction EC50s reliably due to the conformational change mechanism of PXR.

    Returns DataFrame with [smiles, pec50, chembl_id, assay_chembl_id].
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    out_path = cache / "chembl_pxr.parquet"

    if out_path.exists():
        logger.info("Loading ChEMBL PXR data from cache")
        return pd.read_parquet(out_path)

    logger.info(f"Querying ChEMBL for target {target_chembl_id}...")
    from chembl_webresource_client.new_client import new_client

    activity = new_client.activity
    results = activity.filter(
        target_chembl_id=target_chembl_id,
        assay_type=assay_type,
        pchembl_value__gte=min_pchembl,
    ).only(
        "molecule_chembl_id",
        "canonical_smiles",
        "pchembl_value",
        "assay_chembl_id",
        "target_organism",
    )

    records = []
    for r in results:
        if r.get("target_organism", "").lower() != "homo sapiens":
            continue
        smiles = r.get("canonical_smiles")
        pec50 = r.get("pchembl_value")
        if smiles and pec50:
            records.append({
                "smiles": smiles,
                "pec50": float(pec50),
                "chembl_id": r.get("molecule_chembl_id"),
                "assay_chembl_id": r.get("assay_chembl_id"),
            })

    df = pd.DataFrame(records)
    df.to_parquet(out_path, index=False)
    logger.info(f"ChEMBL: {len(df)} human functional PXR records (pChEMBL >= {min_pchembl})")
    return df


def load_toxcast_pxr(
    endpoint: str = "ATG_PXR_TRANS_up",
    cache_dir: str = "data/raw",
) -> pd.DataFrame:
    """
    Loads ToxCast PXR data from the EPA CompTox invitroDB.

    Note: ToxCast uses a different reporter architecture and concentration range
    than the Octant assay. Assign source_weight=0.3 at curation time.

    Returns DataFrame with [smiles, pec50, endpoint, dsstox_sid].
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    out_path = cache / f"toxcast_{endpoint}.parquet"

    if out_path.exists():
        logger.info(f"Loading ToxCast {endpoint} from cache")
        return pd.read_parquet(out_path)

    logger.info(f"Downloading ToxCast {endpoint} from EPA CompTox API...")
    url = "https://comptox.epa.gov/dashboard/api/bioactivity"
    params = {
        "endpoint": endpoint,
        "format": "json",
        "limit": 5000,
    }
    try:
        resp = requests.get(url, params=params, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        records = []
        for item in data.get("results", []):
            ac50_um = item.get("ac50")
            smiles = item.get("smiles") or item.get("qsar_smiles")
            if smiles and ac50_um and float(ac50_um) > 0:
                pec50 = round(6.0 - np.log10(float(ac50_um)), 4)
                records.append({
                    "smiles": smiles,
                    "pec50": pec50,
                    "endpoint": endpoint,
                    "dsstox_sid": item.get("dsstox_substance_id"),
                })

        df = pd.DataFrame(records)
    except Exception as e:
        logger.warning(f"ToxCast download failed: {e}. Returning empty DataFrame.")
        df = pd.DataFrame(columns=["smiles", "pec50", "endpoint", "dsstox_sid"])

    df.to_parquet(out_path, index=False)
    logger.info(f"ToxCast {endpoint}: {len(df)} compounds")
    return df
