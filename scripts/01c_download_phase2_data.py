"""
Download Phase 2 data from HuggingFace.

New files released after Phase 1 close:
  - pxr-challenge_TEST_PHASE_1_UNBLINDED.csv  — Analog Set 1 with ground-truth pEC50
  - pxr-challenge_htchem-libraries_TRAIN.csv  — HTChem combined training data
  - pxr-challenge_96-compound-uscale-semi-pure_TRAIN.csv — 96-compound semi-pure upscale

Run this BEFORE re-running scripts/02_curate_data.py for Phase 2.

Usage:
  python scripts/01c_download_phase2_data.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import os
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from huggingface_hub import hf_hub_download

load_dotenv()

REPO = "openadmet/pxr-challenge-train-test"
CACHE = Path("data/raw")


def download_csv(filename: str, out_stem: str, hf_token: str | None = None) -> pd.DataFrame:
    """Downloads a CSV from the HuggingFace dataset repo and caches as parquet."""
    out_path = CACHE / f"{out_stem}.parquet"
    if out_path.exists():
        logger.info(f"Loading {out_stem} from cache")
        return pd.read_parquet(out_path)

    logger.info(f"Downloading {filename} from {REPO}...")
    local = hf_hub_download(
        repo_id=REPO,
        filename=filename,
        repo_type="dataset",
        token=hf_token or os.getenv("HF_TOKEN"),
    )
    df = pd.read_csv(local)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df.to_parquet(out_path, index=False)
    logger.info(f"  {out_stem}: {len(df)} rows, columns: {list(df.columns)}")
    return df


def _download_chemeleon_weights(
    repo: str = "openadmet/pxr-chemeleon-baseline",
    filename: str = "anvil_training/model.pth",
    local_dir: str = "models/chemeleon",
    hf_token: str | None = None,
) -> None:
    """Downloads CheMeleon pretrained weights for Chemprop 4-task initialization."""
    out = Path(local_dir) / filename
    if out.exists():
        logger.info(f"CheMeleon weights already present: {out}")
        return
    try:
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=repo,
            filename=filename,
            local_dir=local_dir,
            token=hf_token or os.getenv("HF_TOKEN"),
        )
        logger.info(f"CheMeleon weights saved: {out}")
    except Exception as e:
        logger.warning(
            f"CheMeleon download failed: {e}\n"
            f"Manual download: huggingface.co/{repo} → save as {out}"
        )


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    hf_token = os.getenv("HF_TOKEN")

    # Analog Set 1 unblinded (ground-truth pEC50 for ~253 test compounds)
    set1 = download_csv(
        "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
        "analog_set1_unblinded",
        hf_token,
    )
    logger.info(f"Analog Set 1 (unblinded): {len(set1)} compounds")

    # HTChem combined training data (yield-corrected activity)
    htchem = download_csv(
        "pxr-challenge_htchem-libraries_TRAIN.csv",
        "htchem_libraries",
        hf_token,
    )
    logger.info(f"HTChem libraries: {len(htchem)} compounds")

    # 96-compound semi-pure upscale
    semi = download_csv(
        "pxr-challenge_96-compound-uscale-semi-pure_TRAIN.csv",
        "htchem_semi_pure_96",
        hf_token,
    )
    logger.info(f"HTChem semi-pure 96: {len(semi)} compounds")

    # Save Set 1 labels as CSV for scripts/12_phase2_calibrate.py
    # Columns: molecule_name (id), pec50 (ground truth)
    id_col    = "molecule_name" if "molecule_name" in set1.columns else set1.columns[0]
    pec50_col = "pec50" if "pec50" in set1.columns else next(
        (c for c in set1.columns if "pec50" in c.lower()), None
    )
    if pec50_col:
        set1[[id_col, pec50_col]].rename(
            columns={id_col: "compound_id", pec50_col: "pec50"}
        ).to_csv(CACHE / "analog_set1_labels.csv", index=False)
        logger.info(f"Set 1 labels saved → data/raw/analog_set1_labels.csv ({len(set1)} rows)")
    else:
        logger.warning(f"Could not find pEC50 column in Set 1: {list(set1.columns)}")

    # Also download counter-assay and single-concentration if not already cached
    for fname, stem in [
        ("pxr-challenge_counter-assay_TRAIN.csv",       "openadmet_counter_assay"),
        ("pxr-challenge_single_concentration_TRAIN.csv", "openadmet_single_conc"),
    ]:
        p = CACHE / f"{stem}.parquet"
        if not p.exists():
            download_csv(fname, stem, hf_token)

    # CheMeleon pretrained weights for 4-task Chemprop
    _download_chemeleon_weights(hf_token=hf_token)

    logger.info("\n=== Phase 2 Download Summary ===")
    logger.info(f"Analog Set 1 (unblinded):  {len(set1):>5} compounds → data/raw/analog_set1_unblinded.parquet")
    logger.info(f"HTChem libraries:          {len(htchem):>5} compounds → data/raw/htchem_libraries.parquet")
    logger.info(f"HTChem semi-pure 96:       {len(semi):>5} compounds → data/raw/htchem_semi_pure_96.parquet")
    logger.info("\nNext: python scripts/02_curate_data.py")


if __name__ == "__main__":
    main()
