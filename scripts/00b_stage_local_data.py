"""
Stage locally downloaded OpenADMET challenge CSVs into data/raw/ as parquet.

Run this INSTEAD of the HuggingFace download when you already have the CSVs:

    python scripts/00b_stage_local_data.py --src data-challenge/

What this does:
1. Reads the four challenge CSVs and renames columns to pipeline conventions.
2. Joins counter-assay pEC50/Emax onto the main training set by OCNT_ID.
3. Pivots single-concentration activation to per-compound weak-label features.
4. Writes data/raw/openadmet_train.parquet and data/raw/openadmet_test.parquet.

The parquet files match exactly what load_openadmet_dataset() would download
from HuggingFace, so the rest of the pipeline runs unchanged.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import numpy as np
import pandas as pd
from loguru import logger


def stage_train(src: Path, out: Path) -> pd.DataFrame:
    # ── Primary assay ──────────────────────────────────────────────
    train = pd.read_csv(src / "pxr-challenge_TRAIN.csv")

    rename_main = {
        "SMILES":                                       "smiles",
        "OCNT_ID":                                      "compound_id",
        "Molecule Name":                                "molecule_name",
        "pEC50":                                        "pec50",
        "Emax_estimate (log2FC vs. baseline)":          "emax",
        "Emax.vs.pos.ctrl_estimate (dimensionless)":    "emax_vs_ctrl",
        "pEC50_std.error (-log10(molarity))":           "pec50_se",
        "pEC50_ci.lower (-log10(molarity))":            "pec50_ci_lower",
        "pEC50_ci.upper (-log10(molarity))":            "pec50_ci_upper",
        "Emax_std.error (log2FC vs. baseline)":         "emax_se",
        "Split":                                        "split",
        "OCNT Batch":                                   "ocnt_batch",
    }
    train = train.rename(columns=rename_main)
    train["source"] = "openadmet"
    # hill_slope is not output by the fitting; mark as NaN
    train["hill_slope"] = np.nan

    # ── Counter-assay (PXR-null cell line) ────────────────────────
    counter = pd.read_csv(src / "pxr-challenge_counter-assay_TRAIN.csv")

    rename_counter = {
        "OCNT_ID":                                      "compound_id",
        "pEC50":                                        "counter_pec50",
        "Emax_estimate (log2FC vs. baseline)":          "counter_emax",
        "Emax.vs.pos.ctrl_estimate (dimensionless)":    "counter_emax_vs_ctrl",
    }
    counter = counter.rename(columns=rename_counter)[
        ["compound_id", "counter_pec50", "counter_emax", "counter_emax_vs_ctrl"]
    ]
    # 2,859 of 4,139 training compounds have counter-screen data; the rest get NaN
    logger.info(f"Counter-assay: {len(counter)} compounds (of {len(train)} training)")

    # ── Single-concentration weak labels ──────────────────────────
    # Four concentrations in the data: ~1µM, ~8µM, ~33µM, ~99µM
    # We keep the two most common screening concentrations as weak-label features.
    single = pd.read_csv(src / "pxr-challenge_single_concentration_TRAIN.csv")

    CONC_8UM  = 8.251e-06   # ≈ 8 µM plate concentration
    CONC_33UM = 3.3e-05     # ≈ 33 µM plate concentration

    def pivot_conc(df, conc, col_name):
        sub = df[np.isclose(df["concentration_M"], conc, rtol=0.05)].copy()
        sub = sub.rename(columns={"OCNT_ID": "compound_id", "log2_fc_estimate": col_name})
        # Multiple rows per compound (replicates) — take median
        return sub.groupby("compound_id")[col_name].median().reset_index()

    act_8  = pivot_conc(single, CONC_8UM,  "primary_activation_8um")
    act_33 = pivot_conc(single, CONC_33UM, "primary_activation_33um")
    logger.info(
        f"Single-conc: {len(act_8)} compounds at ~8µM, {len(act_33)} at ~33µM"
    )

    # ── Join everything ───────────────────────────────────────────
    df = (
        train
        .merge(counter, on="compound_id", how="left")
        .merge(act_8,   on="compound_id", how="left")
        .merge(act_33,  on="compound_id", how="left")
    )

    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "openadmet_train.parquet", index=False)
    logger.info(f"Saved {len(df)} training compounds → data/raw/openadmet_train.parquet")
    logger.info(f"  Columns: {list(df.columns)}")
    logger.info(f"  pEC50 range: {df['pec50'].min():.2f} – {df['pec50'].max():.2f}")
    logger.info(f"  Counter-screen coverage: {df['counter_pec50'].notna().sum()}/{len(df)}")
    return df


def stage_test(src: Path, out: Path) -> pd.DataFrame:
    test = pd.read_csv(src / "pxr-challenge_TEST_BLINDED.csv")
    test = test.rename(columns={"Molecule Name": "compound_id", "SMILES": "smiles"})
    test["source"] = "openadmet"
    out.mkdir(parents=True, exist_ok=True)
    test.to_parquet(out / "openadmet_test.parquet", index=False)
    logger.info(f"Saved {len(test)} test compounds → data/raw/openadmet_test.parquet")
    return test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data-challenge", help="Path to the challenge CSV folder")
    parser.add_argument("--out", default="data/raw", help="Output parquet directory")
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)

    required = [
        "pxr-challenge_TRAIN.csv",
        "pxr-challenge_TEST_BLINDED.csv",
        "pxr-challenge_counter-assay_TRAIN.csv",
        "pxr-challenge_single_concentration_TRAIN.csv",
    ]
    missing = [f for f in required if not (src / f).exists()]
    if missing:
        logger.error(f"Missing files in {src}: {missing}")
        sys.exit(1)

    logger.info(f"Staging OpenADMET data from {src} → {out}")
    train_df = stage_train(src, out)
    test_df  = stage_test(src, out)

    logger.info("\n=== Staging complete ===")
    logger.info(f"Training: {len(train_df)} compounds")
    logger.info(f"Test:     {len(test_df)} compounds (blinded)")
    logger.info("Next: python scripts/01_download_data.py  (fetches auxiliary sources)")
    logger.info("  or: python scripts/01b_eda.py            (if you skip auxiliary data)")


if __name__ == "__main__":
    main()
