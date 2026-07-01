"""
Phase 2 calibration — run on May 26, 2026 when Analog Set 1 labels drop.

Steps:
1. Load Set 1 labels (downloaded from leaderboard or provided by organizers)
2. Load current ensemble predictions on Set 1
3. Fit linear calibration y_corr = a * y_hat + b
4. Generate calibration diagnostic plot (INSPECT before submitting!)
5. Apply calibration to Set 2 predictions
6. Save calibrated submission

Usage:
  python scripts/12_phase2_calibrate.py \
    --set1_labels data/raw/analog_set1_labels.csv \
    --set2_preds submissions/phase2/ensemble_raw.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import numpy as np
import pandas as pd
from loguru import logger

from openadmet.calibration.residual import (
    fit_residual_calibration,
    apply_residual_calibration,
    diagnostic_calibration_plot,
)
from openadmet.utils.submission import format_submission


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set1_labels", required=True, help="CSV with compound_id,pec50 for Set 1")
    parser.add_argument("--set2_preds", required=True, help="CSV with compound_id,pec50_pred for full 513 compounds")
    args = parser.parse_args()

    # Load Set 1 labels
    set1_labels = pd.read_csv(args.set1_labels)
    logger.info(f"Loaded {len(set1_labels)} Set 1 labels")

    # Load ensemble predictions on all 513 compounds
    all_preds = pd.read_csv(args.set2_preds)

    # Normalise column names — submission uses "Molecule Name"/"pEC50" format
    id_col = "Molecule Name" if "Molecule Name" in all_preds.columns else "compound_id"
    pred_col = "pEC50" if "pEC50" in all_preds.columns else "pec50_pred"
    all_preds = all_preds.rename(columns={id_col: "compound_id", pred_col: "pec50_pred"})

    # Match Set 1 predictions to labels
    set1_preds = all_preds[all_preds["compound_id"].isin(set1_labels["compound_id"])].copy()
    merged = pd.merge(set1_preds, set1_labels, on="compound_id", suffixes=("_pred", "_true"))

    y_pred_set1 = merged["pec50_pred"].values
    y_true_set1 = merged["pec50"].values  # set1_labels column is "pec50"

    logger.info(f"Calibration on {len(y_pred_set1)} Set 1 compounds")
    logger.info(f"  Pre-calibration MAE: {np.mean(np.abs(y_true_set1 - y_pred_set1)):.4f}")

    # Fit calibration
    a, b = fit_residual_calibration(y_pred_set1, y_true_set1)

    # Plot diagnostic — INSPECT THIS before applying
    diagnostic_calibration_plot(
        y_pred_set1, y_true_set1, a, b,
        output_path="submissions/phase2/calibration_diagnostic.png"
    )
    logger.info("Calibration diagnostic plot saved — INSPECT before proceeding!")

    # Apply to all predictions
    all_preds["pec50_pred_calibrated"] = apply_residual_calibration(all_preds["pec50_pred"].values, a, b)

    # Get Set 2 (exclude Set 1 for final blinded submission)
    set2 = all_preds[~all_preds["compound_id"].isin(set1_labels["compound_id"])].copy()

    # Save calibrated submission in challenge format (SMILES, Molecule Name, pEC50)
    # Re-merge with test SMILES for the required format
    out_path = "submissions/phase2/calibrated_submission.csv"
    test_df = pd.read_parquet("data/raw/openadmet_test.parquet")
    smiles_map = dict(zip(test_df["compound_id"].astype(str), test_df["smiles"]))
    all_preds["SMILES"] = all_preds["compound_id"].astype(str).map(smiles_map)
    all_preds.rename(columns={"compound_id": "Molecule Name", "pec50_pred_calibrated": "pEC50"}) \
        [["SMILES", "Molecule Name", "pEC50"]].to_csv(out_path, index=False)

    logger.info(f"\nCalibrated submission saved: {out_path}")
    logger.info(f"  a={a:.4f}, b={b:.4f}")
    logger.info(f"\nNext: python scripts/13_validate_submission.py {out_path}")


if __name__ == "__main__":
    main()
