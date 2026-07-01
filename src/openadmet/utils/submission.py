"""
Submission formatting and validation for the OpenADMET PXR challenge.

Always run validate_submission_format() before uploading to the leaderboard.
The competition has a submission frequency limit — a failed submission wastes
your daily allowance.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger


REQUIRED_COLUMNS = ["SMILES", "Molecule Name", "pEC50"]
VALID_PEC50_RANGE = (3.0, 11.0)  # Outside this range is almost certainly wrong

# Written by scripts/11_ensemble.py after blend/stack selection + dynamic_recal.
PHASE1_CANONICAL_SUBMISSION = "submissions/phase1/ensemble_recal.csv"


def format_submission(
    test_df: pd.DataFrame,
    predictions: np.ndarray,
    compound_id_col: str = "compound_id",
    smiles_col: str = "smiles",
    output_path: str = PHASE1_CANONICAL_SUBMISSION,
) -> pd.DataFrame:
    """
    Formats predictions into the competition submission format.

    Required columns: SMILES, Molecule Name, pEC50 (513 rows, no NaN/inf).
    Validates: no NaN predictions, correct compound count, all IDs present.
    Raises ValueError with clear message on failure.
    """
    if len(predictions) != len(test_df):
        raise ValueError(
            f"Prediction count ({len(predictions)}) != test compound count ({len(test_df)})"
        )

    n_nan = int(np.isnan(predictions).sum())
    if n_nan > 0:
        raise ValueError(f"{n_nan} NaN predictions found — check model output")

    n_oor = int(((predictions < VALID_PEC50_RANGE[0]) | (predictions > VALID_PEC50_RANGE[1])).sum())
    if n_oor > 0:
        logger.warning(
            f"{n_oor} predictions outside valid pEC50 range {VALID_PEC50_RANGE}. "
            f"These may indicate model errors — review before submitting."
        )

    smiles_col_actual = smiles_col if smiles_col in test_df.columns else "smiles"
    submission = pd.DataFrame({
        "SMILES": test_df[smiles_col_actual].values,
        "Molecule Name": test_df[compound_id_col].values,
        "pEC50": predictions,
    })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    logger.info(f"Submission saved: {output_path} ({len(submission)} compounds)")
    return submission


def validate_submission_format(
    submission_path: str,
    expected_compound_ids: list[str],
) -> dict[str, object]:
    """
    Comprehensive pre-submission validation. Run this before EVERY upload.

    Checks:
    1. File exists and is readable
    2. Required columns present
    3. All expected compound IDs present, no extras
    4. No NaN predictions
    5. All predictions in plausible pEC50 range
    6. Count matches expected

    Returns {'valid': bool, 'errors': list[str], 'warnings': list[str]}
    """
    errors = []
    warnings = []
    expected_set = set(expected_compound_ids)

    path = Path(submission_path)
    if not path.exists():
        return {"valid": False, "errors": [f"File not found: {submission_path}"], "warnings": []}

    try:
        df = pd.read_csv(submission_path)
    except Exception as e:
        return {"valid": False, "errors": [f"Cannot read CSV: {e}"], "warnings": []}

    # Column check
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            errors.append(f"Missing required column: '{col}'")

    if errors:
        return {"valid": False, "errors": errors, "warnings": warnings}

    # Count check
    if len(df) != len(expected_compound_ids):
        errors.append(f"Row count {len(df)} != expected {len(expected_compound_ids)}")

    # ID checks (use "Molecule Name" column)
    submitted_set = set(df["Molecule Name"].astype(str))
    missing = expected_set - submitted_set
    extra = submitted_set - expected_set
    if missing:
        errors.append(f"{len(missing)} expected compound IDs missing from submission")
    if extra:
        errors.append(f"{len(extra)} unexpected compound IDs in submission")

    # NaN check
    n_nan = df["pEC50"].isna().sum()
    if n_nan > 0:
        errors.append(f"{n_nan} NaN predictions")

    # Range check
    preds = df["pEC50"].dropna()
    n_oor = ((preds < VALID_PEC50_RANGE[0]) | (preds > VALID_PEC50_RANGE[1])).sum()
    if n_oor > 0:
        warnings.append(f"{n_oor} predictions outside valid pEC50 range {VALID_PEC50_RANGE}")

    valid = len(errors) == 0
    if valid:
        logger.info(f"Submission validation PASSED: {submission_path}")
        if warnings:
            for w in warnings:
                logger.warning(f"  Warning: {w}")
    else:
        logger.error(f"Submission validation FAILED: {submission_path}")
        for e in errors:
            logger.error(f"  Error: {e}")

    return {"valid": valid, "errors": errors, "warnings": warnings}
