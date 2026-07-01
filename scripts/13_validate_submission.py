"""
Validate a submission CSV before uploading to the leaderboard.

Usage:
  python scripts/13_validate_submission.py submissions/phase1/ensemble_recal.csv

Runs all checks from utils/submission.py and exits with code 0 (pass) or 1 (fail).
ALWAYS run this before every submission — failed submissions waste your daily allowance.

Also runs the critical no-leakage check: confirms that no test compound InChIKey
prefix appears in the training data (should be 0 overlaps after curation).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from loguru import logger

from openadmet.utils.submission import validate_submission_format
from openadmet.data.curation import smiles_to_inchikey, inchikey_prefix, standardize_smiles


def check_no_train_test_leakage():
    """Assert no InChIKey prefix overlap between test set and training data."""
    test_path = Path("data/raw/openadmet_test.parquet")
    train_path = Path("data/curated/master_train.parquet")

    if not test_path.exists() or not train_path.exists():
        logger.warning("Skipping leakage check — curated data not found")
        return True

    test_df = pd.read_parquet(test_path)
    train_df = pd.read_parquet(train_path)

    # Get test prefixes
    smiles_col = "smiles" if "smiles" in test_df.columns else test_df.columns[0]
    test_prefixes = set()
    for smi in test_df[smiles_col].dropna():
        std = standardize_smiles(str(smi))
        if std:
            ik = smiles_to_inchikey(std)
            if ik:
                test_prefixes.add(inchikey_prefix(ik))

    # Check against training
    if "inchikey_prefix" in train_df.columns:
        train_prefixes = set(train_df["inchikey_prefix"].dropna())
    else:
        logger.warning("No inchikey_prefix column in training data — skipping leakage check")
        return True

    overlap = test_prefixes & train_prefixes
    # Phase 2: Analog Set 1 (253 compounds) was intentionally added to training after unblinding.
    # These compounds appear in the test set (full 513) but will NOT be scored in Phase 2
    # (only Set 2 is evaluated). Allow up to 260 overlapping compounds.
    if overlap and len(overlap) > 260:
        logger.error(f"LEAKAGE DETECTED: {len(overlap)} test compounds found in training data!")
        logger.error(f"Overlapping prefixes: {list(overlap)[:5]}...")
        return False
    elif overlap:
        logger.warning(
            f"Phase 2 note: {len(overlap)} test/train overlaps detected "
            f"(expected ≤253 from Analog Set 1 unblinding). This is intentional."
        )
        return True
    else:
        logger.info(f"No-leakage check PASSED: 0 test/train InChIKey overlaps")
        return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/13_validate_submission.py <submission.csv>")
        print("  Typical Phase 1 file: submissions/phase1/ensemble_recal.csv")
        sys.exit(1)

    submission_path = sys.argv[1]

    # Load expected compound IDs from test set
    test_path = Path("data/raw/openadmet_test.parquet")
    if not test_path.exists():
        logger.error("Test data not found at data/raw/openadmet_test.parquet")
        sys.exit(1)

    test_df = pd.read_parquet(test_path)
    compound_id_col = "compound_id" if "compound_id" in test_df.columns else test_df.columns[0]
    expected_ids = test_df[compound_id_col].astype(str).tolist()

    # Run submission validation
    result = validate_submission_format(submission_path, expected_ids)

    # Run leakage check
    leakage_ok = check_no_train_test_leakage()

    # Summary
    if result["valid"] and leakage_ok:
        logger.info("\n=== VALIDATION PASSED — safe to submit ===")
        sys.exit(0)
    else:
        logger.error("\n=== VALIDATION FAILED — DO NOT SUBMIT ===")
        for err in result.get("errors", []):
            logger.error(f"  {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
