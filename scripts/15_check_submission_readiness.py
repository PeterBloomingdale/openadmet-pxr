"""
Gate the *next* Hugging Face upload: require Chemprop OOF aligned with Butina splits.

Exit codes:
  0 — Chemprop OOF exists and len == len(butina_folds); safe to run 11_ensemble for canonical submit.
  1 — Chemprop missing or wrong length (next step: GPU 07_train_chemprop.py).

Usage:
  python scripts/15_check_submission_readiness.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from loguru import logger


def main() -> None:
    splits = Path("data/splits/butina_folds.parquet")
    cp_oof = Path("models/chemprop/oof_predictions.npy")

    if not splits.exists():
        logger.error("Missing data/splits/butina_folds.parquet — run 05_build_cv_splits.py")
        sys.exit(1)

    n = len(pd.read_parquet(splits))
    if not cp_oof.exists():
        logger.error(f"Missing {cp_oof} — next: GPU python scripts/07_train_chemprop.py")
        sys.exit(1)

    oof = np.load(cp_oof)
    if len(oof) != n:
        logger.error(
            f"Chemprop OOF length {len(oof)} != training rows {n} — stale checkpoint. "
            f"Next: re-run scripts/07_train_chemprop.py on this machine, then 11_ensemble.py."
        )
        sys.exit(1)

    logger.info(f"Submission readiness OK: Chemprop OOF aligned ({n} rows). Run 11_ensemble → validate.")
    sys.exit(0)


if __name__ == "__main__":
    main()
