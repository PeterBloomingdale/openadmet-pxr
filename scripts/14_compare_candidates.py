"""
Compare two Phase 1 submission CSVs (same test compounds) before choosing an upload.

Prints per-file summary stats, Pearson/Spearman correlation between pEC50 columns,
and mean absolute delta. Does not require labels — selection remains governed by
honest OOF metrics from training scripts; this tool only quantifies how similar two
leaderboard candidates are.

Usage:
  python scripts/14_compare_candidates.py submissions/phase1/ensemble_recal.csv \\
      submissions/phase1/external_sub9.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import pearsonr, spearmanr


def summarize(label: str, p: np.ndarray) -> None:
    logger.info(
        f"{label}: n={len(p)}, mean={p.mean():.4f}, std={p.std():.4f}, "
        f"min={p.min():.4f}, max={p.max():.4f}"
    )


def main() -> None:
    if len(sys.argv) != 3:
        print(
            "Usage: python scripts/14_compare_candidates.py <submission_a.csv> <submission_b.csv>"
        )
        sys.exit(1)

    path_a, path_b = sys.argv[1], sys.argv[2]
    da = pd.read_csv(path_a)
    db = pd.read_csv(path_b)

    for name, df in [("A", da), ("B", db)]:
        for col in ["SMILES", "Molecule Name", "pEC50"]:
            if col not in df.columns:
                logger.error(f"{name}: missing column {col}")
                sys.exit(1)

    set_a = set(da["Molecule Name"].astype(str))
    set_b = set(db["Molecule Name"].astype(str))
    if set_a != set_b:
        logger.error("Compound ID sets differ — align inputs before comparing.")
        sys.exit(1)

    merged = da[["Molecule Name", "pEC50"]].merge(
        db[["Molecule Name", "pEC50"]],
        on="Molecule Name",
        suffixes=("_a", "_b"),
        validate="one_to_one",
    )
    pa = merged["pEC50_a"].astype(float).values
    pb = merged["pEC50_b"].astype(float).values

    summarize(Path(path_a).name, pa)
    summarize(Path(path_b).name, pb)

    diff = pa - pb
    logger.info(f"|A−B|: mean={np.mean(np.abs(diff)):.4f}, max={np.max(np.abs(diff)):.4f}")

    r_p, _ = pearsonr(pa, pb)
    r_s, _ = spearmanr(pa, pb)
    logger.info(f"Correlation A vs B: Pearson={r_p:.4f}, Spearman={r_s:.4f}")
    logger.info("Interpretation: rho≈1 means redundant submits; lower rho means diversify LB probes.")


if __name__ == "__main__":
    main()
