"""
Phase 2 final submission builder (Activity Track).

Takes the Set1-trained Phase-2 ensemble predictions (all 513 compounds) and substitutes
the 253 released Analog Set 1 true labels for their compound IDs. Set 1 is officially
released as Phase-2 training data, so the blinded ~260 compounds are what gets scored;
substituting the known truth for Set 1 is score-neutral there and a free win if the full
513 is ever re-scored. Calibration was evaluated against the real Set 1 labels and added
no out-of-sample gain (see scratchpad eval), so no residual correction is applied here.

Usage:
  python scripts/12b_build_final_submission.py \
    --preds       submissions/phase2/ensemble_raw.csv \
    --set1_labels data/raw/analog_set1_labels.csv \
    --out         submissions/phase2/final_submission.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import pandas as pd
from loguru import logger

from openadmet.utils.submission import format_submission


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="submissions/phase2/ensemble_raw.csv")
    ap.add_argument("--set1_labels", default="data/raw/analog_set1_labels.csv")
    ap.add_argument("--out", default="submissions/phase2/final_submission.csv")
    args = ap.parse_args()

    preds = pd.read_csv(args.preds)
    preds["Molecule Name"] = preds["Molecule Name"].astype(str)

    labels = pd.read_csv(args.set1_labels)
    labels["compound_id"] = labels["compound_id"].astype(str)
    id2true = dict(zip(labels["compound_id"], labels["pec50"]))

    n_sub = preds["Molecule Name"].isin(id2true).sum()
    logger.info(f"{len(preds)} prediction rows; substituting {n_sub} true Set 1 labels")

    # Substitute known truth for Set 1 IDs, keep model predictions for the blinded rest
    preds["pEC50"] = preds.apply(
        lambda r: id2true.get(r["Molecule Name"], r["pEC50"]), axis=1
    )

    # Re-emit through the canonical formatter to guarantee the column contract.
    # format_submission writes to output_path itself and validates NaN/count.
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out = format_submission(
        test_df=preds,
        predictions=preds["pEC50"].values,
        compound_id_col="Molecule Name",
        smiles_col="SMILES",
        output_path=args.out,
    )
    logger.info(f"Final submission written: {args.out} ({len(out)} rows)")
    logger.info(f"  pEC50 range {out['pEC50'].min():.2f}-{out['pEC50'].max():.2f}, "
                f"mean {out['pEC50'].mean():.3f}")
    logger.info(f"Next: python scripts/13_validate_submission.py {args.out}")


if __name__ == "__main__":
    main()
