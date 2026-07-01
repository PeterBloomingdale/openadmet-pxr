"""
Build the mmpdb matched molecular pair database from training SMILES.

Prerequisite: scripts/02_curate_data.py

Output: data/mmps/pxr_training.mmpdb (SQLite database)

Runtime: ~5-15 minutes for 4000 compounds.

mmpdb must be on PATH — verify with: mmpdb --help
If not found, install with: pip install mmpdb
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from loguru import logger

from openadmet.features.mmp import build_mmp_database


def main():
    # Use the full curated training set (n=4,135). The deprecated active-only
    # path (`master_train_active.parquet`, n=1,334) caused Subs 7/8 to lose
    # because it never trained the model on weakly-active analogs that exist
    # in the test set — see Appendix A entry 2026-05-06 in the manuscript.
    train_df = pd.read_parquet("data/curated/master_train.parquet")
    smiles_col = "smiles_std" if "smiles_std" in train_df.columns else "smiles"
    id_col = "inchikey_prefix" if "inchikey_prefix" in train_df.columns else train_df.columns[0]

    smiles = train_df[smiles_col].tolist()
    ids = train_df[id_col].astype(str).tolist()

    logger.info(f"Building MMP database for {len(smiles)} training compounds...")

    try:
        db_path = build_mmp_database(
            smiles_list=smiles,
            ids=ids,
            output_dir="data/mmps",
            mmpdb_executable="mmpdb",
        )
        logger.info(f"MMP database: {db_path}")
    except RuntimeError as e:
        logger.error(f"mmpdb failed: {e}")
        logger.error("Install mmpdb with: pip install mmpdb")
        logger.error("Or check PATH: which mmpdb")
        sys.exit(1)

    logger.info("Next: python scripts/09_train_mmp_delta.py")


if __name__ == "__main__":
    main()
