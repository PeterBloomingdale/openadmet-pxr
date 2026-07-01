"""
Build analog-expansion cross-validation fold assignments.

Prerequisite: scripts/02_curate_data.py

Output: data/splits/butina_folds.parquet
  Contains the training data with 'cluster_id' and 'fold' columns added.

Inspect notebooks/01_eda.ipynb after this to visualize cluster distributions.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from loguru import logger

from openadmet.data.splits import build_analog_expansion_folds


def main():
    # Use full training set (active + censored, n=4135, pEC50 1.61–7.55)
    # Previously used master_train_active.parquet (n=1334, pEC50 5.0–7.55) which caused
    # 3× data deficit vs top leaderboard teams and poor generalization below pEC50=5.0
    train_df = pd.read_parquet("data/curated/master_train.parquet")
    smiles_col = "smiles_std" if "smiles_std" in train_df.columns else "smiles"

    logger.info(f"Building analog-expansion CV splits for {len(train_df)} compounds (pEC50 {train_df['pec50_median'].min():.2f}–{train_df['pec50_median'].max():.2f})...")
    fold_df = build_analog_expansion_folds(
        df=train_df,
        smiles_col=smiles_col,
        n_folds=5,
        tanimoto_threshold=0.40,   # mirrors how the test set was built (Tanimoto > 0.4 to potent parents); the 0.30 workaround for singletons is no longer necessary at n=4,135
        output_path="data/splits/butina_folds.parquet",
    )

    logger.info(f"\nFold distribution:")
    for fold in range(5):
        n = (fold_df["fold"] == fold).sum()
        n_clusters = fold_df[fold_df["fold"] == fold]["cluster_id"].nunique()
        logger.info(f"  Fold {fold}: {n} compounds, {n_clusters} clusters")

    logger.info(f"\nNext: python scripts/06_train_lgbm.py")


if __name__ == "__main__":
    main()
