"""
Cross-validation split construction for the analog-expansion task.
"""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from loguru import logger

from openadmet.cv.clustering import butina_cluster_from_smiles, assign_folds_from_clusters


def build_analog_expansion_folds(
    df: pd.DataFrame,
    smiles_col: str = "smiles_std",
    n_folds: int = 5,
    tanimoto_threshold: float = 0.4,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Builds leave-one-cluster-out CV folds using Butina clustering at Tanimoto 0.4.

    Each cluster is entirely held out in one fold, meaning no compound in the
    validation set has a Tanimoto neighbor > 0.4 in the training set. This mirrors
    the structural separation between the 63 training parents and the 513 test analogs.

    Adds 'cluster_id' and 'fold' columns to df in-place copy.
    If output_path is provided, saves the fold-annotated DataFrame to parquet.
    """
    smiles_list = df[smiles_col].tolist()
    cluster_ids, centroids = butina_cluster_from_smiles(
        smiles_list, tanimoto_threshold=tanimoto_threshold
    )
    fold_ids = assign_folds_from_clusters(cluster_ids, n_folds=n_folds)

    result = df.copy()
    result["cluster_id"] = cluster_ids
    result["fold"] = fold_ids

    logger.info(
        f"Analog-expansion CV: {len(set(cluster_ids[cluster_ids >= 0]))} clusters "
        f"→ {n_folds} folds (threshold={tanimoto_threshold})"
    )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(output_path, index=False)
        logger.info(f"Saved fold assignments to {output_path}")

    return result


def get_train_val_indices(
    fold_df: pd.DataFrame,
    fold_id: int,
    fold_col: str = "fold",
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (train_indices, val_indices) for a given fold number."""
    val_mask = fold_df[fold_col] == fold_id
    train_idx = np.where(~val_mask)[0]
    val_idx = np.where(val_mask)[0]
    return train_idx, val_idx
