"""
MMP-delta baseline model: vanilla LightGBM predicting Δ(pEC50).

Architecture:
  For each compound in the analog test set:
  1. Find the nearest training neighbor (by ECFP4 Tanimoto)
  2. Predict Δ(pEC50) = pEC50(query) - pEC50(neighbor) from MMP features
  3. Final prediction = neighbor_pEC50 + predicted_Δ

This is the "sanity check" delta model that establishes whether MMP features
are useful at all. The antisymmetric Siamese model (delta_siamese.py) builds
on this by enforcing Δ(A→B) = -Δ(B→A) as a hard constraint.

Why this model is essential even as a baseline:
- An absolute-pEC50 model predicts from scratch. The MMP-delta model has the
  advantage of "anchoring" to a known value — it only needs to predict the
  perturbation. For analog series this almost always outperforms absolute models.
- van Tilborg et al. (MoleculeACE, PMC10107580) showed that providing one pair
  compound's activity dramatically improves QSAR accuracy on activity cliffs.

CV note: The MMP-delta CV MUST ensure that the nearest-training-neighbor for
each validation compound comes from the TRAINING fold, not the validation fold.
See test_mmp_delta_cv_no_leakage() in tests/test_cv.py for the assertion.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

from openadmet.features.mmp import build_mmp_feature_matrix
from openadmet.data.splits import get_train_val_indices


DELTA_FEATURE_COLS = [
    "neighbor_pec50", "tanimoto",
    "delta_mw", "delta_logp", "delta_hbd", "delta_hba",
    "delta_tpsa", "delta_rotbonds", "delta_rings", "delta_arom_rings",
]

DEFAULT_PARAMS = {
    "objective": "regression_l1",   # MAE objective aligns with competition metric
    "metric": "mae",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 5,
    "n_estimators": 500,
    "early_stopping_rounds": 30,
    "verbose": -1,
}


def _build_delta_features(
    query_smiles: list[str],
    query_pec50: list[float],
    train_smiles: list[str],
    train_pec50: list[float],
    train_df: pd.DataFrame,
    exclude_self: bool = False,
) -> pd.DataFrame:
    """
    For each query compound, finds its nearest training neighbor and computes
    MMP delta features. Target = query_pec50 - neighbor_pec50.

    exclude_self=True: when query_smiles are a subset of train_smiles (i.e., when
    building training-set MMP features), exclude the query compound itself from the
    neighbor pool. Without this, every compound finds itself (tanimoto=1.0) as its
    nearest neighbor, yielding zero useful training rows.
    """
    from openadmet.features.mmp import find_nearest_training_neighbor, physchem_delta_features

    pec50_col = train_df.columns[train_df.columns.get_loc("pec50_median")] if "pec50_median" in train_df.columns else "pec50"
    smiles_col = "smiles_std" if "smiles_std" in train_df.columns else "smiles"

    records = []
    for q_smi, q_pec50 in zip(query_smiles, query_pec50):
        if exclude_self:
            # Build a neighbor pool that excludes the query compound itself
            pool_smiles = [s for s in train_smiles if s != q_smi]
            pool_pec50  = [p for s, p in zip(train_smiles, train_pec50) if s != q_smi]
        else:
            pool_smiles = train_smiles
            pool_pec50  = train_pec50

        if not pool_smiles:
            continue

        neighbor_smi, neighbor_pec50, tanimoto = find_nearest_training_neighbor(
            q_smi, pool_smiles, pool_pec50
        )

        delta_feats = None
        if neighbor_smi and not np.isnan(neighbor_pec50):
            delta_feats = physchem_delta_features(neighbor_smi, q_smi)

        col_names = ["delta_mw", "delta_logp", "delta_hbd", "delta_hba",
                     "delta_tpsa", "delta_rotbonds", "delta_rings", "delta_arom_rings"]
        rec = {
            "query_smiles": q_smi,
            "neighbor_smiles": neighbor_smi,
            "neighbor_pec50": neighbor_pec50,
            "tanimoto": tanimoto,
            "query_pec50": q_pec50,
        }
        if delta_feats is not None:
            for name, val in zip(col_names, delta_feats):
                rec[name] = float(val)
        else:
            for name in col_names:
                rec[name] = np.nan
        records.append(rec)

    mmp_df = pd.DataFrame(records)
    mmp_df["delta_pec50"] = mmp_df["query_pec50"] - mmp_df["neighbor_pec50"]
    return mmp_df


def train_mmp_delta_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    smiles_col: str = "smiles_std",
    pec50_col: str = "pec50_median",
    params: Optional[dict] = None,
    fold_id: int = 0,
) -> tuple[object, np.ndarray, np.ndarray]:
    """
    Trains one fold of the MMP-delta baseline model.

    LEAKAGE PREVENTION: When building MMP features for validation compounds,
    only train_df is used as the neighbor pool — not val_df. This is enforced
    by passing train_df as the reference set to build_mmp_feature_matrix.

    Returns (booster, val_predictions, val_delta_predictions).
    """
    params = params or DEFAULT_PARAMS.copy()

    train_mmp = _build_delta_features(
        query_smiles=train_df[smiles_col].tolist(),
        query_pec50=train_df[pec50_col].tolist(),
        train_smiles=train_df[smiles_col].tolist(),
        train_pec50=train_df[pec50_col].tolist(),
        train_df=train_df,
        exclude_self=True,  # each compound finds its SECOND nearest neighbor as anchor
    )
    train_mmp = train_mmp.dropna(subset=["delta_pec50"])

    val_mmp = _build_delta_features(
        query_smiles=val_df[smiles_col].tolist(),
        query_pec50=val_df[pec50_col].tolist(),
        train_smiles=train_df[smiles_col].tolist(),
        train_pec50=train_df[pec50_col].tolist(),
        train_df=train_df,
    )
    val_mmp = val_mmp.fillna(0.0)

    feature_cols = [c for c in DELTA_FEATURE_COLS if c in train_mmp.columns]
    X_train = train_mmp[feature_cols].values
    y_train = train_mmp["delta_pec50"].values
    X_val = val_mmp[feature_cols].values

    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=val_mmp.get("delta_pec50", np.zeros(len(X_val))).values)

    n_estimators = params.pop("n_estimators", 500)
    early_stopping_rounds = params.pop("early_stopping_rounds", 30)

    booster = lgb.train(
        params,
        lgb_train,
        num_boost_round=n_estimators,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(-1)],
    )

    delta_preds = booster.predict(X_val)
    # Final prediction = neighbor_pEC50 + predicted_Δ
    final_preds = val_mmp["neighbor_pec50"].values + delta_preds

    logger.info(f"MMP-delta fold {fold_id}: {booster.best_iteration} trees")
    return booster, final_preds, delta_preds


def predict_mmp_delta(
    booster,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    smiles_col: str = "smiles_std",
    pec50_col: str = "pec50_median",
) -> np.ndarray:
    """
    Generates MMP-delta predictions for the test set.

    test_df must have a smiles column; pec50 labels are not required (blinded).
    All training compounds are used as the neighbor pool.
    """
    # Test set may have "smiles" instead of "smiles_std" (standardized col absent in blinded data)
    test_smiles_col = smiles_col if smiles_col in test_df.columns else "smiles"
    test_mmp = build_mmp_feature_matrix(
        query_smiles_list=test_df[test_smiles_col].tolist(),
        train_df=train_df,
    )
    test_mmp = test_mmp.fillna(0.0)

    feature_cols = [c for c in DELTA_FEATURE_COLS if c in test_mmp.columns]
    delta_preds = booster.predict(test_mmp[feature_cols].values)
    return test_mmp["neighbor_pec50"].values + delta_preds
