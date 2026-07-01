"""
Tanimoto k-nearest-neighbor regressor (ECFP4 similarity, precomputed distance matrix).

Why kNN for this challenge:
The test set is constructed as Enamine analogs with Tanimoto > 0.4 to 63 potent
training hits. This structural proximity guarantee makes instance-based prediction
particularly valuable — any test compound has at least one close training neighbor,
and local activity landscapes around scaffolds are smoother than global models assume.

Unlike LightGBM and Chemprop (global models), kNN captures local SAR features that
global models miss, adding genuine diversity to the ensemble.

k=5, weights='distance': inverse-Tanimoto-distance weighting gives closer neighbors
more influence. Tanimoto distance = 1 - Tanimoto similarity (ECFP4, 1024-bit).

Prerequisites:
  - scripts/04_build_features.py (for SMILES; features not needed — kNN uses Tanimoto)
  - scripts/05_build_cv_splits.py → data/splits/butina_folds.parquet

Outputs:
  - models/knn/oof_predictions.npy   (4135,)
  - models/knn/test_predictions.npy  (513,)

Next: python scripts/11_ensemble.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.neighbors import KNeighborsRegressor

from rdkit import DataStructs
from rdkit.Chem import AllChem, MolFromSmiles

from openadmet.cv.oof import evaluate_oof


def _ecfp4_fps(smiles_list: list[str], n_bits: int = 1024):
    """Compute ECFP4 ExplicitBitVect for each SMILES (for Tanimoto similarity)."""
    fps = []
    for smi in smiles_list:
        mol = MolFromSmiles(smi)
        if mol is None:
            fps.append(None)
        else:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits))
    return fps


def _tanimoto_distance_matrix(query_fps, ref_fps) -> np.ndarray:
    """
    Compute (n_query, n_ref) Tanimoto distance matrix.
    distance = 1 - similarity; None fingerprints get distance 1.0 (max dissimilarity).
    """
    n_q, n_r = len(query_fps), len(ref_fps)
    D = np.ones((n_q, n_r), dtype=np.float32)
    for i, qfp in enumerate(query_fps):
        if qfp is None:
            continue
        valid_ref = [(j, fp) for j, fp in enumerate(ref_fps) if fp is not None]
        if not valid_ref:
            continue
        j_idxs = [j for j, _ in valid_ref]
        ref_list = [fp for _, fp in valid_ref]
        sims = DataStructs.BulkTanimotoSimilarity(qfp, ref_list)
        for k, j in enumerate(j_idxs):
            D[i, j] = 1.0 - float(sims[k])
    return D


def main() -> None:
    train_df = pd.read_parquet("data/splits/butina_folds.parquet")
    test_df = pd.read_parquet("data/raw/openadmet_test.parquet")

    smiles_col = "smiles_std" if "smiles_std" in train_df.columns else "smiles"
    test_smiles_col = "smiles"

    pec50_col = "pec50_median" if "pec50_median" in train_df.columns else "pec50"
    y = train_df[pec50_col].values.astype(np.float64)
    folds = train_df["fold"].values

    train_smiles = train_df[smiles_col].tolist()
    test_smiles = test_df[test_smiles_col].tolist()

    logger.info(f"Computing ECFP4 fingerprints for {len(train_smiles)} train + {len(test_smiles)} test ...")
    train_fps = _ecfp4_fps(train_smiles)
    test_fps = _ecfp4_fps(test_smiles)

    oof = np.full(len(train_df), np.nan, dtype=np.float64)
    unique_folds = sorted(np.unique(folds))

    for f in unique_folds:
        train_mask = folds != f
        val_mask = folds == f
        if val_mask.sum() == 0:
            continue

        ref_fps = [train_fps[i] for i in np.where(train_mask)[0]]
        qry_fps = [train_fps[i] for i in np.where(val_mask)[0]]

        D_train = _tanimoto_distance_matrix(ref_fps, ref_fps)
        D_val = _tanimoto_distance_matrix(qry_fps, ref_fps)

        y_ref = y[train_mask]
        knn = KNeighborsRegressor(n_neighbors=5, metric="precomputed", weights="distance")
        knn.fit(D_train, y_ref)
        oof[val_mask] = knn.predict(D_val)

        fold_mae = np.mean(np.abs(oof[val_mask] - y[val_mask]))
        logger.info(f"  Fold {f}: val MAE = {fold_mae:.4f} ({val_mask.sum()} compounds)")

    # Test predictions: fit on all training data
    logger.info("Computing test predictions (full training set)...")
    D_all = _tanimoto_distance_matrix(train_fps, train_fps)
    D_test = _tanimoto_distance_matrix(test_fps, train_fps)

    knn_full = KNeighborsRegressor(n_neighbors=5, metric="precomputed", weights="distance")
    knn_full.fit(D_all, y)
    test_preds = knn_full.predict(D_test).astype(np.float64)

    out = Path("models/knn")
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "oof_predictions.npy", oof)
    np.save(out / "test_predictions.npy", test_preds)

    valid = ~np.isnan(oof)
    metrics = evaluate_oof(y[valid], oof[valid], folds[valid])
    logger.info(
        f"\nkNN OOF: MAE={metrics['mae']:.4f}, RAE={metrics['rae']:.4f}, "
        f"R²={metrics['r2']:.4f}, Spearman={metrics.get('spearman', float('nan')):.4f}"
    )
    logger.info(f"Test predictions: mean={test_preds.mean():.3f}, std={test_preds.std():.3f}")
    logger.info("Saved models/knn/oof_predictions.npy and test_predictions.npy")
    logger.info("Next: python scripts/11_ensemble.py")


if __name__ == "__main__":
    main()
