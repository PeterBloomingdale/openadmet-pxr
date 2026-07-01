"""
TabPFN v2 on frozen CheMeleon embeddings + ECFP4 + RDKit descriptors.

TabPFN performs in-context learning: it receives the entire training set as
"context" and predicts labels for new inputs without gradient-based training.
fit() is near-instantaneous — the training set is just stored in memory.

Constraints:
- TabPFN v2 supports up to ~10K samples, ~500 features — use PCA-256 and full fold data
- Using Jeremy (rank-19)-style feature set: CheMeleon(2048) + ECFP4(2048) + rdkit2d(217)
  → PCA-256 reduction; this is the combination that powers his dominant TabICL model

When to include in ensemble:
- If CheMeleon embeddings are available (scripts/16_extract_chemeleon_embeddings.py ran first)
- If OOF contribution shows unique signal (check blend weight > 0.05)
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

try:
    from tabpfn import TabPFNRegressor
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    logger.warning("TabPFN not available — install with: pip install tabpfn>=7.0.0")

# TabPFN v2 can handle large training sets and many features via PCA reduction.
# Jeremy (rank-19) uses PCA-256 on ~4000 training examples — matching that config.
TABPFN_MAX_TRAIN = 1000   # CPU hard limit in tabpfn 7.x (>1000 raises RuntimeError → silent mean fallback)
TABPFN_MAX_FEATURES = 256  # PCA-256 matches Jeremy's setup; was 100 — underrepresented embedding space


class TabpfnFeatureReducer:
    """
    VarianceThreshold + PCA(max_features) fitted on training rows only.

    Use one instance per CV fold for honest OOF, or one fit on full training for final test preds.
    """

    def __init__(self, max_features: int = TABPFN_MAX_FEATURES):
        self.max_features = max_features
        self._vt: VarianceThreshold | None = None
        self._pca: PCA | None = None

    def fit(self, X: np.ndarray) -> "TabpfnFeatureReducer":
        from sklearn.preprocessing import StandardScaler
        X = np.nan_to_num(X.astype(np.float64), nan=0.0)
        keep_mask = X.var(axis=0) > 0
        X1 = X[:, keep_mask]
        self._vt = keep_mask
        # StandardScaler before PCA: feature scales span [0, 52M] — without scaling,
        # extreme descriptor values cause overflow in randomized SVD (numpy 2.2, float32).
        self._scaler = StandardScaler()
        X1 = self._scaler.fit_transform(X1).astype(np.float32)
        n_comp = min(self.max_features, X1.shape[1])
        self._pca = PCA(n_components=n_comp, random_state=42).fit(X1)
        # Post-PCA standardization so TabPFN sees mean=0, std=1 features
        self._post_scaler = StandardScaler()
        _ = self._post_scaler.fit(self._pca.transform(X1))
        logger.info(
            f"TabpfnFeatureReducer: PCA n_components={n_comp}, "
            f"explained variance={self._pca.explained_variance_ratio_.sum():.2%}"
        )
        return self

    def _vt_transform(self, X: np.ndarray) -> np.ndarray:
        if isinstance(self._vt, np.ndarray):
            return X[:, self._vt]
        return self._vt.transform(X)  # legacy VarianceThreshold objects
        n_comp = min(self.max_features, X1.shape[1])
        self._pca = PCA(n_components=n_comp, random_state=42).fit(X1)
        logger.info(
            f"TabpfnFeatureReducer: PCA n_components={n_comp}, "
            f"explained variance={self._pca.explained_variance_ratio_.sum():.2%}"
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._vt is None or self._pca is None:
            raise RuntimeError("TabpfnFeatureReducer.transform before fit")
        X = np.nan_to_num(X.astype(np.float64), nan=0.0)
        X1 = self._vt_transform(X)
        X1 = self._scaler.transform(X1).astype(np.float32)
        X1 = self._pca.transform(X1).astype(np.float32)
        return self._post_scaler.transform(X1).astype(np.float32)


def prepare_tabpfn_features_matrix(
    X: np.ndarray,
    max_features: int = TABPFN_MAX_FEATURES,
) -> np.ndarray:
    """
    One-shot fit+transform on matrix X (e.g. exploratory fit on full train — **not** honest OOF).
    Prefer `TabpfnFeatureReducer` per fold for CV.
    """
    return TabpfnFeatureReducer(max_features=max_features).fit(X).transform(X)


def prepare_tabpfn_features(
    embeddings: np.ndarray,
    descriptors: np.ndarray,
    max_features: int = TABPFN_MAX_FEATURES,
) -> np.ndarray:
    """
    Combines CheMeleon embeddings + RDKit descriptors and reduces to TabPFN limits.

    Strategy:
    1. Concatenate embeddings + descriptors
    2. Drop near-zero-variance columns
    3. If still > max_features, use PCA to compress to max_features components

    PCA over feature selection: preserves information from all features rather
    than arbitrarily dropping columns. The compressed representation is dense
    but loses interpretability (acceptable for the TabPFN ensemble member).
    """
    X = np.concatenate([embeddings, descriptors], axis=1).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)

    # Variance filter
    sel = VarianceThreshold(threshold=0.01)
    X = sel.fit_transform(X)

    if X.shape[1] > max_features:
        pca = PCA(n_components=max_features, random_state=42)
        X = pca.fit_transform(X)
        logger.info(f"PCA to {max_features} components (explained variance: {pca.explained_variance_ratio_.sum():.2%})")

    return X.astype(np.float32)


def subsample_diverse(
    smiles_list: list[str],
    n: int = TABPFN_MAX_TRAIN,
    seed: int = 42,
) -> np.ndarray:
    """
    Selects n maximally diverse compounds using MaxMin selection on ECFP4.
    Returns indices into smiles_list.
    """
    if len(smiles_list) <= n:
        return np.arange(len(smiles_list))

    from openadmet.features.fingerprints import ecfp4_bitvect
    from rdkit import DataStructs

    fps = [ecfp4_bitvect(s) for s in smiles_list]
    fps_valid = [(i, fp) for i, fp in enumerate(fps) if fp is not None]

    rng = np.random.default_rng(seed)
    # Start with a random compound
    selected = [int(rng.integers(len(fps_valid)))]
    available = list(range(len(fps_valid)))

    while len(selected) < min(n, len(fps_valid)):
        # Find compound with maximum minimum distance to selected set
        selected_fps = [fps_valid[i][1] for i in selected]
        max_min_dist = -1
        best_idx = -1
        for idx in available:
            if idx in selected:
                continue
            sims = DataStructs.BulkTanimotoSimilarity(fps_valid[idx][1], selected_fps)
            min_dist = 1.0 - max(sims)
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_idx = idx
        selected.append(best_idx)

    original_indices = np.array([fps_valid[i][0] for i in selected])
    logger.info(f"MaxMin selected {len(original_indices)} diverse training compounds for TabPFN")
    return original_indices


def train_tabpfn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
) -> "TabPFNRegressor":
    """
    Fits TabPFN regressor.

    If X_train has > TABPFN_MAX_TRAIN rows, randomly subsample.
    (For diverse subsampling, use subsample_diverse before calling this.)

    Returns fitted TabPFNRegressor.
    """
    if not TABPFN_AVAILABLE:
        raise ImportError("TabPFN required — pip install tabpfn>=2.0.3")

    if len(X_train) > TABPFN_MAX_TRAIN:
        idx = np.random.choice(len(X_train), TABPFN_MAX_TRAIN, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]
        logger.warning(f"TabPFN: subsampled to {TABPFN_MAX_TRAIN} training points")

    # Force CPU + allow >1000 samples: v7 raises RuntimeError silently (→ mean predictions)
    # when n > 1000 on CPU without this flag. MPS OOMs with 36 GiB shared pool.
    model = TabPFNRegressor(
        n_estimators=16, random_state=42, device="cpu",
        ignore_pretraining_limits=True,
    )
    model.fit(X_train, y_train)
    logger.info("TabPFN fitted on CPU (in-context, no gradient training)")
    return model


def predict_tabpfn(model: "TabPFNRegressor", X_test: np.ndarray) -> np.ndarray:
    """Returns TabPFN predictions."""
    return model.predict(X_test)
