"""
Tests for cross-validation infrastructure.

Includes the two critical no-leakage tests that MUST pass before any submission.
"""

import numpy as np
import pandas as pd
import pytest

from openadmet.cv.clustering import (
    butina_cluster_from_smiles,
    assign_folds_from_clusters,
)
from openadmet.data.splits import get_train_val_indices


SIMPLE_SMILES = [
    "c1ccccc1",           # Benzene
    "c1ccccc1C",          # Toluene
    "c1ccccc1CC",         # Ethylbenzene
    "c1ccc(cc1)C",        # p-Xylene (similar to toluene)
    "CC(=O)O",            # Acetic acid (very different)
    "CCCCCC",             # Hexane (very different)
    "c1ccc2ccccc2c1",     # Naphthalene
    "O=C(O)c1ccccc1",     # Benzoic acid
]


class TestButinaClustering:
    def test_returns_cluster_per_compound(self):
        cluster_ids, centroids = butina_cluster_from_smiles(SIMPLE_SMILES)
        assert len(cluster_ids) == len(SIMPLE_SMILES)

    def test_aromatic_compounds_cluster_together(self):
        """Benzene, toluene, ethylbenzene, p-xylene should cluster together at Tanimoto 0.4."""
        cluster_ids, _ = butina_cluster_from_smiles(SIMPLE_SMILES, tanimoto_threshold=0.4)
        # Benzene-like compounds should share a cluster
        aromatic_idx = [0, 1, 2, 3]  # benzene, toluene, ethylbenzene, p-xylene
        aromatic_clusters = cluster_ids[aromatic_idx]
        # At least some should cluster together
        assert len(set(aromatic_clusters)) < len(aromatic_idx)

    def test_invalid_smiles_assigned_minus_one(self):
        smiles_with_bad = SIMPLE_SMILES + ["INVALID_SMILES_XYZ"]
        cluster_ids, _ = butina_cluster_from_smiles(smiles_with_bad)
        assert cluster_ids[-1] == -1

    def test_fold_assignment_coverage(self):
        cluster_ids, _ = butina_cluster_from_smiles(SIMPLE_SMILES)
        fold_ids = assign_folds_from_clusters(cluster_ids, n_folds=3)
        assert len(fold_ids) == len(SIMPLE_SMILES)
        assert set(fold_ids).issubset({0, 1, 2})


class TestNoLeakage:
    """
    CRITICAL: These tests must pass before any submission.
    Leakage can silently inflate leaderboard scores without any obvious error.
    """

    def test_train_val_indices_are_disjoint(self):
        """Training and validation indices for a given fold must not overlap."""
        n = 100
        fold_df = pd.DataFrame({"fold": np.tile(np.arange(5), 20)})
        for fold_id in range(5):
            train_idx, val_idx = get_train_val_indices(fold_df, fold_id)
            overlap = set(train_idx) & set(val_idx)
            assert len(overlap) == 0, f"Fold {fold_id}: training/validation indices overlap!"

    def test_mmp_delta_neighbor_leakage(self):
        """
        For MMP-delta CV: the nearest-training-neighbor for each validation
        compound must be selected ONLY from the training fold, not from the
        validation fold itself.

        This is the silent killer of delta models: if a validation compound's
        neighbor is also in the validation fold, the model "knows" the answer
        indirectly through the neighbor's pEC50.
        """
        # Simulate: 10 training compounds, 5 validation compounds
        # All compounds are close to each other (would cluster together)
        train_smiles = [
            "c1ccccc1",          # Benzene
            "c1ccccc1C",         # Toluene
            "c1ccccc1CC",        # Ethylbenzene
            "c1ccc(cc1)C",       # p-Xylene
            "c1cccc(c1)C",       # m-Xylene
            "c1ccc(cc1)Cl",      # Chlorobenzene
            "c1ccc(cc1)F",       # Fluorobenzene
            "c1ccc(cc1)Br",      # Bromobenzene
            "c1ccc(cc1)O",       # Phenol
            "c1ccc(cc1)N",       # Aniline
        ]
        val_smiles = [
            "c1ccc(cc1)CC",      # Should find neighbor in train_smiles
            "c1ccc(cc1)CF",      # Should find neighbor in train_smiles
        ]

        from openadmet.features.mmp import find_nearest_training_neighbor

        train_pec50 = [6.0 + i * 0.1 for i in range(len(train_smiles))]

        for val_smi in val_smiles:
            neighbor_smi, neighbor_pec50, tanimoto = find_nearest_training_neighbor(
                val_smi, train_smiles, train_pec50
            )
            # Neighbor must be from train_smiles, not from val_smiles
            assert neighbor_smi in train_smiles or neighbor_smi == "", \
                f"Neighbor '{neighbor_smi}' is not in training set!"
            assert not np.isnan(neighbor_pec50) or neighbor_smi == "", \
                "Neighbor pEC50 is NaN for a valid neighbor"

    def test_train_test_no_inchikey_overlap(self):
        """
        Test and training sets must share no InChIKey prefixes.
        Run this before EVERY submission.
        """
        # Simulate with a few known identical SMILES
        from openadmet.data.curation import standardize_smiles, smiles_to_inchikey, inchikey_prefix

        train_smiles = ["c1ccccc1", "CC(=O)O", "CCCCCC"]
        test_smiles = ["c1ccccc1C", "CCCCCCC"]  # Different from training

        def get_prefixes(smiles_list):
            prefixes = set()
            for s in smiles_list:
                std = standardize_smiles(s)
                if std:
                    ik = smiles_to_inchikey(std)
                    if ik:
                        prefixes.add(inchikey_prefix(ik))
            return prefixes

        train_prefixes = get_prefixes(train_smiles)
        test_prefixes = get_prefixes(test_smiles)
        overlap = train_prefixes & test_prefixes
        assert len(overlap) == 0, f"Train/test InChIKey overlap detected: {overlap}"


class TestMetrics:
    def test_rae_perfect_predictions(self):
        from openadmet.utils.metrics import compute_rae
        y = np.array([5.0, 6.0, 7.0, 8.0])
        assert compute_rae(y, y) == pytest.approx(0.0)

    def test_rae_uses_dynamic_range(self):
        from openadmet.utils.metrics import compute_rae, compute_dynamic_range
        y_true = np.array([5.0, 8.0])  # Dynamic range = 3.0
        y_pred = np.array([5.5, 7.5])  # MAE = 0.5
        rae = compute_rae(y_true, y_pred)
        assert rae == pytest.approx(0.5 / 3.0)

    def test_rae_zero_dynamic_range_raises(self):
        from openadmet.utils.metrics import compute_rae
        y = np.array([6.0, 6.0, 6.0])
        with pytest.raises(ValueError):
            compute_rae(y, y)
