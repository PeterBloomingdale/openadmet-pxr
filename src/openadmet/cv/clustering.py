"""
Butina (Taylor-Butina) clustering for analog-expansion cross-validation.

Why Butina over scaffold or random splits:
- Random splits are grossly optimistic for analog series (similar compounds
  in both train and val give artificially low error).
- Scaffold splits are too pessimistic — they separate parent/analog pairs
  that we'll actually encounter at test time.
- Butina at Tanimoto 0.4 creates clusters that mirror how the test set was
  constructed (Enamine analogs selected at ECFP4 Tanimoto > 0.4 to training hits).
  A leave-one-cluster-out fold is the closest approximation to the leaderboard
  evaluation we can compute from training data alone.
"""

from typing import Optional
import numpy as np
from loguru import logger

try:
    from rdkit import DataStructs
    from rdkit.Chem import AllChem
    from rdkit.ML.Cluster import Butina
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


def smiles_to_ecfp4(
    smiles: str,
    n_bits: int = 1024,
    radius: int = 2,
) -> Optional[object]:
    """Returns an RDKit ExplicitBitVect ECFP4 fingerprint, or None on failure."""
    try:
        from rdkit.Chem import MolFromSmiles, rdFingerprintGenerator
        mol = MolFromSmiles(smiles)
        if mol is None:
            return None
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        return gen.GetFingerprint(mol)
    except Exception:
        return None


def compute_tanimoto_distance_matrix(
    fps: list,
) -> np.ndarray:
    """
    Computes lower-triangular pairwise Tanimoto distance matrix (1 - similarity).

    RDKit's Butina implementation expects a flat list of the lower-triangle
    (row-major), not a full square matrix.

    Returns 1D array of length n*(n-1)/2 — the format Butina.ClusterData expects.
    """
    n = len(fps)
    dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1.0 - s for s in sims])
    return dists


def butina_cluster_from_smiles(
    smiles_list: list[str],
    tanimoto_threshold: float = 0.4,
    n_bits: int = 1024,
    radius: int = 2,
) -> tuple[np.ndarray, list[int]]:
    """
    Runs Butina clustering on a list of SMILES strings.

    tanimoto_threshold: compounds within this Tanimoto similarity are placed
    in the same cluster. 0.4 is the threshold used by OpenADMET to select
    the 513 test analogs — using the same threshold in CV mirrors the test structure.

    Returns:
    - cluster_ids: int array of length len(smiles_list), one cluster ID per compound.
      Cluster 0 is the largest cluster (Butina's centroid-first ordering).
    - centroid_indices: list of compound indices that are cluster centroids.

    Compounds that fail ECFP4 computation (invalid SMILES) are assigned cluster -1.
    """
    fps = []
    valid_mask = []
    for smi in smiles_list:
        fp = smiles_to_ecfp4(smi, n_bits=n_bits, radius=radius)
        fps.append(fp)
        valid_mask.append(fp is not None)

    valid_fps = [fp for fp in fps if fp is not None]
    n_valid = len(valid_fps)
    n_invalid = len(fps) - n_valid

    if n_invalid > 0:
        logger.warning(f"Butina: {n_invalid} compounds produced invalid ECFP4, assigned cluster -1")

    if n_valid == 0:
        return np.full(len(smiles_list), -1, dtype=int), []

    logger.info(f"Computing Tanimoto distance matrix for {n_valid} compounds...")
    dists = compute_tanimoto_distance_matrix(valid_fps)

    # Butina.ClusterData returns a tuple of tuples: each inner tuple is a cluster's members
    clusters = Butina.ClusterData(dists, n_valid, 1.0 - tanimoto_threshold, isDistData=True)
    logger.info(f"Butina clustering: {len(clusters)} clusters from {n_valid} compounds")

    # Map valid-compound indices back to original indices
    valid_indices = [i for i, v in enumerate(valid_mask) if v]
    cluster_ids = np.full(len(smiles_list), -1, dtype=int)
    centroid_indices = []

    for cluster_id, cluster_members in enumerate(clusters):
        centroid_idx = valid_indices[cluster_members[0]]
        centroid_indices.append(centroid_idx)
        for member in cluster_members:
            original_idx = valid_indices[member]
            cluster_ids[original_idx] = cluster_id

    return cluster_ids, centroid_indices


def assign_folds_from_clusters(
    cluster_ids: np.ndarray,
    n_folds: int = 5,
    strategy: str = "size_balanced",
) -> np.ndarray:
    """
    Assigns fold IDs to compounds, keeping each cluster entirely in one fold.

    strategy="size_balanced":
    Sorts clusters by size (descending) and greedily assigns each cluster
    to the fold with the fewest compounds — like bin-packing. This gives
    more balanced fold sizes than round-robin, which matters for stable
    validation set RAE estimates.

    Compounds with cluster_id == -1 (failed ECFP4) are assigned to fold 0.
    Returns fold_id array aligned with cluster_ids.
    """
    unique_clusters = sorted(set(cluster_ids[cluster_ids >= 0]))
    cluster_sizes = {c: int(np.sum(cluster_ids == c)) for c in unique_clusters}

    # Sort clusters by size descending (largest first for greedy bin-packing)
    sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: -x[1])

    fold_assignments: dict[int, int] = {}
    fold_sizes = [0] * n_folds

    if strategy == "size_balanced":
        for cluster_id, size in sorted_clusters:
            smallest_fold = int(np.argmin(fold_sizes))
            fold_assignments[cluster_id] = smallest_fold
            fold_sizes[smallest_fold] += size
    else:  # round_robin
        for i, (cluster_id, _) in enumerate(sorted_clusters):
            fold_assignments[cluster_id] = i % n_folds

    fold_ids = np.where(cluster_ids >= 0, [fold_assignments.get(c, 0) for c in cluster_ids], 0)

    for fold in range(n_folds):
        n_in_fold = int(np.sum(fold_ids == fold))
        logger.info(f"Fold {fold}: {n_in_fold} compounds")

    return fold_ids
