"""
CheMeleon foundation model embedding extractor.

CheMeleon is a Chemprop v2 BondMessagePassing model pretrained by OpenADMET on large-scale
ChEMBL data. Embeddings capture learned chemical representations that encode ADMET-relevant
pharmacophore context beyond what classical fingerprints capture.

The foundation model weights are downloaded from Zenodo (DOI: 10.48550/arXiv.2506.15792)
and cached at ~/.chemprop/chemeleon_mp.pt by Chemprop itself.

Usage:
    from openadmet.features.chemeleon import extract_chemeleon_embeddings
    emb = extract_chemeleon_embeddings(smiles_list)  # shape (n, d_emb)
"""

from pathlib import Path
from urllib.request import urlretrieve
from typing import Optional

import numpy as np
import torch
from loguru import logger

CHEMELEON_ZENODO_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"
CHEMELEON_CACHE_PATH = Path.home() / ".chemprop" / "chemeleon_mp.pt"


def _download_chemeleon_weights() -> Path:
    """Download CheMeleon foundation MP weights from Zenodo if not cached."""
    CHEMELEON_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not CHEMELEON_CACHE_PATH.exists():
        logger.info(
            f"Downloading CheMeleon foundation model from Zenodo to {CHEMELEON_CACHE_PATH} ..."
        )
        urlretrieve(CHEMELEON_ZENODO_URL, CHEMELEON_CACHE_PATH)
        logger.info("CheMeleon download complete.")
    else:
        logger.info(f"Using cached CheMeleon weights: {CHEMELEON_CACHE_PATH}")
    return CHEMELEON_CACHE_PATH


def extract_chemeleon_embeddings(
    smiles_list: list[str],
    device: Optional[str] = None,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Extract CheMeleon graph-level embeddings for a list of SMILES.

    Uses MPNN.fingerprint() which runs message-passing + aggregation + batch-norm
    but NOT the FFN prediction head. The result is the learned molecular representation
    that CheMeleon's pretraining optimized for chemical property prediction across ChEMBL.

    Parameters
    ----------
    smiles_list : list of str
        Standardized SMILES (use smiles_std column from master_train.parquet).
    device : str or None
        'cuda', 'cpu', or None (auto-detect).
    batch_size : int
        Number of molecules per inference batch.

    Returns
    -------
    np.ndarray, shape (n, d_emb)
        Graph-level embeddings. d_emb is the CheMeleon hidden dimension (typically 2048).
        Failed/invalid SMILES rows are filled with zeros.
    """
    from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
    from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
    from chemprop.models import MPNN
    from chemprop.nn import BondMessagePassing, MeanAggregation, RegressionFFN

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Extracting CheMeleon embeddings for {len(smiles_list)} compounds on {device} ...")

    # Download / load CheMeleon foundation model weights
    ckpt_path = _download_chemeleon_weights()
    chemeleon_data = torch.load(ckpt_path, weights_only=True, map_location="cpu")
    hyper = chemeleon_data["hyper_parameters"]
    state = chemeleon_data["state_dict"]

    # Build BondMessagePassing from CheMeleon's saved hyperparameters
    mp = BondMessagePassing(**hyper)
    mp.load_state_dict(state)
    mp.eval()

    d_h = hyper.get("d_h", hyper.get("hidden_size", 300))

    # Minimal MPNN: MP + aggregation + batch-norm (no FFN head)
    # RegressionFFN is required by the MPNN constructor but its output is discarded.
    agg = MeanAggregation()
    dummy_ffn = RegressionFFN(n_tasks=1, input_dim=d_h)
    model = MPNN(message_passing=mp, agg=agg, predictor=dummy_ffn)
    model = model.to(device)
    model.eval()

    # BatchMolGraph.to(device) returns None in Chemprop 2.2.3 — run extraction
    # on CPU only. For 4-5k compounds this is fast enough (~2-5 min on CPU).
    model = model.cpu()
    model.eval()

    # Build dataset — dummy targets (zeros) since we only need the fingerprint
    featurizer = SimpleMoleculeMolGraphFeaturizer()
    dps = []
    valid_mask = []
    for smi in smiles_list:
        try:
            dp = MoleculeDatapoint.from_smi(smi, np.zeros(1, dtype=np.float32))
            dps.append(dp)
            valid_mask.append(True)
        except Exception:
            dps.append(MoleculeDatapoint.from_smi("C", np.zeros(1, dtype=np.float32)))  # methane placeholder
            valid_mask.append(False)

    dataset = MoleculeDataset(dps, featurizer)
    # drop_last=False: we're in model.eval() so batch-norm uses running stats,
    # not batch stats — safe to include a final batch of any size.
    loader = build_dataloader(dataset, batch_size=batch_size, num_workers=0, shuffle=False, drop_last=False)

    embeddings = []
    with torch.no_grad():
        for batch in loader:
            bmg = batch.bmg  # Do NOT call .to(device) — returns None in chemprop 2.2.3
            H = model.fingerprint(bmg)  # (batch_size, d_h)
            embeddings.append(H.cpu().numpy())

    emb_matrix = np.concatenate(embeddings, axis=0).astype(np.float32)  # (n, d_h)

    # Zero out rows where SMILES parsing failed
    for i, ok in enumerate(valid_mask):
        if not ok:
            emb_matrix[i] = 0.0

    n_failed = sum(1 for ok in valid_mask if not ok)
    if n_failed:
        logger.warning(f"{n_failed} SMILES failed — corresponding embeddings set to zeros")

    logger.info(f"CheMeleon embeddings: shape={emb_matrix.shape}, d_emb={emb_matrix.shape[1]}")
    return emb_matrix
