"""
Hardware-agnostic device detection utilities.

Supports CUDA (NVIDIA), MPS (Apple Silicon), and CPU fallback.
All training scripts should use these helpers instead of hardcoding CUDA.
"""

import torch
from loguru import logger


def get_device() -> str:
    """Returns the best available torch device string: 'mps', 'cuda', or 'cpu'."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def log_device_info() -> str:
    """Logs available hardware and returns the selected device string."""
    device = get_device()
    if device == "mps":
        logger.info("Device: Apple Silicon MPS (Metal Performance Shaders)")
    elif device == "cuda":
        logger.info(f"Device: CUDA — {torch.cuda.get_device_name(0)}")
    else:
        logger.info("Device: CPU (no GPU acceleration available)")
    return device


def clear_device_cache(device: str | None = None) -> None:
    """Clears GPU memory cache for the active device."""
    if device is None:
        device = get_device()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def augment_smiles(smiles_list: list[str], n_aug: int = 10) -> tuple[list[str], list[int]]:
    """
    Generates n_aug randomised SMILES per compound for inference-time augmentation.

    Different atom orderings produce different RDKit conformers via ETKDGv3, giving
    UniMol slightly different 3D views of the same molecule. Averaging predictions
    across n_aug variants reduces prediction variance without additional training.

    Returns:
        augmented_smiles: flat list of len(smiles_list) * n_aug strings
        compound_idx: parallel list mapping each augmented SMILES to its source index
    """
    try:
        from rdkit import Chem
    except ImportError:
        logger.warning("RDKit not available — SMILES augmentation skipped (n_aug=1)")
        return smiles_list, list(range(len(smiles_list)))

    augmented, compound_idx = [], []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            augmented.append(smi)
            compound_idx.append(i)
            continue
        seen: set[str] = set()
        for _ in range(n_aug * 3):   # over-sample to hit n_aug unique orderings
            rand_smi = Chem.MolToSmiles(mol, doRandom=True)
            if rand_smi not in seen:
                seen.add(rand_smi)
                augmented.append(rand_smi)
                compound_idx.append(i)
            if len(seen) >= n_aug:
                break
        # Fill to exactly n_aug if molecule is too small to generate enough variants
        while len(seen) < n_aug:
            augmented.append(smi)
            compound_idx.append(i)
            seen.add(smi + str(len(seen)))
    return augmented, compound_idx


def get_unimol_device_params(base_batch_size: int = 16) -> dict:
    """
    Returns unimol_tools device parameters for the current hardware.

    CUDA: use_gpu="all", use_amp=True (FP16 supported)
    MPS:  use_gpu=True,  use_amp=False (FP16 AMP not supported on MPS)
    CPU:  use_gpu=False, use_amp=False

    base_batch_size: baseline for MPS/CPU. CUDA uses same value since
    M5 Pro unified memory is typically larger than GTX 1080 VRAM.
    """
    if torch.backends.mps.is_available():
        return {
            "use_gpu": True,
            "use_amp": False,     # FP16 AMP unsupported on MPS
            "batch_size": base_batch_size,
        }
    if torch.cuda.is_available():
        return {
            "use_gpu": "all",
            "use_amp": True,
            "batch_size": base_batch_size,
        }
    return {
        "use_gpu": False,
        "use_amp": False,
        "batch_size": base_batch_size,
    }
