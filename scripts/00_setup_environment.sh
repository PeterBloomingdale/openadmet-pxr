#!/usr/bin/env bash
# Environment setup for the OpenADMET PXR Challenge
# Run once from the repo root before anything else.
set -e

echo "=== OpenADMET PXR Challenge Environment Setup ==="
echo ""

# 1. Install uv (fast Python package manager) if not present
if ! command -v uv &> /dev/null; then
    echo "[1/5] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
else
    echo "[1/5] uv already installed ($(uv --version))"
fi

# 2. Create virtual environment
echo "[2/5] Creating Python 3.11 virtual environment..."
uv venv .venv --python 3.11
source .venv/bin/activate

# 3. Install the project
echo "[3/5] Installing openadmet package and all dependencies..."
uv pip install -e ".[dev]"

# 4. Verify critical imports
echo "[4/5] Verifying critical imports..."
python -c "
from rdkit import Chem; print('  ✓ RDKit')
import lightgbm; print('  ✓ LightGBM', lightgbm.__version__)
import torch; print('  ✓ PyTorch', torch.__version__, '| CUDA:', torch.cuda.is_available())
import chemprop; print('  ✓ Chemprop', chemprop.__version__)
import datasets; print('  ✓ HuggingFace datasets')
from chembl_webresource_client.new_client import new_client; print('  ✓ ChEMBL client')
import mapie; print('  ✓ MAPIE')
import wandb; print('  ✓ W&B')
try:
    from tabpfn import TabPFNRegressor; print('  ✓ TabPFN v2')
except ImportError:
    print('  ⚠ TabPFN not available (optional)')
"

# 5. Verify mmpdb CLI
echo "[5/5] Verifying mmpdb CLI..."
if command -v mmpdb &> /dev/null; then
    echo "  ✓ mmpdb CLI found: $(which mmpdb)"
    mmpdb --version 2>/dev/null && echo "  mmpdb version OK" || echo "  ⚠ mmpdb --version failed (may be OK)"
else
    echo "  ⚠ mmpdb CLI not found on PATH"
    echo "    Try: pip install mmpdb  OR  conda install -c conda-forge mmpdb"
    echo "    Then verify with: mmpdb --help"
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. cp .env.example .env  # Fill in HF_TOKEN and WANDB_API_KEY"
echo "  2. python scripts/01_download_data.py"
echo "  3. python scripts/01b_eda.py  # EDA BEFORE any model decisions"
echo "  4. source .venv/bin/activate  # Each new terminal session"
