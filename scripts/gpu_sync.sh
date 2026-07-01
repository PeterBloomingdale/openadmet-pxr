#!/usr/bin/env bash
# Remote GPU job manager for UniMol s4/s5 training on CUDA PC.
#
# Setup: create a .env.gpu file (gitignored, never committed) with:
#   GPU_HOST=<your GPU machine IP or hostname>
#   GPU_USER=<your username on that machine>
#
# Usage:
#   bash scripts/gpu_sync.sh push    — copy data + scripts to PC, install deps
#   bash scripts/gpu_sync.sh run     — launch s4 and s5 training in background
#   bash scripts/gpu_sync.sh status  — tail remote training logs
#   bash scripts/gpu_sync.sh pull    — copy OOF + test predictions back to Mac
#   bash scripts/gpu_sync.sh verify  — check CUDA is available on remote

set -euo pipefail

ENV_FILE=".env.gpu"
if [ ! -f "$ENV_FILE" ]; then
    echo "Error: $ENV_FILE not found."
    echo "Create $ENV_FILE with GPU_HOST and GPU_USER (see header comment)."
    exit 1
fi
# shellcheck source=.env.gpu
source "$ENV_FILE"

: "${GPU_HOST:?GPU_HOST not set in $ENV_FILE}"
: "${GPU_USER:?GPU_USER not set in $ENV_FILE}"

REMOTE="${GPU_USER}@${GPU_HOST}"
REMOTE_DIR="~/unimol_pxr"

MODE="${1:-}"
if [ -z "$MODE" ]; then
    echo "Usage: bash scripts/gpu_sync.sh <push|run|status|pull|verify>"
    exit 1
fi

# ─── push ────────────────────────────────────────────────────────────────────
if [ "$MODE" = "push" ]; then
    echo "=== Creating remote directory structure ==="
    ssh "$REMOTE" "mkdir -p $REMOTE_DIR/data/splits $REMOTE_DIR/data/curated \
        $REMOTE_DIR/scripts \
        $REMOTE_DIR/src/openadmet/cv \
        $REMOTE_DIR/src/openadmet/utils \
        $REMOTE_DIR/models"

    echo "=== Copying data files (~750 KB) ==="
    scp data/splits/butina_folds.parquet \
        "$REMOTE:$REMOTE_DIR/data/splits/"
    scp data/curated/openadmet_test_std.parquet \
        "$REMOTE:$REMOTE_DIR/data/curated/"

    echo "=== Copying training scripts ==="
    scp scripts/35_train_unimol2_s4.py \
        scripts/36_train_unimol2_s5.py \
        "$REMOTE:$REMOTE_DIR/scripts/"

    echo "=== Copying source modules ==="
    scp src/openadmet/cv/oof.py \
        "$REMOTE:$REMOTE_DIR/src/openadmet/cv/"
    scp src/openadmet/utils/device.py \
        "$REMOTE:$REMOTE_DIR/src/openadmet/utils/"

    echo "=== Writing __init__.py stubs so imports resolve ==="
    ssh "$REMOTE" "
        touch $REMOTE_DIR/src/__init__.py
        touch $REMOTE_DIR/src/openadmet/__init__.py
        touch $REMOTE_DIR/src/openadmet/cv/__init__.py
        touch $REMOTE_DIR/src/openadmet/utils/__init__.py
    "

    echo "=== Installing Python dependencies on remote ==="
    ssh "$REMOTE" "pip install --quiet unimol_tools torch pandas loguru scipy pyarrow"

    echo ""
    echo "✅ Push complete. Run: bash scripts/gpu_sync.sh verify"

# ─── verify ──────────────────────────────────────────────────────────────────
elif [ "$MODE" = "verify" ]; then
    echo "=== Checking CUDA availability ==="
    ssh "$REMOTE" "python -c \"
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory/1e9, 1), 'GB')
\""
    echo ""
    echo "=== Checking data files ==="
    ssh "$REMOTE" "ls -lh $REMOTE_DIR/data/splits/ $REMOTE_DIR/data/curated/"

# ─── run ─────────────────────────────────────────────────────────────────────
elif [ "$MODE" = "run" ]; then
    echo "=== Launching UniMol s4 (LR=5e-4) ==="
    ssh "$REMOTE" "cd $REMOTE_DIR && \
        nohup python scripts/35_train_unimol2_s4.py > s4.log 2>&1 & \
        echo \"s4 PID: \$!\""

    echo "=== Launching UniMol s5 (LR=1e-3, batch=4) ==="
    ssh "$REMOTE" "cd $REMOTE_DIR && \
        nohup python scripts/36_train_unimol2_s5.py > s5.log 2>&1 & \
        echo \"s5 PID: \$!\""

    echo ""
    echo "✅ Jobs launched. Check progress: bash scripts/gpu_sync.sh status"
    echo "   Expected runtime: ~4h each on GTX 1080"

# ─── status ──────────────────────────────────────────────────────────────────
elif [ "$MODE" = "status" ]; then
    echo "=== UniMol s4 log (last 20 lines) ==="
    ssh "$REMOTE" "tail -20 $REMOTE_DIR/s4.log 2>/dev/null || echo '(no s4.log yet)'"
    echo ""
    echo "=== UniMol s5 log (last 20 lines) ==="
    ssh "$REMOTE" "tail -20 $REMOTE_DIR/s5.log 2>/dev/null || echo '(no s5.log yet)'"
    echo ""
    echo "=== Running processes ==="
    ssh "$REMOTE" "ps aux | grep train_unimol | grep -v grep || echo '(no unimol processes found)'"

# ─── pull ────────────────────────────────────────────────────────────────────
elif [ "$MODE" = "pull" ]; then
    for variant in s4 s5; do
        case $variant in
            s4) script_num=4 ;;
            s5) script_num=5 ;;
        esac
        REMOTE_MODEL="$REMOTE_DIR/models/unimol2_${variant}"
        LOCAL_MODEL="models/unimol2_${variant}"
        mkdir -p "$LOCAL_MODEL"

        OOF="$REMOTE_MODEL/oof_predictions.npy"
        TEST="$REMOTE_MODEL/test_predictions.npy"
        METRICS="$REMOTE_MODEL/metrics.json"

        if ssh "$REMOTE" "[ -f $OOF ]"; then
            echo "=== Pulling unimol2_${variant} predictions ==="
            scp "$REMOTE:$OOF"     "$LOCAL_MODEL/oof_predictions.npy"
            scp "$REMOTE:$TEST"    "$LOCAL_MODEL/test_predictions.npy"
            scp "$REMOTE:$METRICS" "$LOCAL_MODEL/metrics.json" 2>/dev/null || true
            echo "  ✅ unimol2_${variant}: pulled"
        else
            echo "  ⏳ unimol2_${variant}: OOF not ready yet — check status first"
        fi
    done

    echo ""
    echo "Next: python scripts/39_ensemble_phase2.py"

else
    echo "Unknown mode: $MODE"
    echo "Usage: bash scripts/gpu_sync.sh <push|run|status|pull|verify>"
    exit 1
fi
