#!/bin/bash
#SBATCH --job-name=cnn_binary
#SBATCH --partition=general
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --array=0-41
#SBATCH --output=cnn_binary_%A_%a.out
#SBATCH --error=cnn_binary_%A_%a.err

# Configuration matrix:
# 7 activations: relu, elu, selu, leaky_relu, gelu, swish, mish
# 2 skip configs: no skip, with skip
# 3 depths: 1, 2, 3 conv blocks per resolution (5, 8, 11 total layers)
# Total: 7 × 2 × 3 = 42 experiments
#
# Width constraint: all intermediate layers ≤ 3072D (= 3×32×32)
# Architecture: 3×32×32 → 12×16×16 → 48×8×8 → 192×4×4 → FC(3072) → FC(2)

ACTIVATIONS=(relu elu selu leaky_relu gelu swish mish)
CONV_BLOCKS=(1 2 3)  # Number of conv blocks per resolution

# Map array task ID to configuration
# task_id = activation_idx * 6 + skip_idx * 3 + depth_idx
ACT_IDX=$((SLURM_ARRAY_TASK_ID / 6))
REMAINDER=$((SLURM_ARRAY_TASK_ID % 6))
SKIP_IDX=$((REMAINDER / 3))
DEPTH_IDX=$((REMAINDER % 3))

ACTIVATION=${ACTIVATIONS[$ACT_IDX]}
NUM_CONV_BLOCKS=${CONV_BLOCKS[$DEPTH_IDX]}

if [ $SKIP_IDX -eq 0 ]; then
    USE_SKIP=""
    SKIP_STR="no_skip"
else
    USE_SKIP="--use-skip"
    SKIP_STR="skip"
fi

# Calculate total layers
TOTAL_LAYERS=$((NUM_CONV_BLOCKS * 3 + 2))

# Classify activation type
case $ACTIVATION in
    relu|elu|selu|leaky_relu)
        ACT_TYPE="MONOTONIC"
        ;;
    gelu|swish|mish)
        ACT_TYPE="NON-MONOTONIC"
        ;;
esac

echo "=============================================="
echo "Width-Bounded CNN Binary - Task $SLURM_ARRAY_TASK_ID"
echo "=============================================="
echo "Job Array ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'N/A')"
echo "Activation: $ACTIVATION ($ACT_TYPE)"
echo "Conv blocks per resolution: $NUM_CONV_BLOCKS"
echo "Total layers: $TOTAL_LAYERS"
echo "Skip connections: $SKIP_STR"
echo "Width constraint: ≤ 3072D"
echo "Data augmentation: ENABLED"
echo "Start time: $(date)"
echo "=============================================="

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
DATA_DIR=${DATA_DIR:-"${PROJECT_ROOT}/data"}
OUTPUT_DIR=${OUTPUT_DIR:-"${PROJECT_ROOT}/results/cnn_binary"}

cd "$PROJECT_ROOT"
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-manifold_muon}"
fi
export PYTHONUNBUFFERED=1
mkdir -p "$DATA_DIR" "$OUTPUT_DIR"

python scripts/train_cnn_binary.py \
    --activation $ACTIVATION \
    --num-conv-blocks $NUM_CONV_BLOCKS \
    $USE_SKIP \
    --epochs 100 \
    --lr 0.001 \
    --batch-size 128 \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "End time: $(date)"
