#!/bin/bash
#SBATCH --job-name=cnn10class
#SBATCH --partition=general
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --array=0-41
#SBATCH --output=cnn10class_%A_%a.out
#SBATCH --error=cnn10class_%A_%a.err

# 10-class CIFAR-10 classification
# Hypothesis: Non-monotonic advantage should appear specifically for linked pairs
#
# 7 activations × 2 skip configs × 3 depths = 42 experiments

ACTIVATIONS=(relu elu selu leaky_relu gelu swish mish)
CONV_BLOCKS=(1 2 3)

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

TOTAL_LAYERS=$((NUM_CONV_BLOCKS * 3 + 2))

case $ACTIVATION in
    relu|elu|selu|leaky_relu)
        ACT_TYPE="MONOTONIC"
        ;;
    gelu|swish|mish)
        ACT_TYPE="NON-MONOTONIC"
        ;;
esac

echo "=============================================="
echo "10-CLASS CIFAR-10 CNN - Task $SLURM_ARRAY_TASK_ID"
echo "=============================================="
echo "Job Array ID: $SLURM_ARRAY_JOB_ID"
echo "Activation: $ACTIVATION ($ACT_TYPE)"
echo "Conv blocks: $NUM_CONV_BLOCKS, Total layers: $TOTAL_LAYERS"
echo "Skip: $SKIP_STR"
echo "Start time: $(date)"
echo "=============================================="

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
DATA_DIR=${DATA_DIR:-"${PROJECT_ROOT}/data"}
OUTPUT_DIR=${OUTPUT_DIR:-"${PROJECT_ROOT}/results/cnn_10class"}

cd "$PROJECT_ROOT"
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-manifold_muon}"
fi
export PYTHONUNBUFFERED=1
mkdir -p "$DATA_DIR" "$OUTPUT_DIR"

python scripts/train_cnn_10class.py \
    --activation $ACTIVATION \
    --num-conv-blocks $NUM_CONV_BLOCKS \
    $USE_SKIP \
    --epochs 100 \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR"

echo "End time: $(date)"
