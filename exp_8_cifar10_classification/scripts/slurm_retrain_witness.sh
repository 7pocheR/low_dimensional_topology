#!/bin/bash
#SBATCH --job-name=retrain_witness
#SBATCH --partition=general
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=0-1
#SBATCH --output=retrain_witness_%A_%a.out
#SBATCH --error=retrain_witness_%A_%a.err

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
DATA_DIR=${DATA_DIR:-"${PROJECT_ROOT}/data"}
OUTPUT_DIR=${OUTPUT_DIR:-"${PROJECT_ROOT}/results/cnn_binary_checkpoint"}
cd "$PROJECT_ROOT"

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-cifar10_link}"
fi

export PYTHONPATH="${PROJECT_ROOT}/utils:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
mkdir -p "$DATA_DIR" "$OUTPUT_DIR"

ACTIVATIONS=(relu gelu)
ACT=${ACTIVATIONS[$SLURM_ARRAY_TASK_ID]}

echo "=============================================="
echo "Retraining $ACT L8 no-skip with checkpoint"
echo "Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID"
echo "=============================================="

python scripts/retrain_and_eval_witness.py \
    --activation $ACT \
    --seed 42 \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR"

echo "Done: $(date)"
