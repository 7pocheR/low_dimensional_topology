#!/bin/bash
#SBATCH --job-name=eps_min_aug
#SBATCH --partition=general
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=eps_min_aug_%j.out
#SBATCH --error=eps_min_aug_%j.err

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
DATA_DIR=${DATA_DIR:-"${PROJECT_ROOT}/data"}
OUTPUT_DIR=${OUTPUT_DIR:-"${PROJECT_ROOT}/results/eps_min_augmented"}

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-cifar10_link}"
fi

export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/utils:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
mkdir -p "$DATA_DIR" "$OUTPUT_DIR"

cd "$PROJECT_ROOT"

echo "=============================================="
echo "ε_min Binary Search (AUGMENTED data, matched params)"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

python analysis/eps_min_all_pairs_augmented.py \
    --n-aug 20 \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --eps-cap 2.0 \
    --precision 0.005

echo "End time: $(date)"
