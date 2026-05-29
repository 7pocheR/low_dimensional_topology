#!/bin/bash
#SBATCH --job-name=eps_aug
#SBATCH --partition=general
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-7
#SBATCH --output=eps_aug_%A_%a.out
#SBATCH --error=eps_aug_%A_%a.err

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

# 45 pairs split across 8 jobs: 6,6,6,6,6,6,6,3
PAIRS_PER_JOB=6
START=$((SLURM_ARRAY_TASK_ID * PAIRS_PER_JOB))
END=$((START + PAIRS_PER_JOB))
if [ $END -gt 45 ]; then
    END=45
fi

echo "=============================================="
echo "ε_min Binary Search — Shard ${SLURM_ARRAY_TASK_ID} (pairs ${START}-$((END-1)))"
echo "Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}, $(date)"
echo "=============================================="

python analysis/eps_min_all_pairs_augmented.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --pair-start $START \
    --pair-end $END \
    --eps-start 0.05 \
    --eps-cap 1.0 \
    --precision 0.005 \
    --top-k 500

echo "Done: $(date)"
