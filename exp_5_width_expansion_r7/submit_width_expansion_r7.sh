#!/bin/bash
#SBATCH --job-name=width_r7
#SBATCH --output=width_r7_%A_%a.out
#SBATCH --error=width_r7_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-119

# Width expansion in R^7: third data point for multiplicative scaling
# n=3 (R^7, critical width=7), S^3 ⊔ S^3 linked
# widths 7, 8, 10, 14, 21, 28, 35, 49  (8 widths x 15 seeds = 120 jobs)
# k=10 copies, matching R^5 setup
#
# If multiplicative: need ~7x = 49 for ~100%
# If additive (+7): need ~14 for ~100%

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-manifold_muon}"
fi
OUTPUT_BASE=${OUTPUT_BASE:-"${SCRIPT_DIR}/results_width_r7"}
mkdir -p "$OUTPUT_BASE"

WIDTHS=(7 8 10 14 21 28 35 49)
N_SEEDS=15

WIDTH_IDX=$((SLURM_ARRAY_TASK_ID / N_SEEDS))
SEED=$((SLURM_ARRAY_TASK_ID % N_SEEDS))
WIDTH=${WIDTHS[$WIDTH_IDX]}

COPIES=10

OUTDIR="${OUTPUT_BASE}/n3_w${WIDTH}_d5_relu_c${COPIES}_s${SEED}"
mkdir -p "$OUTDIR"

if [ -f "$OUTDIR/results.json" ]; then
    echo "Already done: $OUTDIR"
    exit 0
fi

echo "Width expansion R7: n=3, width=${WIDTH}, depth=5, relu, copies=${COPIES}, seed=${SEED}"

python train_width_scaling_v7.py \
    --n 3 \
    --depth 5 \
    --activation relu \
    --width $WIDTH \
    --num_copies $COPIES \
    --seed $SEED \
    --train_samples 100000 \
    --val_test_samples 10000 \
    --epochs 500 \
    --patience 110 \
    --output_dir "$OUTDIR"
