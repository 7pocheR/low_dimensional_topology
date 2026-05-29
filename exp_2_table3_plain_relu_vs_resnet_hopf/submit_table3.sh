#!/bin/bash
#SBATCH --job-name=cam_t3
#SBATCH --output=cam_t3_%A_%a.out
#SBATCH --error=cam_t3_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:4
#SBATCH --array=0-29

# Camera-ready Table 3: 6 depths x 2 archs x 10 seeds = 120 trials
# 4 trials per array task -> 30 array tasks
# Seeds: 42-51 (same as original relu_resnet_comparison.py)

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUTPUT_BASE=${OUTPUT_BASE:-"${SCRIPT_DIR}/results/table3"}

DEPTHS=(3 4 5 6 7 8)
ARCHS=(relu resnet)
SEEDS=(42 43 44 45 46 47 48 49 50 51)

NUM_DEPTHS=${#DEPTHS[@]}
NUM_ARCHS=${#ARCHS[@]}
NUM_SEEDS=${#SEEDS[@]}
TOTAL=$((NUM_DEPTHS * NUM_ARCHS * NUM_SEEDS))

mkdir -p "$OUTPUT_BASE"

cd "$SCRIPT_DIR"
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-manifold_muon}"
fi

BASE=$((SLURM_ARRAY_TASK_ID * 4))

decode() {
    local exp=$1
    local d_idx=$((exp / (NUM_ARCHS * NUM_SEEDS)))
    local rem=$((exp % (NUM_ARCHS * NUM_SEEDS)))
    local a_idx=$((rem / NUM_SEEDS))
    local s_idx=$((rem % NUM_SEEDS))
    echo "$d_idx $a_idx $s_idx"
}

for gpu in 0 1 2 3; do
    EXP=$((BASE + gpu))
    if [ $EXP -ge $TOTAL ]; then continue; fi
    read D_IDX A_IDX S_IDX <<<"$(decode $EXP)"
    DEPTH=${DEPTHS[$D_IDX]}
    ARCH=${ARCHS[$A_IDX]}
    SEED=${SEEDS[$S_IDX]}
    OUT="${OUTPUT_BASE}/d${DEPTH}_${ARCH}_s${SEED}"
    mkdir -p "$OUT"
    if [ -f "$OUT/result.json" ]; then
        echo "gpu $gpu: skip (already done) $OUT"
        continue
    fi
    echo "gpu $gpu: depth=$DEPTH arch=$ARCH seed=$SEED -> $OUT"
    CUDA_VISIBLE_DEVICES=$gpu python -u run_table3_trial.py \
        --depth $DEPTH --arch $ARCH --seed $SEED \
        --output_dir "$OUT" \
        > "$OUT/train.log" 2>&1 &
done

wait
echo "Array task $SLURM_ARRAY_TASK_ID done at $(date)"
