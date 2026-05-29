#!/bin/bash
#SBATCH --job-name=cam_t2
#SBATCH --output=cam_t2_%A_%a.out
#SBATCH --error=cam_t2_%A_%a.err
#SBATCH --time=8:00:00
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:4
#SBATCH --array=0-89

# Camera-ready Table 2: 6 depths x 2 activations x 30 seeds = 360 trials
# 4 trials per array task -> 90 array tasks
# Seeds: np.random.seed(42); np.random.randint(1, 10000, 30)

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUTPUT_BASE=${OUTPUT_BASE:-"${SCRIPT_DIR}/results/table2"}

DEPTHS=(3 5 8 12 16 20)
ACTS=(relu gelu)
SEEDS=(7271 861 5391 5192 5735 6266 467 4427 5579 8323 \
       1686 770 6950 2434 5312 5052 6421 1185 4556 3386 \
       6397 8667 9275 2559 7850 2048 2748 9168 9999 190)

NUM_DEPTHS=${#DEPTHS[@]}
NUM_ACTS=${#ACTS[@]}
NUM_SEEDS=${#SEEDS[@]}
TOTAL=$((NUM_DEPTHS * NUM_ACTS * NUM_SEEDS))

mkdir -p "$OUTPUT_BASE"

cd "$SCRIPT_DIR"
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-manifold_muon}"
fi

BASE=$((SLURM_ARRAY_TASK_ID * 4))

decode() {
    local exp=$1
    # Layout (slowest -> fastest): depth, activation, seed
    local d_idx=$((exp / (NUM_ACTS * NUM_SEEDS)))
    local rem=$((exp % (NUM_ACTS * NUM_SEEDS)))
    local a_idx=$((rem / NUM_SEEDS))
    local s_idx=$((rem % NUM_SEEDS))
    echo "$d_idx $a_idx $s_idx"
}

for gpu in 0 1 2 3; do
    EXP=$((BASE + gpu))
    if [ $EXP -ge $TOTAL ]; then continue; fi
    read D_IDX A_IDX S_IDX <<<"$(decode $EXP)"
    DEPTH=${DEPTHS[$D_IDX]}
    ACT=${ACTS[$A_IDX]}
    SEED=${SEEDS[$S_IDX]}
    OUT="${OUTPUT_BASE}/d${DEPTH}_${ACT}_s${SEED}"
    mkdir -p "$OUT"
    if [ -f "$OUT/result.json" ]; then
        echo "gpu $gpu: skip (already done) $OUT"
        continue
    fi
    echo "gpu $gpu: depth=$DEPTH act=$ACT seed=$SEED -> $OUT"
    CUDA_VISIBLE_DEVICES=$gpu python -u run_table2_trial.py \
        --depth $DEPTH --activation $ACT --seed $SEED \
        --output_dir "$OUT" \
        > "$OUT/train.log" 2>&1 &
done

wait
echo "Array task $SLURM_ARRAY_TASK_ID done at $(date)"
