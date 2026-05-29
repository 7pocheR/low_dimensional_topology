#!/bin/bash
#SBATCH --job-name=lk_v7
#SBATCH --output=lk_v7_%A_%a.out
#SBATCH --error=lk_v7_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH --array=0-519

# V7: Same as V5 but with L1-ordered grid placement
# 13 copies × 8 models × 20 seeds = 2080 experiments
# 4 experiments per job → 520 array tasks
#
# ORDERING: copies first, so we see all models at small k before large k
#
# Models (8 total):
#   Monotonic no-skip: relu, elu, leaky_relu
#   Non-monotonic: gelu, swish
#   Skip connections: relu_skip, elu_skip, leaky_relu_skip

set -e
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Configuration (same as v5)
N=2
DEPTH=5
SPACING=10.0
RHO=0.5
BASE_TRAIN=10000
BASE_VAL_TEST=1000
OUTPUT_BASE=${OUTPUT_BASE:-"${SCRIPT_DIR}/results/linking_seeds_v7"}

# Model configurations: "activation skip_flag"
MODELS=(
    "relu 0"
    "relu 1"
    "elu 0"
    "elu 1"
    "leaky_relu 0"
    "leaky_relu 1"
    "gelu 0"
    "swish 0"
)

# New k values: 1-10, 20, 50, 100
COPIES=(1 2 3 4 5 6 7 8 9 10 20 50 100)
SEEDS=(42 123 456 789 1000 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)

NUM_MODELS=${#MODELS[@]}
NUM_COPIES=${#COPIES[@]}
NUM_SEEDS=${#SEEDS[@]}

# Total: 13 × 8 × 20 = 2080 experiments
TOTAL_EXPERIMENTS=$((NUM_COPIES * NUM_MODELS * NUM_SEEDS))

echo "============================================================"
echo "Linking Number Scaling V7 (L1-grid, v5 settings)"
echo "============================================================"
echo "Job ID: $SLURM_ARRAY_JOB_ID, Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "GPUs: 4"
echo "Total experiments: $TOTAL_EXPERIMENTS"
echo "Started: $(date)"
echo "============================================================"

cd "$SCRIPT_DIR"
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-manifold_muon}"
fi
mkdir -p "$OUTPUT_BASE"

# Each array task handles 4 consecutive experiment IDs
BASE_EXP_ID=$((SLURM_ARRAY_TASK_ID * 4))

# Decode experiment ID to (copies_idx, model_idx, seed_idx)
# ORDERING: copies first → model → seed
decode_exp_id() {
    local exp_id=$1
    # Layout: for each copies, for each model, for each seed
    local copies_idx=$((exp_id / (NUM_MODELS * NUM_SEEDS)))
    local remainder=$((exp_id % (NUM_MODELS * NUM_SEEDS)))
    local model_idx=$((remainder / NUM_SEEDS))
    local seed_idx=$((remainder % NUM_SEEDS))
    echo "$copies_idx $model_idx $seed_idx"
}

# Run 4 experiments in parallel, one per GPU
for gpu_id in 0 1 2 3; do
    EXP_ID=$((BASE_EXP_ID + gpu_id))

    if [ $EXP_ID -ge $TOTAL_EXPERIMENTS ]; then
        echo "GPU $gpu_id: Skipping experiment $EXP_ID (out of range)"
        continue
    fi

    # Decode configuration
    read COPIES_IDX MODEL_IDX SEED_IDX <<< $(decode_exp_id $EXP_ID)

    # Parse model config
    MODEL_CONFIG="${MODELS[$MODEL_IDX]}"
    ACT=$(echo $MODEL_CONFIG | cut -d' ' -f1)
    SKIP=$(echo $MODEL_CONFIG | cut -d' ' -f2)

    COPY_COUNT=${COPIES[$COPIES_IDX]}
    SEED=${SEEDS[$SEED_IDX]}

    if [ "$SKIP" = "1" ]; then
        SKIP_STR="skip"
        SKIP_FLAG="--skip"
    else
        SKIP_STR="noskip"
        SKIP_FLAG=""
    fi

    OUTPUT_DIR="${OUTPUT_BASE}/n${N}_d${DEPTH}_${ACT}_${SKIP_STR}_c${COPY_COUNT}_s${SEED}"

    # Skip if already done
    if [ -f "${OUTPUT_DIR}/results.json" ]; then
        echo "GPU $gpu_id: Exp $EXP_ID already done ($ACT $SKIP_STR c=$COPY_COUNT s=$SEED)"
        continue
    fi

    # Scale samples with copies (capped) - same as v5
    TRAIN_SAMPLES=$((BASE_TRAIN * COPY_COUNT))
    VAL_TEST_SAMPLES=$((BASE_VAL_TEST * COPY_COUNT))
    [ $TRAIN_SAMPLES -gt 10000000 ] && TRAIN_SAMPLES=10000000
    [ $VAL_TEST_SAMPLES -gt 500000 ] && VAL_TEST_SAMPLES=500000

    # Patience scales with copies - same as v5
    PATIENCE=$((100 + COPY_COUNT))
    [ $PATIENCE -gt 300 ] && PATIENCE=300

    mkdir -p $OUTPUT_DIR

    echo "GPU $gpu_id: Exp $EXP_ID -> $ACT $SKIP_STR c=$COPY_COUNT s=$SEED"

    CUDA_VISIBLE_DEVICES=$gpu_id python -u train_width_scaling_v7.py \
        --n $N \
        --depth $DEPTH \
        --activation $ACT \
        --num_copies $COPY_COUNT \
        --spacing $SPACING \
        --rho $RHO \
        --train_samples $TRAIN_SAMPLES \
        --val_test_samples $VAL_TEST_SAMPLES \
        --patience $PATIENCE \
        --seed $SEED \
        --output_dir $OUTPUT_DIR \
        $SKIP_FLAG \
        > "${OUTPUT_DIR}/train.log" 2>&1 &
done

wait

echo "============================================================"
echo "All experiments completed"
echo "Finished: $(date)"
echo "============================================================"
