#!/bin/bash
#SBATCH --job-name=lk_track
#SBATCH --output=lk_track_%j.out
#SBATCH --error=lk_track_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=general
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

# Track linking number through layers for the best seed of each architecture
# Picks the highest-accuracy non-collapsed seed for each model type

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-manifold_muon}"
fi
RESULTS_BASE=${RESULTS_BASE:-"${SCRIPT_DIR}/results_width_r3"}
OUTPUT_BASE=${OUTPUT_BASE:-"${SCRIPT_DIR}/results_linking_tracking"}
mkdir -p "$OUTPUT_BASE"

echo "=========================================="
echo "Linking Number Layer-by-Layer Tracking"
echo "=========================================="

# For each architecture, find the best seed and track
for MODEL_DIR_PATTERN in \
    "n1_w3_d5_relu_c1" \
    "n1_w3_d5_gelu_noskip_c1" \
    "n1_w3_d5_relu_skip_c1"; do

    echo ""
    echo "=== Pattern: $MODEL_DIR_PATTERN ==="

    # Find the seed with highest test accuracy
    BEST_DIR=""
    BEST_ACC=0
    for d in "${RESULTS_BASE}"/${MODEL_DIR_PATTERN}_s*/; do
        if [ -f "$d/results.json" ]; then
            acc=$(python3 -c "import json; print(json.load(open('$d/results.json'))['test_acc'])" 2>/dev/null)
            # Skip collapsed runs (acc ~ 0.5)
            is_better=$(python3 -c "print(1 if $acc > $BEST_ACC and $acc > 0.55 else 0)" 2>/dev/null)
            if [ "$is_better" = "1" ]; then
                BEST_ACC=$acc
                BEST_DIR=$d
            fi
        fi
    done

    if [ -z "$BEST_DIR" ]; then
        echo "  No valid model found for $MODEL_DIR_PATTERN"
        continue
    fi

    echo "  Best seed: $BEST_DIR (acc=$BEST_ACC)"

    # Run tracking with multiple point counts for robustness
    for NPTS in 100 200 500; do
        echo "  Tracking with n_points=$NPTS..."
        python track_linking_through_layers.py \
            --results-dir "$BEST_DIR" \
            --n-points $NPTS \
            --seed 42 \
            --output-dir "${OUTPUT_BASE}/$(basename $BEST_DIR)_npts${NPTS}"
    done
done

echo ""
echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
