#!/bin/bash
#SBATCH --job-name=validate_witness
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=validate_witness_%j.out
#SBATCH --error=validate_witness_%j.err

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
DATA_DIR=${DATA_DIR:-"${PROJECT_ROOT}/data"}
RESULTS_FILE=${RESULTS_FILE:-"${PROJECT_ROOT}/results/cifar10/linking_results.json"}
OUTPUT_DIR=${OUTPUT_DIR:-"${PROJECT_ROOT}/results/validation/witness_cycles"}

echo "=============================================="
echo "Validate ACTUAL Witness Cycles from Linked Pairs"
echo "=============================================="

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-cifar10_link}"
fi

cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/utils:${PYTHONPATH:-}"
mkdir -p "$OUTPUT_DIR"

# Use the epsilon=1.0 results (which has witness info)
python analysis/validate_witness_cycles.py \
    --results-file "$RESULTS_FILE" \
    --n-pairs 15 \
    --n-edges 6 \
    --n-interp 5 \
    --output-dir "$OUTPUT_DIR" \
    --data-dir "$DATA_DIR"

echo ""
echo "Results in: ${OUTPUT_DIR}"
