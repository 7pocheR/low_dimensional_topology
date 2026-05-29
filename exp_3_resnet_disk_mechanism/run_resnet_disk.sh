#!/bin/bash
#SBATCH --job-name=resnet_disk
#SBATCH --partition=general
#SBATCH --gres=gpu:1                  # 1 GPU
#SBATCH --cpus-per-task=8             # 8 CPUs
#SBATCH --mem=32G                     # 32GB RAM
#SBATCH --time=02:00:00               # 2 hours should be plenty
#SBATCH --output=resnet_disk_%j.out
#SBATCH --error=resnet_disk_%j.err

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Print job info
echo "=========================================="
echo "ResNet Disk/Annulus Visualization Experiment"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "GPU allocated: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo ""

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-cifar10_link}"
fi

# Set environment variables for PyTorch
export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Test PyTorch and GPU
echo "Testing PyTorch setup:"
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
echo ""

cd "$SCRIPT_DIR"
echo "Running colorful_resnet_disk.py..."
echo ""

python colorful_resnet_disk.py

echo ""
echo "=========================================="
echo "Experiment completed!"
echo "End time: $(date)"
echo "=========================================="
