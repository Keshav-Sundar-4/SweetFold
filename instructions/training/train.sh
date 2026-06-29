#!/bin/bash
#SBATCH -A your-account-name
#SBATCH -p your-gpu-partition
#SBATCH --gres=gpu:H100:4
#SBATCH --job-name=sweetfold_training_4gpu
#SBATCH --time=5-00:00:00

set -euo pipefail

# ---------------------------------------------------------------------
# User-configurable paths
# ---------------------------------------------------------------------

# Directory containing full.yaml, train.py, train.sh, and update_checkpoint.py
TRAINING_DIR="/path/to/sweetfold_training"

# Conda or Python environment containing the SweetFold/Boltz installation
SWEETFOLD_ENV="/path/to/sweetfold_env"

# ---------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------

module --ignore-cache load cuda/12.4

echo "Changing to training directory:"
echo "  ${TRAINING_DIR}"
cd "${TRAINING_DIR}"

echo "Activating SweetFold environment:"
echo "  ${SWEETFOLD_ENV}"
source activate "${SWEETFOLD_ENV}"

echo "Starting SweetFold training..."
echo "Using config:"
echo "  ${TRAINING_DIR}/full.yaml"

# ---------------------------------------------------------------------
# Main training command
# ---------------------------------------------------------------------

python train.py full.yaml

echo "Training script finished."
