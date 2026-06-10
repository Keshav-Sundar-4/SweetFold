#!/bin/bash
#SBATCH -A hagan-lab
#SBATCH -p hagan-gpu
#SBATCH --gres=gpu:H100:4
#SBATCH --job-name=boltz_glycanbias_module_training_4GPU
#SBATCH --time=5-00:00:00

# Ensure the output directory exists (optional, train.py does it too)
# mkdir -p /work/keshavsundar/work_sundar/glycan_test

# Load necessary modules (if applicable on your cluster)
# The following line loads the CUDA module version 12.4 and ignores any cached version.
module --ignore-cache load cuda/12.4
# You can also load additional modules, for example:
# module load anaconda/XXXX.XX

echo "Loading Conda environment..."
source activate /work/keshavsundar/env/sweetfold
echo "Conda environment loaded. Starting training..."

# --- Main Execution Line ---
# Replace structure.yaml with the actual path if it's not in the CWD
# Add any overrides after the yaml path if needed, e.g., python train.py structure.yaml trainer.devices=2 data.batch_size=2
python train.py full.yaml

echo "Training script finished."
