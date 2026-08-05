#!/bin/bash
#....

# Activate the Conda environment
source activate /work/keshavsundar/env/sweetfold

# Run Boltz prediction
boltz predict example.yaml \
  --cache /work/keshavsundar/env/sweetfold/weights \
  --checkpoint /work/keshavsundar/env/sweetfold/weights/boltz1_glycan_epoch_23.ckpt \
  --no_potentials \
  --sampling_steps 200 \
  --diffusion_samples 10

# Checkpoint refers to the checkpoint that the Boltz is being ran on. 
#Potentials (physical steering) from Boltz is turned of. This script predicts 10 diffusion samples
# It runs for 200 diffusion steps
