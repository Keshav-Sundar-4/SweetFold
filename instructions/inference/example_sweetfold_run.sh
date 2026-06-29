#!/bin/bash
#....

# Activate the Conda environment
source activate /work/keshavsundar/env/sweetfold

# Run Boltz prediction
boltz predict glycan_test.yaml \
  --cache /work/keshavsundar/env/sweetfold/weights \
  --checkpoint /work/keshavsundar/env/sweetfold/weights/boltz1_glycan_epoch_23.ckpt \
  --no_potentials \
  --sampling_steps 200 \
  --diffusion_samples 10
