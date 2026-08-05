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

# The cache refers to the current folder containing the weights
# The checkpoint refers to the actual weights being used by the model
# Potentials, which is Boltz's method of physical steering, are turned off. The model cannot predict accurate structures with them on
# The number of sampling steps is set to 200, which is the default
# The number of diffusion samples is set to 10, meaning you will get 10 output cif files
# More info can be found at: https://github.com/jwohlwend/boltz/blob/v1.0.0/docs/prediction.md
