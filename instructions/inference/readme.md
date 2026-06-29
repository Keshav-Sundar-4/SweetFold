# SweetFold Inference and Training Setup Instructions

This guide explains how to set up SweetFold for inference and training.

SweetFold is built by installing Boltz v1.0.0 first, then replacing the installed Boltz source-code folder with the SweetFold-modified Boltz folder from the SweetFold GitHub repository.

The SweetFold-modified Boltz source code is located here:

https://github.com/Keshav-Sundar-4/SweetFold/tree/main/src/boltz

Important: inside the SweetFold GitHub repository, the folder path is `src/boltz`. However, when copied into your Python environment, the folder must be named exactly `boltz`, not `src/boltz`.

---

# Part 1: SweetFold Inference Setup Instructions

These instructions explain how to set up SweetFold for inference using the epoch 23 SweetFold checkpoint.

Inference is simpler than training. You only need:

1. A working SweetFold/Boltz environment.
2. Boltz v1.0.0 installed.
3. The SweetFold-modified `boltz` source-code folder.
4. The SweetFold epoch 23 checkpoint.
5. An example inference shell script or your own command.

---

## 1. Important Inference Paths

Replace every `/path/to/...` placeholder with the correct absolute path on your system.

### SweetFold environment

```text
/path/to/sweetfold_env
```

This is the Python or Conda environment where Boltz/SweetFold is installed.

### SweetFold weights directory

```text
/path/to/sweetfold_env/weights
```

This is where you should place the SweetFold inference checkpoint.

You may need to create this folder yourself.

### Installed Boltz source-code directory

```text
/path/to/sweetfold_env/lib/python3.10/site-packages/boltz
```

This is the installed Boltz package folder inside the Python environment. This folder must be replaced with the SweetFold-modified `boltz` folder from:

```text
SweetFold/src/boltz
```

### SweetFold GitHub repository

```text
https://github.com/Keshav-Sundar-4/SweetFold
```

The SweetFold-modified Boltz folder is specifically located at:

```text
https://github.com/Keshav-Sundar-4/SweetFold/tree/main/src/boltz
```

### SweetFold checkpoint page

```text
https://huggingface.co/Keshav-Sundar-4/SweetFold/tree/main
```

This Hugging Face repository contains the SweetFold checkpoint files.

---

## 2. Create the SweetFold Environment

Create and activate a Python environment.

Example using Conda:

```bash
conda create -n sweetfold_env python=3.10 -y
conda activate sweetfold_env
python -m pip install --upgrade pip
```

Install Boltz v1.0.0 specifically with CUDA support:

```bash
pip install "boltz[cuda]==1.0.0"
```

Do not install a different Boltz version unless you know exactly what you are doing. SweetFold expects the Boltz v1.0.0 codebase as the base package.

---

## 3. Create the Weights Folder

Create a weights folder inside the SweetFold environment.

```bash
mkdir -p /path/to/sweetfold_env/weights
```

This folder will store the SweetFold checkpoint used for inference.

---

## 4. Download the SweetFold Epoch 23 Checkpoint

Open the SweetFold Hugging Face repository in your browser:

```text
https://huggingface.co/Keshav-Sundar-4/SweetFold/tree/main
```

Download the epoch 23 SweetFold checkpoint from that page.

After downloading it, move the checkpoint into:

```text
/path/to/sweetfold_env/weights
```

For example, your weights folder may look like:

```text
/path/to/sweetfold_env/weights/
└── boltz1_glycan_epoch_23.ckpt
```

The exact filename should match the checkpoint you downloaded. Do not rename it unless your inference script expects a specific filename.

---

## 5. Download the SweetFold-Modified `boltz` Folder

The SweetFold-modified Boltz source code lives inside the SweetFold GitHub repository at:

```text
https://github.com/Keshav-Sundar-4/SweetFold/tree/main/src/boltz
```

The folder path inside GitHub is:

```text
src/boltz
```

However, the folder you copy into your environment must be named:

```text
boltz
```

It must not be named:

```text
src/boltz
```

The final installed package path must look like:

```text
/path/to/sweetfold_env/lib/python3.10/site-packages/boltz
```

not:

```text
/path/to/sweetfold_env/lib/python3.10/site-packages/src/boltz
```

### Recommended method: clone the full SweetFold repository

```bash
cd /path/to/project
git clone https://github.com/Keshav-Sundar-4/SweetFold.git sweetfold_repo
```

After cloning, the SweetFold-modified Boltz folder will be located at:

```text
/path/to/project/sweetfold_repo/src/boltz
```

This is the folder you will use to replace the installed Boltz package.

---

## 6. Replace the Installed Boltz Folder

Activate the SweetFold environment:

```bash
conda activate sweetfold_env
```

Find the installed Boltz source-code folder:

```bash
python -c "import boltz, pathlib; print(pathlib.Path(boltz.__file__).parent)"
```

This should print something like:

```text
/path/to/sweetfold_env/lib/python3.10/site-packages/boltz
```

Back up the original Boltz folder:

```bash
mv /path/to/sweetfold_env/lib/python3.10/site-packages/boltz /path/to/sweetfold_env/lib/python3.10/site-packages/boltz_original
```

Copy the SweetFold-modified Boltz folder into the environment:

```bash
cp -r /path/to/project/sweetfold_repo/src/boltz /path/to/sweetfold_env/lib/python3.10/site-packages/boltz
```

Verify that the copied folder is named correctly:

```bash
ls /path/to/sweetfold_env/lib/python3.10/site-packages/boltz
```

You should see the contents of the Boltz source-code package directly inside that folder.

Verify that Python can import the SweetFold-modified Boltz package:

```bash
python -c "import boltz; print('SweetFold-modified Boltz import successful:', boltz.__file__)"
```

---

## 7. Run SweetFold Inference

After the environment is installed, the SweetFold-modified `boltz` folder has been copied into place, and the epoch 23 checkpoint has been placed in the weights folder, you can run SweetFold inference.

An example HPC shell script may be provided in the same folder as the inference files. Before running it, update the paths inside the script so they point to:

```text
/path/to/sweetfold_env
/path/to/sweetfold_env/weights/boltz1_glycan_epoch_23.ckpt
/path/to/your/input.yaml
/path/to/your/output_directory
```

A typical inference command will need to point to:

1. The SweetFold input YAML file.
2. The SweetFold checkpoint.
3. The desired output directory.
4. Any inference-specific flags required by the provided script.

If using an HPC shell script, run something like:

```bash
sbatch run_inference.sh
```

If running locally or interactively, activate the environment first:

```bash
conda activate sweetfold_env
```

Then run the inference command provided by your shell script or notebook.

---

## 8. Inference Setup Summary

The overall inference setup flow is:

```bash
# Create and activate environment
conda create -n sweetfold_env python=3.10 -y
conda activate sweetfold_env
python -m pip install --upgrade pip

# Install Boltz v1.0.0 specifically
pip install "boltz[cuda]==1.0.0"

# Create weights folder
mkdir -p /path/to/sweetfold_env/weights

# Manually download the epoch 23 checkpoint from:
# https://huggingface.co/Keshav-Sundar-4/SweetFold/tree/main
# Then move it into:
# /path/to/sweetfold_env/weights/

# Clone SweetFold repository
cd /path/to/project
git clone https://github.com/Keshav-Sundar-4/SweetFold.git sweetfold_repo

# Replace installed Boltz source code with SweetFold-modified Boltz
python -c "import boltz, pathlib; print(pathlib.Path(boltz.__file__).parent)"
mv /path/to/sweetfold_env/lib/python3.10/site-packages/boltz /path/to/sweetfold_env/lib/python3.10/site-packages/boltz_original
cp -r /path/to/project/sweetfold_repo/src/boltz /path/to/sweetfold_env/lib/python3.10/site-packages/boltz

# Verify import
python -c "import boltz; print('SweetFold-modified Boltz import successful:', boltz.__file__)"

# Run inference using the provided shell script or command
sbatch run_inference.sh
```

---

# Part 2: SweetFold Training Setup Instructions

This guide explains how to set up and run SweetFold training using four files:

```text
full.yaml
train.py
train.sh
update_checkpoint.py
```

All file paths below are written as general placeholders. Before running training, replace every `/path/to/...` path with the correct absolute path on your system.

---

## 1. Expected File Layout

Place the following four files in the same working directory:

```text
/path/to/sweetfold_training/
├── full.yaml
├── train.py
├── train.sh
└── update_checkpoint.py
```

For example, if your working directory is:

```text
/path/to/sweetfold_training/
```

then all commands should be run from inside that folder unless otherwise specified.

---

## 2. Important Training Paths

The instructions use the following general paths.

### SweetFold environment

```text
/path/to/sweetfold_env
```

This is the Python or Conda environment where SweetFold/Boltz is installed.

### SweetFold weights directory

```text
/path/to/sweetfold_env/weights
```

This is where the required checkpoint and metadata files should be placed.

You may need to create this folder yourself.

### Installed Boltz source-code directory

```text
/path/to/sweetfold_env/lib/python3.10/site-packages/boltz
```

This is the installed `boltz` package folder inside the Python environment. This folder must be replaced with the SweetFold-modified `boltz` folder from:

```text
SweetFold/src/boltz
```

### SweetFold GitHub repository

```text
https://github.com/Keshav-Sundar-4/SweetFold
```

The SweetFold-modified Boltz folder is specifically located at:

```text
https://github.com/Keshav-Sundar-4/SweetFold/tree/main/src/boltz
```

### Training dataset directory

```text
/path/to/glycan_dataset
```

This is the training dataset directory.

### MSA directory

```text
/path/to/glycan_dataset/msa
```

This is the MSA directory corresponding to the training dataset.

### Training output directory

```text
/path/to/training_outputs
```

This is where training outputs and checkpoints will be written.

---

## 3. Create the SweetFold Training Environment

Create and activate a Python environment.

Example using Conda:

```bash
conda create -n sweetfold_env python=3.10 -y
conda activate sweetfold_env
python -m pip install --upgrade pip
```

Install Boltz v1.0.0 specifically with CUDA support:

```bash
pip install "boltz[cuda]==1.0.0"
```

Install helper packages if needed:

```bash
python -m pip install tqdm httpx scipy numpy pandas biopython
```

If RDKit is missing, install it with Conda:

```bash
conda install -c conda-forge rdkit -y
```

---

## 4. Download and Install the SweetFold-Modified `boltz` Folder

The SweetFold-modified Boltz source code lives inside the SweetFold GitHub repository at:

```text
https://github.com/Keshav-Sundar-4/SweetFold/tree/main/src/boltz
```

The folder path inside GitHub is:

```text
src/boltz
```

However, the folder copied into the Python environment must be named exactly:

```text
boltz
```

The final installed package path must be:

```text
/path/to/sweetfold_env/lib/python3.10/site-packages/boltz
```

### Clone the full SweetFold repository

```bash
cd /path/to/project
git clone https://github.com/Keshav-Sundar-4/SweetFold.git sweetfold_repo
```

After cloning, the SweetFold-modified Boltz folder will be located at:

```text
/path/to/project/sweetfold_repo/src/boltz
```

### Replace the installed Boltz folder

Activate the environment:

```bash
conda activate sweetfold_env
```

Find the installed Boltz source-code folder:

```bash
python -c "import boltz, pathlib; print(pathlib.Path(boltz.__file__).parent)"
```

Back up the original Boltz folder:

```bash
mv /path/to/sweetfold_env/lib/python3.10/site-packages/boltz /path/to/sweetfold_env/lib/python3.10/site-packages/boltz_original
```

Copy the SweetFold-modified Boltz folder into the environment:

```bash
cp -r /path/to/project/sweetfold_repo/src/boltz /path/to/sweetfold_env/lib/python3.10/site-packages/boltz
```

Verify the import:

```bash
python -c "import boltz; print('SweetFold-modified Boltz import successful:', boltz.__file__)"
```

---

## 5. Create the Weights Folder

Create the weights folder inside the SweetFold environment:

```bash
mkdir -p /path/to/sweetfold_env/weights
```

The final weights folder should be:

```text
/path/to/sweetfold_env/weights
```

This folder is required because the training configuration points to checkpoint and metadata files inside it.

---

## 6. Required Files for Training

The following files are required in the weights directory:

```text
/path/to/sweetfold_env/weights/boltz1_conf_converted.ckpt
/path/to/sweetfold_env/weights/symmetry.pkl
/path/to/sweetfold_env/weights/ccd.pkl
```

The `full.yaml` file directly points to:

```text
/path/to/sweetfold_env/weights/boltz1_conf_converted.ckpt
/path/to/sweetfold_env/weights/symmetry.pkl
```

The `ccd.pkl` file is also required by the Boltz/SweetFold codebase, even if it is not explicitly listed in `full.yaml`.

---

## 7. Download `boltz1_conf_converted.ckpt`

Download `boltz1_conf_converted.ckpt` manually from Hugging Face.

Open this link in a browser:

```text
https://huggingface.co/Keshav-Sundar-4/SweetFold/blob/main/boltz1_conf_converted.ckpt
```

On the Hugging Face page, click the download button for `boltz1_conf_converted.ckpt`.

After downloading it, move the file into:

```text
/path/to/sweetfold_env/weights/boltz1_conf_converted.ckpt
```

Do not rename the file.

---

## 8. Place `symmetry.pkl` and `ccd.pkl`

Place the following two files in the same weights directory:

```text
/path/to/sweetfold_env/weights/symmetry.pkl
/path/to/sweetfold_env/weights/ccd.pkl
```

The final weights directory should contain:

```text
/path/to/sweetfold_env/weights/
├── boltz1_conf_converted.ckpt
├── symmetry.pkl
└── ccd.pkl
```

If the weights folder does not exist yet, create it first:

```bash
mkdir -p /path/to/sweetfold_env/weights
```

---

## 9. Configure `full.yaml`

In `full.yaml`, update the output path:

```yaml
output: /path/to/training_outputs
```

This directory controls where training outputs and checkpoints are saved.

Update the pretrained checkpoint path:

```yaml
pretrained: /path/to/sweetfold_env/weights/boltz1_conf_converted.ckpt
```

Update the dataset paths:

```yaml
target_dir: /path/to/glycan_dataset
msa_dir: /path/to/glycan_dataset/msa
```

There are two `target_dir` fields in `full.yaml`. Both should point to the same glycan dataset directory:

```yaml
target_dir: /path/to/glycan_dataset
```

Update the symmetry file path:

```yaml
symmetries: /path/to/sweetfold_env/weights/symmetry.pkl
```

The training checkpoints generated during training will be saved under:

```text
/path/to/training_outputs/checkpoints/
```

The most recent checkpoint will usually be:

```text
/path/to/training_outputs/checkpoints/last.ckpt
```

---

## 10. Configure `train.sh`

In `train.sh`, update the training directory:

```bash
TRAINING_DIR="/path/to/sweetfold_training"
```

This should be the folder containing:

```text
full.yaml
train.py
train.sh
update_checkpoint.py
```

Update the SweetFold environment path:

```bash
SWEETFOLD_ENV="/path/to/sweetfold_env"
```

The script will enter the training directory, activate the environment, and run:

```bash
python train.py full.yaml
```

---

## 11. Running Training

From the directory containing the four training files, submit the SLURM job:

```bash
cd /path/to/sweetfold_training
sbatch train.sh
```

Training outputs will be written to:

```text
/path/to/training_outputs
```

Checkpoints will be written to:

```text
/path/to/training_outputs/checkpoints
```

The latest checkpoint will usually be:

```text
/path/to/training_outputs/checkpoints/last.ckpt
```

---

## 12. Create Final Checkpoints After Training

After training finishes, update the paths at the top of `update_checkpoint.py`.

The base checkpoint should be:

```text
/path/to/sweetfold_env/weights/boltz1_conf_converted.ckpt
```

The trained checkpoint should usually be:

```text
/path/to/training_outputs/checkpoints/last.ckpt
```

The inference checkpoint output should be:

```text
/path/to/sweetfold_env/weights/boltz1_glycan.ckpt
```

The resume checkpoint output should be:

```text
/path/to/sweetfold_env/weights/boltz1_glycan_resume.ckpt
```

Then run:

```bash
python update_checkpoint.py
```

This creates two checkpoint files.

### Inference checkpoint

```text
boltz1_glycan.ckpt
```

Use this checkpoint for inference or future pretrained loading.

### Resume checkpoint

```text
boltz1_glycan_resume.ckpt
```

Use this checkpoint if you want to resume training.

---

## 13. Training Setup Summary

The overall training setup flow is:

```bash
# Create and activate environment
conda create -n sweetfold_env python=3.10 -y
conda activate sweetfold_env
python -m pip install --upgrade pip

# Install Boltz v1.0.0 specifically
pip install "boltz[cuda]==1.0.0"

# Install helper packages if needed
python -m pip install tqdm httpx scipy numpy pandas biopython
conda install -c conda-forge rdkit -y

# Clone SweetFold repository
cd /path/to/project
git clone https://github.com/Keshav-Sundar-4/SweetFold.git sweetfold_repo

# Replace installed Boltz source code with SweetFold-modified Boltz
python -c "import boltz, pathlib; print(pathlib.Path(boltz.__file__).parent)"
mv /path/to/sweetfold_env/lib/python3.10/site-packages/boltz /path/to/sweetfold_env/lib/python3.10/site-packages/boltz_original
cp -r /path/to/project/sweetfold_repo/src/boltz /path/to/sweetfold_env/lib/python3.10/site-packages/boltz
python -c "import boltz; print('SweetFold-modified Boltz import successful:', boltz.__file__)"

# Create weights folder
mkdir -p /path/to/sweetfold_env/weights

# Manually download boltz1_conf_converted.ckpt from:
# https://huggingface.co/Keshav-Sundar-4/SweetFold/blob/main/boltz1_conf_converted.ckpt
# Then place it at:
# /path/to/sweetfold_env/weights/boltz1_conf_converted.ckpt

# Place symmetry.pkl and ccd.pkl in:
# /path/to/sweetfold_env/weights/

# Run training
cd /path/to/sweetfold_training
sbatch train.sh

# After training, create final checkpoints
python update_checkpoint.py
```

---

## 14. Summary of Training Files

| File | Purpose |
|---|---|
| `full.yaml` | Main training configuration. Controls dataset paths, output directory, checkpoint paths, model settings, and training hyperparameters. |
| `train.py` | Python training entry point. Loads `full.yaml`, initializes the model and data module, loads pretrained weights, and starts PyTorch Lightning training. |
| `train.sh` | SLURM job script. Requests GPUs, activates the environment, enters the working directory, and runs `python train.py full.yaml`. |
| `update_checkpoint.py` | Post-training checkpoint utility. Merges the trained checkpoint with the base checkpoint to create inference and resume checkpoints. |
