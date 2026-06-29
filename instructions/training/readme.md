# SweetFold Training Setup Instructions

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

## 2. Important Paths

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

### Installed Boltz source-code directory

```text
/path/to/sweetfold_env/lib/python3.10/site-packages/boltz
```

This is the installed `boltz` package folder inside the Python environment. This folder must be replaced with the `boltz/` folder from the SweetFold GitHub repository.

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

## 3. Required Files for Training

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

## 4. Download `boltz1_conf_converted.ckpt`

Download `boltz1_conf_converted.ckpt` manually from Hugging Face using the website.

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

## 5. Place `symmetry.pkl` and `ccd.pkl`

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

---

## 6. Replace the Installed `boltz` Folder

After installing the environment, locate the installed `boltz` package folder:

```text
/path/to/sweetfold_env/lib/python3.10/site-packages/boltz
```

Replace this folder with the `boltz/` folder from the SweetFold GitHub repository.

This step is required because the SweetFold version of the `boltz` source code contains the glycan-specific training changes. If the original Boltz package is left in place, the training code may not match the SweetFold checkpoint or dataset format.

---

## 7. Configure `full.yaml`

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

## 8. Configure `train.sh`

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

## 9. Running Training

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

## 10. Create Final Checkpoints After Training

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

## 11. Summary of Files

| File | Purpose |
|---|---|
| `full.yaml` | Main training configuration. Controls dataset paths, output directory, checkpoint paths, model settings, and training hyperparameters. |
| `train.py` | Python training entry point. Loads `full.yaml`, initializes the model and data module, loads pretrained weights, and starts PyTorch Lightning training. |
| `train.sh` | SLURM job script. Requests GPUs, activates the environment, enters the working directory, and runs `python train.py full.yaml`. |
| `update_checkpoint.py` | Post-training checkpoint utility. Merges the trained checkpoint with the base checkpoint to create inference and resume checkpoints. |
