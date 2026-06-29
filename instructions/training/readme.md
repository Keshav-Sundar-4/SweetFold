# SweetFold Training Setup Instructions

SweetFold training uses four files:

```
full.yaml
train.py
train.sh
update_checkpoint.py
```

Training requires:

1. Boltz v1.0.0 installed.
2. The installed Boltz source-code folder replaced with SweetFold’s `boltz` folder.
3. A `weights` folder containing the required checkpoint and metadata files.
4. A processed glycan training dataset.
5. A configured output directory for training checkpoints.
6. The four training files listed above.

---

## 1. Install Boltz v1.0.0

Create a fresh Python or Conda environment for SweetFold training.

Inside that environment, install Boltz v1.0.0 exactly:

```
pip install "boltz[cuda]==1.0.0"
```

Use this exact version.

Install any additional dependencies required by your system or scripts, such as RDKit, NumPy, SciPy, pandas, tqdm, Biopython, and HTTPX.

---

## 2. Download the SweetFold `boltz` Folder

Download the SweetFold `boltz` folder from GitHub:

```
https://github.com/Keshav-Sundar-4/SweetFold/tree/main/src/boltz
```

This is the SweetFold-modified version of the Boltz source code.

---

## 3. Replace the Installed Boltz Folder

After installing Boltz v1.0.0, locate the installed `boltz` source-code folder inside your environment.

Replace that installed `boltz` folder with the SweetFold `boltz` folder downloaded from GitHub.

The folder name should remain:

```
boltz
```

This replacement is required because SweetFold training depends on the SweetFold-modified Boltz source code.

---

## 4. Create a Weights Folder

Create a folder named:

```
weights
```

inside the SweetFold environment.

This folder should contain the checkpoint and metadata files required for training.

The final `weights` folder should contain:

```
weights/
├── boltz1_conf_converted.ckpt
├── symmetry.pkl
└── ccd.pkl
```

---

## 5. Download `boltz1_conf_converted.ckpt`

Download `boltz1_conf_converted.ckpt` from Hugging Face:

```
https://huggingface.co/Keshav-Sundar-4/SweetFold/blob/main/boltz1_conf_converted.ckpt
```

Place it inside the `weights` folder.

Do not rename this file unless you also update every config file or script that points to it.

---

## 6. Add `symmetry.pkl` and `ccd.pkl`

Place the following files in the same `weights` folder:

```
symmetry.pkl
ccd.pkl
```

The training configuration directly points to:

```
boltz1_conf_converted.ckpt
symmetry.pkl
```

The SweetFold/Boltz code also requires:

```
ccd.pkl
```

even if it is not explicitly listed in `full.yaml`.

---

## 7. Prepare the Training Files

Place the following four files in the same working directory:

```
full.yaml
train.py
train.sh
update_checkpoint.py
```

This folder is the SweetFold training directory.

Run training from this directory unless your shell script is configured otherwise.

---

## 8. Prepare the Training Dataset

Before running training, prepare the processed glycan dataset.

The dataset should contain the structure files and metadata expected by SweetFold/Boltz training.

A typical processed dataset should look like:

```
glycan_dataset/
├── structures/
├── records/
├── manifest.json
├── validation_ids.txt
└── msa/
```

The `msa` folder should correspond to the same dataset used for training.

---

## 9. Choose a Training Output Directory

Choose a directory where training outputs and checkpoints will be saved.

This directory is controlled by the `output` field in `full.yaml`.

During training, checkpoints will be written under:

```
training_outputs/checkpoints/
```

The latest checkpoint will usually be:

```
training_outputs/checkpoints/last.ckpt
```

---

## 10. Configure `full.yaml`

Open `full.yaml` and update the training output location:

```
output: /path/to/training_outputs
```

Update the pretrained checkpoint path so it points to the converted Boltz checkpoint inside the `weights` folder:

```
pretrained: /path/to/sweetfold_env/weights/boltz1_conf_converted.ckpt
```

Update the dataset paths:

```
target_dir: /path/to/glycan_dataset
msa_dir: /path/to/glycan_dataset/msa
```

There are two `target_dir` fields in `full.yaml`.

Both should point to the same processed glycan dataset directory:

```
target_dir: /path/to/glycan_dataset
```

Update the symmetry file path so it points to `symmetry.pkl` inside the `weights` folder:

```
symmetries: /path/to/sweetfold_env/weights/symmetry.pkl
```

---

## 11. Configure `train.sh`

Open `train.sh`.

Set the training directory to the folder containing:

```
full.yaml
train.py
train.sh
update_checkpoint.py
```

Set the SweetFold environment path to the environment where Boltz v1.0.0 was installed and where the SweetFold `boltz` folder replacement was performed.

The script should activate the SweetFold environment, enter the training directory, and run:

```
python train.py full.yaml
```

---

## 12. Run Training

Submit or run `train.sh` according to your system’s normal workflow.

For SLURM-based HPC systems, this usually means submitting the shell script as a job.

Training outputs will be written to the output directory specified in `full.yaml`.

The latest checkpoint will usually be saved as:

```
training_outputs/checkpoints/last.ckpt
```

---

## 13. Create Final Checkpoints After Training

After training finishes, use `update_checkpoint.py` to create the final inference and resume checkpoints.

Before running it, update the paths at the top of `update_checkpoint.py`.

The base checkpoint should be:

```
boltz1_conf_converted.ckpt
```

The trained checkpoint should usually be:

```
last.ckpt
```

The final inference checkpoint should be named:

```
boltz1_glycan.ckpt
```

The final resume checkpoint should be named:

```
boltz1_glycan_resume.ckpt
```

The inference checkpoint is used for inference or future pretrained loading.

The resume checkpoint is used if you want to resume training.

---

## 14. Summary of Training Files

| File                   | Purpose                                                                                                                                                 |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `full.yaml`            | Main training configuration. Controls dataset paths, output directory, checkpoint paths, model settings, and training hyperparameters.                  |
| `train.py`             | Python training entry point. Loads `full.yaml`, initializes the model and data module, loads pretrained weights, and starts PyTorch Lightning training. |
| `train.sh`             | Shell script for launching training. Usually used for HPC or SLURM jobs.                                                                                |
| `update_checkpoint.py` | Post-training checkpoint utility. Creates final inference and resume checkpoints from the trained checkpoint.                                           |
