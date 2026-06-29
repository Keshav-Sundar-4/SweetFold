# SweetFold Training Setup Instructions

SweetFold training uses four files:

    full.yaml
    train.py
    train.sh
    update_checkpoint.py

---

## 1. Install Boltz v1.0.0

Create a fresh Python or Conda environment.

Install Boltz v1.0.0:

    pip install "boltz[cuda]==1.0.0"

Install any additional dependencies required by the scripts, including RDKit, NumPy, SciPy, pandas, tqdm, Biopython, and HTTPX.

---

## 2. Download the SweetFold `boltz` Folder

Download the SweetFold `boltz` folder from GitHub:

    https://github.com/Keshav-Sundar-4/SweetFold/tree/main/src/boltz

---

## 3. Replace the Installed Boltz Folder

Locate the installed `boltz` source-code folder inside your environment.

Replace that installed `boltz` folder with the SweetFold `boltz` folder downloaded from GitHub.

The folder should still be named:

    boltz

---

## 4. Create a `weights` Folder

Create a folder named:

    weights

inside the SweetFold environment.

The final `weights` folder should contain:

    weights/
    ├── boltz1_conf_converted.ckpt
    ├── symmetry.pkl
    └── ccd.pkl

---

## 5. Download `boltz1_conf_converted.ckpt`

Download `boltz1_conf_converted.ckpt`:

    https://huggingface.co/Keshav-Sundar-4/SweetFold/blob/main/boltz1_conf_converted.ckpt

Place it inside the `weights` folder.

Do not rename it.

---

## 6. Download `symmetry.pkl`

Download `symmetry.pkl`:

    https://huggingface.co/Keshav-Sundar-4/SweetFold/blob/main/symmetry.pkl

Place it inside the `weights` folder.

Do not rename it.

---

## 7. Download `ccd.pkl`

Download `ccd.pkl`:

    https://huggingface.co/Keshav-Sundar-4/SweetFold/blob/main/ccd.pkl

Place it inside the `weights` folder.

Do not rename it.

---

## 8. Prepare the Training Files

Place these four files in the same working directory:

    full.yaml
    train.py
    train.sh
    update_checkpoint.py

This folder is the SweetFold training directory.

---

## 9. Prepare the Training Dataset

Prepare the processed glycan dataset.

The dataset should contain:

    glycan_dataset/
    ├── structures/
    ├── records/
    ├── manifest.json
    ├── validation_ids.txt
    └── msa/

---

## 10. Choose a Training Output Directory

Choose a directory for training outputs and checkpoints.

Training checkpoints will be written under:

    training_outputs/checkpoints/

The latest checkpoint will usually be:

    training_outputs/checkpoints/last.ckpt

---

## 11. Configure `full.yaml`

Open `full.yaml`.

Set the training output directory:

    output: /path/to/training_outputs

Set the pretrained checkpoint path:

    pretrained: /path/to/sweetfold_env/weights/boltz1_conf_converted.ckpt

Set the dataset paths:

    target_dir: /path/to/glycan_dataset
    msa_dir: /path/to/glycan_dataset/msa

There are two `target_dir` fields in `full.yaml`.

Set both to:

    target_dir: /path/to/glycan_dataset

Set the symmetry file path:

    symmetries: /path/to/sweetfold_env/weights/symmetry.pkl

---

## 12. Configure `train.sh`

Open `train.sh`.

Set the training directory to the folder containing:

    full.yaml
    train.py
    train.sh
    update_checkpoint.py

Set the SweetFold environment path to the environment where Boltz v1.0.0 is installed and where the SweetFold `boltz` folder replacement was performed.

The script should run:

    python train.py full.yaml

---

## 13. Run Training

Run or submit `train.sh`.

For SLURM-based HPC systems, submit the shell script as a job.

Training outputs will be written to the output directory specified in `full.yaml`.

The latest checkpoint will usually be saved as:

    training_outputs/checkpoints/last.ckpt

---

## 14. Create Final Checkpoints After Training

After training finishes, open `update_checkpoint.py`.

Set the base checkpoint path to:

    /path/to/sweetfold_env/weights/boltz1_conf_converted.ckpt

Set the trained checkpoint path to:

    /path/to/training_outputs/checkpoints/last.ckpt

Set the final inference checkpoint output path to:

    /path/to/sweetfold_env/weights/boltz1_glycan.ckpt

Set the final resume checkpoint output path to:

    /path/to/sweetfold_env/weights/boltz1_glycan_resume.ckpt

Run:

    python update_checkpoint.py

This creates:

    boltz1_glycan.ckpt
    boltz1_glycan_resume.ckpt

---

## 15. Summary of Training Files

| File | Purpose |
|---|---|
| `full.yaml` | Main training configuration. |
| `train.py` | Python training entry point. |
| `train.sh` | Shell script for launching training. |
| `update_checkpoint.py` | Creates final inference and resume checkpoints. |
