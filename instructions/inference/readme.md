# SweetFold Inference and Training Setup Instructions

This guide explains how to set up SweetFold for inference and training.

SweetFold uses Boltz v1.0.0 as the base package. After installing Boltz, you replace the installed Boltz source-code folder with the SweetFold-modified `boltz` folder from GitHub.

SweetFold `boltz` folder:

https://github.com/Keshav-Sundar-4/SweetFold/tree/main/src/boltz

Download that folder. The downloaded folder should be named:

    boltz

---

# Part 1: SweetFold Inference Setup Instructions

Inference is simple. You need to:

1. Create a Python environment.
2. Install Boltz v1.0.0.
3. Download the SweetFold `boltz` folder.
4. Replace the installed Boltz folder with the SweetFold folder.
5. Create a weights folder.
6. Download the SweetFold epoch 23 checkpoint.
7. Run inference using your shell script or command.

---

## 1. Create a SweetFold Environment

Create a fresh Conda environment:

    conda create -n sweetfold_env python=3.10 -y
    conda activate sweetfold_env
    python -m pip install --upgrade pip

Install Boltz v1.0.0 with CUDA support:

    pip install "boltz[cuda]==1.0.0"

Use this exact version.

You can find the full path to your active environment by running:

    echo $CONDA_PREFIX

In the rest of these instructions, that environment path is written as:

    /path/to/sweetfold_env

Replace `/path/to/sweetfold_env` with the path printed by `echo $CONDA_PREFIX`.

---

## 2. Download the SweetFold `boltz` Folder

Go to the SweetFold `boltz` folder on GitHub:

    https://github.com/Keshav-Sundar-4/SweetFold/tree/main/src/boltz

Download that folder.

After downloading, place the downloaded `boltz` folder somewhere convenient, for example:

    /path/to/project/boltz

The folder should be named:

    boltz

You will use this downloaded `boltz` folder to replace the original Boltz folder that was installed by:

    pip install "boltz[cuda]==1.0.0"

---

## 3. Replace the Installed Boltz Folder

Activate the SweetFold environment:

    conda activate sweetfold_env

Find the installed Boltz folder:

    python -c "import boltz, pathlib; print(pathlib.Path(boltz.__file__).parent)"

This will print something like:

    /path/to/sweetfold_env/lib/python3.10/site-packages/boltz

Back up the original installed Boltz folder:

    mv /path/to/sweetfold_env/lib/python3.10/site-packages/boltz /path/to/sweetfold_env/lib/python3.10/site-packages/boltz_original

Copy the downloaded SweetFold `boltz` folder into the environment:

    cp -r /path/to/project/boltz /path/to/sweetfold_env/lib/python3.10/site-packages/boltz

Verify that Python can import the replaced package:

    python -c "import boltz; print('SweetFold-modified Boltz import successful:', boltz.__file__)"

---

## 4. Create the Weights Folder

Create a weights folder inside the SweetFold environment:

    mkdir -p /path/to/sweetfold_env/weights

This folder will store the checkpoint used for inference.

---

## 5. Download the SweetFold Epoch 23 Checkpoint

Open the SweetFold Hugging Face repository:

    https://huggingface.co/Keshav-Sundar-4/SweetFold/tree/main

Download the SweetFold epoch 23 checkpoint.

Place the checkpoint inside:

    /path/to/sweetfold_env/weights

For example, the final checkpoint path may look like:

    /path/to/sweetfold_env/weights/boltz1_glycan_epoch_23.ckpt

Use the exact checkpoint filename expected by your inference script. If your script points to a different checkpoint filename, either update the script or rename the checkpoint consistently.

---

## 6. Run SweetFold Inference

Before running inference, update your shell script or inference command so it points to:

    /path/to/sweetfold_env
    /path/to/sweetfold_env/weights/boltz1_glycan_epoch_23.ckpt
    /path/to/your/input.yaml
    /path/to/your/output_directory

If using an HPC shell script, submit it with:

    sbatch run_inference.sh

If running interactively, activate the environment first:

    conda activate sweetfold_env

Then run the inference command used by your script.

---

## 7. Inference Setup Summary

The inference setup flow is:

    conda create -n sweetfold_env python=3.10 -y
    conda activate sweetfold_env
    python -m pip install --upgrade pip
    pip install "boltz[cuda]==1.0.0"

    echo $CONDA_PREFIX

    # Download the SweetFold boltz folder from:
    # https://github.com/Keshav-Sundar-4/SweetFold/tree/main/src/boltz
    # Put it somewhere like:
    # /path/to/project/boltz

    python -c "import boltz, pathlib; print(pathlib.Path(boltz.__file__).parent)"

    mv /path/to/sweetfold_env/lib/python3.10/site-packages/boltz /path/to/sweetfold_env/lib/python3.10/site-packages/boltz_original
    cp -r /path/to/project/boltz /path/to/sweetfold_env/lib/python3.10/site-packages/boltz

    python -c "import boltz; print('SweetFold-modified Boltz import successful:', boltz.__file__)"

    mkdir -p /path/to/sweetfold_env/weights

    # Download the epoch 23 checkpoint from:
    # https://huggingface.co/Keshav-Sundar-4/SweetFold/tree/main
    # Place it inside:
    # /path/to/sweetfold_env/weights

    sbatch run_inference.sh

---

# Part 2: SweetFold Training Setup Instructions

Training uses four files:

    full.yaml
    train.py
    train.sh
    update_checkpoint.py

Training requires:

1. A SweetFold environment.
2. Boltz v1.0.0.
3. The SweetFold `boltz` folder.
4. A weights folder.
5. `boltz1_conf_converted.ckpt`.
6. `symmetry.pkl`.
7. `ccd.pkl`.
8. A processed glycan training dataset.

---

## 1. Create the Training Folder

Create a folder for the training files:

    mkdir -p /path/to/sweetfold_training

Place these four files inside that folder:

    /path/to/sweetfold_training/
    ├── full.yaml
    ├── train.py
    ├── train.sh
    └── update_checkpoint.py

From this point onward, `/path/to/sweetfold_training` means the folder containing those four files.

---

## 2. Create the SweetFold Environment

Create a fresh Conda environment:

    conda create -n sweetfold_env python=3.10 -y
    conda activate sweetfold_env
    python -m pip install --upgrade pip

Install Boltz v1.0.0 with CUDA support:

    pip install "boltz[cuda]==1.0.0"

Install helper packages if needed:

    python -m pip install tqdm httpx scipy numpy pandas biopython

If RDKit is missing, install it with Conda:

    conda install -c conda-forge rdkit -y

Find the full path to the environment:

    echo $CONDA_PREFIX

In the rest of these instructions, that path is written as:

    /path/to/sweetfold_env

Replace `/path/to/sweetfold_env` with your actual environment path.

---

## 3. Download the SweetFold `boltz` Folder

Go to the SweetFold `boltz` folder on GitHub:

    https://github.com/Keshav-Sundar-4/SweetFold/tree/main/src/boltz

Download that folder.

Place the downloaded folder somewhere convenient, for example:

    /path/to/project/boltz

The folder should be named:

    boltz

---

## 4. Replace the Installed Boltz Folder

Activate the SweetFold environment:

    conda activate sweetfold_env

Find the installed Boltz folder:

    python -c "import boltz, pathlib; print(pathlib.Path(boltz.__file__).parent)"

This will print something like:

    /path/to/sweetfold_env/lib/python3.10/site-packages/boltz

Back up the original installed Boltz folder:

    mv /path/to/sweetfold_env/lib/python3.10/site-packages/boltz /path/to/sweetfold_env/lib/python3.10/site-packages/boltz_original

Copy the downloaded SweetFold `boltz` folder into the environment:

    cp -r /path/to/project/boltz /path/to/sweetfold_env/lib/python3.10/site-packages/boltz

Verify that Python can import the replaced package:

    python -c "import boltz; print('SweetFold-modified Boltz import successful:', boltz.__file__)"

---

## 5. Create the Weights Folder

Create a weights folder inside the SweetFold environment:

    mkdir -p /path/to/sweetfold_env/weights

This folder must contain the checkpoint and metadata files required for training.

The final folder should be:

    /path/to/sweetfold_env/weights

---

## 6. Download `boltz1_conf_converted.ckpt`

Download `boltz1_conf_converted.ckpt` from Hugging Face:

    https://huggingface.co/Keshav-Sundar-4/SweetFold/blob/main/boltz1_conf_converted.ckpt

Move it into the weights folder:

    /path/to/sweetfold_env/weights/boltz1_conf_converted.ckpt

Do not rename the file unless you also update every config file or script that points to it.

---

## 7. Add `symmetry.pkl` and `ccd.pkl`

Place these two files in the same weights folder:

    /path/to/sweetfold_env/weights/symmetry.pkl
    /path/to/sweetfold_env/weights/ccd.pkl

The final weights folder should contain:

    /path/to/sweetfold_env/weights/
    ├── boltz1_conf_converted.ckpt
    ├── symmetry.pkl
    └── ccd.pkl

The `full.yaml` file points directly to:

    boltz1_conf_converted.ckpt
    symmetry.pkl

The SweetFold/Boltz code also needs:

    ccd.pkl

even if it is not always explicitly listed in `full.yaml`.

---

## 8. Prepare the Training Dataset

Your processed glycan dataset should already exist before training.

In these instructions, the dataset folder is written as:

    /path/to/glycan_dataset

The MSA folder should be:

    /path/to/glycan_dataset/msa

These are placeholders. Replace them with the real dataset paths on your system.

---

## 9. Create a Training Output Folder

Choose or create a folder where training outputs should be written:

    mkdir -p /path/to/training_outputs

This folder will store logs and checkpoints.

The latest checkpoint will usually be written to:

    /path/to/training_outputs/checkpoints/last.ckpt

---

## 10. Configure `full.yaml`

Open `full.yaml`.

Update the output path:

    output: /path/to/training_outputs

Update the pretrained checkpoint path:

    pretrained: /path/to/sweetfold_env/weights/boltz1_conf_converted.ckpt

Update the dataset paths:

    target_dir: /path/to/glycan_dataset
    msa_dir: /path/to/glycan_dataset/msa

There are two `target_dir` fields in `full.yaml`. Both should point to the same glycan dataset directory:

    target_dir: /path/to/glycan_dataset

Update the symmetry file path:

    symmetries: /path/to/sweetfold_env/weights/symmetry.pkl

---

## 11. Configure `train.sh`

Open `train.sh`.

Set the training directory to the folder containing the four training files:

    TRAINING_DIR="/path/to/sweetfold_training"

Set the SweetFold environment path:

    SWEETFOLD_ENV="/path/to/sweetfold_env"

The script should enter the training directory, activate the environment, and run:

    python train.py full.yaml

---

## 12. Run Training

Submit the SLURM job from the training folder:

    cd /path/to/sweetfold_training
    sbatch train.sh

Training outputs will be written to:

    /path/to/training_outputs

Checkpoints will be written to:

    /path/to/training_outputs/checkpoints

The latest checkpoint will usually be:

    /path/to/training_outputs/checkpoints/last.ckpt

---

## 13. Create Final Checkpoints After Training

After training finishes, update the paths at the top of `update_checkpoint.py`.

The base checkpoint should be:

    /path/to/sweetfold_env/weights/boltz1_conf_converted.ckpt

The trained checkpoint should usually be:

    /path/to/training_outputs/checkpoints/last.ckpt

The inference checkpoint output should be:

    /path/to/sweetfold_env/weights/boltz1_glycan.ckpt

The resume checkpoint output should be:

    /path/to/sweetfold_env/weights/boltz1_glycan_resume.ckpt

Run:

    cd /path/to/sweetfold_training
    python update_checkpoint.py

This creates two checkpoint files.

The inference checkpoint is:

    boltz1_glycan.ckpt

Use this checkpoint for inference or future pretrained loading.

The resume checkpoint is:

    boltz1_glycan_resume.ckpt

Use this checkpoint if you want to resume training.

---

## 14. Training Setup Summary

The training setup flow is:

    mkdir -p /path/to/sweetfold_training

    conda create -n sweetfold_env python=3.10 -y
    conda activate sweetfold_env
    python -m pip install --upgrade pip
    pip install "boltz[cuda]==1.0.0"

    python -m pip install tqdm httpx scipy numpy pandas biopython
    conda install -c conda-forge rdkit -y

    echo $CONDA_PREFIX

    # Download the SweetFold boltz folder from:
    # https://github.com/Keshav-Sundar-4/SweetFold/tree/main/src/boltz
    # Put it somewhere like:
    # /path/to/project/boltz

    python -c "import boltz, pathlib; print(pathlib.Path(boltz.__file__).parent)"

    mv /path/to/sweetfold_env/lib/python3.10/site-packages/boltz /path/to/sweetfold_env/lib/python3.10/site-packages/boltz_original
    cp -r /path/to/project/boltz /path/to/sweetfold_env/lib/python3.10/site-packages/boltz

    python -c "import boltz; print('SweetFold-modified Boltz import successful:', boltz.__file__)"

    mkdir -p /path/to/sweetfold_env/weights

    # Download boltz1_conf_converted.ckpt from:
    # https://huggingface.co/Keshav-Sundar-4/SweetFold/blob/main/boltz1_conf_converted.ckpt
    # Place it at:
    # /path/to/sweetfold_env/weights/boltz1_conf_converted.ckpt

    # Place symmetry.pkl and ccd.pkl in:
    # /path/to/sweetfold_env/weights/

    mkdir -p /path/to/training_outputs

    cd /path/to/sweetfold_training
    sbatch train.sh

    python update_checkpoint.py

---

## 15. Summary of Training Files

| File | Purpose |
|---|---|
| `full.yaml` | Main training configuration. Controls dataset paths, output directory, checkpoint paths, model settings, and training hyperparameters. |
| `train.py` | Python training entry point. Loads `full.yaml`, initializes the model and data module, loads pretrained weights, and starts PyTorch Lightning training. |
| `train.sh` | SLURM job script. Requests GPUs, activates the environment, enters the working directory, and runs `python train.py full.yaml`. |
| `update_checkpoint.py` | Post-training checkpoint utility. Merges the trained checkpoint with the base checkpoint to create inference and resume checkpoints. |
