# SweetFold Inference Setup Instructions

SweetFold inference requires four things:

1. Install Boltz v1.0.0.
2. Download the SweetFold `boltz` folder.
3. Replace the installed Boltz folder with the SweetFold version.
4. Download the SweetFold checkpoint and run inference.

---

## 1. Create a Python Environment

Create and activate a fresh environment:

    conda create -n sweetfold_env python=3.10 -y
    conda activate sweetfold_env
    python -m pip install --upgrade pip

Install Boltz v1.0.0 exactly:

    pip install "boltz[cuda]==1.0.0"

---

## 2. Download the SweetFold `boltz` Folder

Go here:

    https://github.com/Keshav-Sundar-4/SweetFold/tree/main/src/boltz

Download the `boltz` folder.

Put it somewhere convenient, for example:

    /path/to/project/boltz

---

## 3. Replace the Installed Boltz Folder

Find the Boltz folder that pip installed:

    python -c "import boltz, pathlib; print(pathlib.Path(boltz.__file__).parent)"

The command will print the folder you need to replace.

Back it up:

    mv /printed/path/to/boltz /printed/path/to/boltz_original

Copy the SweetFold `boltz` folder into its place:

    cp -r /path/to/project/boltz /printed/path/to/boltz

Verify the replacement worked:

    python -c "import boltz; print(boltz.__file__)"

---

## 4. Create a Weights Folder

Find your active environment path:

    echo $CONDA_PREFIX

Create a weights folder inside it:

    mkdir -p "$CONDA_PREFIX/weights"

---

## 5. Download the SweetFold Checkpoint

Open:

    https://huggingface.co/Keshav-Sundar-4/SweetFold/tree/main

Download the epoch 23 checkpoint.

Move it into:

    "$CONDA_PREFIX/weights"

For example:

    mv /path/to/downloaded/checkpoint.ckpt "$CONDA_PREFIX/weights/"

---

## 6. Run Inference

Update your inference script so the checkpoint path points to the file you just placed in:

    "$CONDA_PREFIX/weights"

Then run your script, for example:

    sbatch run_inference.sh
