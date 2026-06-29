# SweetFold Inference Setup Instructions

SweetFold inference requires:

1. Boltz v1.0.0 installed.
2. The installed Boltz source-code folder replaced with SweetFold’s `boltz` folder.
3. A `weights` folder containing the SweetFold epoch 23 checkpoint.
4. An input YAML file.
5. An inference command or shell script.

---

## 1. Install Boltz v1.0.0

Create a fresh Python or Conda environment for SweetFold inference.

Inside that environment, install Boltz v1.0.0 exactly:

    pip install "boltz[cuda]==1.0.0"

Use this exact version.

---

## 2. Download the SweetFold `boltz` Folder

Download the SweetFold `boltz` folder from GitHub:

    https://github.com/Keshav-Sundar-4/SweetFold/tree/main/src/boltz

This is the SweetFold-modified version of the Boltz source code.

---

## 3. Replace the Installed Boltz Folder

After installing Boltz v1.0.0, locate the installed `boltz` source-code folder inside your environment.

Replace that installed `boltz` folder with the SweetFold `boltz` folder downloaded from GitHub.

The folder name should remain:

    boltz

This replacement is required because SweetFold inference depends on the SweetFold-modified Boltz source code.

---

## 4. Create a Weights Folder

Create a folder named:

    weights

inside the SweetFold environment.

This folder will store the SweetFold inference checkpoint.

---

## 5. Download the SweetFold Epoch 23 Checkpoint

Download the SweetFold epoch 23 checkpoint from Hugging Face:

    https://huggingface.co/Keshav-Sundar-4/SweetFold/tree/main

Place the checkpoint inside the `weights` folder.

For example, the environment should contain:

    sweetfold_env/
    └── weights/
        └── boltz1_glycan_epoch_23.ckpt

Use the exact checkpoint filename expected by your inference script.

---

## 6. Run SweetFold Inference

Run SweetFold using your inference command or the provided HPC shell script.

Before running, make sure the script points to:

1. The SweetFold environment.
2. The SweetFold checkpoint inside the `weights` folder.
3. The input YAML file.
4. The desired output directory.

An example HPC shell script may be provided alongside the inference files. Update its paths for your system, then run it using your normal HPC workflow.
