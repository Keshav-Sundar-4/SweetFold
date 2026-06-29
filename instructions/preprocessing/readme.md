# SweetFold Glycan Dataset Processing Instructions

These instructions explain how to set up the SweetFold/Boltz environment and run the glycan dataset processing pipeline from raw PDB discovery to final validation-set creation.

The scripts are written as standalone Python scripts. Some can run on a normal local Python environment, while others require the SweetFold/Boltz environment because they import `boltz`, `rdkit`, and related dependencies.

---

## 1. Overview

The full pipeline is:

1. Find PDB IDs that contain glycans.
2. Download the matching PDB files.
3. Clean protein-glycan PDB files.
4. Separately extract or clean free-floating glycan structures.
5. Combine the cleaned folders into one dataset folder.
6. Convert cleaned `.pdb` files into Boltz/SweetFold `.npz` training structures.
7. Create a validation ID file from the processed `.npz` files.

---

## 2. Important Placeholder Paths

Replace these placeholder paths with real paths on your machine or cluster.

`/path/to/sweetfold_env`

This is the Python or Conda environment where Boltz/SweetFold is installed.

`/path/to/sweetfold_repo`

This is the local folder created after cloning the SweetFold GitHub repository.

`/path/to/project`

This is your working project folder where you want datasets, outputs, and scripts to live.

`/path/to/project/scripts`

This is where the processing scripts are stored.

`/path/to/project/data`

This is the parent folder for all generated datasets.

`/path/to/sweetfold_env/weights`

This is where `ccd.pkl` or other required metadata/checkpoint files should be placed if a script requires them.

`/path/to/sweetfold_env/lib/python3.10/site-packages/boltz`

This is the installed `boltz` source-code folder inside the Python environment.

Important: the exact `site-packages` path can vary by Python version and environment type. If you are unsure where `boltz` is installed, activate the environment and run:

    python -c "import boltz, pathlib; print(pathlib.Path(boltz.__file__).parent)"

---

## 3. Create and Install the SweetFold/Boltz Environment

You must first create the environment and install Boltz before replacing the installed `boltz/` folder with the SweetFold-modified version.

These instructions assume Conda and Python 3.10.

    conda create -n sweetfold_env python=3.10 -y
    conda activate sweetfold_env
    python -m pip install --upgrade pip

Install Boltz version `1.0.0` specifically:

    python -m pip install boltz==1.0.0

Install the additional helper packages used by the dataset scripts:

    python -m pip install tqdm httpx scipy numpy pandas biopython

RDKit may already be installed by Boltz. If RDKit is missing, install it with Conda:

    conda install -c conda-forge rdkit -y

---

## 4. Download the SweetFold Code Folder

Clone the SweetFold repository somewhere on your machine.

Replace the URL below with the actual SweetFold GitHub repository URL.

    cd /path/to/project
    git clone https://github.com/YOUR_USERNAME/YOUR_SWEETFOLD_REPO.git sweetfold_repo

After cloning, your local SweetFold folder should look something like this:

    /path/to/project/sweetfold_repo
    /path/to/project/sweetfold_repo/boltz

The key folder is:

    /path/to/project/sweetfold_repo/boltz

That folder is the SweetFold-modified `boltz/` source folder.

---

## 5. Replace the Installed Boltz Folder with the SweetFold Boltz Folder

First, activate the SweetFold environment:

    conda activate sweetfold_env

Find the installed Boltz package location:

    python -c "import boltz, pathlib; print(pathlib.Path(boltz.__file__).parent)"

This will print something like:

    /path/to/sweetfold_env/lib/python3.10/site-packages/boltz

Back up the original installed Boltz folder:

    mv /path/to/sweetfold_env/lib/python3.10/site-packages/boltz /path/to/sweetfold_env/lib/python3.10/site-packages/boltz_original

Copy the SweetFold-modified `boltz/` folder into the environment:

    cp -r /path/to/project/sweetfold_repo/boltz /path/to/sweetfold_env/lib/python3.10/site-packages/boltz

Verify that Python can import the replaced package:

    python -c "import boltz; print('Boltz/SweetFold import successful:', boltz.__file__)"

---

## 6. Suggested Project Folder Layout

A clean project layout is:

    /path/to/project/
    ├── scripts/
    │   ├── pdb_api_call.py
    │   ├── pdb_api.py
    │   ├── phase1_cleaner.py
    │   ├── free_glycan_data.py
    │   ├── lectinz_clean.py
    │   ├── preprocess_glycans.py
    │   └── validation_dataset_creation.py
    ├── data/
    │   ├── pdb_id_lists/
    │   ├── raw_pdb_downloads/
    │   ├── phase1_cleaned_pdbs/
    │   ├── free_glycan_raw_pdbs/
    │   ├── free_glycan_cleaned_pdbs/
    │   ├── lectinz_cleaned_pdbs/
    │   ├── combined_cleaned_pdbs/
    │   └── final_processed_dataset/
    └── sweetfold_repo/

You can use different names, but the pipeline is much easier to debug if every output folder has a clear name.

---

## 7. Scripts That Do Not Require the SweetFold Environment

The first two scripts do not require the SweetFold/Boltz environment.

They can be run in a normal Python environment as long as the required packages are installed.

Install basic local dependencies if needed:

    python -m pip install tqdm httpx

These scripts are:

    pdb_api_call.py
    pdb_api.py

---

## 8. Scripts That Require the SweetFold Environment

The later scripts should be run after activating the SweetFold environment:

    conda activate sweetfold_env

These scripts require the SweetFold/Boltz environment:

    phase1_cleaner.py
    free_glycan_data.py
    lectinz_clean.py
    preprocess_glycans.py
    validation_dataset_creation.py

`free_glycan_data.py` may work outside the environment if its dependencies are installed, but it is simplest to run it inside the SweetFold environment.

---

## 9. Step-by-Step Pipeline

### Step 1: Find PDB IDs Containing Glycans

Script:

    pdb_api_call.py

Purpose:

This script queries the RCSB PDB API and writes a text file containing PDB IDs that contain glycan CCD codes and satisfy the resolution cutoff.

Before running, edit the configuration near the top of `pdb_api_call.py`:

    OUT_DIR = Path("/path/to/project/data/pdb_id_lists")
    OUT_FILE = OUT_DIR / "pdb_ids_with_glycans.txt"

Run:

    cd /path/to/project/scripts
    python pdb_api_call.py

Expected output:

    /path/to/project/data/pdb_id_lists/pdb_ids_with_glycans.txt

---

### Step 2: Download the Matching PDB Files

Script:

    pdb_api.py

Purpose:

This script reads the PDB ID text file from Step 1 and downloads the matching `.pdb` files.

Before running, edit the configuration near the top of `pdb_api.py`:

    PDB_ID_FILE = Path("/path/to/project/data/pdb_id_lists/pdb_ids_with_glycans.txt")
    OUTPUT_DIR = Path("/path/to/project/data/raw_pdb_downloads")

Run:

    cd /path/to/project/scripts
    python pdb_api.py

Expected output:

    /path/to/project/data/raw_pdb_downloads/

This folder should contain downloaded `.pdb` files.

---

### Step 3a: Clean Protein-Glycan PDB Files

Script:

    phase1_cleaner.py

Purpose:

This script cleans raw protein-glycan PDB files. It removes irrelevant atoms, keeps valid protein and glycan residues, resolves alternate conformations, checks for clashes, removes invalid glycan components, and writes cleaned PDB files.

Activate the SweetFold environment first:

    conda activate sweetfold_env

Before running, edit the hardcoded configuration near the top of `phase1_cleaner.py`:

    INPUT_FOLDER = "/path/to/project/data/raw_pdb_downloads"
    OUTPUT_FOLDER = "/path/to/project/data/phase1_cleaned_pdbs"

Run:

    cd /path/to/project/scripts
    python phase1_cleaner.py

Expected output:

    /path/to/project/data/phase1_cleaned_pdbs/

---

### Step 3b: Generate Free-Floating Glycan Data

Script:

    free_glycan_data.py

Purpose:

This script extracts glycan-only structures from PDB files that contain glycans but no protein residues. It removes hydrogens, removes non-glycan atoms, splits disconnected glycan components, and writes each glycan component as its own PDB file.

This script uses command-line arguments, so you do not need to edit paths inside the script.

Run:

    conda activate sweetfold_env
    cd /path/to/project/scripts

    python free_glycan_data.py \
      --input_folder /path/to/project/data/raw_pdb_downloads \
      --output_folder /path/to/project/data/free_glycan_cleaned_pdbs

Expected output:

    /path/to/project/data/free_glycan_cleaned_pdbs/

Note:

If you only use the PDB-derived dataset, you may get a relatively small number of free-floating glycans. If you also have an MD-derived glycan dataset, process that as well and add it to the combined dataset later.

---

### Step 3c: Clean MD-Derived or Nonstandard Glycan PDB Files

Script:

    lectinz_clean.py

Purpose:

This script is designed for MD-derived or otherwise nonstandard glycan PDB files. It removes hydrogens, normalizes atom names, merges fragmented monosaccharide residues, checks against the CCD dictionary, and writes cleaned PDB files.

Before running, edit the configuration near the top of `lectinz_clean.py`:

    TARGET_FOLDER = Path("/path/to/project/data/free_glycan_raw_pdbs")
    OUTPUT_FOLDER = Path("/path/to/project/data/lectinz_cleaned_pdbs")
    CCD_FILE = Path("/path/to/sweetfold_env/weights/ccd.pkl")

Run:

    conda activate sweetfold_env
    cd /path/to/project/scripts
    python lectinz_clean.py

Expected output:

    /path/to/project/data/lectinz_cleaned_pdbs/

If you do not have MD-derived or nonstandard glycan PDB files, you can skip this step.

---

### Step 3d: Combine the Cleaned PDB Folders

Purpose:

The preprocessing script expects one folder of cleaned `.pdb` files. Combine the cleaned protein-glycan files, free-floating glycan files, and optional MD-cleaned glycan files into one folder.

Create the combined folder:

    mkdir -p /path/to/project/data/combined_cleaned_pdbs

Copy protein-glycan cleaned files:

    cp /path/to/project/data/phase1_cleaned_pdbs/*.pdb /path/to/project/data/combined_cleaned_pdbs/

Copy free-floating glycan files:

    cp /path/to/project/data/free_glycan_cleaned_pdbs/*.pdb /path/to/project/data/combined_cleaned_pdbs/

Optional: copy MD/nonstandard cleaned glycan files:

    cp /path/to/project/data/lectinz_cleaned_pdbs/*.pdb /path/to/project/data/combined_cleaned_pdbs/

Expected combined folder:

    /path/to/project/data/combined_cleaned_pdbs/

---

### Step 4: Convert Cleaned PDB Files into `.npz` Dataset Files

Script:

    preprocess_glycans.py

Purpose:

This script converts cleaned `.pdb` files into Boltz/SweetFold-compatible `.npz` structure files and JSON records.

Important:

This step requires the SweetFold/Boltz environment and the replaced SweetFold `boltz/` folder.

Run:

    conda activate sweetfold_env
    cd /path/to/project/scripts

    python preprocess_glycans.py \
      --datadir /path/to/project/data/combined_cleaned_pdbs \
      --ccd-path /path/to/sweetfold_env/weights/ccd.pkl \
      --outdir /path/to/project/data/final_processed_dataset \
      --num-processes 8

Arguments:

`--datadir`

Folder containing the cleaned `.pdb` files.

`--ccd-path`

Path to the `ccd.pkl` file.

`--outdir`

Output folder for the processed dataset.

`--num-processes`

Number of CPU processes to use. Use a smaller number if your machine has limited RAM.

Expected output:

    /path/to/project/data/final_processed_dataset/structures/
    /path/to/project/data/final_processed_dataset/records/
    /path/to/project/data/final_processed_dataset/manifest.json

The `structures/` folder should contain `.npz` files.

---

### Step 5: Create the Validation ID File

Script:

    validation_dataset_creation.py

Purpose:

This script randomly selects a fraction of processed `.npz` files and writes their IDs to a validation text file.

Before running, edit the configuration near the top of `validation_dataset_creation.py`:

    NPZ_DIR = "/path/to/project/data/final_processed_dataset/structures"
    OUTPUT_FILE = "/path/to/project/data/final_processed_dataset/validation_ids.txt"
    VALIDATION_SET_FRACTION = 0.05

Run:

    conda activate sweetfold_env
    cd /path/to/project/scripts
    python validation_dataset_creation.py

Expected output:

    /path/to/project/data/final_processed_dataset/validation_ids.txt

---

## 10. Final Expected Dataset Folder

After the full pipeline, the final processed dataset should look like:

    /path/to/project/data/final_processed_dataset/
    ├── structures/
    │   ├── example_1.npz
    │   ├── example_2.npz
    │   └── ...
    ├── records/
    │   ├── example_1.json
    │   ├── example_2.json
    │   └── ...
    ├── manifest.json
    └── validation_ids.txt

---

## 11. Full Command Summary

The overall command flow is:

    # Create and activate the environment
    conda create -n sweetfold_env python=3.10 -y
    conda activate sweetfold_env
    python -m pip install --upgrade pip
    python -m pip install boltz==1.0.0
    python -m pip install tqdm httpx scipy numpy pandas biopython
    conda install -c conda-forge rdkit -y

    # Download SweetFold code
    cd /path/to/project
    git clone https://github.com/YOUR_USERNAME/YOUR_SWEETFOLD_REPO.git sweetfold_repo

    # Replace installed Boltz with SweetFold Boltz
    python -c "import boltz, pathlib; print(pathlib.Path(boltz.__file__).parent)"
    mv /path/to/sweetfold_env/lib/python3.10/site-packages/boltz /path/to/sweetfold_env/lib/python3.10/site-packages/boltz_original
    cp -r /path/to/project/sweetfold_repo/boltz /path/to/sweetfold_env/lib/python3.10/site-packages/boltz
    python -c "import boltz; print('Boltz/SweetFold import successful:', boltz.__file__)"

    # Step 1: Get PDB IDs
    cd /path/to/project/scripts
    python pdb_api_call.py

    # Step 2: Download PDB files
    python pdb_api.py

    # Step 3a: Clean protein-glycan PDB files
    python phase1_cleaner.py

    # Step 3b: Extract free-floating glycans
    python free_glycan_data.py \
      --input_folder /path/to/project/data/raw_pdb_downloads \
      --output_folder /path/to/project/data/free_glycan_cleaned_pdbs

    # Step 3c: Optional MD/nonstandard glycan cleanup
    python lectinz_clean.py

    # Step 3d: Combine cleaned folders
    mkdir -p /path/to/project/data/combined_cleaned_pdbs
    cp /path/to/project/data/phase1_cleaned_pdbs/*.pdb /path/to/project/data/combined_cleaned_pdbs/
    cp /path/to/project/data/free_glycan_cleaned_pdbs/*.pdb /path/to/project/data/combined_cleaned_pdbs/
    cp /path/to/project/data/lectinz_cleaned_pdbs/*.pdb /path/to/project/data/combined_cleaned_pdbs/

    # Step 4: Preprocess into NPZ dataset
    python preprocess_glycans.py \
      --datadir /path/to/project/data/combined_cleaned_pdbs \
      --ccd-path /path/to/sweetfold_env/weights/ccd.pkl \
      --outdir /path/to/project/data/final_processed_dataset \
      --num-processes 8

    # Step 5: Create validation IDs
    python validation_dataset_creation.py

---

## 12. File Descriptions

### `pdb_api_call.py`

Finds PDB entries that contain at least one allowed glycan CCD code and satisfy the resolution cutoff. It writes the matching PDB IDs to a text file.

### `pdb_api.py`

Reads a text file of PDB IDs and downloads the corresponding `.pdb` files from RCSB.

### `phase1_cleaner.py`

Cleans raw protein-glycan PDB files. It keeps protein and glycan residues, removes unrelated atoms, handles alternate conformations, checks for atomic clashes, removes invalid glycan components, and writes cleaned PDB files.

### `free_glycan_data.py`

Extracts free-floating glycan-only structures from PDB files. It removes hydrogens, removes non-glycan atoms, splits disconnected glycan components, and writes each glycan component as a separate PDB file.

### `lectinz_clean.py`

Cleans MD-derived or nonstandard glycan PDB files. It removes hydrogens, normalizes atom names, merges fragmented monosaccharide residues, checks residue completeness using `ccd.pkl`, and writes corrected PDB files.

### `preprocess_glycans.py`

Converts cleaned `.pdb` files into SweetFold/Boltz-compatible `.npz` files. It parses atoms, residues, chains, glycan connectivity, glycosylation sites, anomeric configuration, and writes both structure files and JSON records.

### `validation_dataset_creation.py`

Randomly selects a fraction of the processed `.npz` files and writes their IDs to a validation text file.
