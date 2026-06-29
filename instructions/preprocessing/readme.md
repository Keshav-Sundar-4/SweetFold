# SweetFold Glycan Dataset Preprocessing Instructions

This guide explains how to create the SweetFold glycan dataset from raw PDB structures.

The preprocessing pipeline uses these scripts:

    pdb_api_call.py
    pdb_api.py
    phase1_cleaner.py
    free_glycan_data.py
    lectinz_clean.py
    preprocess_glycans.py
    validation_dataset_creation.py

---

## 1. Install Boltz v1.0.0

Create a fresh Python or Conda environment.

Install Boltz v1.0.0:

    pip install "boltz[cuda]==1.0.0"

Install any additional dependencies required by the scripts, including RDKit, NumPy, SciPy, pandas, tqdm, Biopython, HTTPX, and any other missing package.

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

---

## 5. Download `ccd.pkl`

Download `ccd.pkl`:

    https://huggingface.co/Keshav-Sundar-4/SweetFold/blob/main/ccd.pkl

Place it inside the `weights` folder.

Do not rename it.

---

## 6. Put the Preprocessing Scripts Together

Place these scripts in the same working directory:

    pdb_api_call.py
    pdb_api.py
    phase1_cleaner.py
    free_glycan_data.py
    lectinz_clean.py
    preprocess_glycans.py
    validation_dataset_creation.py

This folder is the preprocessing scripts directory.

---

## 7. Create a Data Folder

Create a data folder for all preprocessing outputs.

A clean layout is:

    glycan_data/
    ├── pdb_id_lists/
    ├── raw_pdb_downloads/
    ├── phase1_cleaned_pdbs/
    ├── free_glycan_cleaned_pdbs/
    ├── md_glycan_raw_pdbs/
    ├── md_glycan_cleaned_pdbs/
    ├── combined_cleaned_pdbs/
    └── final_processed_dataset/

---

## 8. Find PDB IDs Containing Glycans

Script:

    pdb_api_call.py

Open `pdb_api_call.py`.

Set:

    OUT_DIR = Path("/path/to/glycan_data/pdb_id_lists")
    OUT_FILE = OUT_DIR / "pdb_ids_with_glycans.txt"

Run:

    python pdb_api_call.py

Output:

    glycan_data/pdb_id_lists/pdb_ids_with_glycans.txt

---

## 9. Download the PDB Files

Script:

    pdb_api.py

Open `pdb_api.py`.

Set:

    PDB_ID_FILE = Path("/path/to/glycan_data/pdb_id_lists/pdb_ids_with_glycans.txt")
    OUTPUT_DIR = Path("/path/to/glycan_data/raw_pdb_downloads")

Run:

    python pdb_api.py

Output:

    glycan_data/raw_pdb_downloads/

---

## 10. Clean Protein-Glycan PDB Files

Script:

    phase1_cleaner.py

Open `phase1_cleaner.py`.

Set:

    INPUT_FOLDER = "/path/to/glycan_data/raw_pdb_downloads"
    OUTPUT_FOLDER = "/path/to/glycan_data/phase1_cleaned_pdbs"

Run:

    python phase1_cleaner.py

Output:

    glycan_data/phase1_cleaned_pdbs/

---

## 11. Generate Free-Floating Glycan Files

Script:

    free_glycan_data.py

Run:

    python free_glycan_data.py \
      --input_folder /path/to/glycan_data/raw_pdb_downloads \
      --output_folder /path/to/glycan_data/free_glycan_cleaned_pdbs

Output:

    glycan_data/free_glycan_cleaned_pdbs/

---

## 12. Clean MD-Derived Glycan Files

Script:

    lectinz_clean.py

Skip this step if you do not have MD-derived or nonstandard glycan PDB files.

Open `lectinz_clean.py`.

Set:

    TARGET_FOLDER = Path("/path/to/glycan_data/md_glycan_raw_pdbs")
    OUTPUT_FOLDER = Path("/path/to/glycan_data/md_glycan_cleaned_pdbs")
    CCD_FILE = Path("/path/to/sweetfold_env/weights/ccd.pkl")

Run:

    python lectinz_clean.py

Output:

    glycan_data/md_glycan_cleaned_pdbs/

---

## 13. Combine Cleaned PDB Files

Combine the cleaned PDB files into one folder:

    glycan_data/combined_cleaned_pdbs/

Include files from:

    glycan_data/phase1_cleaned_pdbs/
    glycan_data/free_glycan_cleaned_pdbs/
    glycan_data/md_glycan_cleaned_pdbs/

Skip the MD-derived folder if you did not use it.

---

## 14. Convert Cleaned PDB Files into `.npz` Files

Script:

    preprocess_glycans.py

Run:

    python preprocess_glycans.py \
      --datadir /path/to/glycan_data/combined_cleaned_pdbs \
      --ccd-path /path/to/sweetfold_env/weights/ccd.pkl \
      --outdir /path/to/glycan_data/final_processed_dataset \
      --num-processes 8

Output:

    glycan_data/final_processed_dataset/
    ├── structures/
    ├── records/
    └── manifest.json

Use fewer processes if memory is limited.

---

## 15. Create the Validation ID File

Script:

    validation_dataset_creation.py

Open `validation_dataset_creation.py`.

Set:

    NPZ_DIR = "/path/to/glycan_data/final_processed_dataset/structures"
    OUTPUT_FILE = "/path/to/glycan_data/final_processed_dataset/validation_ids.txt"
    VALIDATION_SET_FRACTION = 0.05

Run:

    python validation_dataset_creation.py

Output:

    glycan_data/final_processed_dataset/validation_ids.txt

---

## 16. Final Dataset Layout

The final processed dataset should look like:

    final_processed_dataset/
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

## 17. Script Summary

| Script | Purpose |
|---|---|
| `pdb_api_call.py` | Finds PDB IDs containing glycans. |
| `pdb_api.py` | Downloads PDB files from the PDB IDs. |
| `phase1_cleaner.py` | Cleans protein-glycan PDB files. |
| `free_glycan_data.py` | Extracts free-floating glycan-only PDB files. |
| `lectinz_clean.py` | Cleans MD-derived or nonstandard glycan PDB files. |
| `preprocess_glycans.py` | Converts cleaned PDB files into `.npz` dataset files. |
| `validation_dataset_creation.py` | Creates the validation ID file. |
