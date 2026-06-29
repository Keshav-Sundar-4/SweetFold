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

```text
/path/to/sweetfold_env
