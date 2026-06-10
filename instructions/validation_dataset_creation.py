#!/usr/bin/env python3

"""
validation_dataset_creation.py

Builds a validation set from processed .npz files by taking a simple
random sample of the available files. This script is designed for speed
and simplicity.

The process is as follows:
1.  Scan the target directory for all .npz files.
2.  Calculate the number of files corresponding to the desired validation
    set percentage (e.g., 5%).
3.  Randomly shuffle the list of all file IDs.
4.  Select the first N files from the shuffled list.
5.  Write these selected IDs to the output text file.
"""

import os
import random
import math
from tqdm import tqdm

# Reproducibility
random.seed(42)

# === CONFIGURATION ===
# The folder containing the final .npz structure files
NPZ_DIR = '/work/keshavsundar/work_sundar/glycan_data/Final_Dataset/structures'

# The output file where the list of validation IDs will be saved
OUTPUT_FILE = '/work/keshavsundar/work_sundar/glycan_data/Final_Dataset/validation_ids.txt'

# The fraction of total files to allocate to the validation set
VALIDATION_SET_FRACTION = 0.05  # This is 5%

def main():
    """
    Main execution function to create the validation set.
    """
    print("="*60)
    print("      Simple Random Validation Set Creator      ")
    print("="*60)

    # --- 1. Validate Source Folder and Scan for Files ---
    if not os.path.isdir(NPZ_DIR):
        print(f"[ERROR] Source NPZ directory not found: {NPZ_DIR}")
        exit(1)
    
    print(f"[INFO] Scanning for .npz files in: {NPZ_DIR}")
    
    try:
        # Get a list of all file IDs (without the .npz extension)
        all_ids = sorted([
            os.path.splitext(f)[0] for f in os.listdir(NPZ_DIR) if f.endswith('.npz')
        ])
        total_files = len(all_ids)

        if total_files == 0:
            print("[WARNING] The source directory is empty. No validation set will be created.")
            return

    except Exception as e:
        print(f"[ERROR] Failed to read files from the source directory: {e}")
        exit(1)

    # --- 2. Calculate Sample Size ---
    # Use math.ceil to ensure we get at least one file for small datasets
    target_size = math.ceil(total_files * VALIDATION_SET_FRACTION)

    print(f"[INFO] Found {total_files} total files.")
    print(f"[INFO] Target validation set size: {target_size} files ({VALIDATION_SET_FRACTION:.0%})")

    # --- 3. Randomly Select Files ---
    print("\n--- Selecting Validation Set Files ---")
    
    # Shuffle the list of all IDs in-place to ensure a random sample
    random.shuffle(all_ids)
    
    # Take the first N files from the now-randomized list
    selected_ids = all_ids[:target_size]
    
    print(f"Randomly selected {len(selected_ids)} files for the validation set.")

    # --- 4. Writing Output File ---
    print("\n--- Writing Output ---")
    
    # Sort the final list alphabetically for a clean, deterministic output file
    final_selection = sorted(selected_ids)
    
    try:
        with open(OUTPUT_FILE, 'w') as f:
            for cid in tqdm(final_selection, desc="Writing file IDs"):
                f.write(cid + '\n')
            
        print(f"\nSuccessfully created validation set with {len(final_selection)} IDs.")
        print(f"File list written to: {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"[ERROR] Failed to write to the output file: {e}")
        exit(1)

    print("="*60)
    print("Operation complete.")
    print("="*60)

if __name__ == '__main__':
    main()