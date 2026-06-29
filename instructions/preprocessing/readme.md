# Processing Procedure:
1. Use pdb_api_call.py to obtain a folder of PDB files (Does not require SweetFold environment, can be done on local machine)
2. Use pdb_api.py call on the text file created via step one to generate the PDB glycan dataset (Does not require SweetFold environment, can be done on local machine)
3.
      a) Use the phase1_cleaner.py script on the dataset obtained in step 2
      b) Generate a dataset of free-floating glycans via free_glycan_data.py. (Note that if you use the PDB glycan dataset you will get a small number of free-floating glycans. It is suggested you also            generate an MD dataset as well.)
      c) Use lectinz_clean.py and point it at the free-floating dataset that was created in Step 3b
      d) Combine the folders generated in steps 3a-c 
4. Run preprocess_glycans.py on the folder generated in step 3d
5. Run validation_dataset_creation.py on the folder generated in step 4


# Processing Instructions:

### Script Execution Guide

---

#### `pdb_api_call.py`
This script uses hardcoded directories to save the retrieved PDB IDs. 

**Command:**
`python pdb_api_call.py`

**Lines to Change:**
Before running, open the file and modify the following lines to match your local directory structure:
* **Line 20:** `OUT_DIR = Path("Boltz_Data_Files")` -> Change `"Boltz_Data_Files"` to your desired output folder path. 
* *Note: `OUT_FILE` (Line 21) will automatically be created inside whatever you set `OUT_DIR` to.*

---

#### `pdb_api.py`
This script uses hardcoded directories for both its input list and its output destination.

**Command:**
`python pdb_api.py`

**Lines to Change:**
Update the configuration section to point to your specific paths:
* **Line 23:** `PDB_ID_FILE = Path("Boltz_Data_Files/pdb_ids_with_glycans.txt")` -> Change this to the path where your input text file of PDB IDs is located.
* **Line 24:** `OUTPUT_DIR = Path("PDB_Downloads")` -> Change this to the directory where you want the `.pdb` files to be downloaded.

---

#### `phase1_cleaner.py`
This script processes data using hardcoded absolute paths. 

**Command:**
`python phase1_cleaner.py`

**Lines to Change:**
Locate the `--- HARDCODED CONFIGURATION ---` section and update these paths:
* **Line 50:** `INPUT_FOLDER = '/work/keshavsundar/.../PDB_Raw_Dataset'` -> Change to the path of your raw PDB files.
* **Line 51:** `OUTPUT_FOLDER = '/work/keshavsundar/.../Phase1_Raw_Dataset'` -> Change to the path where you want the cleaned files saved.

---

#### `free_glycan_data.py`
This script is executed using command-line arguments (CLI) to specify the input and output folders dynamically.

**Command:**
`python free_glycan_data.py --input_folder /path/to/your/raw/PDBs --output_folder /path/to/your/cleaned/output`

**CLI Arguments:**
* `--input_folder`: Provide the absolute or relative path to the directory containing your raw `.pdb` or `.ent` files.
* `--output_folder`: Provide the path to the directory where the glycan-only output files should be saved.

---

#### `lectinz_clean.py`
This script uses hardcoded paths for its input, output, and a necessary `.pkl` file.

**Command:**
`python lectinz_clean.py`

**Lines to Change:**
Update the `--- CONFIGURATION ---` block to match your system:
* **Line 21:** `TARGET_FOLDER = Path("...")` -> Change to the folder containing the input PDBs you want to clean.
* **Line 22:** `OUTPUT_FOLDER = Path("...")` -> Change to the destination folder for the cleaned data.
* **Line 23:** `CCD_FILE = Path("...")` -> Change to the specific file path pointing to your `ccd.pkl` weights file.

---

#### `preprocess_glycnas.py`
This script relies heavily on CLI arguments to dictate where data is read from, where it goes, and how much compute to use.

**Command:**
`python preprocess_glycnas.py --datadir /path/to/cleaned/glycan/pdbs --ccd-path /path/to/ccd.pkl --outdir /path/to/npz/output/folder --num-processes 8`

**CLI Arguments:**
* `--datadir`: The directory containing your pre-processed/cleaned PDB files.
* `--ccd-path`: The direct file path to your `ccd.pkl` dictionary file.
* `--outdir`: The directory where the final `.npz` structures and `.json` records will be saved.
* `--num-processes` *(Optional)*: The number of CPU cores to dedicate to the task. If omitted, it defaults to half of your available CPU cores.

---

#### `validation_data_creation.py`
This script uses hardcoded paths to locate the `.npz` files and determine where to drop the text file containing the validation IDs.

**Command:**
`python validation_data_creation.py`

**Lines to Change:**
Update the `=== CONFIGURATION ===` block:
* **Line 29:** `NPZ_DIR = '...'` -> Change to the directory containing the `.npz` structure files generated in the previous step.
* **Line 32:** `OUTPUT_FILE = '...'` -> Change to the exact file path (including the `.txt` extension) where you want the validation IDs to be written.

---

# File Descriptions:
- pdb_api_call.py: A script that obtains all PDB files from the PDB directory that are both <9 angstroms in resolution and contain 1 or more CCD codes that are considered sugars. Outputs a text file with all the pdb IDs used.

- pdb_api.py: A script that uses a text file of PDB ids and generates a folder of PDB files consisting of the codes in said text file

- phase1_cleaner.py: This script is used to clean the folder of raw PDB files obtained from pdb_api.py. The script discards all atoms other than those belonging to protein or glycan residues. The script checks for atomic clashes (heavy atoms closer than 0.5A to one another) and removes such files. The script also handles models with multiple conformations by choosing one of the provided conformations. Furthermore, if a glycan residue has less than 6 heavy atoms only the glycan is scrubbed from the file. The script then outputs a clean PDB file with the valid protein/glycan atoms as well as SEQRES information to correctly inform the model of any unresolved residues or unnatural amino acids. 

-  free_glycan_data.py: A cleaning script specialized for free-floating glycans. It works with PDB files and begins by stripping hydrogen atoms from the file. It generates a connectivty file from scratch and deals with any PDB files with mutliple conformations or poses. No steroehemical cleaning etc. is done.

- lectinz_clean.py: This script is used to manage and clean MD-derived PDB files. Such files often defy common PDB conventions and thus require special handling. Hydrogens are scrubbed, and atom names are normalized. Moreover, such files often break up a single monosaccharide into distinct residues, so a connectivity derived alhorithm is used to generate unified residue naming schemes. The corrected file is then converted into standard PDB format. 

-  preprocess_glycans.py: This script converts .pdb files into .npz files. The general flow is that it reads .pdb files and parses the atoms, groups residues/chains, and then converts the necessary features into arrays. Necessary features include atom names, elements, residue names, chain IDs, bond connectivity, glycan semantics, protein sequences, etc.
      - Glycans are often 'chained' as part of the protein. The script generates connectivity via sci-py's cKDTree algorithm and identifies protein and glycan molecules, creating chains based on                    connectivity. It generates chain ID's from scratch based on these re-chained entities. 
      - Glycosylation sites are specially detected and placed into the connections array
      - The script obtains glycan anomeric configuration using a 3d approach. Anomeric carbons of sugars are identified based on their residue name. The dihedral calculation is found from 4 atoms,                  which consist of [Anomeric_Oxygen, Anomeric_Carbon, Ring Oxygen, Ring_Neighboring_Carbon]. If the dihedral is between -95 and 95 degrees then the anomericity is classified as alpha. If the                  dihedral is outside of this range it is classified as beta.
      - Glycosylation Filtering is also applied due to inconsistencies in glycosylation structure prediction. If the [CG, ND2, C1] angle is outside of 110-130 degrees, the glycan is filtered out. If                the [OD2, CG, ND2, C1] dihedral angle is not within the -20 < X < 20 range the glycan is filtered out. (Note that filtering the glycan out is equivalent to removing the glycosylation site. The              protein sample is maintained as there are often many glycosylation sites in one structure. To maximize data only the glycan is removed. 
      - Occasionally, there are PDB artifacts where the oxygen involved in a glycosidic bond is not the only oxygen bound to the acceptor monosaccharide's carbon. In this case we remove the extra oxygen
      - For certain MD files, the atom names are not consistent with standard PDB atom names. A name mapping is used to modify the file in these edge cases
 
-  validation_dataset_creation.py: This script generates a validation text file containing of 3.5% of the processes in a given file. 
