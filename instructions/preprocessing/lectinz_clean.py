import os
import sys
import re
import pickle
import shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree
from concurrent.futures import ProcessPoolExecutor, as_completed

# RDKit imports are required for reading the ccd.pkl
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    print("Error: RDKit is not installed. Please install it to run this script.")
    sys.exit(1)

# --- CONFIGURATION ---
TARGET_FOLDER = Path("/work/keshavsundar/work_sundar/glycan_data/raw_glycan_data")
OUTPUT_FOLDER = Path("/work/keshavsundar/work_sundar/glycan_data/cleaned_glycan_data")
CCD_FILE = Path("/work/keshavsundar/env/boltz1x/weights/ccd.pkl")

# Standard Amino Acids
STANDARD_AMINO_ACIDS = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
    "TYR", "VAL"
}

# The bespoke map ported from the preprocessing script to fix naming mismatches
NAME_MAP = {
    "A2G": {"C2N": "C7", "CME": "C8", "O2N": "O7"}, 
    "NAG": {"C2N": "C7", "CME": "C8", "O2N": "O7"},
    "NGA": {"C2N": "C7", "CME": "C8", "O2N": "O7"}, 
    "NDG": {"C2N": "C7", "CME": "C8", "O2N": "O7"},
    "SIA": {"C5N": "C10", "CME": "C11", "O5N": "O10"},
    "NGC": {"C5N": "C10", "CME": "C11", "O5N": "O10", "OHG": "O11"},
    "TOA": {'O3': 'N3', 'O6A': 'O1', 'O6B': 'O6'}
}

def load_ccd(path):
    """Loads the Chemical Component Dictionary (CCD) and pre-calculates heavy atom counts."""
    if not path.exists():
        print(f"Error: CCD file not found at {path}")
        sys.exit(1)
        
    print(f"Loading CCD from {path}...")
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
    with open(path, "rb") as f: 
        ccd_data = pickle.load(f)
    
    ccd_expected_heavy = {}
    for res_name, mol in ccd_data.items():
        try: 
            mol_no_h = AllChem.RemoveHs(mol, sanitize=False)
        except Exception: 
            mol_no_h = mol
            
        ccd_expected_heavy[res_name] = sum(1 for atom in mol_no_h.GetAtoms() if atom.GetAtomicNum() > 1)

    return ccd_expected_heavy

def extract_num(res_num_str):
    """Safely extracts the integer value from a PDB residue number string."""
    match = re.search(r'\d+', res_num_str)
    return int(match.group()) if match else 0

def infer_element(atom_name):
    """Derives the correct chemical element from the atom name."""
    match = re.search(r'[a-zA-Z]+', atom_name)
    if match:
        chars = match.group().upper()
        # Explicitly added 2-letter elements starting with H (HE, HF, HG, HO) to protect them from being purged
        if len(chars) >= 2 and chars[:2] in ["CL", "BR", "FE", "ZN", "CA", "MG", "MN", "NA", "CU", "SE", "HE", "HF", "HG", "HO"]:
            return chars[:2]
        return chars[0]
    return "C"

def parse_pdb_full(file_path):
    """Parses a PDB file, mapping custom atom names and inferring correct elements."""
    atoms = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                try:
                    orig_atom_name = line[12:16].strip()
                    res_name = line[17:20].strip()
                    chain_id = line[21].strip()
                    res_num = line[22:26].strip()
                    
                    element = line[76:78].strip().upper()
                    
                    # If element is blank, firmly establish what it is
                    if not element: 
                        element = infer_element(orig_atom_name)
                    
                    # Rigorously drop Hydrogens and Deuterium, avoiding false positives like Mercury (HG)
                    if element in ['H', 'D']: 
                        continue
                        
                    atom_name = orig_atom_name
                    if res_name in NAME_MAP and atom_name in NAME_MAP[res_name]:
                        atom_name = NAME_MAP[res_name][atom_name]

                    final_element = infer_element(atom_name)

                    atoms.append({
                        'atom_name': atom_name,
                        'res_name': res_name,
                        'chain_id': chain_id,
                        'res_num': res_num,
                        'element': final_element,
                        'x': float(line[30:38]), 
                        'y': float(line[38:46]), 
                        'z': float(line[46:54]),
                        'line': line,
                        'res_key': (chain_id, res_num, res_name)
                    })
                except Exception:
                    continue
    return atoms

def process_pdb_file(input_path, output_path, ccd_expected):
    """Integrates fragments, scrubs incomplete residues, and outputs clean PDB coordinates."""
    atoms = parse_pdb_full(input_path)
    if not atoms:
        return False, 0, 0
        
    coords = np.array([[a['x'], a['y'], a['z']] for a in atoms])
    tree = cKDTree(coords)
    bonded_pairs = tree.query_pairs(r=2.0)
    
    res_parent = {a['res_key']: a['res_key'] for a in atoms}
    res_sizes = defaultdict(int)
    for a in atoms:
        res_sizes[a['res_key']] += 1

    def find_res(k):
        if res_parent[k] == k: return k
        res_parent[k] = find_res(res_parent[k])
        return res_parent[k]

    merged_count = 0
    for i, j in bonded_pairs:
        k1 = find_res(atoms[i]['res_key'])
        k2 = find_res(atoms[j]['res_key'])
        
        if k1 != k2 and k1[2] == k2[2]:
            expected_size = ccd_expected.get(k1[2], 0)
            if expected_size > 0 and (res_sizes[k1] + res_sizes[k2]) <= (expected_size + 2):
                if extract_num(k1[1]) < extract_num(k2[1]):
                    root, child = k1, k2
                else:
                    root, child = k2, k1
                    
                res_parent[child] = root
                res_sizes[root] += res_sizes[child]
                res_sizes[child] = 0
                merged_count += 1

    final_residues = defaultdict(list)
    for atom in atoms:
        u_key = find_res(atom['res_key'])
        atom['u_key'] = u_key
        final_residues[u_key].append(atom)

    deleted_count = 0
    valid_atoms = []
    
    u_key_order = {}
    for i, atom in enumerate(atoms):
        if atom['u_key'] not in u_key_order:
            u_key_order[atom['u_key']] = i
            
    for u_key, res_atoms in final_residues.items():
        res_name = u_key[2]
        if res_name in STANDARD_AMINO_ACIDS or res_name not in ccd_expected:
            valid_atoms.extend(res_atoms)
            continue
            
        expected = ccd_expected[res_name]
        actual = len(res_atoms)
        
        if (expected - actual) >= 2:
            deleted_count += 1
        else:
            valid_atoms.extend(res_atoms)

    valid_atoms.sort(key=lambda a: (u_key_order[a['u_key']], a['u_key']))
    
    new_serial = 1
    with open(output_path, 'w') as f_out:
        for atom in valid_atoms:
            u_key = atom['u_key']
            
            serial_str = f"{new_serial:>5}"
            name = atom['atom_name']
            name_str = f" {name:<3}" if len(name) < 4 else f"{name:<4}"
            res_str = f"{u_key[2]:>3}"
            chain_str = f"{u_key[0]:>1}"
            num_str = f"{u_key[1]:>4}"
            element_str = f"{atom['element']:>2}"
            
            padded_line = atom['line'].rstrip('\n\r').ljust(80)
            
            # Extract exactly from column 27 (insertion code) up to column 66 (end of Temp Factor).
            middle_part = padded_line[26:66]
            
            # 10 spaces explicitly blank out the legacy Segment ID (columns 67-76).
            # This perfectly isolates the element symbol to 77-78 and charge to 79-80.
            new_line = f"{padded_line[:6]}{serial_str} {name_str}{padded_line[16]}{res_str} {chain_str}{num_str}{middle_part}          {element_str}  \n"
            
            f_out.write(new_line)
            new_serial += 1
            
        f_out.write("END\n")

    return True, merged_count, deleted_count


def main():
    if not TARGET_FOLDER.exists():
        print(f"Error: Target folder '{TARGET_FOLDER}' not found.")
        sys.exit(1)

    if not OUTPUT_FOLDER.exists():
        OUTPUT_FOLDER.mkdir(parents=True)
        print(f"Created output directory: {OUTPUT_FOLDER}")

    ccd_expected = load_ccd(CCD_FILE)
    
    # We only care about PDB files per your prompt
    all_files = list(TARGET_FOLDER.glob("*.pdb"))
    if not all_files:
        print("No PDB files found in the target folder.")
        return
        
    print(f"\nFound {len(all_files)} files. Engaging hyper-efficient multiprocessing over {os.cpu_count()} CPU cores...")

    processed, total_merged, total_deleted = 0, 0, 0

    # Execute parsing and graph building in parallel across all available CPUs
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {
            executor.submit(process_pdb_file, file_path, OUTPUT_FOLDER / file_path.name, ccd_expected): file_path 
            for file_path in all_files
        }
        
        # Track completion as futures resolve
        for i, future in enumerate(as_completed(futures), 1):
            file_path = futures[future]
            try:
                was_cleaned, m_count, d_count = future.result()
                if was_cleaned:
                    processed += 1
                    total_merged += m_count
                    total_deleted += d_count
                    
                # Print a clean progress tracker instead of spamming every file
                if i % 100 == 0 or i == len(all_files):
                    sys.stdout.write(f"\rProgress: [{i}/{len(all_files)}] files complete.")
                    sys.stdout.flush()
            except Exception as exc:
                print(f"\n[!] File {file_path.name} generated an exception: {exc}")

    print(f"\n\nOperation complete.")
    print(f"  - Files repaired/cleaned: {processed}")
    print(f"  - Total fragments mapped & integrated: {total_merged}")
    print(f"  - Total unrecoverable residues deleted: {total_deleted}")

if __name__ == "__main__":
    main()
