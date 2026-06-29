#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PDB Protein-Glycan Cleaner

This script performs a fast, multi-stage filtering of raw PDB files. It correctly
handles multi-model (NMR) and multi-conformation (altloc) files. Its primary
purpose is to remove non-protein and non-glycan entities and to validate the
structural integrity of the remaining atoms.

Core Logic:
1.  Scans the input directory for PDB files.
2.  Uses a multiprocessing Pool for parallel processing.
3.  **PRE-VALIDATION STAGE (per file):**
    -   **Protein Context Check:** Files without any standard amino acid residues
        are SKIPPED immediately.
    -   **Atomic Clash Check:** For each model, it checks for atomic clashes using
        a highly efficient cKDTree. If any two heavy atoms are within 0.5 Å,
        the ENTIRE source PDB file is SKIPPED.
4.  **PROCESSING STAGE (per model):**
    -   Splits multi-model files into separate blocks.
    -   Collapses multi-conformation (altloc) atoms by highest occupancy.
    -   **Simple Filtering:** It keeps only atoms belonging to standard amino
        acids or a predefined list of allowed glycan residues. All other atoms
        (water, ions, ligands, etc.) are DISCARDED. Original chain IDs and
        residue numbers are preserved.
    -   **Glycan Completeness Validation:** For each model, it checks EVERY glycan
        residue. If any glycan has 6 or fewer resolved heavy atoms, that
        specific model is SKIPPED.
    -   Valid, cleaned structures are written to new, uniquely named PDB files,
        with atom serials renumbered sequentially for correctness.
"""
import os
import shutil
import time
import multiprocessing
from multiprocessing import Pool
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree
import sys
import string
from collections import deque
import traceback
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

# --- HARDCODED CONFIGURATION ---
INPUT_FOLDER = '/work/keshavsundar/work_sundar/glycan_data/PDB_Raw_Dataset'
OUTPUT_FOLDER = '/work/keshavsundar/work_sundar/glycan_data/Phase1_Raw_Dataset'
CLASH_THRESHOLD = 0.5      # Angstroms

# --- RESIDUE DEFINITIONS ---
AMINO_ACIDS = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"}
ALLOWED_GLYCANS = {"05L", "07E", "0HX", "0LP", "0MK", "0NZ", "0UB", "0WK", "0XY", "0YT", "12E", "145", "147", "149", "14T", "15L", "16F", "16G", "16O", "17T", "18D", "18O", "1CF", "1GL", "1GN", "1S3", "1S4", "1SD", "1X4", "20S", "20X", "22O", "22S", "23V", "24S", "25E", "26O", "27C", "289", "291", "293", "2DG", "2DR", "2F8", "2FG", "2FL", "2GL", "2GS", "2H5", "2M5", "2M8", "2WP", "32O", "34V", "38J", "3DO", "3FM", "3HD", "3J3", "3J4", "3LJ", "3MG", "3MK", "3R3", "3S6", "3YW", "42D", "445", "44S", "46Z", "475", "491", "49A", "49S", "49T", "49V", "4AM", "4CQ", "4GL", "4GP", "4JA", "4N2", "4NN", "4QY", "4R1", "4SG", "4U0", "4U1", "4U2", "4UZ", "4V5", "50A", "510", "51N", "56N", "57S", "5DI", "5GF", "5GO", "5KQ", "5KV", "5L2", "5L3", "5LS", "5LT", "5N6", "5QP", "5TH", "5TJ", "5TK", "5TM", "604", "61J", "62I", "64K", "66O", "6BG", "6C2", "6GB", "6GP", "6GR", "6K3", "6KH", "6KL", "6KS", "6KU", "6KW", "6LS", "6LW", "6MJ", "6MN", "6PY", "6PZ", "6S2", "6UD", "6Y6", "6YR", "6ZC", "73E", "79J", "7CV", "7D1", "7GP", "7JZ", "7K2", "7K3", "7NU", "83Y", "89Y", "8B7", "8B9", "8EX", "8GA", "8GG", "8GP", "8LM", "8LR", "8OQ", "8PK", "8S0", "95Z", "96O", "9AM", "9C1", "9CD", "9GP", "9KJ", "9MR", "9OK", "9PG", "9QG", "9QZ", "9S7", "9SG", "9SJ", "9SM", "9SP", "9T1", "9T7", "9VP", "9WJ", "9WN", "9WZ", "9YW", "A0K", "A1Q", "A2G", "A5C", "A6P", "AAL", "ABD", "ABE", "ABF", "ABL", "AC1", "ACR", "ACX", "ADA", "AF1", "AFD", "AFO", "AFP", "AFR", "AGL", "AGR", "AH2", "AH8", "AHG", "AHM", "AHR", "AIG", "ALL", "ALX", "AMG", "AMN", "AMU", "AMV", "ANA", "AOG", "AQA", "ARA", "ARB", "ARI", "ARW", "ASC", "ASG", "ASO", "AXP", "AXR", "AY9", "AZC", "B0D", "B16", "B1H", "B1N", "B6D", "B7G", "B8D", "B9D", "BBK", "BBV", "BCD", "BCW", "BDF", "BDG", "BDP", "BDR", "BDZ", "BEM", "BFN", "BG6", "BG8", "BGC", "BGL", "BGN", "BGP", "BGS", "BHG", "BM3", "BM7", "BMA", "BMX", "BND", "BNG", "BNX", "BO1", "BOG", "BQY", "BS7", "BTG", "BTU", "BWG", "BXF", "BXX", "BXY", "BZD", "C3B", "C3G", "C3X", "C4B", "C4W", "C5X", "CBF", "CBI", "CBK", "CDR", "CE5", "CE6", "CE8", "CEG", "CEX", "CEY", "CEZ", "CGF", "CJB", "CKB", "CKP", "CNP", "CR1", "CR6", "CRA", "CT3", "CTO", "CTR", "CTT", "D0N", "D1M", "D5E", "D6G", "DAF", "DAG", "DAN", "DDA", "DDL", "DEG", "DEL", "DFR", "DFX", "DGO", "DGS", "DJB", "DJE", "DK4", "DKX", "DKZ", "DL6", "DLD", "DLF", "DLG", "DO8", "DOM", "DPC", "DQR", "DR2", "DR3", "DR5", "DRI", "DSR", "DT6", "DVC", "DYM", "E3M", "E5G", "EAG", "EBG", "EBQ", "EEN", "EEQ", "EGA", "EMP", "EMZ", "EPG", "EQP", "EQV", "ERE", "ERI", "ETT", "F1P", "F1X", "F55", "F58", "F6P", "FBP", "FCA", "FCB", "FCT", "FDP", "FDQ", "FFC", "FFX", "FIF", "FK9", "FKD", "FMF", "FMO", "FNG", "FNY", "FRU", "FSA", "FSI", "FSM", "FSR", "FSW", "FUB", "FUC", "FUF", "FUL", "FUY", "FVQ", "FX1", "FYJ", "G0S", "G16", "G1P", "G20", "G28", "G2F", "G3F", "G4D", "G4S", "G6D", "G6P", "G6S", "G7P", "G8Z", "GAA", "GAC", "GAD", "GAF", "GAL", "GAT", "GBH", "GC1", "GC4", "GC9", "GCB", "GCD", "GCN", "GCO", "GCS", "GCT", "GCU", "GCV", "GCW", "GDA", "GDL", "GE1", "GE3", "GFP", "GIV", "GL0", "GL1", "GL2", "GL4", "GL5", "GL6", "GL7", "GL9", "GLA", "GLC", "GLD", "GLF", "GLG", "GLO", "GLP", "GLS", "GLT", "GM0", "GMB", "GMH", "GMT", "GMZ", "GN1", "GN4", "GNS", "GNX", "GP0", "GP1", "GP4", "GPH", "GPK", "GPM", "GPO", "GPQ", "GPU", "GPV", "GPW", "GQ1", "GRF", "GRX", "GS1", "GS9", "GTK", "GTM", "GTR", "GU0", "GU1", "GU2", "GU3", "GU4", "GU5", "GU6", "GU8", "GU9", "GUF", "GUL", "GUP", "GUZ", "GXL", "GYE", "GYG", "GYP", "GYU", "GYV", "GZL", "H1M", "H1S", "H2P", "H53", "H6Q", "H6Z", "HBZ", "HD4", "HNV", "HNW", "HSG", "HSH", "HSJ", "HSQ", "HSX", "HSY", "HTG", "HTM", "I57", "IAB", "IDC", "IDF", "IDG", "IDR", "IDS", "IDU", "IDX", "IDY", "IEM", "IN1", "IPT", "ISD", "ISL", "ISX", "IXD", "J5B", "JFZ", "JHM", "JLT", "JS2", "JV4", "JVA", "JVS", "JZR", "K5B", "K99", "KBA", "KBG", "KD5", "KDA", "KDB", "KDD", "KDE", "KDF", "KDM", "KDN", "KDO", "KDR", "KFN", "KG1", "KGM", "KHP", "KME", "KO1", "KO2", "KOT", "KTU", "L1L", "L6S", "LAH", "LAK", "LAO", "LAT", "LB2", "LBS", "LBT", "LCN", "LDY", "LEC", "LFR", "LGC", "LGU", "LKA", "LKS", "LNV", "LOG", "LOX", "LRH", "LVO", "LVZ", "LXB", "LXC", "LXZ", "LZ0", "M1F", "M1P", "M2F", "M3N", "M55", "M6D", "M6P", "M7B", "M7P", "M8C", "MA1", "MA2", "MA3", "MA8", "MAF", "MAG", "MAL", "MAN", "MAT", "MAV", "MAW", "MBE", "MBF", "MBG", "MCU", "MDA", "MDP", "MFB", "MFU", "MG5", "MGC", "MGL", "MGS", "MJJ", "MLB", "MLR", "MMA", "MN0", "MNA", "MQG", "MQT", "MRH", "MRP", "MSX", "MTT", "MUB", "MUR", "MVP", "MXY", "MXZ", "MYG", "N1L", "N9S", "NA1", "NAA", "NAG", "NBG", "NBX", "NBY", "NDG", "NFG", "NG1", "NG6", "NGA", "NGC", "NGE", "NGK", "NGR", "NGS", "NGY", "NGZ", "NHF", "NLC", "NM6", "NM9", "NNG", "NPF", "NSQ", "NT1", "NTF", "NTO", "NTP", "NXD", "NYT", "O1G", "OAK", "OEL", "OI7", "OPM", "OSU", "OTG", "OTN", "OTU", "OX2", "P53", "P6P", "PA1", "PAV", "PDX", "PH5", "PKM", "PNA", "PNG", "PNJ", "PNW", "PPC", "PRP", "PSG", "PSV", "PUF", "PZU", "QIF", "QKH", "QPS", "R1P", "R1X", "R2B", "R2G", "RAE", "RAF", "RAM", "RAO", "RCD", "RER", "RF5", "RGG", "RHA", "RHC", "RI2", "RIB", "RIP", "RM4", "RP3", "RP5", "RP6", "RR7", "RRJ", "RRY", "RST", "RTG", "RTV", "RUG", "RUU", "RV7", "RVG", "RVM", "RWI", "RY7", "RZM", "S7P", "S81", "SA0", "SCG", "SCR", "SDY", "SEJ", "SF6", "SF9", "SFJ", "SFU", "SG4", "SG5", "SG6", "SG7", "SGA", "SGC", "SGD", "SGN", "SHB", "SHD", "SHG", "SIA", "SID", "SIO", "SIZ", "SLB", "SLM", "SLT", "SMD", "SN5", "SNG", "SOE", "SOG", "SOR", "SR1", "SSG", "STZ", "SUC", "SUP", "SUS", "SWE", "SZZ", "T68", "T6P", "T6T", "TA6", "TCB", "TCG", "TDG", "TEU", "TF0", "TFU", "TGA", "TGK", "TGR", "TGY", "TH1", "TMR", "TMX", "TNX", "TOA", "TOC", "TQY", "TRE", "TRV", "TS8", "TT7", "TTV", "TTZ", "TU4", "TUG", "TUJ", "TUP", "TUR", "TVD", "TVG", "TVM", "TVS", "TVV", "TVY", "TW7", "TWA", "TWD", "TWG", "TWJ", "TWY", "TXB", "TYV", "U1Y", "U2A", "U2D", "U63", "U8V", "U97", "U9A", "U9D", "U9G", "U9J", "U9M", "UAP", "UCD", "UDC", "UEA", "V3M", "V3P", "V71", "VG1", "VTB", "W9T", "WIA", "WOO", "WUN", "X0X", "X1P", "X1X", "X2F", "X6X", "XDX", "XGP", "XIL", "XLF", "XLS", "XMM", "XXM", "XXR", "XXX", "XYF", "XYL", "XYP", "XYS", "XYT", "XYZ", "YIO", "YJM", "YKR", "YO5", "YX0", "YX1", "YYB", "YYH", "YYJ", "YYK", "YYM", "YYQ", "YZ0", "Z0F", "Z15", "Z16", "Z2D", "Z2T", "Z3K", "Z3L", "Z3Q", "Z3U", "Z4K", "Z4R", "Z4S", "Z4U", "Z4V", "Z4W", "Z4Y", "Z57", "Z5J", "Z5L", "Z61", "Z6H", "Z6J", "Z6W", "Z8H", "Z8T", "Z9D", "Z9E", "Z9H", "Z9K", "Z9L", "Z9M", "Z9N", "Z9W", "ZB0", "ZB1", "ZB2", "ZB3", "ZCD", "ZCZ", "ZD0", "ZDC", "ZDO", "ZEE", "ZEL", "ZGE", "ZMR"}
VALID_RESIDUES = AMINO_ACIDS.union(ALLOWED_GLYCANS)

def find_nearby_pairs(atom_data, threshold):
    """
    Finds all pairs of atoms within a given distance threshold using scipy's cKDTree.
    This is highly efficient for proximity searches.
    """
    if len(atom_data) < 2:
        return []
    
    coords = np.array([atom['coords'] for atom in atom_data])
    tree = cKDTree(coords)
    
    # query_pairs returns a set of (i, j) tuples where i < j
    index_pairs = tree.query_pairs(r=threshold, output_type='set')
    
    return [(atom_data[i], atom_data[j]) for i, j in index_pairs]

def collapse_altlocs_and_clean(lines):
    """
    Resolves alternate locations, keeps only valid residues, and removes hydrogens.
    Returns a list of PDB lines for the chosen, resolved heavy atoms.
    """
    best_atom_line_for_key = {}
    def altloc_priority(ch): return (0, 'A') if ch == 'A' else ((1, ch) if ch and ch != ' ' else (2, ' '))

    for line in lines:
        if not line.startswith(("ATOM", "HETATM")): continue
        
        # Pre-filter to only consider atoms from residues we might keep
        res_name = line[17:20].strip()
        if res_name not in VALID_RESIDUES: continue

        # Key uniquely identifies an atom's position, ignoring occupancy and altloc
        key = (line[21:22], line[22:26], line[26:27], line[12:16].strip()) # chain, res_num, icode, atom_name
        alt_loc = line[16]
        try: occ = float(line[54:60])
        except (ValueError, IndexError): occ = 0.0

        if key not in best_atom_line_for_key:
            best_atom_line_for_key[key] = (occ, alt_loc, line)
        else:
            best_occ, best_alt, _ = best_atom_line_for_key[key]
            if occ > best_occ or (occ == best_occ and altloc_priority(alt_loc) < altloc_priority(best_alt)):
                best_atom_line_for_key[key] = (occ, alt_loc, line)

    # Filter out hydrogens from the chosen atoms
    final_lines = []
    for _, _, line in best_atom_line_for_key.values():
        if not line[12:16].strip().startswith('H'):
            final_lines.append(line)
            
    return final_lines

def split_pdb_by_model(lines):
    """Splits a PDB file's lines into a list of blocks, one for each MODEL."""
    if not any(line.startswith("MODEL") for line in lines): return [lines]
    
    model_blocks, current_model_lines, in_model = [], [], False
    for line in lines:
        if line.startswith("MODEL"):
            if current_model_lines: model_blocks.append(current_model_lines)
            current_model_lines, in_model = [], True
        elif line.startswith("ENDMDL"):
            if in_model:
                model_blocks.append(current_model_lines)
                in_model, current_model_lines = False, []
        elif in_model: current_model_lines.append(line)
    
    if in_model and current_model_lines: model_blocks.append(current_model_lines)
    return model_blocks if model_blocks else [lines]

def find_glycan_components(atoms_by_residue: dict, inter_residue_threshold: float = 2.0):
    """
    Identifies connected glycan components based on atomic proximity.
    This does NOT rechain, it just identifies the groups of connected glycan residues.

    Args:
        atoms_by_residue: A dictionary mapping residue keys to lists of PDB atom lines.
        inter_residue_threshold: The max distance in Angstroms to consider two atoms bonded.

    Returns:
        A list of frozensets, where each frozenset contains the residue keys
        of a single, connected glycan component.
    """
    glycan_residue_keys = [
        key for key, atoms in atoms_by_residue.items()
        if atoms and atoms[0][17:20].strip() in ALLOWED_GLYCANS
    ]

    if not glycan_residue_keys:
        return []

    # Create a flat list of all heavy atom coordinates from all glycan residues
    flat_glycan_coords = []
    atom_idx_to_res_key = []
    for res_key in glycan_residue_keys:
        for line in atoms_by_residue[res_key]:
            try:
                coords = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                flat_glycan_coords.append(coords)
                atom_idx_to_res_key.append(res_key)
            except (ValueError, IndexError):
                continue

    # Build an adjacency graph between glycan RESIDUES using atomic proximity
    adj_residues = defaultdict(set)
    if flat_glycan_coords:
        coords_array = np.array(flat_glycan_coords)
        if coords_array.shape[0] < 2:
            # Not enough atoms to form pairs, treat each residue as its own component
            return [frozenset([key]) for key in glycan_residue_keys]

        tree = cKDTree(coords_array)
        pairs = tree.query_pairs(r=inter_residue_threshold)
        for i, j in pairs:
            res_key1 = atom_idx_to_res_key[i]
            res_key2 = atom_idx_to_res_key[j]
            if res_key1 != res_key2:
                adj_residues[res_key1].add(res_key2)
                adj_residues[res_key2].add(res_key1)

    # Find connected components using Breadth-First Search (BFS)
    visited_residues = set()
    components = []
    for res_key in glycan_residue_keys:
        if res_key not in visited_residues:
            component = set()
            q = deque([res_key])
            visited_residues.add(res_key)
            while q:
                current_res_key = q.popleft()
                component.add(current_res_key)
                for neighbor_res_key in adj_residues.get(current_res_key, []):
                    if neighbor_res_key not in visited_residues:
                        visited_residues.add(neighbor_res_key)
                        q.append(neighbor_res_key)
            components.append(frozenset(component)) # Use frozenset for hashability

    return components

def process_pdb_file(filepath):
    """
    Worker function that filters PDB files by removing bad glycan components
    and correctly handles clashes on a per-model basis.
    """
    filename = os.path.basename(filepath)
    pdb_id, _ = os.path.splitext(filename)

    try:
        with open(filepath, 'r', encoding='latin-1') as f:
            all_lines = f.readlines()
    except Exception:
        return 0, 0, 0, False, False, [f"Skipped {filename}: File could not be read."], False, filename, 0

    seqres_lines = [line for line in all_lines if line.startswith("SEQRES")]
    has_seqres = bool(seqres_lines)

    # File-level pre-check: Skip if no protein is present at all.
    if not any(line.startswith('ATOM') and line[17:20].strip() in AMINO_ACIDS for line in all_lines):
        return 0, 0, 0, False, has_seqres, [f"Skipped {filename}: No protein residues found."], False, filename, 0

    model_blocks = split_pdb_by_model(all_lines)
    num_models_in_file = len(model_blocks)

    files_written, processed_confs, is_multiconf, skip_reasons = 0, 0, False, []
    file_resorted_to_double_char = False # Kept for signature, not used
    glycans_removed_count = 0

    for model_num, model_block in enumerate(model_blocks, 1):
        try:
            if any(line.startswith(("ATOM", "HETATM")) and line[16].strip() for line in model_block):
                is_multiconf = True

            # --- MODEL-LEVEL PROCESSING ---
            # 1. Resolve altlocs and remove hydrogens for this specific model
            resolved_model_lines = collapse_altlocs_and_clean(model_block)
            if not resolved_model_lines:
                continue

            # 2. Perform clash check ON THIS MODEL ONLY.
            atom_data_clash = []
            for line in resolved_model_lines:
                try:
                    atom_data_clash.append({'coords': (float(line[30:38]), float(line[38:46]), float(line[46:54]))})
                except (ValueError, IndexError): continue
            
            if len(find_nearby_pairs(atom_data_clash, CLASH_THRESHOLD)) > 0:
                # If clash found, skip THIS MODEL, not the whole file.
                reason = f"Skipped {filename} (Model {model_num}): Atomic clash (< {CLASH_THRESHOLD} Å) detected."
                skip_reasons.append(reason)
                continue # Move to the next model

            # 3. Group atoms by residue for glycan analysis
            atoms_by_residue = defaultdict(list)
            for line in resolved_model_lines:
                residue_key = (line[21:22].strip(), line[22:26].strip(), line[26].strip())
                atoms_by_residue[residue_key].append(line)

            # 4. Identify all unique glycan components in this model
            glycan_components = find_glycan_components(atoms_by_residue)
            
            # 5. Find all "bad" glycan residues based on the two filter conditions
            bad_residue_keys = set()
            for res_key, atom_lines in atoms_by_residue.items():
                res_name = atom_lines[0][17:20].strip()
                if res_name in ALLOWED_GLYCANS:
                    if len(atom_lines) <= 6:
                        bad_residue_keys.add(res_key)
                        continue
                    if len(atom_lines) > 1:
                        coords = [np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]) for line in atom_lines]
                        tree = cKDTree(np.array(coords))
                        min_distances = tree.query(coords, k=2, workers=-1)[0][:, 1]
                        if np.any(min_distances > 2.0):
                            bad_residue_keys.add(res_key)

            # 6. Determine which full glycan components to remove
            components_to_remove = set()
            if bad_residue_keys:
                for component in glycan_components:
                    if not bad_residue_keys.isdisjoint(component):
                        components_to_remove.add(component)

            glycans_removed_count += len(components_to_remove)

            # 7. Filter the model lines to create the final, clean structure
            final_model_lines = []
            if components_to_remove:
                residues_to_delete = set().union(*components_to_remove)
                for line in resolved_model_lines:
                    residue_key = (line[21:22].strip(), line[22:26].strip(), line[26].strip())
                    if residue_key not in residues_to_delete:
                        final_model_lines.append(line)
            else:
                final_model_lines = resolved_model_lines

            if not any(line[17:20].strip() in AMINO_ACIDS for line in final_model_lines):
                continue

            # 8. Write the cleaned model to a file
            model_suffix = f"_model{model_num}" if len(model_blocks) > 1 else ""
            output_filename = f"{pdb_id}{model_suffix}.pdb"
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)

            with open(output_path, 'w') as fw:
                if has_seqres: fw.writelines(seqres_lines)
                
                def safe_sort_key(line_str):
                    try: res_num = int(line_str[22:26])
                    except ValueError: res_num = 0
                    return (line_str[21:22].strip(), res_num)

                sorted_lines = sorted(final_model_lines, key=safe_sort_key)
                
                for i, line in enumerate(sorted_lines, 1):
                    fw.write(f"{line[:6]}{i:5}{line[11:]}")
                fw.write("END\n")
            
            files_written += 1
            processed_confs += 1

        except Exception as e:
            reason = f"Skipped {filename} (Model {model_num}): Error during processing. See stderr for full debug report."
            skip_reasons.append(reason)
            continue

    # Return aggregated results for the entire file
    return files_written, num_models_in_file, processed_confs, is_multiconf, has_seqres, skip_reasons, file_resorted_to_double_char, filename, glycans_removed_count

def main():
    """ Main execution function """
    print("="*60)
    print(" PDB Protein-Glycan Cleaner ")
    print("="*60)

    if os.path.exists(OUTPUT_FOLDER): shutil.rmtree(OUTPUT_FOLDER)
    os.makedirs(OUTPUT_FOLDER)

    pdb_files = [os.path.join(r, f) for r, _, fs in os.walk(INPUT_FOLDER) for f in fs if f.lower().endswith((".pdb", ".ent"))]
    if not pdb_files:
        print(f"[ERROR] No PDB files found in {INPUT_FOLDER}")
        return

    num_files = len(pdb_files)
    num_workers = multiprocessing.cpu_count()
    print(f"[INFO] Found {num_files} source PDB files. Using {num_workers} workers.\n")

    start_time = time.time()
    
    total_written, total_skipped, total_models, total_confs = 0, 0, 0, 0
    total_multi_conf, total_without_seqres, total_glycans_removed = 0, 0, 0
    debug_reasons = []

    with Pool(processes=num_workers) as pool:
        results_iterator = pool.imap_unordered(process_pdb_file, pdb_files)
        
        # Wrap the iterator with tqdm for a clean progress bar
        pbar = tqdm(results_iterator, total=num_files, desc="Processing PDBs")
        
        for written, models, confs, is_multi, has_seqres, reasons, used_double, fname, glycans_removed in pbar:
            total_models += models
            total_confs += confs
            total_glycans_removed += glycans_removed
            
            if is_multi: total_multi_conf += 1
            
            if written > 0:
                total_written += written
                if not has_seqres:
                    total_without_seqres += 1
            else:
                total_skipped += 1
                if reasons and len(debug_reasons) < 10:
                    debug_reasons.append(reasons[0])

            # Update the tqdm progress bar's postfix with live statistics
            pbar.set_postfix(
                written=f"{total_written}",
                skipped=f"{total_skipped}",
                glycans_removed=f"{total_glycans_removed}",
                refresh=False # Set to False for HPC environments where \r might not work well
            )

    end_time = time.time()
    
    if debug_reasons:
        print("\n--- Sample of Skipped File Reasons ---")
        for reason in debug_reasons: print(f"[DEBUG] {reason}")
    
    print("\n" + "="*60)
    print("--- Processing Complete ---")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    print(f"  - Source PDB files handled: {num_files}")
    print(f"  - Total output files written: {total_written}")
    print(f"  - Source files with no valid structures: {total_skipped}")
    print(f"  - Source files written without SEQRES records: {total_without_seqres}")
    print(f"  - Total glycan components removed: {total_glycans_removed}") # New stat
    
    print("\n--- Conformational Metrics ---")
    print(f"  - Total NMR-style models scanned: {total_models}")
    print(f"  - Total valid conformations written: {total_confs}")
    
    percent_altlocs = (total_multi_conf / num_files) if num_files > 0 else 0
    print(f"  - Source files containing altlocs: {total_multi_conf} ({percent_altlocs:.2%})")

    print(f"  - Average models per source file: {total_models/num_files if num_files > 0 else 0:.2f}")
    print(f"  - Average valid conformations written per source file: {total_confs/num_files if num_files > 0 else 0:.2f}")
    
    print("="*60)

if __name__ == "__main__":
    main()
