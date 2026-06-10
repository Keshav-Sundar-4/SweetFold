#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Glycan-Only File Cleaner & Extractor (No Stereochem Filter)

What this script does:

1) Scans PDB/ENT files in an input folder.
2) Keeps only files that:
   - contain NO standard amino-acid residues, and
   - contain AT LEAST ONE glycan residue from ALLOWED_GLYCANS.
3) Within those files:
   - removes ALL atoms not belonging to ALLOWED_GLYCANS
   - removes ALL hydrogens (robustly)
4) Splits remaining glycan atoms into individual covalently connected glycan
   components using a distance threshold.
5) Writes each glycan component to its own renumbered PDB file.

Notes:
- No stereochemistry / RDKit / CCD logic is used.
- If a file has any amino acid residue anywhere, it is skipped entirely.
"""

import os
import re
import time
import shutil
import argparse
from functools import partial
from collections import Counter, defaultdict
import multiprocessing

import numpy as np
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# --------------------------------------------------------
# Configuration
# --------------------------------------------------------
COVALENT_THRESHOLD = 2.0  # Å; used to split glycan components

# --------------------------------------------------------
# Allowed Residue Definitions
# --------------------------------------------------------
ALLOWED_GLYCANS = set([
    "05L", "07E", "0HX", "0LP", "0MK", "0NZ", "0UB", "0WK", "0XY", "0YT",
    "12E", "145", "147", "149", "14T", "15L", "16F", "16G", "16O", "17T",
    "18D", "18O", "1CF", "1GL", "1GN", "1S3", "1S4", "1SD", "1X4", "20S",
    "20X", "22O", "22S", "23V", "24S", "25E", "26O", "27C", "289", "291",
    "293", "2DG", "2DR", "2F8", "2FG", "2FL", "2GL", "2GS", "2H5", "2M5",
    "2M8", "2WP", "32O", "34V", "38J", "3DO", "3FM", "3HD", "3J3", "3J4",
    "3LJ", "3MG", "3MK", "3R3", "3S6", "3YW", "42D", "445", "44S", "46Z",
    "475", "491", "49A", "49S", "49T", "49V", "4AM", "4CQ", "4GL", "4GP",
    "4JA", "4N2", "4NN", "4QY", "4R1", "4SG", "4U0", "4U1", "4U2", "4UZ",
    "4V5", "50A", "510", "51N", "56N", "57S", "5DI", "5GF", "5GO", "5KQ",
    "5KV", "5L2", "5L3", "5LS", "5LT", "5N6", "5QP", "5TH", "5TJ", "5TK",
    "5TM", "604", "61J", "62I", "64K", "66O", "6BG", "6C2", "6GB", "6GP",
    "6GR", "6K3", "6KH", "6KL", "6KS", "6KU", "6KW", "6LS", "6LW", "6MJ",
    "6MN", "6PY", "6PZ", "6S2", "6UD", "6Y6", "6YR", "6ZC", "73E", "79J",
    "7CV", "7D1", "7GP", "7JZ", "7K2", "7K3", "7NU", "83Y", "89Y", "8B7",
    "8B9", "8EX", "8GA", "8GG", "8GP", "8LM", "8LR", "8OQ", "8PK", "8S0",
    "95Z", "96O", "9AM", "9C1", "9CD", "9GP", "9KJ", "9MR", "9OK", "9PG",
    "9QG", "9QZ", "9S7", "9SG", "9SJ", "9SM", "9SP", "9T1", "9T7", "9VP",
    "9WJ", "9WN", "9WZ", "9YW", "A0K", "A1Q", "A2G", "A5C", "A6P", "AAL",
    "ABD", "ABE", "ABF", "ABL", "AC1", "ACR", "ACX", "ADA", "AF1", "AFD",
    "AFO", "AFP", "AFR", "AGL", "AGR", "AH2", "AH8", "AHG", "AHM", "AHR",
    "AIG", "ALL", "ALX", "AMG", "AMN", "AMU", "AMV", "ANA", "AOG", "AQA",
    "ARA", "ARB", "ARI", "ARW", "ASC", "ASG", "ASO", "AXP", "AXR", "AY9",
    "AZC", "B0D", "B16", "B1H", "B1N", "B6D", "B7G", "B8D", "B9D", "BBK",
    "BBV", "BCD", "BCW", "BDF", "BDG", "BDP", "BDR", "BDZ", "BEM", "BFN",
    "BG6", "BG8", "BGC", "BGL", "BGN", "BGP", "BGS", "BHG", "BM3", "BM7",
    "BMA", "BMX", "BND", "BNG", "BNX", "BXY", "BZD", "C3B", "C3G", "C3X",
    "C4B", "C4W", "C5X", "CBF", "CBI", "CBK", "CDR", "CE5", "CE6", "CE8",
    "CEG", "CEX", "CEY", "CEZ", "CGF", "CJB", "CKB", "CKP", "CNP", "CR1",
    "CR6", "CRA", "CT3", "CTO", "CTR", "CTT", "D0N", "D1M", "D5E", "D6G",
    "DAF", "DAG", "DAN", "DDA", "DDL", "DEG", "DEL", "DFR", "DFX", "DGO",
    "DGS", "DJB", "DJE", "DK4", "DKX", "DKZ", "DL6", "DLD", "DLF", "DLG",
    "DO8", "DOM", "DPC", "DQR", "DR2", "DR3", "DR5", "DRI", "DSR", "DT6",
    "DVC", "DYM", "E3M", "E5G", "EAG", "EBG", "EBQ", "EEN", "EEQ", "EGA",
    "EMP", "EMZ", "EPG", "EQP", "EQV", "ERE", "ERI", "ETT", "F1P", "F1X",
    "F55", "F58", "F6P", "FBP", "FCA", "FCB", "FCT", "FDP", "FDQ", "FFC",
    "FFX", "FIF", "FK9", "FKD", "FMF", "FMO", "FNG", "FNY", "FRU", "FSA",
    "FSI", "FSM", "FSR", "FSW", "FUB", "FUC", "FUF", "FUL", "FUY", "FVQ",
    "FX1", "FYJ", "G0S", "G16", "G1P", "G20", "G28", "G2F", "G3F", "G4D",
    "G4S", "G6D", "G6P", "G6S", "G7P", "G8Z", "GAA", "GAC", "GAD", "GAF",
    "GAL", "GAT", "GBH", "GC1", "GC4", "GC9", "GCB", "GCD", "GCN", "GCO",
    "GCS", "GCT", "GCU", "GCV", "GCW", "GDA", "GDL", "GE1", "GE3", "GFP",
    "GIV", "GL0", "GL1", "GL2", "GL4", "GL5", "GL6", "GL7", "GL9", "GLA",
    "GLC", "GLD", "GLF", "GLG", "GLO", "GLP", "GLS", "GLT", "GM0", "GMB",
    "GMH", "GMT", "GMZ", "GN1", "GN4", "GNS", "GNX", "GP0", "GP1", "GP4",
    "GPH", "GPK", "GPM", "GPO", "GPQ", "GPU", "GPV", "GPW", "GQ1", "GRF",
    "GRX", "GS1", "GS9", "GTK", "GTM", "GTR", "GU0", "GU1", "GU2", "GU3",
    "GU4", "GU5", "GU6", "GU8", "GU9", "GUF", "GUL", "GUP", "GUZ", "GXL",
    "GYE", "GYG", "GYP", "GYU", "GYV", "GZL", "H1M", "H1S", "H2P", "H53",
    "H6Q", "H6Z", "HBZ", "HD4", "HNV", "HNW", "HSG", "HSH", "HSJ", "HSQ",
    "HSX", "HSY", "HTG", "HTM", "I57", "IAB", "IDC", "IDF", "IDG", "IDR",
    "IDS", "IDU", "IDX", "IDY", "IEM", "IN1", "IPT", "ISD", "ISL", "ISX",
    "IXD", "J5B", "JFZ", "JHM", "JLT", "JS2", "JV4", "JVA", "JVS", "JZR",
    "K5B", "K99", "KBA", "KBG", "KD5", "KDA", "KDB", "KDD", "KDE", "KDF",
    "KDM", "KDN", "KDO", "KDR", "KFN", "KG1", "KGM", "KHP", "KME", "KO1",
    "KO2", "KOT", "KTU", "L1L", "L6S", "LAH", "LAK", "LAO", "LAT", "LB2",
    "LBS", "LBT", "LCN", "LDY", "LEC", "LFR", "LGC", "LGU", "LKA", "LKS",
    "LNV", "LOG", "LOX", "LRH", "LVO", "LVZ", "LXB", "LXC", "LXZ", "LZ0",
    "M1F", "M1P", "M2F", "M3N", "M55", "M6D", "M6P", "M7B", "M7P", "M8C",
    "MA1", "MA2", "MA3", "MA8", "MAF", "MAG", "MAL", "MAN", "MAT", "MAV",
    "MAW", "MBE", "MBF", "MBG", "MCU", "MDA", "MDP", "MFB", "MFU", "MG5",
    "MGC", "MGL", "MGS", "MJJ", "MLB", "MLR", "MMA", "MN0", "MNA", "MQG",
    "MQT", "MRH", "MRP", "MSX", "MTT", "MUB", "MUR", "MVP", "MXY", "MXZ",
    "MYG", "N1L", "N9S", "NA1", "NAA", "NAG", "NBG", "NBX", "NBY", "NDG",
    "NFG", "NG1", "NG6", "NGA", "NGC", "NGE", "NGK", "NGR", "NGS", "NGY",
    "NGZ", "NHF", "NLC", "NM6", "NM9", "NNG", "NPF", "NSQ", "NT1", "NTF",
    "NTO", "NTP", "NXD", "NYT", "O1G", "OAK", "OEL", "OI7", "OPM", "OSU",
    "OTG", "OTN", "OTU", "OX2", "P53", "P6P", "PA1", "PAV", "PDX", "PH5",
    "PKM", "PNA", "PNG", "PNJ", "PNW", "PPC", "PRP", "PSG", "PSV", "PUF",
    "PZU", "QIF", "QKH", "QPS", "R1P", "R1X", "R2B", "R2G", "RAE", "RAF",
    "RAM", "RAO", "RCD", "RER", "RF5", "RGG", "RHA", "RHC", "RI2", "RIB",
    "RIP", "RM4", "RP3", "RP5", "RP6", "RR7", "RRJ", "RRY", "RST", "RTG",
    "RTV", "RUG", "RUU", "RV7", "RVG", "RVM", "RWI", "RY7", "RZM", "S7P",
    "S81", "SA0", "SCG", "SCR", "SDY", "SEJ", "SF6", "SF9", "SFJ", "SFU",
    "SG4", "SG5", "SG6", "SG7", "SGA", "SGC", "SGD", "SGN", "SHB", "SHD",
    "SHG", "SIA", "SID", "SIO", "SIZ", "SLB", "SLM", "SLT", "SMD", "SN5",
    "SNG", "SOE", "SOG", "SOR", "SR1", "SSG", "STZ", "SUC", "SUP", "SUS",
    "SWE", "SZZ", "T68", "T6P", "T6T", "TA6", "TCB", "TCG", "TDG", "TEU",
    "TF0", "TFU", "TGA", "TGK", "TGR", "TGY", "TH1", "TMR", "TMX", "TNX",
    "TOA", "TOC", "TQY", "TRE", "TRV", "TS8", "TT7", "TTV", "TTZ", "TU4",
    "TUG", "TUJ", "TUP", "TUR", "TVD", "TVG", "TVM", "TVS", "TVV", "TVY",
    "TW7", "TWA", "TWD", "TWG", "TWJ", "TWY", "TXB", "TYV", "U1Y", "U2A",
    "U2D", "U63", "U8V", "U97", "U9A", "U9D", "U9G", "U9J", "U9M", "UAP",
    "UCD", "UDC", "UEA", "V3M", "V3P", "V71", "VG1", "VTB", "W9T", "WIA",
    "WOO", "WUN", "X0X", "X1P", "X1X", "X2F", "X6X", "XDX", "XGP", "XIL",
    "XLF", "XLS", "XMM", "XXM", "XXR", "XXX", "XYF", "XYL", "XYP", "XYS",
    "XYT", "XYZ", "YIO", "YJM", "YKR", "YO5", "YX0", "YX1", "YYB", "YYH",
    "YYJ", "YYK", "YYM", "YYQ", "YZ0", "Z0F", "Z15", "Z16", "Z2D", "Z2T",
    "Z3K", "Z3L", "Z3Q", "Z3U", "Z4K", "Z4R", "Z4S", "Z4U", "Z4V", "Z4W",
    "Z4Y", "Z57", "Z5J", "Z5L", "Z61", "Z6H", "Z6J", "Z6W", "Z8H", "Z8T",
    "Z9D", "Z9E", "Z9H", "Z9K", "Z9L", "Z9M", "Z9N", "Z9W", "ZB0", "ZB1",
    "ZB2", "ZB3", "ZCD", "ZCZ", "ZD0", "ZDC", "ZDO", "ZEE", "ZEL", "ZGE",
    "ZMR"
])

AMINO_ACIDS = set([
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"
])

# --------------------------------------------------------
# Helper & I/O Functions
# --------------------------------------------------------
def is_hydrogen_atom_line(line: str) -> bool:
    """
    Robust hydrogen detector for PDB ATOM/HETATM lines.
    - Checks atom name patterns (H, 1H, 2H, H1, etc).
    - Checks the element column (77-78).
    Removes deuterium too (D).
    """
    atom_name = line[12:16].strip()
    element = line[76:78].strip().upper() if len(line) >= 78 else ""

    if element in ("H", "D"):
        return True
    if re.match(r'^\d*H', atom_name.upper()):  # e.g., 1H, 2H, H...
        return True
    if atom_name.upper().startswith("H"):
        return True
    if atom_name.upper().startswith("D"):
        return True
    return False


def atom_sort_key(atom_dict):
    try:
        numeric_res_num = int(re.sub(r'[A-Z]', '', atom_dict['res_num']))
    except (ValueError, TypeError):
        numeric_res_num = 0
    return (atom_dict['chain'], numeric_res_num, atom_dict['atom_number'])


def reformat_and_renumber_atoms(sorted_atom_list):
    """
    Formats atom dictionaries into PDB-compliant strings, renumbering sequentially.
    Uses corrected element derivation.
    """
    new_lines = []
    for i, atom in enumerate(sorted_atom_list, 1):
        record_type = "HETATM"  # glycans are hetero residues
        atom_name_str = atom['atom_name']
        formatted_atom_name = atom_name_str if len(atom_name_str) == 4 else f" {atom_name_str:<3}"

        element = ''.join(filter(str.isalpha, atom_name_str))
        element = element[0] if element else ' '

        line_out = (
            f"{record_type:<6}{i:5} {formatted_atom_name} {atom['res_name']:>3} "
            f"{atom['chain']:1}{atom['res_num']:>4}    "
            f"{atom['coords'][0]:8.3f}{atom['coords'][1]:8.3f}{atom['coords'][2]:8.3f}"
            f"{1.00:6.2f}{0.00:6.2f}          {element:>2}"
        )

        new_lines.append(f"{line_out:<80}")
    return new_lines


def get_unique_filename(output_folder, base_filename):
    base_path = os.path.join(output_folder, base_filename)
    if not os.path.exists(base_path):
        return base_path
    name, ext = os.path.splitext(base_filename)
    n = 1
    while True:
        candidate = os.path.join(output_folder, f"{name}_v{n}{ext}")
        if not os.path.exists(candidate):
            return candidate
        n += 1


def split_pdb_by_model(lines):
    if not any(line.startswith("MODEL") for line in lines):
        return [lines]

    model_blocks, header_lines, current_model_lines = [], [], []
    in_header, in_model = True, False

    for line in lines:
        if line.startswith("MODEL"):
            in_header, in_model = False, True
            current_model_lines = []
        elif line.startswith("ENDMDL"):
            if current_model_lines:
                model_blocks.append(header_lines + current_model_lines)
            in_model = False
        elif in_header:
            header_lines.append(line)
        elif in_model:
            current_model_lines.append(line)

    return model_blocks if model_blocks else [lines]


# --------------------------------------------------------
# Core Processing Logic
# --------------------------------------------------------
def process_pdb_worker_glycan_only(filepath, output_folder):
    """
    Worker that:
    - skips file if any amino acid residue is present
    - skips file if no glycan residue is present
    - otherwise extracts glycan-only components and writes each to PDB
    """
    try:
        with open(filepath, 'r', encoding='latin-1') as f:
            all_lines = f.readlines()
    except Exception:
        return Counter()

    pdb_id = os.path.basename(filepath).split('.')[0][:4].upper()
    model_line_blocks = split_pdb_by_model(all_lines)
    local_counter = Counter()

    for model_num, lines in enumerate(model_line_blocks, 1):
        # Determine altLocs
        all_alt_locs = {
            line[16] for line in lines
            if line.startswith(("ATOM", "HETATM")) and line[16].strip()
        }
        all_alt_locs.add(' ')
        conformations = [' '] if len(all_alt_locs) == 1 else sorted([loc for loc in all_alt_locs if loc.strip()])

        for conf_id in conformations:
            glycan_atoms = []
            found_amino_acid = False
            found_glycan = False

            for line in lines:
                if not line.startswith(("ATOM", "HETATM")):
                    continue
                if line[16] not in (' ', conf_id):
                    continue

                # Remove hydrogens robustly
                if is_hydrogen_atom_line(line):
                    continue

                res_name = line[17:20].strip()

                if res_name in AMINO_ACIDS:
                    found_amino_acid = True
                    break  # file+model+conf invalid, stop early

                if res_name in ALLOWED_GLYCANS:
                    found_glycan = True
                    atom_name = line[12:16].strip()

                    try:
                        atom_dict = {
                            'res_name': res_name,
                            'chain': line[21:22].strip(),
                            'res_num': line[22:26].strip(),
                            'atom_name': atom_name,
                            'coords': np.array([
                                float(line[30:38]),
                                float(line[38:46]),
                                float(line[46:54])
                            ]),
                            'atom_number': int(line[6:11])
                        }
                        glycan_atoms.append(atom_dict)
                    except (ValueError, IndexError):
                        continue

            # Apply file-level filter
            if found_amino_acid:
                local_counter['files_skipped_contains_amino_acids'] += 1
                continue
            if not found_glycan or not glycan_atoms:
                local_counter['files_skipped_no_glycans'] += 1
                continue

            # Split into covalent components
            glycan_coords = np.array([a['coords'] for a in glycan_atoms])
            kdtree_glycan = KDTree(glycan_coords)
            connections = kdtree_glycan.query_pairs(r=COVALENT_THRESHOLD)

            if connections:
                rows, cols = zip(*connections)
                data = np.ones(len(connections))
                adj_matrix = csr_matrix((data, (rows, cols)), shape=(len(glycan_atoms), len(glycan_atoms)))
            else:
                adj_matrix = csr_matrix((len(glycan_atoms), len(glycan_atoms)))

            n_components, labels = connected_components(csgraph=adj_matrix, directed=False, return_labels=True)

            glycan_components = [
                [g for i, g in enumerate(glycan_atoms) if labels[i] == j]
                for j in range(n_components)
            ]

            # Write each component
            for comp_idx, comp_atoms in enumerate(glycan_components, 1):
                final_atoms = sorted(comp_atoms, key=atom_sort_key)
                lines_out = reformat_and_renumber_atoms(final_atoms)

                model_suffix = f"_model{model_num}" if len(model_line_blocks) > 1 else ""
                conf_suffix = f"_conf{conf_id}" if conf_id.strip() else ""
                comp_suffix = f"_comp{comp_idx}" if n_components > 1 else ""

                base_name = f"{pdb_id}_glycan_only{model_suffix}{conf_suffix}{comp_suffix}.pdb"
                out_path = get_unique_filename(output_folder, base_name)

                with open(out_path, "w") as fw:
                    fw.write("\n".join(lines_out) + "\nEND\n")

                local_counter['glycan_components_written'] += 1

    return local_counter


# --------------------------------------------------------
# Main Execution Block
# --------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Glycan-Only File Cleaner & Extractor (No stereochem filter).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input_folder", type=str, required=True,
                        help="Path to the folder with raw PDB/ENT files.")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Path to the folder to save cleaned glycan-only PDB files.")
    args = parser.parse_args()

    if os.path.exists(args.output_folder):
        shutil.rmtree(args.output_folder)
    os.makedirs(args.output_folder)

    pdb_files = [
        os.path.join(r, f)
        for r, _, fs in os.walk(args.input_folder)
        for f in fs
        if f.lower().endswith((".pdb", ".ent"))
    ]

    if not pdb_files:
        print("[ERROR] No PDB/ENT files found.")
        return

    try:
        num_workers = len(os.sched_getaffinity(0))
    except AttributeError:
        num_workers = os.cpu_count() or 1

    print("=" * 70)
    print("  Glycan-Only File Cleaner & Extractor (No stereochem filter) ")
    print("=" * 70)
    print(f"Found {len(pdb_files)} files. Processing with {num_workers} workers...\n")

    start = time.time()
    worker_func = partial(process_pdb_worker_glycan_only, output_folder=args.output_folder)
    summary = Counter()

    with multiprocessing.Pool(processes=num_workers) as pool:
        for i, cnt in enumerate(pool.imap_unordered(worker_func, pdb_files), 1):
            summary.update(cnt)
            progress = i / len(pdb_files)
            bar = '█' * int(40 * progress) + '-' * (40 - int(40 * progress))
            print(f"\rProgress: |{bar}| {i}/{len(pdb_files)} ({progress:.1%})",
                  end="", flush=True)

    elapsed = time.time() - start
    print("\n\n--- Processing Complete ---")
    print(f"Total time: {elapsed:.2f} seconds")
    print("=" * 60)
    print("Final Summary:")
    print(f"  - Glycan components written: {summary.get('glycan_components_written', 0)}")
    print(f"  - Files skipped (contained amino acids): {summary.get('files_skipped_contains_amino_acids', 0)}")
    print(f"  - Files skipped (no glycans): {summary.get('files_skipped_no_glycans', 0)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
