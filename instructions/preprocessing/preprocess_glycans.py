# pre_process_glycan.py

import argparse
import json
import multiprocessing
import pickle
import traceback
import re
import os
import time
import sys
import site
from dataclasses import asdict, dataclass, replace
from collections import defaultdict, deque
from functools import partial
from pathlib import Path
from typing import Any, Optional, Dict, List, Tuple, Mapping, Set
from scipy.spatial import cKDTree
import random
from contextlib import redirect_stdout, redirect_stderr
import string

import numpy as np
import rdkit
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol, Conformer

import boltz
from boltz.data import const
from boltz.data.feature.featurizer import MONO_TYPE_MAP
from boltz.data.types import (
    Atom, Bond, Residue, Chain, Connection, Interface, Structure, StructureInfo,
    Record, Target, ChainInfo, InterfaceInfo, GlycosylationSite
    # We don't need MSA types here
)
# Import necessary helper functions (adapt paths if needed)
from boltz.data.parse.schema import (
    get_conformer, convert_atom_name, parse_ccd_residue,
)
# Need ParsedAtom, ParsedBond, ParsedResidue, ParsedChain for intermediate representation
#from boltz.data.parse.mmcif import ParsedAtom, ParsedBond, ParsedResidue, ParsedChain


from tqdm import tqdm

@dataclass(frozen=True, slots=True)
class ParsedAtom:
    """A parsed atom object."""
    name: str
    element: int
    charge: int
    coords: tuple[float, float, float]
    conformer: tuple[float, float, float]
    is_present: bool
    chirality: int

@dataclass(frozen=True, slots=True)
class ParsedBond:
    """A parsed bond object."""
    atom_1: int
    atom_2: int
    type: int

@dataclass(frozen=True, slots=True)
class ParsedResidue:
    """A parsed residue object."""
    name: str
    type: int
    idx: int
    atoms: list[ParsedAtom]
    bonds: list[ParsedBond]
    orig_idx: Optional[int]
    atom_center: int
    atom_disto: int
    is_standard: bool
    is_present: bool

@dataclass(frozen=True, slots=True)
class ParsedChain:
    """A parsed chain object."""
    name: str
    entity: str
    type: str
    residues: list[ParsedResidue]
    sequence: list[str]

# Define connection types BEFORE they are used in ParsedGlycoproteinData
@dataclass(frozen=True, slots=True)
class GlycoConnection:
    """Represents a detected glycosidic linkage."""
    parent_chain_id: str
    child_chain_id: str
    parent_res_id: int
    child_res_id: int
    parent_acceptor_atom_name: str # e.g., "O4"
    child_donor_atom_name: str     # e.g., "C1"
    anomeric: Optional[str]        # 'a', 'b', or None

@dataclass(frozen=True, slots=True)
class GlycosylationSiteConnection:
    """Represents a detected protein-glycan linkage."""
    protein_chain_id: str
    protein_res_id: int
    protein_atom_name: str
    glycan_chain_id: str
    glycan_res_id: int
    glycan_atom_name: str
    anomeric: Optional[str] = None

# Now, define the main data container, which can safely reference the classes above.
@dataclass
class ParsedGlycoproteinData:
    """Intermediate storage for parsed glycoprotein data."""
    pdb_id: str
    chains: Dict[str, ParsedChain]
    glycosidic_connections: List[GlycoConnection]
    glycosylation_sites: List[GlycosylationSiteConnection]

@dataclass(frozen=True, slots=True)
class PDBFile:
    """Represents a raw PDB input file."""
    id: str
    path: Path
    cluster_num: int
    frame_num: int

@dataclass(frozen=True)
class AnomericAtom:
    """Represents an atom for anomeric configuration analysis."""
    idx: int
    name: str
    res_name: str
    coords: np.ndarray
    element: str



worker_ccd_data = None # Global variable placeholder for worker processes

standard_amino_acids = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
    "TYR", "VAL"
}

ATOMIC_NUMBERS = { 'H': 1, 'HE': 2, 'LI': 3, 'BE': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'NE': 10, 'NA': 11, 'MG': 12, 'AL': 13, 'SI': 14, 'P': 15, 'S': 16, 'CL': 17, 'AR': 18, 'K': 19, 'CA': 20, 'SC': 21, 'TI': 22, 'V': 23, 'CR': 24, 'MN': 25, 'FE': 26, 'CO': 27, 'NI': 28, 'CU': 29, 'ZN': 30, 'GA': 31, 'GE': 32, 'AS': 33, 'SE': 34, 'BR': 35, 'KR': 36, 'RB': 37, 'SR': 38, 'Y': 39, 'ZR': 40, 'NB': 41, 'MO': 42, 'TC': 43, 'RU': 44, 'RH': 45, 'PD': 46, 'AG': 47, 'CD': 48, 'IN': 49, 'SN': 50, 'SB': 51, 'TE': 52, 'I': 53, 'XE': 54, 'CS': 55, 'BA': 56, 'LA': 57, 'CE': 58, 'PR': 59, 'ND': 60, 'PM': 61, 'SM': 62, 'EU': 63, 'GD': 64, 'TB': 65, 'DY': 66, 'HO': 67, 'ER': 68, 'TM': 69, 'YB': 70, 'LU': 71, 'HF': 72, 'TA': 73, 'W': 74, 'RE': 75, 'OS': 76, 'IR': 77, 'PT': 78, 'AU': 79, 'HG': 80, 'TL': 81, 'PB': 82, 'BI': 83, 'PO': 84, 'AT': 85, 'RN': 86 }


# --- Constants ---
GLYCOSIDIC_BOND_THRESHOLD = 2.0 # Angstrom distance to detect potential glycosidic linkages
BOND_TYPE_SINGLE = const.bond_type_ids.get("SINGLE", 1) # Default to 1 if not found

class NumpyJSONEncoder(json.JSONEncoder):
    """
    A custom JSON encoder that converts NumPy data types to native Python types,
    making them serializable.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyJSONEncoder, self).default(obj)

# These dataclasses are temporary, for use by the IUPAC generation logic.
@dataclass(frozen=True)
class AnomericTestAtom:
    idx: int; name: str; res_name: str; chain_id: str; res_num: int; element: str; coords: np.ndarray

@dataclass(frozen=True)
class AnomericTestGlycoConnection:
    parent_res_key: Tuple[str, int]; child_res_key: Tuple[str, int]; parent_acceptor_atom: AnomericTestAtom
    child_donor_atom: AnomericTestAtom; anomeric_config: Optional[str]

def _generate_chain_ids(exclude=None):
    """
    Efficiently generates a sequence of unique chain IDs, skipping any
    that are in the 'exclude' set (e.g., existing protein chains).
    """
    if exclude is None:
        exclude = set()

    # 1. Uppercase letters (A-Z)
    for char in string.ascii_uppercase:
        if char not in exclude: yield char
        
    # 2. Lowercase letters (a-z)
    for char in string.ascii_lowercase:
        if char not in exclude: yield char
        
    # 3. Digits (0-9)
    for char in string.digits:
        if char not in exclude: yield char
    
    # 4. Two-character combinations (for > 62 chains)
    two_char_pool = string.ascii_uppercase + string.digits
    for char1 in two_char_pool:
        for char2 in two_char_pool:
            combo = char1 + char2
            if combo not in exclude: yield combo

def _rechain_glycan_components(pdb_lines: List[str], inter_residue_threshold: float = 2.0) -> List[str]:
    """
    (PORTED LOGIC) Separates glycan trees into new chains based on residue-level
    proximity, ensuring new Chain IDs do not conflict with existing protein chains.
    """
    if not pdb_lines:
        return []

    # 1. Group atoms by residue, identify glycans, and find used protein chains
    atoms_by_residue = defaultdict(list)
    protein_chain_ids = set()

    for line in pdb_lines:
        res_name = line[17:20].strip()
        chain_id = line[21]
        
        # Use MONO_TYPE_MAP as the definition of a glycan for consistency
        is_glycan = res_name in MONO_TYPE_MAP and MONO_TYPE_MAP[res_name] != "OTHER"
        
        if not is_glycan and res_name in standard_amino_acids:
            protein_chain_ids.add(chain_id)

        # Key: (chain, res_num, insertion_code)
        residue_key = (line[21], line[22:26].strip(), line[26])
        atoms_by_residue[residue_key].append({
            'line': line,
            'coords': np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
            'is_glycan': is_glycan
        })
    
    glycan_residue_keys = [key for key, atoms in atoms_by_residue.items() if atoms and atoms[0]['is_glycan']]

    if not glycan_residue_keys:
        return pdb_lines # No glycans to rechain, return original lines

    # 2. Build inter-residue graph for GLYCANS ONLY
    flat_glycan_atoms = []
    atom_idx_to_res_key = []
    for res_key in glycan_residue_keys:
        for atom in atoms_by_residue[res_key]:
            # Only consider heavy atoms for linkage detection
            if not atom['line'][12:16].strip().startswith('H'):
                flat_glycan_atoms.append(atom['coords'])
                atom_idx_to_res_key.append(res_key)
    
    adj_residues = defaultdict(set)
    if flat_glycan_atoms:
        tree = cKDTree(np.array(flat_glycan_atoms))
        pairs = tree.query_pairs(r=inter_residue_threshold)
        for i, j in pairs:
            res_key1 = atom_idx_to_res_key[i]
            res_key2 = atom_idx_to_res_key[j]
            if res_key1 != res_key2:
                adj_residues[res_key1].add(res_key2)
                adj_residues[res_key2].add(res_key1)

    # 3. Find connected components and assign new, unique chain IDs
    visited_residues = set()
    chain_id_gen = _generate_chain_ids(exclude=protein_chain_ids)
    final_lines = []
    processed_glycan_residue_keys = set()

    for res_key in glycan_residue_keys:
        if res_key not in visited_residues:
            component_residue_keys = set()
            q = deque([res_key])
            visited_residues.add(res_key)
            
            while q:
                current_res_key = q.popleft()
                component_residue_keys.add(current_res_key)
                for neighbor_res_key in adj_residues.get(current_res_key, []):
                    if neighbor_res_key not in visited_residues:
                        visited_residues.add(neighbor_res_key)
                        q.append(neighbor_res_key)
            
            new_chain_id = next(chain_id_gen)

            for key in component_residue_keys:
                processed_glycan_residue_keys.add(key)
                for atom in atoms_by_residue[key]:
                    line = atom['line']
                    # PDB format handles 1 and 2 character chain IDs differently
                    if len(new_chain_id) == 1:
                        new_line = f"{line[:21]}{new_chain_id}{line[22:]}"
                    else: # Two-character chain ID
                        new_line = f"{line[:20]}{new_chain_id.ljust(2)}{line[22:]}"
                    final_lines.append(new_line)

    # 4. Append all original non-glycan lines
    for res_key, atoms_data in atoms_by_residue.items():
        if res_key not in processed_glycan_residue_keys:
            for atom in atoms_data:
                final_lines.append(atom['line'])
                
    return final_lines

def parse_pdb_atoms_by_residue(pdb_lines: List[str]) -> Tuple[Dict[Tuple[str, int], List[Dict]], Dict[str, str]]:
    """
    (MODIFIED) Parses a list of PDB lines, grouping atoms by their 
    (chain_id, residue_number) key. It also determines the type of each chain
    (PROTEIN or GLYCAN).

    Args:
        pdb_lines: A list of strings, where each string is a line from a PDB file.

    Returns:
        A tuple containing:
        - A dictionary mapping residue keys to lists of atom dictionaries.
        - A dictionary mapping chain IDs to their determined type ('PROTEIN' or 'GLYCAN').
    """
    atoms_by_residue = defaultdict(list)
    chain_has_protein = defaultdict(bool)
    chain_has_glycan = defaultdict(bool)

    for line in pdb_lines:
        if line.startswith(('ATOM', 'HETATM')):
            try:
                res_name = line[17:20].strip().upper()
                # Correctly parse chain IDs that may now be two characters
                chain_id = line[20:22].strip() or "A" # Default to 'A' if chain is blank
                res_num = int(line[22:26].strip())
                res_key = (chain_id, res_num)

                atom_dict = {
                    'atom_number':   int(line[6:11].strip()),
                    'atom_name':     line[12:16].strip(),
                    'residue_name':  res_name,
                    'chain_id':      chain_id,
                    'residue_num':   res_num,
                    'x': float(line[30:38].strip()),
                    'y': float(line[38:46].strip()),
                    'z': float(line[46:54].strip()),
                    'element': (line[76:78].strip() or line[12:16].strip()[0]).upper()
                }
                atoms_by_residue[res_key].append(atom_dict)

                if res_name in standard_amino_acids:
                    chain_has_protein[chain_id] = True
                elif res_name in MONO_TYPE_MAP and MONO_TYPE_MAP[res_name] != "OTHER":
                    chain_has_glycan[chain_id] = True

            except (ValueError, IndexError):
                continue

    chain_types = {}
    all_chains = set(chain_has_protein.keys()) | set(chain_has_glycan.keys())
    for chain_id in all_chains:
        if chain_has_protein[chain_id]:
            chain_types[chain_id] = "PROTEIN"
        elif chain_has_glycan[chain_id]:
            chain_types[chain_id] = "GLYCAN"

    return dict(atoms_by_residue), chain_types

def detect_all_connections(
    atoms_by_residue: Dict[Tuple[str, int], List[Dict]],
    pdb_id: str = "UNKNOWN"
) -> Tuple[List[GlycoConnection], List[GlycosylationSiteConnection], Dict[str, Dict[str, int]], List[Tuple[str, str, int, str]], List[Tuple[str, str, int]]]:
    """
    (CORRECTED) Detects linkages.
    - Enforces Intra-chain only for Glycan-Glycan.
    - Prioritizes valid configs > plausible errors > distance errors.
    """
    threshold = 2.0
    
    # Key: Tuple[sorted((chain1, res1), (chain2, res2))]
    # Value: GlycoConnection
    best_glycosidic_conns: Dict[Tuple[Tuple[str, int], Tuple[str, int]], GlycoConnection] = {}
    
    glycosylation_sites_temp: List[GlycosylationSiteConnection] = []
    site_stats = defaultdict(lambda: defaultdict(int))
    anomalous_sites = []
    no_ring_errors = []

    flat_atoms = []
    all_coords = []
    
    is_standard_aa = {}
    for key, atoms in atoms_by_residue.items():
        if atoms:
            is_standard_aa[key] = atoms[0]['residue_name'] in standard_amino_acids

    for res_key, atoms in atoms_by_residue.items():
        if not atoms: continue
        is_aa = is_standard_aa.get(res_key, False)
        for atom_dict in atoms:
            if atom_dict['atom_name'].startswith('H'): continue
            coords = np.array([atom_dict['x'], atom_dict['y'], atom_dict['z']])
            all_coords.append(coords)
            flat_atoms.append({'res_key': res_key, 'is_aa': is_aa, 'atom_dict': atom_dict})

    if len(flat_atoms) < 2: return [], [], {}, [], []

    tree = cKDTree(np.array(all_coords))
    nearby_pairs = tree.query_pairs(r=threshold, output_type='set')

    for i, j in nearby_pairs:
        atom1_info, atom2_info = flat_atoms[i], flat_atoms[j]
        res1_key, is_aa1 = atom1_info['res_key'], atom1_info['is_aa']
        res2_key, is_aa2 = atom2_info['res_key'], atom2_info['is_aa']

        if res1_key == res2_key: continue
        
        # Skip Protein-Protein (irrelevant for glycan logic)
        if is_aa1 and is_aa2: continue

        # --- Element-Based Directionality ---
        a1 = atom1_info['atom_dict']
        a2 = atom2_info['atom_dict']
        el1 = a1['element'].upper()
        el2 = a2['element'].upper()
        
        donor_info, acceptor_info = None, None
        donor_atom, acceptor_atom = None, None
        
        if el1 == 'C' and el2 != 'C':
            donor_info, acceptor_info = atom1_info, atom2_info
            donor_atom, acceptor_atom = a1, a2
        elif el2 == 'C' and el1 != 'C':
            donor_info, acceptor_info = atom2_info, atom1_info
            donor_atom, acceptor_atom = a2, a1
        else:
            continue

        # Case 1: Protein-Glycan linkage
        if donor_info['is_aa'] == False and acceptor_info['is_aa'] == True:
            glycan_info = donor_info
            protein_info = acceptor_info
            
            config_result = determine_anomeric_config_universal(
                child_residue_dicts=atoms_by_residue[glycan_info['res_key']],
                acceptor_atom_dict=acceptor_atom,
                donor_atom_dict=donor_atom
            )
            
            final_anom = config_result if config_result in ['a', 'b'] else "other"
            
            # Stats logging
            p_name = protein_info['atom_dict']['residue_name']
            p_id = protein_info['res_key'][1]
            site_stats[p_name][final_anom] += 1
            
            if 'no ring' in config_result:
                no_ring_errors.append((pdb_id, p_name, p_id))

            glycosylation_sites_temp.append(GlycosylationSiteConnection(
                protein_chain_id=protein_info['res_key'][0], protein_res_id=protein_info['res_key'][1],
                protein_atom_name=acceptor_atom['atom_name'],
                glycan_chain_id=glycan_info['res_key'][0], glycan_res_id=glycan_info['res_key'][1],
                glycan_atom_name=donor_atom['atom_name'],
                anomeric=config_result 
            ))

        # Case 2: Monosaccharide-Monosaccharide linkage
        elif not donor_info['is_aa'] and not acceptor_info['is_aa']:
            # STRICT FILTER: Glycan-Glycan bonds must be Intra-Chain
            # (Assuming rechaining logic has already grouped connected glycans)
            if donor_info['res_key'][0] != acceptor_info['res_key'][0]:
                continue

            config_result = determine_anomeric_config_universal(
                child_residue_dicts=atoms_by_residue[donor_info['res_key']],
                acceptor_atom_dict=acceptor_atom,
                donor_atom_dict=donor_atom
            )
            
            new_conn = GlycoConnection(
                parent_chain_id=acceptor_info['res_key'][0], child_chain_id=donor_info['res_key'][0],
                parent_res_id=acceptor_info['res_key'][1], child_res_id=donor_info['res_key'][1],
                parent_acceptor_atom_name=acceptor_atom['atom_name'],
                child_donor_atom_name=donor_atom['atom_name'], 
                anomeric=config_result
            )
            
            pair_key = tuple(sorted(((acceptor_info['res_key']), (donor_info['res_key']))))
            
            if pair_key not in best_glycosidic_conns:
                best_glycosidic_conns[pair_key] = new_conn
            else:
                existing = best_glycosidic_conns[pair_key]
                existing_valid = existing.anomeric in ['a', 'b']
                new_valid = new_conn.anomeric in ['a', 'b']
                
                # Prioritization Logic
                if new_valid and not existing_valid:
                    # Always prefer valid config over error
                    best_glycosidic_conns[pair_key] = new_conn
                elif not new_valid and not existing_valid:
                    # Both are errors. Prefer the one that is NOT "too far".
                    # "too far" indicates a random steric contact.
                    # Other errors (e.g. "no ring") indicate the correct bond but bad topology.
                    existing_too_far = "too far" in str(existing.anomeric)
                    new_too_far = "too far" in str(new_conn.anomeric)
                    
                    if existing_too_far and not new_too_far:
                        # Upgrade from a distance error to a topology error (implies correct atoms found)
                        best_glycosidic_conns[pair_key] = new_conn

    unique_glycosidic_conns = list(best_glycosidic_conns.values())
            
    return unique_glycosidic_conns, glycosylation_sites_temp, dict(site_stats), anomalous_sites, no_ring_errors

def find_atom_by_name(atoms: List[Dict], name: str) -> Optional[Dict]:
    """Finds the first atom matching the name (case-insensitive)."""
    for atom in atoms:
        if atom['atom_name'].upper() == name.upper():
            return atom
    return None


def build_residue_graph(atoms: List[AnomericAtom]) -> Dict[int, List[int]]:
    """Builds a covalent bond graph for a list of atoms using a distance threshold."""
    COVALENT_BOND_THRESHOLD_INTERNAL = 2.0  # CHANGED AS REQUESTED
    graph = defaultdict(list)
    if len(atoms) < 2:
        return graph
    coords = np.array([a.coords for a in atoms])
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=COVALENT_BOND_THRESHOLD_INTERNAL)
    for i, j in pairs:
        graph[atoms[i].idx].append(atoms[j].idx)
        graph[atoms[j].idx].append(atoms[i].idx)
    return graph

def get_atom_mass(element: str) -> float:
    """
    Returns heuristic mass for priority sorting: O > N > C > others.
    """
    table = {'O': 16.0, 'N': 14.0, 'C': 12.0, 'S': 32.0, 'P': 31.0, 'H': 1.0}
    return table.get(element.upper(), 0.0)

def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Calculates the angle (in degrees) between three points p1-p2-p3, 
    where p2 is the center vertex.
    """
    v1 = p1 - p2
    v2 = p3 - p2

    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 < 1e-6 or norm_v2 < 1e-6:
        return 0.0

    cosine_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
    
    # Clip to handle floating point errors slightly outside domain [-1, 1]
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def calculate_dihedral(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> float:
    """
    Calculate the dihedral angle (in degrees) defined by 4 points.
    Range: -180 to 180.
    """
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3

    # Normalize b2
    norm_b2 = np.linalg.norm(b2)
    if norm_b2 < 1e-6:
        return 0.0
    b2_u = b2 / norm_b2

    # Normals to the planes
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    # Cartesian coordinates of the angle
    x = np.dot(n1, n2)
    y = np.dot(np.cross(n1, n2), b2_u)

    angle_rad = np.arctan2(y, x)
    return np.degrees(angle_rad)

# --- Graph and Topology Functions ---

def find_cycle_indices(graph: Dict[int, List[int]]) -> Set[int]:
    """
    Finds a single cycle in the graph using DFS.
    Returns a set of atom indices comprising the ring.
    Adapted from anomeric_test.py.
    """
    visited = set()
    
    # Sort keys for deterministic behavior
    for start_node in sorted(graph.keys()):
        if start_node in visited:
            continue
            
        # Stack: (current_node, parent_node, path_list)
        stack = [(start_node, -1, [start_node])]
        
        while stack:
            curr, parent, path = stack.pop()
            
            if curr not in visited:
                visited.add(curr)
                
            for neighbor in sorted(graph[curr]):
                if neighbor == parent:
                    continue
                
                if neighbor in path:
                    # Cycle detected
                    cycle_start_index = path.index(neighbor)
                    cycle_path = path[cycle_start_index:]
                    # Basic chemical ring filter (e.g. usually 5 or 6 atoms)
                    if len(cycle_path) >= 3:
                        return set(cycle_path)
                else:
                    stack.append((neighbor, curr, path + [neighbor]))
    
    return set()

def determine_anomeric_config_universal(
    child_residue_dicts: List[Dict],
    acceptor_atom_dict: Dict,
    donor_atom_dict: Dict
) -> str:
    """
    Determines Alpha/Beta configuration using the dihedral angle quartet method.
    Returns 'a', 'b', or a descriptive error string.
    """
    if not child_residue_dicts:
        return "error (empty residue)"
        
    # 1. Adapter to convert dicts to AnomericAtom objects
    heavy_atoms = []
    for a in child_residue_dicts:
        if a['element'].upper() == 'H': continue
        heavy_atoms.append(AnomericAtom(
            idx=a['atom_number'], name=a['atom_name'], res_name=a['residue_name'],
            coords=np.array([a['x'], a['y'], a['z']]), element=a['element']
        ))
    
    acceptor_atom = AnomericAtom(
        idx=acceptor_atom_dict['atom_number'], name=acceptor_atom_dict['atom_name'], res_name=acceptor_atom_dict['residue_name'],
        coords=np.array([acceptor_atom_dict['x'], acceptor_atom_dict['y'], acceptor_atom_dict['z']]), element=acceptor_atom_dict['element']
    )
    
    if not heavy_atoms:
        return "error (no heavy atoms)"

    atoms_map = {a.idx: a for a in heavy_atoms}

    try:
        # 2. Find the cycle
        graph = build_residue_graph(heavy_atoms)
        ring_indices = find_cycle_indices(graph)
        
        if not ring_indices:
            return "error (no ring detected)"
            
        # 3. Find Ring Non-Carbon Atom (Atom 3)
        ring_non_carbons = [idx for idx in ring_indices if atoms_map[idx].element.upper() != 'C']
        if not ring_non_carbons:
            return "error (no ring heteroatom)"
        atom_3_idx = ring_non_carbons[0] 
        atom_3 = atoms_map[atom_3_idx]
        
        # 4. Find Anomeric Carbon (Atom 2)
        atom_2_idx = -1
        c1_candidates = [idx for idx in ring_indices if atoms_map[idx].name.strip().upper() == 'C1']
        if c1_candidates:
            atom_2_idx = c1_candidates[0]
        else:
            c2_candidates = [idx for idx in ring_indices if atoms_map[idx].name.strip().upper() == 'C2']
            if c2_candidates:
                atom_2_idx = c2_candidates[0]
        
        if atom_2_idx == -1:
            return f"error (no anomeric C1/C2 found in ring {list(atoms_map[i].name for i in ring_indices)})"
        atom_2 = atoms_map[atom_2_idx]
        
        # 5. Find Other Endocyclic Atom (Atom 4)
        neighbors_of_3 = graph[atom_3_idx]
        atom_4_idx = -1
        for n_idx in neighbors_of_3:
            if n_idx in ring_indices and n_idx != atom_2_idx:
                atom_4_idx = n_idx
                break
        if atom_4_idx == -1:
            return "error (topology mismatch for Atom 4)"
        atom_4 = atoms_map[atom_4_idx]
        
        # 6. Find Anomeric Substituent (Atom 1)
        # Check distance to ensure the acceptor is actually attached to the anomeric carbon
        dist = np.linalg.norm(acceptor_atom.coords - atom_2.coords)
        if dist > 2.5: 
            return f"error (acceptor {acceptor_atom.name} is {dist:.1f}A from anomeric {atom_2.name}, too far)"
            
        neighbors_of_2 = graph[atom_2_idx]
        exocyclic_candidates = []
        for n_idx in neighbors_of_2:
            if n_idx not in ring_indices:
                exocyclic_candidates.append(atoms_map[n_idx])
        
        existing_indices = {a.idx for a in exocyclic_candidates}
        if acceptor_atom.idx not in existing_indices:
            exocyclic_candidates.append(acceptor_atom)
            
        if not exocyclic_candidates:
            return "error (no exocyclic substituent)"
            
        exocyclic_candidates.sort(key=lambda a: get_atom_mass(a.element), reverse=True)
        atom_1 = exocyclic_candidates[0]
        
        # 7. Calculate Dihedral Angle
        angle = calculate_dihedral(atom_1.coords, atom_2.coords, atom_3.coords, atom_4.coords)
        
        # 8. Determine Configuration
        if -95.0 <= angle <= 95.0:
            return 'a'
        elif (angle > 95.0 and angle < 225.0) or (angle < -95.0 and angle > -225.0):
            return 'b'
        else:
            if abs(abs(angle) - 180.0) < 1e-3: 
                return 'b'
            return f"error (angle {angle:.1f} out of bounds)"

    except Exception as e:
        return f"error (exception: {str(e)})"

def _determine_root_anomeric_config(
    atoms_by_residue: Dict[Tuple[str, int], List[Dict]],
    existing_connections: List[GlycoConnection]
) -> List[GlycoConnection]:
    """
    Determines the anomeric configuration for root or lone monosaccharides.
    Adapts the quartet logic to find the local exocyclic substituent (e.g. O1) 
    to serve as the 'acceptor' for the calculation.
    """
    all_glycan_residues = {
        key for key, atoms in atoms_by_residue.items()
        if atoms and atoms[0]['residue_name'] not in standard_amino_acids
    }
    
    child_residues = {
        (conn.child_chain_id, conn.child_res_id) for conn in existing_connections
    }
    
    root_and_lone_residues = all_glycan_residues - child_residues
    
    pseudo_connections = []
    
    for res_key in root_and_lone_residues:
        chain_id, res_id = res_key
        residue_dicts = atoms_by_residue[res_key]
        
        # Filter heavy atoms for topology check
        heavy_atoms = []
        for a in residue_dicts:
            if a['element'].upper() == 'H': continue
            heavy_atoms.append(AnomericAtom(
                idx=a['atom_number'], name=a['atom_name'], res_name=a['residue_name'],
                coords=np.array([a['x'], a['y'], a['z']]), element=a['element']
            ))

        if not heavy_atoms:
            continue

        atoms_map = {a.idx: a for a in heavy_atoms}
        # Build graph using existing utility
        graph = build_residue_graph(heavy_atoms)
        
        # Find ring using new DFS utility
        ring_indices = find_cycle_indices(graph)
        if not ring_indices:
            continue

        donor_atom_dict = None
        acceptor_atom_dict = None

        # Determine Anomeric Carbon (C1 or C2 in ring)
        c1_in_ring = next((idx for idx in ring_indices if atoms_map[idx].name.strip().upper() == 'C1'), None)
        c2_in_ring = next((idx for idx in ring_indices if atoms_map[idx].name.strip().upper() == 'C2'), None)
        
        anomeric_idx = c1_in_ring if c1_in_ring is not None else c2_in_ring
        
        if anomeric_idx is not None:
            anomeric_atom = atoms_map[anomeric_idx]
            
            # Find heaviest local exocyclic substituent to act as 'Acceptor'
            neighbors = graph[anomeric_idx]
            exocyclic_candidates = []
            for n_idx in neighbors:
                if n_idx not in ring_indices:
                    exocyclic_candidates.append(atoms_map[n_idx])
            
            if exocyclic_candidates:
                exocyclic_candidates.sort(key=lambda a: get_atom_mass(a.element), reverse=True)
                best_substituent = exocyclic_candidates[0]
                
                # Retrieve original dicts
                donor_atom_dict = next(a for a in residue_dicts if a['atom_number'] == anomeric_idx)
                acceptor_atom_dict = next(a for a in residue_dicts if a['atom_number'] == best_substituent.idx)

        # Calculate configuration if pair found
        if donor_atom_dict and acceptor_atom_dict:
            anom_config = determine_anomeric_config_universal(
                child_residue_dicts=residue_dicts,
                acceptor_atom_dict=acceptor_atom_dict,
                donor_atom_dict=donor_atom_dict
            )
            
            pseudo_connections.append(GlycoConnection(
                parent_chain_id=chain_id,
                child_chain_id=chain_id,
                parent_res_id=res_id,
                child_res_id=res_id,
                parent_acceptor_atom_name=acceptor_atom_dict['atom_name'],
                child_donor_atom_name=donor_atom_dict['atom_name'],
                anomeric=anom_config
            ))
            
    return pseudo_connections

def parse_seqres(filepath: Path) -> Dict[str, List[str]]:
    """
    Parses SEQRES records from a PDB file to get the full polymer sequence.

    Args:
        filepath: Path to the PDB file.

    Returns:
        A dictionary mapping chain ID to a list of 3-letter residue codes.
    """
    seqres_data = defaultdict(list)
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith("SEQRES"):
                try:
                    chain_id = line[11]
                    res_names = line[19:].strip().split()
                    seqres_data[chain_id].extend(res_names)
                except IndexError:
                    continue
    return dict(seqres_data)

def parse_glycoprotein_pdb(
    pdb_file: PDBFile,
    ccd: Mapping[str, Mol]
) -> Tuple[Optional[ParsedGlycoproteinData], Optional[Dict], Optional[Dict], Optional[List], Optional[List], Dict[str, int]]:
    """
    (MODIFIED) Parses a PDB file.
    1. Rechains glycans.
    2. Detects connections and anomeric configurations.
    3. Resolves the TRUE residue identity BEFORE any parsing occurs.
    4. Filters sites based on Geometry and Config.
    5. Removes Anomeric Oxygens.
    6. Parses the true structures from the CCD.
    """
    pdb_id = pdb_file.id
    
    # Initialize removal counters
    removal_counts = {
        'bad_geom_asn': 0,
        'alpha_asn': 0, 'other_asn': 0,
        'beta_ser': 0, 'other_ser': 0,
        'beta_thr': 0, 'other_thr': 0
    }

    try:
        name_map = {
            "A2G": {"C2N": "C7", "CME": "C8", "O2N": "O7"}, "NAG": {"C2N": "C7", "CME": "C8", "O2N": "O7"},
            "NGA": {"C2N": "C7", "CME": "C8", "O2N": "O7"}, "NDG": {"C2N": "C7", "CME": "C8", "O2N": "O7"},
            "SIA": {"C5N": "C10", "CME": "C11", "O5N": "O10"},
            "NGC": {"C5N": "C10", "CME": "C11", "O5N": "O10", "OHG": "O11"},
            "TOA": {'O3': 'N3', 'O6A': 'O1', 'O6B': 'O6'}
        }
        inverse_name_map = {res: {v: k for k, v in nmap.items()} for res, nmap in name_map.items()}

        with open(pdb_file.path, 'r') as f:
            raw_atom_lines = [line for line in f if line.startswith(('ATOM', 'HETATM'))]
        
        if not raw_atom_lines:
            return None, {"type": "Parsing Error", "pdb_id": pdb_id, "message": "File contains no ATOM/HETATM records."}, {}, [], [], removal_counts
            
        rechained_atom_lines = _rechain_glycan_components(raw_atom_lines)
        atoms_by_residue, chain_types = parse_pdb_atoms_by_residue(rechained_atom_lines)
        seqres_data = parse_seqres(pdb_file.path)

        # --- STEP 1: Detect Connections and Anomeric Configurations ---
        glycosidic_conns, glycosylation_sites, site_stats, anomalous_sites, no_ring_errors = detect_all_connections(
            atoms_by_residue, pdb_id=pdb_id
        )

        root_pseudo_conns = _determine_root_anomeric_config(atoms_by_residue, glycosidic_conns)
        all_conns_for_lookup = glycosidic_conns + root_pseudo_conns

        # --- STEP 2: RESOLVE TRUE RESIDUE IDENTITIES FIRST ---
        # We must figure out the exact CCD code before we generate any ref_pos or bond data.
        for res_key, pdb_atoms in atoms_by_residue.items():
            if not pdb_atoms: continue
            
            orig_name = pdb_atoms[0]['residue_name']
            if orig_name in standard_amino_acids or orig_name not in MONO_TYPE_MAP:
                continue # Skip non-glycans
                
            anomeric_config = None
            
            # Check if this glycan acts as a child in any sugar-sugar or root connection
            for conn in all_conns_for_lookup:
                if (conn.child_chain_id, conn.child_res_id) == res_key:
                    if conn.anomeric in ['a', 'b']:
                        anomeric_config = conn.anomeric
                    break
                    
            # Check if this glycan acts as a child in a protein-sugar connection
            if anomeric_config is None:
                for site in glycosylation_sites:
                    if (site.glycan_chain_id, site.glycan_res_id) == res_key:
                        if site.anomeric in ['a', 'b']:
                            anomeric_config = site.anomeric
                        break
            
            # Map identicals and resolve anomers
            true_name = const.IDENTICAL_MAP.get(orig_name, orig_name)
            if anomeric_config and true_name in const.ANOMER_MAP:
                true_name = const.ANOMER_MAP[true_name][anomeric_config]
                
            # Mutate the raw dictionaries so the parser builds the correct ref_pos/bonds
            for atom_dict in pdb_atoms:
                atom_dict['residue_name'] = true_name

        # --- STEP 3: Apply Biochemical and Geometric Filters ---
        valid_glycosylation_sites = []
        invalid_glycan_chain_ids: Set[str] = set()

        for site in glycosylation_sites:
            prot_key = (site.protein_chain_id, site.protein_res_id)
            glycan_key = (site.glycan_chain_id, site.glycan_res_id)
            
            prot_atoms = atoms_by_residue.get(prot_key, [])
            glycan_atoms = atoms_by_residue.get(glycan_key, [])
            
            if not prot_atoms or not glycan_atoms:
                valid_glycosylation_sites.append(site)
                continue

            res_name = prot_atoms[0]['residue_name']
            config = site.anomeric

            # --- A. Biochemical Filters (Anomeric Config) ---
            if res_name == 'ASN':
                if config == 'a':
                    removal_counts['alpha_asn'] += 1
                    invalid_glycan_chain_ids.add(site.glycan_chain_id)
                    continue
                elif config != 'b': 
                    removal_counts['other_asn'] += 1
                    invalid_glycan_chain_ids.add(site.glycan_chain_id)
                    continue
            
            elif res_name == 'SER':
                if config == 'b':
                    removal_counts['beta_ser'] += 1
                    invalid_glycan_chain_ids.add(site.glycan_chain_id)
                    continue
                elif config != 'a': 
                    removal_counts['other_ser'] += 1
                    invalid_glycan_chain_ids.add(site.glycan_chain_id)
                    continue

            elif res_name == 'THR':
                if config == 'b':
                    removal_counts['beta_thr'] += 1
                    invalid_glycan_chain_ids.add(site.glycan_chain_id)
                    continue
                elif config != 'a': 
                    removal_counts['other_thr'] += 1
                    invalid_glycan_chain_ids.add(site.glycan_chain_id)
                    continue

            # --- B. Geometric Filter (ASN Only) ---
            if res_name == "ASN":
                atom_map_prot = {a['atom_name'].upper(): np.array([a['x'], a['y'], a['z']]) for a in prot_atoms}
                atom_map_glyc = {a['atom_name'].upper(): np.array([a['x'], a['y'], a['z']]) for a in glycan_atoms}
                
                od1 = atom_map_prot.get("OD1")
                cg = atom_map_prot.get("CG")
                nd2 = atom_map_prot.get("ND2")
                c_glycan = atom_map_glyc.get(site.glycan_atom_name.upper()) 
                
                if od1 is not None and cg is not None and nd2 is not None and c_glycan is not None:
                    angle_val = calculate_angle(cg, nd2, c_glycan)
                    dihedral_val = calculate_dihedral(od1, cg, nd2, c_glycan)
                    
                    angle_pass = 110.0 < angle_val < 130.0
                    dihedral_pass = -20.0 < dihedral_val < 20.0
                    
                    if not angle_pass or not dihedral_pass:
                        removal_counts['bad_geom_asn'] += 1
                        invalid_glycan_chain_ids.add(site.glycan_chain_id)
                        continue 
            
            valid_glycosylation_sites.append(site)
        
        glycosylation_sites = valid_glycosylation_sites

        # --- STEP 4: Remove Anomeric Oxygens ---
        preprocess_remove_anomeric_oxygen(atoms_by_residue, glycosidic_conns, glycosylation_sites)
        
        # Add root connections back in so they are tracked for assembly later
        glycosidic_conns.extend(root_pseudo_conns)

        # --- STEP 5: Parse Data Using True CCD Targets ---
        chains_to_residues: Dict[str, List[ParsedResidue]] = defaultdict(list)
        
        for (chain_id, res_num_pdb), pdb_atoms in atoms_by_residue.items():
            if chain_id in invalid_glycan_chain_ids:
                continue

            if not pdb_atoms: continue
            
            # This name is inherently correct now, pointing directly to the true CCD template
            res_name = pdb_atoms[0]['residue_name'] 
            temp_res_idx = len(chains_to_residues[chain_id])
            parsed_res = None
            
            try:
                if res_name in standard_amino_acids:
                    parsed_res = parse_protein_residue(res_name, res_num_pdb, temp_res_idx, pdb_atoms, ccd)
                elif res_name in ccd: 
                    parsed_res = parse_glycan_residue(res_name, res_num_pdb, temp_res_idx, pdb_atoms, ccd, name_map, inverse_name_map)
            except ValueError as ve:
                if "CRITICAL_MISSING_ATOMS" in str(ve):
                    print(f"[{pdb_id}] Dropping Glycan Chain '{chain_id}' due to malformed residue {res_name} (ID: {res_num_pdb}): {ve}")
                    invalid_glycan_chain_ids.add(chain_id)
                    if chain_id in chains_to_residues:
                        del chains_to_residues[chain_id]
                    parsed_res = None
                    continue
                else:
                    raise ve

            if parsed_res:
                chains_to_residues[chain_id].append(parsed_res)
        
        for chain_id, mol_type in chain_types.items():
            if chain_id in invalid_glycan_chain_ids: continue 

            if mol_type == "PROTEIN" and chain_id in seqres_data:
                full_sequence = seqres_data[chain_id]
                observed_res_nums = {r.orig_idx for r in chains_to_residues[chain_id]}
                for res_idx, res_name in enumerate(full_sequence):
                    res_num_pdb = res_idx + 1
                    if res_num_pdb not in observed_res_nums and res_name in standard_amino_acids:
                        parsed_res = parse_protein_residue(res_name, res_num_pdb, res_idx, None, ccd)
                        if parsed_res:
                            chains_to_residues[chain_id].append(parsed_res)

        filtered_glycosidic_conns = [
            c for c in glycosidic_conns 
            if c.parent_chain_id not in invalid_glycan_chain_ids 
            and c.child_chain_id not in invalid_glycan_chain_ids
        ]
        
        filtered_glycosylation_sites = [
            s for s in glycosylation_sites
            if s.glycan_chain_id not in invalid_glycan_chain_ids
        ]

        parsed_chains: Dict[str, ParsedChain] = {}
        for chain_id, res_list in chains_to_residues.items():
            if not res_list: continue
            res_list.sort(key=lambda r: r.orig_idx)
            for i, res in enumerate(res_list):
                res_list[i] = replace(res, idx=i)
            
            chain_type_str = chain_types.get(chain_id, "GLYCAN")
            chain_type_id = const.chain_type_ids.get(chain_type_str, const.chain_type_ids["NONPOLYMER"])

            parsed_chains[chain_id] = ParsedChain(
                name=chain_id, entity="", type=chain_type_id,
                residues=res_list, sequence=[r.name for r in res_list]
            )

        return ParsedGlycoproteinData(
            pdb_id=pdb_id, chains=parsed_chains,
            glycosidic_connections=filtered_glycosidic_conns,
            glycosylation_sites=filtered_glycosylation_sites
        ), None, site_stats, anomalous_sites, no_ring_errors, removal_counts

    except Exception as e:
        error_message = f"{type(e).__name__}: {e}"
        return None, {"type": "Processing Exception", "pdb_id": pdb_id, "message": error_message, "traceback": traceback.format_exc()}, {}, [], [], removal_counts

def _parse_unresolved_ccd_residue(
    name: str,
    ccd: Mapping[str, Mol],
    res_idx: int,
    res_num_pdb: int,
) -> Optional[ParsedResidue]:
    """
    Creates a ParsedResidue for an unresolved (missing) non-standard residue,
    using only the CCD as a template. All atoms will be marked as not present.
    """
    if name not in ccd:
        return None

    ref_mol = ccd[name]
    ref_mol = AllChem.RemoveHs(ref_mol, sanitize=False)
    
    unk_chirality = const.chirality_type_ids.get(const.unk_chirality_type, 0)
    
    parsed_atoms_list = [
        ParsedAtom(
            name=atom.GetProp("name"),
            element=atom.GetAtomicNum(),
            charge=atom.GetFormalCharge(),
            coords=(0.0, 0.0, 0.0),
            conformer=(0.0, 0.0, 0.0), # Conformer also 0 as it's unresolved
            is_present=False, # CRITICAL: Mark as not present
            chirality=const.chirality_type_ids.get(str(atom.GetChiralTag()), unk_chirality),
        ) for atom in ref_mol.GetAtoms()
    ]

    parsed_bonds_list = [
        ParsedBond(
            atom_1=bond.GetBeginAtomIdx(), atom_2=bond.GetEndAtomIdx(),
            type=const.bond_type_ids.get(bond.GetBondType().name, const.bond_type_ids[const.unk_bond_type])
        ) for bond in ref_mol.GetBonds()
    ]
    
    return ParsedResidue(
        name=name, type=const.token_ids.get("UNK"), atoms=parsed_atoms_list,
        bonds=parsed_bonds_list, idx=res_idx, orig_idx=res_num_pdb,
        atom_center=0, atom_disto=min(1, len(parsed_atoms_list)-1), 
        is_standard=False, is_present=False
    )

def preprocess_remove_anomeric_oxygen(
    atoms_by_residue: Dict[Tuple[str, int], List[Dict]],
    glycosidic_conns: List[GlycoConnection],
    glycosylation_sites: List[GlycosylationSiteConnection]
) -> None:
    """
    (MODIFIED) Removes the anomeric oxygen (O1 or O2) ONLY if the monosaccharide
    is a donor (child) in a glycosidic linkage or a protein-glycan bond.
    
    If the residue is a Root or Lone monosaccharide, the anomeric oxygen is preserved.
    """
    # 1. Identify all residues that are children (donors)
    # These residues MUST have their anomeric oxygen removed to form the bond.
    child_residue_keys = set()

    for conn in glycosidic_conns:
        child_residue_keys.add((conn.child_chain_id, conn.child_res_id))
    
    for site in glycosylation_sites:
        child_residue_keys.add((site.glycan_chain_id, site.glycan_res_id))

    # 2. Iterate through all residues
    for res_key, atoms in atoms_by_residue.items():
        if not atoms:
            continue
            
        res_name = atoms[0]['residue_name']
        
        # Only process Glycans
        if res_name not in MONO_TYPE_MAP or MONO_TYPE_MAP[res_name] == "OTHER":
            continue

        # CRITICAL CHANGE: If this glycan is NOT a child (it is a root or lone),
        # we skip deletion to preserve the O1/O2.
        if res_key not in child_residue_keys:
            continue

        # Convert to temporary objects for graph building
        heavy_atoms = []
        for a in atoms:
            if a['element'].upper() == 'H': continue
            heavy_atoms.append(AnomericAtom(
                idx=a['atom_number'], name=a['atom_name'], res_name=a['residue_name'],
                coords=np.array([a['x'], a['y'], a['z']]), element=a['element']
            ))

        if len(heavy_atoms) < 3:
            continue

        atoms_map = {a.idx: a for a in heavy_atoms}
        
        # 3. Build Graph & Find Ring
        graph = build_residue_graph(heavy_atoms)
        ring_indices = find_cycle_indices(graph)
        
        if not ring_indices:
            continue
            
        ring_atom_names = {atoms_map[idx].name.strip().upper() for idx in ring_indices}

        target_removal_atom = None

        # 4. Determine Anomeric Carbon based on Ring Composition
        if 'C1' in ring_atom_names:
            # Standard Aldose (Glucose, etc.) -> C1 is anomeric -> Remove O1
            target_removal_atom = 'O1'
        elif 'C2' in ring_atom_names and 'C1' not in ring_atom_names:
            # Ketose / Sialic Acid (SIA, KDO) -> C2 is anomeric -> Remove O2
            target_removal_atom = 'O2'

        # 5. Delete the specific oxygen if it exists
        if target_removal_atom:
            atoms_to_keep = []
            for atom in atoms:
                if atom['atom_name'].strip().upper() == target_removal_atom:
                    continue # Skip (Delete) this atom
                atoms_to_keep.append(atom)
            
            # Update the dictionary in place
            atoms_by_residue[res_key] = atoms_to_keep
            
# --- This is a new helper function for parsing glycan residues (extracted from old logic) ---
def parse_glycan_residue(
    res_name: str,
    res_num_pdb: int,
    res_idx: int,
    pdb_atoms: List[Dict[str, Any]],
    ccd: Mapping[str, Mol],
    name_map: Dict,
    inverse_name_map: Dict,
) -> ParsedResidue:
    """
    (MODIFIED) Parses a glycan residue.
    
    Includes a strict check: If MORE than 1 atom is missing relative to the CCD,
    it raises a ValueError to stop parsing the file.
    """
    # 1. Determine the Ground Truth Reference Name
    lookup_name = res_name

    if lookup_name not in ccd:
        return ParsedResidue(
            name=res_name, type=const.token_ids["UNK"], idx=res_idx,
            atoms=[], bonds=[], orig_idx=res_num_pdb,
            atom_center=0, atom_disto=0, is_standard=False, is_present=False
        )

    # 2. Prepare Reference Molecule
    ref_mol_no_h = AllChem.RemoveHs(ccd[lookup_name], sanitize=False)
    for atom in ref_mol_no_h.GetAtoms():
        if not atom.HasProp("name"):
            atom.SetProp("name", f"{atom.GetSymbol()}{atom.GetIdx()+1}")

    try:
        ref_conformer = get_conformer(ref_mol_no_h)
    except ValueError:
        print(f"Warning: No CCD conformer found for glycan residue '{lookup_name}'.", file=sys.stderr)
        ref_conformer = None

    parsed_atoms_list: List[ParsedAtom] = []
    pdb_atom_map = {a['atom_name'].upper(): a for a in pdb_atoms}
    unk_chirality = const.chirality_type_ids.get(const.unk_chirality_type, 0)

    ccd_idx_to_new_idx: Dict[int, int] = {}
    new_idx_counter = 0
    
    # --- NEW: Missing Atom Counter ---
    missing_atom_count = 0 

    for ref_idx, ref_atom in enumerate(ref_mol_no_h.GetAtoms()):
        ref_atom_name_ccd = ref_atom.GetProp("name")
        ref_atom_name_ccd_upper = ref_atom_name_ccd.upper()
        
        pdb_atom_dict = pdb_atom_map.get(ref_atom_name_ccd_upper)
        if pdb_atom_dict is None:
            unmapped_pdb_name = inverse_name_map.get(res_name, {}).get(ref_atom_name_ccd_upper)
            if unmapped_pdb_name:
                pdb_atom_dict = pdb_atom_map.get(unmapped_pdb_name)

        # If still None, it is missing
        if pdb_atom_dict is None:
            missing_atom_count += 1
            continue

        coords = tuple(pdb_atom_dict.get(c, 0.0) for c in 'xyz')

        if ref_conformer:
            ref_coords_rdkit = ref_conformer.GetAtomPosition(ref_atom.GetIdx())
            conformer_coords = (ref_coords_rdkit.x, ref_coords_rdkit.y, ref_coords_rdkit.z)
        else:
            conformer_coords = (0.0, 0.0, 0.0)

        parsed_atoms_list.append(ParsedAtom(
            name=ref_atom_name_ccd, 
            element=ref_atom.GetAtomicNum(), 
            charge=ref_atom.GetFormalCharge(),
            coords=coords,
            conformer=conformer_coords, 
            is_present=True,
            chirality=const.chirality_type_ids.get(ref_atom.GetChiralTag(), unk_chirality),
        ))
        
        ccd_idx_to_new_idx[ref_atom.GetIdx()] = new_idx_counter
        new_idx_counter += 1

    # --- NEW: Strict Validation Check ---
    # We expect exactly 1 missing atom (the O1 or O2 we deleted). 
    # If missing > 1, it means the PDB was already missing atoms.
    if missing_atom_count >= 2:
        raise ValueError(f"CRITICAL_MISSING_ATOMS: Residue {res_name} is missing {missing_atom_count} atoms (Threshold: < 2).")

    parsed_bonds_list = []
    unk_bond_type = const.bond_type_ids.get(const.unk_bond_type, 1)
    
    for bond in ref_mol_no_h.GetBonds():
        old_idx1 = bond.GetBeginAtomIdx()
        old_idx2 = bond.GetEndAtomIdx()
        
        if old_idx1 in ccd_idx_to_new_idx and old_idx2 in ccd_idx_to_new_idx:
            new_idx1 = ccd_idx_to_new_idx[old_idx1]
            new_idx2 = ccd_idx_to_new_idx[old_idx2]
            bond_val = const.bond_type_ids.get(bond.GetBondType().name, unk_bond_type)
            parsed_bonds_list.append(ParsedBond(new_idx1, new_idx2, bond_val))

    try:
        center_idx = next(i for i, pa in enumerate(parsed_atoms_list) if pa.name.upper() == 'C1')
    except StopIteration:
        center_idx = 0
        
    try:
        disto_idx = next(i for i, pa in enumerate(parsed_atoms_list) if pa.name.upper() in ["C4'", "C4"])
    except StopIteration:
        disto_idx = center_idx if len(parsed_atoms_list) <= 1 else 1

    return ParsedResidue(
        name=res_name, type=const.token_ids.get(res_name, const.token_ids["UNK"]), idx=res_idx,
        atoms=parsed_atoms_list, bonds=parsed_bonds_list, orig_idx=res_num_pdb,
        atom_center=center_idx, atom_disto=disto_idx, is_standard=False, is_present=True
    )
    
def parse_protein_residue(
    res_name: str,
    res_num_pdb: int,
    res_idx: int,
    pdb_atoms: Optional[List[Dict[str, Any]]], 
    ccd: Mapping[str, Mol],
) -> ParsedResidue:
    """
    (MODIFIED) Parses a standard amino acid residue without attempting to track
    its internal bonds, as standard AAs are now seamlessly tokenized.
    """
    is_present = pdb_atoms is not None
    pdb_atom_map = {a['atom_name'].upper(): a for a in pdb_atoms} if is_present else {}

    if res_name not in ccd:
        if res_name == "MSE": res_name = "MET"
        else: raise ValueError(f"Standard residue '{res_name}' not found in CCD.")

    ref_mol = ccd[res_name]
    ref_mol = AllChem.RemoveHs(ref_mol, sanitize=False)
    
    try:
        ref_conformer = get_conformer(ref_mol)
    except ValueError:
        print(f"Warning: No CCD conformer for '{res_name}'. Using zero vectors.", file=sys.stderr)
        ref_conformer = None

    ref_atom_names_ordered = const.ref_atoms.get(res_name, [])
    ref_name_to_atom_obj = {atom.GetProp("name"): atom for atom in ref_mol.GetAtoms()}
    
    parsed_atoms_list: List[ParsedAtom] = []
    unk_chirality = const.chirality_type_ids.get(const.unk_chirality_type, 0)

    for atom_name in ref_atom_names_ordered:
        pdb_atom_dict = pdb_atom_map.get(atom_name.upper())
        ref_atom_obj = ref_name_to_atom_obj.get(atom_name)

        if ref_atom_obj is None: continue

        atom_is_present = bool(pdb_atom_dict)
        coords = (
            (pdb_atom_dict['x'], pdb_atom_dict['y'], pdb_atom_dict['z'])
            if atom_is_present else (0.0, 0.0, 0.0)
        )
        conformer_coords = (0.0, 0.0, 0.0)
        if ref_conformer:
            ref_coords_rdkit = ref_conformer.GetAtomPosition(ref_atom_obj.GetIdx())
            conformer_coords = (ref_coords_rdkit.x, ref_coords_rdkit.y, ref_coords_rdkit.z)

        parsed_atoms_list.append(ParsedAtom(
            name=atom_name, element=ref_atom_obj.GetAtomicNum(), charge=ref_atom_obj.GetFormalCharge(),
            coords=coords, conformer=conformer_coords, is_present=atom_is_present,
            chirality=unk_chirality,
        ))

    # Standard representations don't need explicitly tracked internal bonds anymore
    parsed_bonds_list: List[ParsedBond] = []

    center_idx = const.res_to_center_atom_id.get(res_name, 0)
    disto_idx = const.res_to_disto_atom_id.get(res_name, 1)

    return ParsedResidue(
        name=res_name, type=const.token_ids.get(res_name, const.token_ids["UNK"]),
        idx=res_idx, atoms=parsed_atoms_list, bonds=parsed_bonds_list, orig_idx=res_num_pdb,
        atom_center=center_idx, atom_disto=disto_idx, is_standard=True,
        is_present=is_present,
    )

def convert_atom_name(name: str) -> tuple[int, int, int, int]:
    """
    (Required) Convert an atom name to a standard numerical format.
    This function must be added to the script.
    """
    name = name.strip()
    name = [ord(c) - 32 for c in name]
    name = name + [0] * (4 - len(name))
    return tuple(name)

def _find_glycan_components(
    all_glycan_residues: List[Tuple[str, ParsedResidue]],
    glycosidic_connections: List[GlycoConnection],
) -> List[List[Tuple[str, int]]]:
    """
    Finds connected components of glycans using a graph traversal.

    Each component represents a single, complete, independent glycan tree.

    Args:
        all_glycan_residues: A list of tuples (original_chain_name, ParsedResidue).
        glycosidic_connections: A list of detected covalent bonds between monosaccharides.

    Returns:
        A list of components, where each component is a list of unique residue keys
        (original_chain_name, original_residue_id).
    """
    if not all_glycan_residues:
        return []

    # Create a map of residue key -> ParsedResidue for easy lookup
    res_map = {(chain_name, res.orig_idx): res for chain_name, res in all_glycan_residues}
    all_res_keys = set(res_map.keys())

    # Build the adjacency list for the graph
    adj = defaultdict(list)
    for conn in glycosidic_connections:
        parent_key = (conn.parent_chain_id, conn.parent_res_id)
        child_key = (conn.child_chain_id, conn.child_res_id)
        # Ensure we only consider edges between known glycan residues
        if parent_key in all_res_keys and child_key in all_res_keys:
            adj[parent_key].append(child_key)
            adj[child_key].append(parent_key)

    # Find connected components using BFS
    visited = set()
    components = []
    for res_key in all_res_keys:
        if res_key not in visited:
            new_component = []
            q = deque([res_key])
            visited.add(res_key)
            while q:
                current_key = q.popleft()
                new_component.append(current_key)
                for neighbor_key in adj.get(current_key, []):
                    if neighbor_key not in visited:
                        visited.add(neighbor_key)
                        q.append(neighbor_key)
            components.append(new_component)

    return components

def assemble_glycoprotein_structure(
    parsed_data: ParsedGlycoproteinData,
    cluster_id: int,
) -> Tuple[Dict[str, Any], Record]:
    """
    (MODIFIED) Assembles structure arrays. Name-swapping logic has been removed 
    because ParsedResidues are now built with their true identities from the start.
    """
    pdb_id = parsed_data.pdb_id
    atom_rows, res_rows, chain_rows_tuples = [], [], []
    bond_rows, connection_rows = [], []
    glycosylation_site_tuples = []
    final_glycan_feature_map, final_atom_to_mono_idx_map = {}, {}

    atom_map: Dict[Tuple[str, int, str], int] = {}
    res_lookup_info: Dict[Tuple[str, int], Tuple[int, int, int]] = {} 
    
    chain_map: Dict[str, int] = {}
    atom_offset, res_offset, chain_offset = 0, 0, 0

    protein_residues_by_chain = defaultdict(list)
    all_glycan_residues_with_orig_chain = []
    res_obj_map = {} 

    for chain_name, chain in parsed_data.chains.items():
        for res in chain.residues:
            res_key = (chain_name, res.orig_idx if res.orig_idx is not None else res.idx)
            res_obj_map[res_key] = res
            
            is_true_glycan = res.name in MONO_TYPE_MAP and MONO_TYPE_MAP[res.name] != "OTHER"

            if res.is_standard or not is_true_glycan:
                protein_residues_by_chain[chain_name].append(res)
            else:
                all_glycan_residues_with_orig_chain.append((chain_name, res))

    # --- Step 1: Process PROTEIN chains ---
    for chain_name in sorted(protein_residues_by_chain.keys()):
        chain_residues = protein_residues_by_chain[chain_name]
        chain_residues.sort(key=lambda r: r.orig_idx)

        current_chain_idx = chain_offset
        chain_map[chain_name] = current_chain_idx
        
        num_atoms_in_chain = sum(len(r.atoms) for r in chain_residues)
        
        chain_rows_tuples.append((
            chain_name, const.chain_type_ids["PROTEIN"], current_chain_idx, current_chain_idx, current_chain_idx,
            atom_offset, num_atoms_in_chain, res_offset, len(chain_residues), 0
        ))
        
        for res in chain_residues:
            res_key = (chain_name, res.orig_idx if res.orig_idx is not None else res.idx)
            res_lookup_info[res_key] = (res_offset, atom_offset, current_chain_idx)
            
            res_rows.append((
                res.name, res.type, res.idx, atom_offset, len(res.atoms),
                atom_offset + res.atom_center, atom_offset + res.atom_disto,
                res.is_standard, res.is_present
            ))
            
            for local_atom_idx, atom in enumerate(res.atoms):
                atom_map[res_key + (atom.name,)] = atom_offset + local_atom_idx
                atom_rows.append((
                    convert_atom_name(atom.name), atom.element, atom.charge, 
                    atom.coords, atom.conformer,
                    atom.is_present, atom.chirality
                ))
            
            for bond in res.bonds:
                bond_rows.append((min(bond.atom_1, bond.atom_2) + atom_offset, max(bond.atom_1, bond.atom_2) + atom_offset, bond.type))
            
            atom_offset += len(res.atoms)
            res_offset += 1

        for prev_res, next_res in zip(chain_residues, chain_residues[1:]):
            if not (prev_res.is_present and next_res.is_present): continue
            prev_key = (chain_name, prev_res.orig_idx if prev_res.orig_idx is not None else prev_res.idx)
            next_key = (chain_name, next_res.orig_idx if next_res.orig_idx is not None else next_res.idx)
            
            c_atom_key, n_atom_key = prev_key + ('C',), next_key + ('N',)

            if c_atom_key in atom_map and n_atom_key in atom_map:
                atom_idx_c, atom_idx_n = atom_map[c_atom_key], atom_map[n_atom_key]
                res_idx_prev, _, _ = res_lookup_info[prev_key]
                res_idx_next, _, _ = res_lookup_info[next_key]
                connection_rows.append((current_chain_idx, current_chain_idx, res_idx_prev, res_idx_next, atom_idx_c, atom_idx_n))
        
        chain_offset += 1

    # --- Step 2: Process GLYCAN components ---
    glycan_components = _find_glycan_components(all_glycan_residues_with_orig_chain, parsed_data.glycosidic_connections)
    
    for component_res_keys in glycan_components:
        current_chain_idx = chain_offset
        agg_chain_name = f"G{current_chain_idx}" 
        
        sorted_keys = sorted(component_res_keys)
        
        total_atoms_in_tree = 0
        for k in sorted_keys:
             res = res_obj_map[k]
             total_atoms_in_tree += len(res.atoms)

        chain_rows_tuples.append((
            agg_chain_name, const.chain_type_ids["NONPOLYMER"], current_chain_idx, current_chain_idx, current_chain_idx,
            atom_offset, total_atoms_in_tree, res_offset, len(sorted_keys), 0
        ))

        atom_to_mono_idx_list = []
        
        for mono_idx_in_chain, res_key in enumerate(sorted_keys):
            res = res_obj_map[res_key]
            
            # --- ANOMERIC CONFIGURATION & LOGGING ---
            anomeric_config = None
            found_conn = None
            
            # 1. Check if Child in Sugar-Sugar Bond
            for conn in parsed_data.glycosidic_connections:
                if (conn.child_chain_id, conn.child_res_id) == res_key:
                    if conn.parent_chain_id == conn.child_chain_id and conn.parent_res_id == conn.child_res_id:
                        continue
                    found_conn = conn
                    break
            
            if found_conn:
                raw_config = found_conn.anomeric
                if raw_config in ['a', 'b']:
                    anomeric_config = raw_config
                else:
                    print(f"[{pdb_id}] [Assemble] CONFIG ERROR: {res.name} {res_key[0]}:{res_key[1]} linked to {found_conn.parent_chain_id}:{found_conn.parent_res_id}. Reason: {raw_config}")
                    anomeric_config = None
            
            # 2. Check if Child in Protein-Sugar Bond
            if anomeric_config is None and not found_conn:
                found_site = None
                for site in parsed_data.glycosylation_sites:
                    if (site.glycan_chain_id, site.glycan_res_id) == res_key:
                        found_site = site
                        break
                
                if found_site:
                    raw_config = found_site.anomeric
                    if raw_config in ['a', 'b']:
                        anomeric_config = raw_config
                    else:
                        print(f"[{pdb_id}] [Assemble] CONFIG ERROR: {res.name} {res_key[0]}:{res_key[1]} linked to Protein {found_site.protein_chain_id}:{found_site.protein_res_id}. Reason: {raw_config}")
                        anomeric_config = None
            
            res_lookup_info[res_key] = (res_offset, atom_offset, current_chain_idx)
            
            res_rows.append((
                res.name, res.type, mono_idx_in_chain, atom_offset, len(res.atoms),
                atom_offset + res.atom_center, atom_offset + res.atom_disto,
                False, res.is_present 
            ))
            
            for local_atom_idx, atom in enumerate(res.atoms):
                atom_map[res_key + (atom.name,)] = atom_offset + local_atom_idx
                atom_rows.append((
                    convert_atom_name(atom.name), atom.element, atom.charge, 
                    atom.coords, atom.conformer,
                    atom.is_present, atom.chirality
                ))
                atom_to_mono_idx_list.append(mono_idx_in_chain)

            for bond in res.bonds:
                bond_rows.append((
                    min(bond.atom_1, bond.atom_2) + atom_offset, 
                    max(bond.atom_1, bond.atom_2) + atom_offset, 
                    bond.type
                ))

            # The feature map is populated correctly because res.name is inherently accurate
            final_glycan_feature_map[(current_chain_idx, mono_idx_in_chain)] = boltz.data.parse.schema.MonosaccharideFeatures(
                asym_id=current_chain_idx,
                ccd_code=res.name,
                source_glycan_idx=0,
                anomeric_config=anomeric_config
            )

            atom_offset += len(res.atoms)
            res_offset += 1

        final_atom_to_mono_idx_map[current_chain_idx] = np.array(atom_to_mono_idx_list, dtype=np.int32)
        chain_offset += 1

    # --- Step 3: Create Connections ---
    for conn in parsed_data.glycosidic_connections:
        if conn.parent_chain_id == conn.child_chain_id and conn.parent_res_id == conn.child_res_id:
            continue

        parent_key = (conn.parent_chain_id, conn.parent_res_id)
        child_key = (conn.child_chain_id, conn.child_res_id)
        
        if parent_key in res_lookup_info and child_key in res_lookup_info:
            parent_res_idx, parent_atom_start, parent_chain_idx = res_lookup_info[parent_key]
            child_res_idx, child_atom_start, child_chain_idx = res_lookup_info[child_key]
            
            parent_res = res_obj_map[parent_key]
            child_res = res_obj_map[child_key]
            
            p_atom_local = next((i for i, a in enumerate(parent_res.atoms) if a.name == conn.parent_acceptor_atom_name), None)
            c_atom_local = next((i for i, a in enumerate(child_res.atoms) if a.name == conn.child_donor_atom_name), None)
            
            if p_atom_local is not None and c_atom_local is not None:
                p_atom_global = parent_atom_start + p_atom_local
                c_atom_global = child_atom_start + c_atom_local
                connection_rows.append((parent_chain_idx, child_chain_idx, parent_res_idx, child_res_idx, p_atom_global, c_atom_global))

    for i, conn in enumerate(parsed_data.glycosylation_sites):
        prot_key = (conn.protein_chain_id, conn.protein_res_id)
        glycan_key = (conn.glycan_chain_id, conn.glycan_res_id)
        
        if prot_key in res_lookup_info and glycan_key in res_lookup_info:
            prot_res_idx, prot_atom_start, prot_chain_idx = res_lookup_info[prot_key]
            glycan_res_idx, glycan_atom_start, glycan_chain_idx = res_lookup_info[glycan_key]
            
            prot_res = res_obj_map[prot_key]
            glycan_res = res_obj_map[glycan_key]
            
            p_atom_local = next((i for i, a in enumerate(prot_res.atoms) if a.name == conn.protein_atom_name), None)
            g_atom_local = next((i for i, a in enumerate(glycan_res.atoms) if a.name == conn.glycan_atom_name), None)
            
            if p_atom_local is not None and g_atom_local is not None:
                p_atom_global = prot_atom_start + p_atom_local
                g_atom_global = glycan_atom_start + g_atom_local
                
                connection_rows.append((prot_chain_idx, glycan_chain_idx, prot_res_idx, glycan_res_idx, p_atom_global, g_atom_global))
                
                mono_idx = res_rows[glycan_res_idx][2] 
                prot_chain_tuple = chain_rows_tuples[prot_chain_idx]
                prot_chain_start_res = prot_chain_tuple[7] 
                
                glycosylation_site_tuples.append((
                    prot_chain_idx, prot_res_idx - prot_chain_start_res, conn.protein_atom_name,
                    glycan_chain_idx, mono_idx, conn.glycan_atom_name
                ))

    # --- Step 4: Finalize Arrays ---
    atoms = np.array(atom_rows, dtype=Atom)
    bonds = np.array(sorted(list(set(bond_rows))), dtype=Bond) if bond_rows else np.array([], dtype=Bond)
    residues = np.array(res_rows, dtype=Residue)
    chains = np.array(chain_rows_tuples, dtype=Chain)
    connections = np.array(connection_rows, dtype=Connection) if connection_rows else np.array([], dtype=Connection)
    glycosylation_sites_arr = np.array(glycosylation_site_tuples, dtype=GlycosylationSite) if glycosylation_site_tuples else None

    npz_data = {
        'atoms': atoms, 'bonds': bonds, 'residues': residues, 'chains': chains,
        'connections': connections, 'interfaces': np.array([], dtype=Interface),
        'mask': np.ones(len(chains), dtype=bool), 'glycosylation_sites': glycosylation_sites_arr,
        'glycan_feature_map': final_glycan_feature_map, 'atom_to_mono_idx_map': final_atom_to_mono_idx_map,
    }
    
    chain_info_list = [ChainInfo(chain_id=i, chain_name=c['name'].strip(), mol_type=c['mol_type'], num_residues=c['res_num'], valid=True, entity_id=c['entity_id'], msa_id='', cluster_id=cluster_id) for i, c in enumerate(chains)]
    record = Record(id=pdb_id, structure=StructureInfo(num_chains=len(chains)), chains=chain_info_list, interfaces=[], inference_options=None)
    
    return npz_data, record
            
# --- Main Processing Functions --#
def finalize(outdir: Path) -> None:
    """
    Aggregates all individual record .json files into a single, RANDOMLY SHUFFLED
    manifest.json. This is critical for ensuring that downstream dataloaders
    which iterate sequentially still produce representative training batches.
    """
    records_dir = outdir / "records"
    if not records_dir.is_dir():
        print(f"Warning: Records directory not found: {records_dir}")
        return

    final_manifest_entries = []
    record_files = list(records_dir.glob("*.json"))
    
    if not record_files:
        print("No record files found to aggregate.")
        return

    print(f"Aggregating {len(record_files)} record files...")
    for record_path in tqdm(record_files, desc="Creating manifest"):
        try:
            with record_path.open("r") as f:
                record_data = json.load(f)
                final_manifest_entries.append(record_data)
        except Exception as e:
            print(f"Warning: Failed to parse record file {record_path}. Skipping. Error: {e}")
            continue

    print("Randomly shuffling manifest entries...")
    random.shuffle(final_manifest_entries)

    outpath = outdir / "manifest.json"
    print(f"Saving shuffled manifest with {len(final_manifest_entries)} entries to: {outpath}")
    with outpath.open("w") as f:
        json.dump(final_manifest_entries, f, indent=2)
    print("Manifest saved successfully.")

def process_pdb_file(
    task: Tuple[PDBFile, int], 
    outdir: Path
) -> Tuple[bool, Optional[Dict], bool, Optional[Dict], Optional[List], Optional[List], Dict[str, int]]:
    """
    (REVISED) Processes a single PDB file and returns the removal stats dictionary.
    """
    pdb_file, cluster_id = task
    
    global worker_ccd_data
    if worker_ccd_data is None:
        error_info = {"type": "Worker Error", "pdb_id": "N/A", "message": "Worker CCD data not loaded."}
        # Return empty dict for counters
        return False, error_info, False, {}, [], [], {}

    pdb_id = pdb_file.id
    struct_path = outdir / "structures" / f"{pdb_id}.npz"
    record_path = outdir / "records" / f"{pdb_id}.json"

    struct_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Unpack the tuple including removal_counts dict
        parsed_data, error, site_stats, anomalous_sites, no_ring_errors, removal_counts = parse_glycoprotein_pdb(pdb_file, worker_ccd_data)
        
        if error:
            return False, error, False, {}, [], [], removal_counts
        if not parsed_data:
            return False, {"type": "Unknown Parse Error", "pdb_id": pdb_id, "message": "Parsing returned no data."}, False, {}, [], [], removal_counts

        npz_data, record = assemble_glycoprotein_structure(parsed_data, cluster_id)
        
        has_site = (
            npz_data.get('glycosylation_sites') is not None and
            npz_data['glycosylation_sites'].size > 0
        )
        
        np.savez_compressed(struct_path, **npz_data, allow_pickle=True)
        
        with open(record_path, 'w') as f:
            json.dump(asdict(record), f, indent=4, cls=NumpyJSONEncoder)

        return True, None, has_site, site_stats, anomalous_sites, no_ring_errors, removal_counts

    except Exception as e:
        tb_str = traceback.format_exc()
        error_type = type(e).__name__
        error_msg = str(e)
        
        print(f"\n--- FAILURE processing {pdb_id} ---\n", file=sys.stderr, flush=True)
        print(f"  REASON: [{error_type}] {error_msg}", file=sys.stderr, flush=True)
        print(f"  TRACEBACK:\n{tb_str}", file=sys.stderr, flush=True)
        print(f"--- END {pdb_id} ---\n", file=sys.stderr, flush=True)

        return False, {
            "type": "Processing Exception",
            "pdb_id": pdb_id,
            "message": f"{error_type}: {error_msg}",
            "traceback": tb_str
        }, False, {}, [], [], {}

def main(args):
    """
    (MODIFIED) Main loop. Aggregates and prints all filtration stats.
    """
    script_overall_start_time = time.time()
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "structures").mkdir(parents=True, exist_ok=True)
    (args.outdir / "records").mkdir(parents=True, exist_ok=True)

    if not args.ccd_path.is_file():
        print(f"Error: CCD file not found at {args.ccd_path}")
        return 1

    all_pdb_paths = list(args.datadir.rglob("*.pdb"))
    if not all_pdb_paths:
        print("No PDB files found. Exiting.")
        return 0
    
    initial_tasks = [PDBFile(id=p.stem, path=p, cluster_num=0, frame_num=0) for p in all_pdb_paths]
    print(f"Found {len(initial_tasks)} PDB files to process.")

    use_parallel = args.num_processes > 1 and len(initial_tasks) > 1
    ccd_path_str = str(args.ccd_path.resolve())
    
    print("\n--- Processing files (No Clustering / 1 Cluster per File) ---")
    
    final_processing_tasks = [(task, i) for i, task in enumerate(initial_tasks)]

    results = []
    if final_processing_tasks:
        if use_parallel:
            process_func = partial(process_pdb_file, outdir=args.outdir)
            with multiprocessing.Pool(args.num_processes, initializer=worker_init, initargs=(ccd_path_str,)) as pool:
                results = list(tqdm(pool.imap_unordered(process_func, final_processing_tasks), total=len(final_processing_tasks), desc="Processing"))
        else:
            worker_init(ccd_path_str)
            results = [process_pdb_file(task, args.outdir) for task in tqdm(final_processing_tasks, desc="Processing (Serial)")]
    
    success_count = 0
    failed_files = []
    global_site_stats = defaultdict(lambda: defaultdict(int))
    global_anomalous_sites = []
    global_no_ring_errors = []
    
    # Global accumulator for removals
    global_removal_counts = defaultdict(int)

    # Unpack the 7-element tuple from results
    for success, error_info, has_site, site_stats_for_file, anomalous_sites_for_file, no_ring_errors_for_file, removal_counts_file in results:
        # We accumulate removal stats even if the file failed later (e.g. assembly error),
        # but usually we only get them if parse_glycoprotein_pdb returns.
        if removal_counts_file:
            for k, v in removal_counts_file.items():
                global_removal_counts[k] += v

        if success:
            success_count += 1
            if site_stats_for_file:
                for res_name, counts in site_stats_for_file.items():
                    for config, count in counts.items():
                        global_site_stats[res_name][config] += count
            if anomalous_sites_for_file:
                global_anomalous_sites.extend(anomalous_sites_for_file)
            if no_ring_errors_for_file:
                global_no_ring_errors.extend(no_ring_errors_for_file)
        else:
            failed_files.append(error_info)

    print("\n--- Aggregating Records into Manifest ---")
    finalize(args.outdir)

    print("\n\n--- GLOBAL Glycosylation Site Anomeric Configuration Summary ---")
    if not global_site_stats:
        print("No glycosylation sites were detected across all successfully processed files.")
    else:
        print("Summary of all detected protein-glycan linkages across the dataset:")
        for res_name, counts in sorted(global_site_stats.items()):
            alpha_count = counts.get('a', 0)
            beta_count = counts.get('b', 0)
            summary_line = f"For {res_name}: found {alpha_count} alpha and {beta_count} beta configurations."
            other_configs = {k: v for k, v in counts.items() if k not in ['a', 'b']}
            if other_configs:
                other_parts = ", ".join([f"{v} '{k}'" for k, v in other_configs.items()])
                summary_line += f" (Additionally found: {other_parts})"
            print(summary_line)
    print("----------------------------------------------------------------")

    print("\n\n--- Summary of Biochemically Anomalous Glycosylation Sites (Survivors) ---")
    if not global_anomalous_sites:
        print("No anomalous linkages (alpha-ASN, beta-THR, beta-SER) were found remaining in the dataset.")
    else:
        # Note: These numbers should ideally be zero if the new filter works perfectly,
        # unless 'anomalous_sites' in 'detect_connections' flags things differently than the filter logic.
        alpha_asn_examples = [s for s in global_anomalous_sites if s[1] == 'ASN']
        beta_ser_examples = [s for s in global_anomalous_sites if s[1] == 'SER']
        beta_thr_examples = [s for s in global_anomalous_sites if s[1] == 'THR']

        print(f"Found {len(alpha_asn_examples)} alpha-ASN, {len(beta_ser_examples)} beta-SER, and {len(beta_thr_examples)} beta-THR linkages remaining.")
        
        if alpha_asn_examples:
            print("\nDisplaying up to 3 examples of alpha-ASN linkages:")
            for i, (pdb_id, res_name, res_num, config) in enumerate(alpha_asn_examples[:3]):
                print(f"  - PDB ID: {pdb_id}, Residue: {res_name}{res_num}, Detected Config: {config}")
        
    print("----------------------------------------------------------------")

    print("\n\n--- Summary of 'No Ring' Calculation Errors ---")
    if not global_no_ring_errors:
        print("No 'no ring' errors were encountered during processing.")
    else:
        print(f"Found {len(global_no_ring_errors)} sites where a ring could not be determined. Displaying up to 5 examples:")
        for i, (pdb_id, res_name, res_num) in enumerate(global_no_ring_errors[:5]):
            print(f"  - PDB ID: {pdb_id}, Glycosylated Residue: {res_name}{res_num}")
        if len(global_no_ring_errors) > 5:
             print(f"... and {len(global_no_ring_errors) - 5} more.")
    print("----------------------------------------------------------------")
    
    # --- FLUSHED PRINT STATEMENT FOR REMOVED GLYCANS ---
    print("\n--- Summary of Removed Glycans (Filtering) ---")
    print(f"Total glycans removed due to 'bad-ASN' geometry: {global_removal_counts['bad_geom_asn']}")
    print(f"Total glycans removed due to 'alpha-ASN' config: {global_removal_counts['alpha_asn']}")
    print(f"Total glycans removed due to 'other-ASN' config: {global_removal_counts['other_asn']}")
    print(f"Total glycans removed due to 'beta-SER' config:  {global_removal_counts['beta_ser']}")
    print(f"Total glycans removed due to 'other-SER' config: {global_removal_counts['other_ser']}")
    print(f"Total glycans removed due to 'beta-THR' config:  {global_removal_counts['beta_thr']}")
    print(f"Total glycans removed due to 'other-THR' config: {global_removal_counts['other_thr']}")
    sys.stdout.flush()
    print("----------------------------------------------------------------")


    print("\n\n--- SCRIPT COMPLETE ---")
    print(f"Total files processed and saved successfully: {success_count}")
    print(f"Total files that were skipped or failed: {len(failed_files)}")
    
    if failed_files:
        print("\n\n--- FAILURE SUMMARY ---")
        errors_by_reason = defaultdict(list)
        for err in failed_files:
            if err:
                reason = f"[{err.get('type', 'Unknown')}] {err.get('message', 'No details provided.')}"
                errors_by_reason[reason].append(err.get('pdb_id', 'N/A'))
        for reason, pdb_ids in sorted(errors_by_reason.items(), key=lambda item: len(item[1]), reverse=True):
            print("-" * 70)
            print(f"REASON: {reason} ({len(pdb_ids)} files)")
            display_limit = 10
            if len(pdb_ids) > display_limit:
                print(f"  Examples: {', '.join(pdb_ids[:display_limit])}, ...")
            else:
                print(f"  Affected Files: {', '.join(pdb_ids)}")
        print("-" * 70)

    print(f"\nTotal script execution time: {time.time() - script_overall_start_time:.2f} seconds.")
    return 0

def worker_init(ccd_path_str: str):
    """
    Initializer for each worker process. It now raises an exception on failure
    instead of calling sys.exit(), allowing the main process to catch it.
    """
    global worker_ccd_data
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    
    try:
        rdkit.Chem.SetDefaultPickleProperties(rdkit.Chem.PropertyPickleOptions.AllProps)
        with open(ccd_path_str, "rb") as f:
            worker_ccd_data = pickle.load(f)
        
        if not worker_ccd_data or not isinstance(worker_ccd_data, dict):
            # This is a critical failure. Raise an exception to terminate the pool.
            raise RuntimeError("Worker failed to load or received empty/invalid CCD data.")
            
    except Exception as e:
        # Re-raise the exception with more context. This will be caught by the main process.
        raise RuntimeError(f"A worker process failed to initialize: {e}") from e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-process specialized glycan PDB dataset.")
    parser.add_argument(
        "--datadir",
        type=Path,
        required=True,
        help="Directory containing the input PDB files.",
    )
    parser.add_argument(
        "--ccd-path",
        type=Path,
        required=True,
        help="Path to the pickled CCD dictionary (ccd.pkl).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default="glycan_processed",
        help="The output directory for processed .npz files.",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=max(1, multiprocessing.cpu_count() // 2), # Default to half the cores
        help="Number of parallel processes to use.",
    )
    args = parser.parse_args()

    # Basic validation
    if not args.datadir.is_dir():
        print(f"Error: Input data directory not found: {args.datadir}")
        exit(1)
    elif not args.ccd_path.is_file():
         print(f"Error: CCD file not found: {args.ccd_path}")
         exit(1)
    else:
        # Add traceback import if not already present at top level
        import traceback
        exit_code = main(args)
        exit(exit_code)
