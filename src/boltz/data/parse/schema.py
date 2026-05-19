from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Optional, Tuple, List, Mapping, Dict
import sys

import click
import re
import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondStereo, Conformer, Mol
from rdkit.Chem.rdDistGeom import GetMoleculeBoundsMatrix

from boltz.data import const
from boltz.data.types import (
    Atom,
    Bond,
    Chain,
    ChainInfo,
    ChiralAtomConstraint,
    Connection,
    InferenceOptions,
    Interface,
    PlanarBondConstraint,
    PlanarRing5Constraint,
    PlanarRing6Constraint,
    RDKitBoundsConstraint,
    Record,
    Residue,
    ResidueConstraints,
    StereoBondConstraint,
    Structure,
    GlycosylationSite,
    StructureInfo,
    Target,
)

####################################################################################################
# DATACLASSES
####################################################################################################

@dataclass
class MonosaccharideFeatures:
    """Stores detailed features for a specific monosaccharide instance (chain)."""
    asym_id: int
    ccd_code: str
    source_glycan_idx: int
    anomeric_config: Optional[str] = None  # 'a' or 'b' from the donating linkage spec

# Add a type hint for the global monosaccharide map for clarity
MonosaccharideFeatureMapType = Dict[Tuple[int, int], MonosaccharideFeatures]

@dataclass(frozen=True)
class ParsedAtom:
    """A parsed atom object."""

    name: str
    element: int
    charge: int
    coords: tuple[float, float, float]
    conformer: tuple[float, float, float]
    is_present: bool
    chirality: int


@dataclass(frozen=True)
class ParsedBond:
    """A parsed bond object."""

    atom_1: int
    atom_2: int
    type: int


@dataclass(frozen=True)
class ParsedRDKitBoundsConstraint:
    """A parsed RDKit bounds constraint object."""

    atom_idxs: tuple[int, int]
    is_bond: bool
    is_angle: bool
    upper_bound: float
    lower_bound: float


@dataclass(frozen=True)
class ParsedChiralAtomConstraint:
    """A parsed chiral atom constraint object."""

    atom_idxs: tuple[int, int, int, int]
    is_reference: bool
    is_r: bool


@dataclass(frozen=True)
class ParsedStereoBondConstraint:
    """A parsed stereo bond constraint object."""

    atom_idxs: tuple[int, int, int, int]
    is_check: bool
    is_e: bool


@dataclass(frozen=True)
class ParsedPlanarBondConstraint:
    """A parsed planar bond constraint object."""

    atom_idxs: tuple[int, int, int, int, int, int]


@dataclass(frozen=True)
class ParsedPlanarRing5Constraint:
    """A parsed planar bond constraint object."""

    atom_idxs: tuple[int, int, int, int, int]


@dataclass(frozen=True)
class ParsedPlanarRing6Constraint:
    """A parsed planar bond constraint object."""

    atom_idxs: tuple[int, int, int, int, int, int]


@dataclass(frozen=True)
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
    rdkit_bounds_constraints: Optional[list[ParsedRDKitBoundsConstraint]] = None
    chiral_atom_constraints: Optional[list[ParsedChiralAtomConstraint]] = None
    stereo_bond_constraints: Optional[list[ParsedStereoBondConstraint]] = None
    planar_bond_constraints: Optional[list[ParsedPlanarBondConstraint]] = None
    planar_ring_5_constraints: Optional[list[ParsedPlanarRing5Constraint]] = None
    planar_ring_6_constraints: Optional[list[ParsedPlanarRing6Constraint]] = None


@dataclass(frozen=True)
class ParsedChain:
    """A parsed chain object."""

    entity: str
    type: str
    residues: list[ParsedResidue]
    cyclic_period: int


####################################################################################################
# HELPERS
####################################################################################################


def convert_atom_name(name: str) -> tuple[int, int, int, int]:
    """Convert an atom name to a standard format.

    Parameters
    ----------
    name : str
        The atom name.

    Returns
    -------
    Tuple[int, int, int, int]
        The converted atom name.

    """
    name = name.strip()
    name = [ord(c) - 32 for c in name]
    name = name + [0] * (4 - len(name))
    return tuple(name)


def compute_3d_conformer(mol: Mol, version: str = "v3") -> bool:
    """Generate 3D coordinates using EKTDG method.

    Taken from `pdbeccdutils.core.component.Component`.

    Parameters
    ----------
    mol: Mol
        The RDKit molecule to process
    version: str, optional
        The ETKDG version, defaults ot v3

    Returns
    -------
    bool
        Whether computation was successful.

    """
    if version == "v3":
        options = AllChem.ETKDGv3()
    elif version == "v2":
        options = AllChem.ETKDGv2()
    else:
        options = AllChem.ETKDGv2()

    options.clearConfs = False
    conf_id = -1

    try:
        conf_id = AllChem.EmbedMolecule(mol, options)

        if conf_id == -1:
            print(
                f"WARNING: RDKit ETKDGv3 failed to generate a conformer for molecule "
                f"{Chem.MolToSmiles(AllChem.RemoveHs(mol))}, so the program will start with random coordinates. "
                f"Note that the performance of the model under this behaviour was not tested."
            )
            options.useRandomCoords = True
            conf_id = AllChem.EmbedMolecule(mol, options)

        AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=1000)

    except RuntimeError:
        pass  # Force field issue here
    except ValueError:
        pass  # sanitization issue here

    if conf_id != -1:
        conformer = mol.GetConformer(conf_id)
        conformer.SetProp("name", "Computed")
        conformer.SetProp("coord_generation", f"ETKDG{version}")

        return True

    return False


def get_conformer(mol: Mol) -> Conformer:
    """Retrieve an rdkit object for a deemed conformer.

    Inspired by `pdbeccdutils.core.component.Component`.

    Parameters
    ----------
    mol: Mol
        The molecule to process.

    Returns
    -------
    Conformer
        The desired conformer, if any.

    Raises
    ------
    ValueError
        If there are no conformers of the given tyoe.

    """
    # Try using the computed conformer
    for c in mol.GetConformers():
        try:
            if c.GetProp("name") == "Computed":
                return c
        except KeyError:  # noqa: PERF203
            pass

    # Fallback to the ideal coordinates
    for c in mol.GetConformers():
        try:
            if c.GetProp("name") == "Ideal":
                return c
        except KeyError:  # noqa: PERF203
            pass

    msg = "Conformer does not exist."
    raise ValueError(msg)


def compute_geometry_constraints(mol: Mol, idx_map):
    if mol.GetNumAtoms() <= 1:
        return []

    bounds = GetMoleculeBoundsMatrix(
        mol,
        set15bounds=True,
        scaleVDW=True,
        doTriangleSmoothing=True,
        useMacrocycle14config=False,
    )
    bonds = set(
        tuple(sorted(b)) for b in mol.GetSubstructMatches(Chem.MolFromSmarts("*~*"))
    )
    angles = set(
        tuple(sorted([a[0], a[2]]))
        for a in mol.GetSubstructMatches(Chem.MolFromSmarts("*~*~*"))
    )

    constraints = []
    for i, j in zip(*np.triu_indices(mol.GetNumAtoms(), k=1)):
        if i in idx_map and j in idx_map:
            constraint = ParsedRDKitBoundsConstraint(
                atom_idxs=(idx_map[i], idx_map[j]),
                is_bond=tuple(sorted([i, j])) in bonds,
                is_angle=tuple(sorted([i, j])) in angles,
                upper_bound=bounds[i, j],
                lower_bound=bounds[j, i],
            )
            constraints.append(constraint)
    return constraints


def compute_chiral_atom_constraints(mol, idx_map):
    constraints = []
    if all([atom.HasProp("_CIPRank") for atom in mol.GetAtoms()]):
        for center_idx, orientation in Chem.FindMolChiralCenters(
            mol, includeUnassigned=False
        ):
            center = mol.GetAtomWithIdx(center_idx)
            neighbors = [
                (neighbor.GetIdx(), int(neighbor.GetProp("_CIPRank")))
                for neighbor in center.GetNeighbors()
            ]
            neighbors = sorted(
                neighbors, key=lambda neighbor: neighbor[1], reverse=True
            )
            neighbors = tuple(neighbor[0] for neighbor in neighbors)
            is_r = orientation == "R"

            if len(neighbors) > 4:
                continue

            atom_idxs = (*neighbors[:3], center_idx)
            if all(i in idx_map for i in atom_idxs):
                constraints.append(
                    ParsedChiralAtomConstraint(
                        atom_idxs=tuple(idx_map[i] for i in atom_idxs),
                        is_reference=True,
                        is_r=is_r,
                    )
                )

            if len(neighbors) == 4:
                for skip_idx in range(3):
                    chiral_set = neighbors[:skip_idx] + neighbors[skip_idx + 1 :]
                    if skip_idx % 2 == 0:
                        atom_idxs = chiral_set[::-1] + (center_idx,)
                    else:
                        atom_idxs = chiral_set + (center_idx,)
                    if all(i in idx_map for i in atom_idxs):
                        constraints.append(
                            ParsedChiralAtomConstraint(
                                atom_idxs=tuple(idx_map[i] for i in atom_idxs),
                                is_reference=False,
                                is_r=is_r,
                            )
                        )
    return constraints


def compute_stereo_bond_constraints(mol, idx_map):
    constraints = []
    if all([atom.HasProp("_CIPRank") for atom in mol.GetAtoms()]):
        for bond in mol.GetBonds():
            stereo = bond.GetStereo()
            if stereo in {BondStereo.STEREOE, BondStereo.STEREOZ}:
                start_atom_idx, end_atom_idx = (
                    bond.GetBeginAtomIdx(),
                    bond.GetEndAtomIdx(),
                )
                start_neighbors = [
                    (neighbor.GetIdx(), int(neighbor.GetProp("_CIPRank")))
                    for neighbor in mol.GetAtomWithIdx(start_atom_idx).GetNeighbors()
                    if neighbor.GetIdx() != end_atom_idx
                ]
                start_neighbors = sorted(
                    start_neighbors, key=lambda neighbor: neighbor[1], reverse=True
                )
                start_neighbors = [neighbor[0] for neighbor in start_neighbors]
                end_neighbors = [
                    (neighbor.GetIdx(), int(neighbor.GetProp("_CIPRank")))
                    for neighbor in mol.GetAtomWithIdx(end_atom_idx).GetNeighbors()
                    if neighbor.GetIdx() != start_atom_idx
                ]
                end_neighbors = sorted(
                    end_neighbors, key=lambda neighbor: neighbor[1], reverse=True
                )
                end_neighbors = [neighbor[0] for neighbor in end_neighbors]
                is_e = stereo == BondStereo.STEREOE

                atom_idxs = (
                    start_neighbors[0],
                    start_atom_idx,
                    end_atom_idx,
                    end_neighbors[0],
                )
                if all(i in idx_map for i in atom_idxs):
                    constraints.append(
                        ParsedStereoBondConstraint(
                            atom_idxs=tuple(idx_map[i] for i in atom_idxs),
                            is_check=True,
                            is_e=is_e,
                        )
                    )

                if len(start_neighbors) == 2 and len(end_neighbors) == 2:
                    atom_idxs = (
                        start_neighbors[1],
                        start_atom_idx,
                        end_atom_idx,
                        end_neighbors[1],
                    )
                    if all(i in idx_map for i in atom_idxs):
                        constraints.append(
                            ParsedStereoBondConstraint(
                                atom_idxs=tuple(idx_map[i] for i in atom_idxs),
                                is_check=False,
                                is_e=is_e,
                            )
                        )
    return constraints


def compute_flatness_constraints(mol, idx_map):
    planar_double_bond_smarts = Chem.MolFromSmarts("[C;X3;^2](*)(*)=[C;X3;^2](*)(*)")
    aromatic_ring_5_smarts = Chem.MolFromSmarts("[ar5^2]1[ar5^2][ar5^2][ar5^2][ar5^2]1")
    aromatic_ring_6_smarts = Chem.MolFromSmarts(
        "[ar6^2]1[ar6^2][ar6^2][ar6^2][ar6^2][ar6^2]1"
    )

    planar_double_bond_constraints = []
    aromatic_ring_5_constraints = []
    aromatic_ring_6_constraints = []
    for match in mol.GetSubstructMatches(planar_double_bond_smarts):
        if all(i in idx_map for i in match):
            planar_double_bond_constraints.append(
                ParsedPlanarBondConstraint(atom_idxs=tuple(idx_map[i] for i in match))
            )
    for match in mol.GetSubstructMatches(aromatic_ring_5_smarts):
        if all(i in idx_map for i in match):
            aromatic_ring_5_constraints.append(
                ParsedPlanarRing5Constraint(atom_idxs=tuple(idx_map[i] for i in match))
            )
    for match in mol.GetSubstructMatches(aromatic_ring_6_smarts):
        if all(i in idx_map for i in match):
            aromatic_ring_6_constraints.append(
                ParsedPlanarRing6Constraint(atom_idxs=tuple(idx_map[i] for i in match))
            )

    return (
        planar_double_bond_constraints,
        aromatic_ring_5_constraints,
        aromatic_ring_6_constraints,
    )


####################################################################################################
# PARSING
####################################################################################################


def parse_ccd_residue(
    name: str,
    ref_mol: Mol,
    res_idx: int,
) -> Optional[ParsedResidue]:
    """Parse an MMCIF ligand.

    First tries to get the SMILES string from the RCSB.
    Then, tries to infer atom ordering using RDKit.

    Parameters
    ----------
    name: str
        The name of the molecule to parse.
    ref_mol: Mol
        The reference molecule to parse.
    res_idx : int
        The residue index.

    Returns
    -------
    ParsedResidue, optional
       The output ParsedResidue, if successful.

    """
    unk_chirality = const.chirality_type_ids[const.unk_chirality_type]

    # Remove hydrogens
    ref_mol = AllChem.RemoveHs(ref_mol, sanitize=False)
    Chem.AssignStereochemistry(ref_mol, cleanIt=True, force=True)

    # Check if this is a single atom CCD residue
    if ref_mol.GetNumAtoms() == 1:
        pos = (0, 0, 0)
        ref_atom = ref_mol.GetAtoms()[0]
        chirality_type = const.chirality_type_ids.get(
            str(ref_atom.GetChiralTag()), unk_chirality
        )
        atom = ParsedAtom(
            name=ref_atom.GetProp("name"),
            element=ref_atom.GetAtomicNum(),
            charge=ref_atom.GetFormalCharge(),
            coords=pos,
            conformer=(0, 0, 0),
            is_present=True,
            chirality=chirality_type,
        )
        unk_prot_id = const.unk_token_ids["PROTEIN"]
        residue = ParsedResidue(
            name=name,
            type=unk_prot_id,
            atoms=[atom],
            bonds=[],
            idx=res_idx,
            orig_idx=None,
            atom_center=0,  # Placeholder, no center
            atom_disto=0,  # Placeholder, no center
            is_standard=False,
            is_present=True,
        )
        return residue

    # Get reference conformer coordinates
    conformer = get_conformer(ref_mol)

    # Parse each atom in order of the reference mol
    atoms = []
    atom_idx = 0
    idx_map = {}  # Used for bonds later

    for i, atom in enumerate(ref_mol.GetAtoms()):
        # Get atom name, charge, element and reference coordinates
        atom_name = atom.GetProp("name")
        charge = atom.GetFormalCharge()
        element = atom.GetAtomicNum()
        ref_coords = conformer.GetAtomPosition(atom.GetIdx())
        ref_coords = (ref_coords.x, ref_coords.y, ref_coords.z)
        chirality_type = const.chirality_type_ids.get(
            str(atom.GetChiralTag()), unk_chirality
        )

        # Get PDB coordinates, if any
        coords = (0, 0, 0)
        atom_is_present = True

        # Add atom to list
        atoms.append(
            ParsedAtom(
                name=atom_name,
                element=element,
                charge=charge,
                coords=coords,
                conformer=ref_coords,
                is_present=atom_is_present,
                chirality=chirality_type,
            )
        )
        idx_map[i] = atom_idx
        atom_idx += 1  # noqa: SIM113

    # Load bonds
    bonds = []
    unk_bond = const.bond_type_ids[const.unk_bond_type]
    for bond in ref_mol.GetBonds():
        idx_1 = bond.GetBeginAtomIdx()
        idx_2 = bond.GetEndAtomIdx()

        # Skip bonds with atoms ignored
        if (idx_1 not in idx_map) or (idx_2 not in idx_map):
            continue

        idx_1 = idx_map[idx_1]
        idx_2 = idx_map[idx_2]
        start = min(idx_1, idx_2)
        end = max(idx_1, idx_2)
        bond_type = bond.GetBondType().name
        bond_type = const.bond_type_ids.get(bond_type, unk_bond)
        bonds.append(ParsedBond(start, end, bond_type))

    rdkit_bounds_constraints = compute_geometry_constraints(ref_mol, idx_map)
    chiral_atom_constraints = compute_chiral_atom_constraints(ref_mol, idx_map)
    stereo_bond_constraints = compute_stereo_bond_constraints(ref_mol, idx_map)
    planar_bond_constraints, planar_ring_5_constraints, planar_ring_6_constraints = (
        compute_flatness_constraints(ref_mol, idx_map)
    )

    unk_prot_id = const.unk_token_ids["PROTEIN"]
    return ParsedResidue(
        name=name,
        type=unk_prot_id,
        atoms=atoms,
        bonds=bonds,
        idx=res_idx,
        atom_center=0,
        atom_disto=0,
        orig_idx=None,
        is_standard=False,
        is_present=True,
        rdkit_bounds_constraints=rdkit_bounds_constraints,
        chiral_atom_constraints=chiral_atom_constraints,
        stereo_bond_constraints=stereo_bond_constraints,
        planar_bond_constraints=planar_bond_constraints,
        planar_ring_5_constraints=planar_ring_5_constraints,
        planar_ring_6_constraints=planar_ring_6_constraints,
    )


def parse_polymer(
    sequence: list[str],
    entity: str,
    chain_type: str,
    components: dict[str, Mol],
    cyclic: bool,
    glycosylated_residue_indices: set = frozenset(), # Preserved to avoid breaking callers, but ignored internally
) -> Optional[ParsedChain]:
    """(FINAL CORRECTED VERSION) Process a sequence into a chain object."""
    ref_res = set(const.tokens)
    unk_chirality = const.chirality_type_ids[const.unk_chirality_type]

    parsed = []
    for res_idx, res_name in enumerate(sequence):
        res_corrected = res_name if res_name != "MSE" else "MET"

        # If it's a true non-standard residue (ligand in sequence), use the generic CCD parser
        if res_corrected not in ref_res:
            if res_corrected not in components:
                raise ValueError(f"Component definition for '{res_corrected}' not found.")
            
            residue = parse_ccd_residue(
                name=res_corrected,
                ref_mol=components[res_corrected],
                res_idx=res_idx,
            )
            parsed.append(residue)
            continue

        # This is the standard path for ALL protein residues, glycosylated or otherwise
        ref_mol = components[res_corrected]
        ref_mol = AllChem.RemoveHs(ref_mol, sanitize=False)
        ref_conformer = get_conformer(ref_mol)
        ref_name_to_atom = {a.GetProp("name"): a for a in ref_mol.GetAtoms()}
        ref_atoms = [ref_name_to_atom[a] for a in const.ref_atoms[res_corrected]]

        atoms: list[ParsedAtom] = []
        for ref_atom in ref_atoms:
            atom_name = ref_atom.GetProp("name")
            idx = ref_atom.GetIdx()
            ref_coords = ref_conformer.GetAtomPosition(idx)
            ref_coords = (ref_coords.x, ref_coords.y, ref_coords.z)
            atoms.append(
                ParsedAtom(
                    name=atom_name,
                    element=ref_atom.GetAtomicNum(),
                    charge=ref_atom.GetFormalCharge(),
                    coords=(0, 0, 0),
                    conformer=ref_coords,
                    is_present=True,
                    chirality=const.chirality_type_ids.get(str(ref_atom.GetChiralTag()), unk_chirality),
                )
            )

        parsed.append(
            ParsedResidue(
                name=res_corrected,
                type=const.token_ids[res_corrected],
                atoms=atoms,
                bonds=[],
                idx=res_idx,
                atom_center=const.res_to_center_atom_id[res_corrected],
                atom_disto=const.res_to_disto_atom_id[res_corrected],
                is_standard=True,
                is_present=True,
                orig_idx=None,
            )
        )

    cyclic_period = len(sequence) if cyclic else 0

    return ParsedChain(
        entity=entity,
        residues=parsed,
        type=chain_type,
        cyclic_period=cyclic_period,
    )

def parse_boltz_schema(  # noqa: C901, PLR0915, PLR0912
    name: str,
    schema: dict,
    ccd: Mapping[str, Mol],
) -> Target:
    """Parse a Boltz input yaml / json.

    Integrates:
    1. Protein/DNA/RNA parsing.
    2. Ligand parsing (CCD and SMILES).
    3. Glycan parsing (IUPAC with Anomeric/Stoichiometry logic).
    4. Constraints (Bonds, Pockets, Glycosylation).
    """
    # Assert version 1
    version = schema.get("version", 1)
    if version != 1:
        msg = f"Invalid version {version} in input!"
        raise ValueError(msg)

    # Disable rdkit warnings
    blocker = rdBase.BlockLogs()  # noqa: F841

    # --- 1. Group items by entity type and sequence ---
    items_to_group = {}
    glycan_unique_counter = 0  # FIX: Counter to force unique entities for glycans

    for item in schema["sequences"]:
        # Get entity type
        entity_type = next(iter(item.keys())).lower()
        if entity_type not in {"protein", "dna", "rna", "ligand", "glycan"}:
            msg = f"Invalid entity type: {entity_type}"
            raise ValueError(msg)

        # Get sequence definition for grouping
        if entity_type in {"protein", "dna", "rna"}:
            seq = str(item[entity_type]["sequence"])
        elif entity_type == "ligand":
            assert "smiles" in item[entity_type] or "ccd" in item[entity_type]
            assert "smiles" not in item[entity_type] or "ccd" not in item[entity_type]
            if "smiles" in item[entity_type]:
                seq = f"smiles:{item[entity_type]['smiles']}"
            else:
                seq = f"ccd:{item[entity_type]['ccd']}"
        elif entity_type == "glycan":
            # FIX: Append unique ID to key to force every glycan input to be a unique entity.
            # This prevents the IHM writer from merging identical glycan chains into one entity,
            # which causes "Duplicate entity" crashes if not handled perfectly.
            seq = f"iupac:{item[entity_type]['iupac']}_{glycan_unique_counter}"
            glycan_unique_counter += 1
            
        items_to_group.setdefault((entity_type, seq), []).append(item)

    # --- 2. Parse Entities into Chains ---
    chains: dict[str, ParsedChain] = {}
    chain_to_msa: dict[str, str] = {}
    entity_to_seq: dict[str, str] = {}
    
    # Storage for extra glycan data that doesn't fit into standard ParsedChain
    # Map: entity_id -> { 'internal_connections': ..., 'feature_map': ..., 'atom_map': ... }
    glycan_extra_data = {} 

    is_msa_custom = False
    is_msa_auto = False

    for entity_id, items in enumerate(items_to_group.values()):
        # Get entity type
        entity_type = next(iter(items[0].keys())).lower()

        # MSA Handling (Proteins)
        msa = -1
        if entity_type == "protein":
            msa = items[0][entity_type].get("msa", 0)
            if (msa is None) or (msa == ""):
                msa = 0

            # Check consistency
            for item in items:
                item_msa = item[entity_type].get("msa", 0)
                if (item_msa is None) or (item_msa == ""):
                    item_msa = 0
                if item_msa != msa:
                    raise ValueError("All proteins with the same sequence must share the same MSA!")

            if msa == "empty":
                msa = -1
                click.echo("Found explicit empty MSA, running in single sequence mode.")

            if msa not in (0, -1):
                is_msa_custom = True
            elif msa == 0:
                is_msa_auto = True

        # --- A. Parse Polymer (Protein, DNA, RNA) ---
        if entity_type in {"protein", "dna", "rna"}:
            if entity_type == "rna":
                token_map = const.rna_letter_to_token
            elif entity_type == "dna":
                token_map = const.dna_letter_to_token
            elif entity_type == "protein":
                token_map = const.prot_letter_to_token

            chain_type = const.chain_type_ids[entity_type.upper()]
            unk_token = const.unk_token[entity_type.upper()]

            seq_str = items[0][entity_type]["sequence"]
            entity_to_seq[entity_id] = seq_str
            seq_tokens = [token_map.get(c, unk_token) for c in list(seq_str)]

            # Apply modifications
            for mod in items[0][entity_type].get("modifications", []):
                code = mod["ccd"]
                idx = mod["position"] - 1
                seq_tokens[idx] = code

            cyclic = items[0][entity_type].get("cyclic", False)

            # Pass glycosylation site indices to parser if protein (to avoid OXT addition)
            # We need to scan ALL chains of this entity to find all potential sites
            glyco_indices = set()
            if entity_type == "protein" and "glycosylation" in schema:
                potential_chain_ids = []
                for item in items:
                    ids = item[entity_type]["id"]
                    if isinstance(ids, str): ids = [ids]
                    potential_chain_ids.extend(ids)
                
                potential_chain_ids = set(potential_chain_ids)
                
                for site in schema.get("glycosylation", []):
                    try:
                        p_chain = site["site"]["protein"][0]
                        p_res = site["site"]["protein"][1] - 1 # 0-based
                        if p_chain in potential_chain_ids:
                            glyco_indices.add(p_res)
                    except:
                        pass

            parsed_chain = parse_polymer(
                sequence=seq_tokens,
                entity=entity_id,
                chain_type=chain_type,
                components=ccd,
                cyclic=cyclic,
                glycosylated_residue_indices=glyco_indices
            )

        # --- B. Parse Ligand (CCD) ---
        elif (entity_type == "ligand") and "ccd" in (items[0][entity_type]):
            seq = items[0][entity_type]["ccd"]
            if isinstance(seq, str):
                seq = [seq]

            residues = []
            for res_idx, code in enumerate(seq):
                if code not in ccd:
                    raise ValueError(f"CCD component {code} not found!")
                residue = parse_ccd_residue(name=code, ref_mol=ccd[code], res_idx=res_idx)
                residues.append(residue)

            parsed_chain = ParsedChain(
                entity=entity_id,
                residues=residues,
                type=const.chain_type_ids["NONPOLYMER"],
                cyclic_period=0,
            )
            assert not items[0][entity_type].get("cyclic", False), "Cyclic flag not supported for ligands"

        # --- C. Parse Ligand (SMILES) ---
        elif (entity_type == "ligand") and ("smiles" in items[0][entity_type]):
            seq = items[0][entity_type]["smiles"]
            mol = AllChem.MolFromSmiles(seq)
            if mol is None:
                raise ValueError(f"Invalid SMILES string: {seq}")
            mol = AllChem.AddHs(mol)

            canonical_order = AllChem.CanonicalRankAtoms(mol, breakTies=True)
            for atom, can_idx in zip(mol.GetAtoms(), canonical_order):
                atom_name = atom.GetSymbol().upper() + str(can_idx + 1)
                if len(atom_name) > 4:
                    raise ValueError(f"{seq} has atom name > 4 chars: {atom_name}")
                atom.SetProp("name", atom_name)

            success = compute_3d_conformer(mol)
            if not success:
                raise ValueError(f"Failed to compute 3D conformer for {seq}")

            mol_no_h = AllChem.RemoveHs(mol)
            Chem.AssignStereochemistry(mol_no_h, cleanIt=True, force=True)
            residue = parse_ccd_residue(name="LIG", ref_mol=mol_no_h, res_idx=0)
            
            parsed_chain = ParsedChain(
                entity=entity_id,
                residues=[residue],
                type=const.chain_type_ids["NONPOLYMER"],
                cyclic_period=0,
            )
            assert not items[0][entity_type].get("cyclic", False), "Cyclic flag not supported for ligands"

        # --- D. Parse Glycan (IUPAC) ---
        elif entity_type == "glycan":
            iupac_str = items[0][entity_type]["iupac"]
            # Store original IUPAC for reference, stripping the unique counter suffix if needed for display
            # (though strictly entity_to_seq is just internal storage)
            entity_to_seq[entity_id] = iupac_str
            
            # Use the new parse_glycan function
            residues, internal_conns, feat_map, atom_map_arr = parse_glycan(iupac_str, ccd)
            
            parsed_chain = ParsedChain(
                entity=entity_id,
                residues=residues,
                type=const.chain_type_ids["NONPOLYMER"],
                cyclic_period=0
            )
            
            # Store auxiliary data needed for flattening
            glycan_extra_data[entity_id] = {
                "internal_connections": internal_conns,
                "feature_map": feat_map,
                "atom_map": atom_map_arr
            }

        else:
            raise ValueError(f"Invalid entity type: {entity_type}")

        # Add chains
        for item in items:
            ids = item[entity_type]["id"]
            if isinstance(ids, str):
                ids = [ids]
            for chain_name in ids:
                chains[chain_name] = parsed_chain
                chain_to_msa[chain_name] = msa

    if is_msa_custom and is_msa_auto:
        raise ValueError("Cannot mix custom and auto-generated MSAs!")
    if not chains:
        raise ValueError("No chains parsed!")

    # --- 3. Flatten Chains into Tables ---
    atom_data, bond_data, res_data, chain_data = [], [], [], []
    connections_list = []
    
    # RDKit Constraints
    rdkit_bounds_constraint_data = []
    chiral_atom_constraint_data = []
    stereo_bond_constraint_data = []
    planar_bond_constraint_data = []
    planar_ring_5_constraint_data = []
    planar_ring_6_constraint_data = []
    
    # Glycan specific global maps
    global_glycan_feature_map = {}
    global_atom_to_mono_idx_map = {}

    atom_idx = 0
    res_idx = 0
    asym_id = 0
    sym_count = {}
    chain_to_idx = {}
    atom_idx_map = {} # (chain_name, res_idx, atom_name) -> (asym_id, res_idx, atom_idx)

    for asym_id, (chain_name, chain) in enumerate(chains.items()):
        res_num = len(chain.residues)
        atom_num = sum(len(res.atoms) for res in chain.residues)
        entity_id = int(chain.entity)
        sym_id = sym_count.get(entity_id, 0)
        
        chain_data.append((
            chain_name, chain.type, entity_id, sym_id, asym_id,
            atom_idx, atom_num, res_idx, res_num, chain.cyclic_period
        ))
        
        chain_to_idx[chain_name] = asym_id
        sym_count[entity_id] = sym_id + 1

        # GLYCAN HANDLING: If this entity has extra data, map it to global indices
        if entity_id in glycan_extra_data:
            g_data = glycan_extra_data[entity_id]
            
            # 1. Feature Map: (chain_idx, mono_idx) -> features
            # 'feat_map' keys are just (mono_idx). We add the current asym_id.
            for mono_i, feats in g_data["feature_map"].items():
                # Re-wrap as MonosaccharideFeatures dataclass if not already
                if isinstance(feats, dict):
                    feat_obj = MonosaccharideFeatures(
                        asym_id=asym_id,
                        ccd_code=feats["ccd_code"],
                        source_glycan_idx=0, # Placeholder
                        anomeric_config=feats["anomeric_config"]
                    )
                else:
                    feat_obj = feats
                global_glycan_feature_map[(asym_id, mono_i)] = feat_obj
            
            # 2. Atom Map: chain_idx -> array
            global_atom_to_mono_idx_map[asym_id] = g_data["atom_map"]
            
            # 3. Internal Connections
            # stored as (parent_mono_idx, child_mono_idx, p_atom_local, c_atom_local)
            # We need to calculate global atom indices.
            
            # First, map local mono indices to global atom start indices for this chain
            # We do this by iterating the residues right here
            current_chain_residue_atom_starts = []
            temp_atom_counter = atom_idx
            for res in chain.residues:
                current_chain_residue_atom_starts.append(temp_atom_counter)
                temp_atom_counter += len(res.atoms)

            for p_mono, c_mono, p_atom_local, c_atom_local in g_data["internal_connections"]:
                # Global Atom Idx = Start of Residue + Local Offset
                p_atom_global = current_chain_residue_atom_starts[p_mono] + p_atom_local
                c_atom_global = current_chain_residue_atom_starts[c_mono] + c_atom_local
                
                # We need the global residue indices for the connection struct
                p_res_global = res_idx + p_mono
                c_res_global = res_idx + c_mono
                
                connections_list.append((asym_id, asym_id, p_res_global, c_res_global, p_atom_global, c_atom_global))

        # Flatten Residues
        for res in chain.residues:
            atom_center = atom_idx + res.atom_center
            atom_disto = atom_idx + res.atom_disto
            res_data.append((
                res.name, res.type, res.idx, atom_idx, len(res.atoms),
                atom_center, atom_disto, res.is_standard, res.is_present
            ))

            # Constraints (Bounds, Chiral, Stereo, Planar) - Copy from old code
            if res.rdkit_bounds_constraints:
                for c in res.rdkit_bounds_constraints:
                    rdkit_bounds_constraint_data.append((tuple(a + atom_idx for a in c.atom_idxs), c.is_bond, c.is_angle, c.upper_bound, c.lower_bound))
            if res.chiral_atom_constraints:
                for c in res.chiral_atom_constraints:
                    chiral_atom_constraint_data.append((tuple(a + atom_idx for a in c.atom_idxs), c.is_reference, c.is_r))
            if res.stereo_bond_constraints:
                for c in res.stereo_bond_constraints:
                    stereo_bond_constraint_data.append((tuple(a + atom_idx for a in c.atom_idxs), c.is_check, c.is_e))
            if res.planar_bond_constraints:
                for c in res.planar_bond_constraints:
                    planar_bond_constraint_data.append((tuple(a + atom_idx for a in c.atom_idxs),))
            if res.planar_ring_5_constraints:
                for c in res.planar_ring_5_constraints:
                    planar_ring_5_constraint_data.append((tuple(a + atom_idx for a in c.atom_idxs),))
            if res.planar_ring_6_constraints:
                for c in res.planar_ring_6_constraints:
                    planar_ring_6_constraint_data.append((tuple(a + atom_idx for a in c.atom_idxs),))

            # Internal Bonds (e.g. within a ligand or residue)
            for bond in res.bonds:
                bond_data.append((atom_idx + bond.atom_1, atom_idx + bond.atom_2, bond.type))

            # Atoms
            for atom in res.atoms:
                atom_idx_map[(chain_name, res.idx, atom.name)] = (asym_id, res_idx, atom_idx)
                atom_data.append((
                    convert_atom_name(atom.name), atom.element, atom.charge,
                    atom.coords, atom.conformer, atom.is_present, atom.chirality
                ))
                atom_idx += 1
            res_idx += 1

    # --- 4. Parse Constraints (Bonds, Pockets, Glycosylation) ---
    pocket_binders = []
    pocket_residues = []
    constraints = schema.get("constraints", [])
    
    # Explicit Bonds
    for constraint in constraints:
        if "bond" in constraint:
            if "atom1" not in constraint["bond"] or "atom2" not in constraint["bond"]:
                raise ValueError("Bond constraint improperly specified")
            c1, r1, a1 = tuple(constraint["bond"]["atom1"])
            c2, r2, a2 = tuple(constraint["bond"]["atom2"])
            c1_idx, r1_idx, a1_idx = atom_idx_map[(c1, r1 - 1, a1)]
            c2_idx, r2_idx, a2_idx = atom_idx_map[(c2, r2 - 1, a2)]
            connections_list.append((c1_idx, c2_idx, r1_idx, r2_idx, a1_idx, a2_idx))
        
        elif "pocket" in constraint:
            if "binder" not in constraint["pocket"] or "contacts" not in constraint["pocket"]:
                raise ValueError("Pocket constraint improperly specified")
            binder = constraint["pocket"]["binder"]
            contacts = constraint["pocket"]["contacts"]
            
            if len(pocket_binders) > 0:
                if pocket_binders[-1] != chain_to_idx[binder]:
                    raise ValueError("Only one pocket binder supported!")
                else:
                    pocket_residues[-1].extend([(chain_to_idx[c], r - 1) for c, r in contacts])
            else:
                pocket_binders.append(chain_to_idx[binder])
                pocket_residues.append([(chain_to_idx[c], r - 1) for c, r in contacts])
        else:
            raise ValueError(f"Invalid constraint: {constraint}")

    # Glycosylation Sites (Protein-Glycan covalent bonds)
    glycosylation_sites_data = []
    if "glycosylation" in schema:
        for site in schema.get("glycosylation", []):
            try:
                p_spec = site["site"]["protein"]
                p_chain, p_res_1based = p_spec[0], p_spec[1]
                p_res_0based = p_res_1based - 1
                
                g_spec = site["site"]["glycan"]
                g_chain, g_mono_idx, g_atom_name = g_spec[0], g_spec[1], g_spec[2]
                
                # Determine Protein Atom Name
                if len(p_spec) == 3:
                    p_atom_name = p_spec[2]
                else:
                    # Infer standard attachment point based on residue type (N-linked vs O-linked)
                    # We need to look up the residue type. 
                    # atom_idx_map keys are (chain, res, atom). 
                    # We iterate the map keys (inefficient but safe) or check commonly
                    found = False
                    for possible_atom in ["ND2", "OG", "OG1"]: # ASN, SER, THR
                        if (p_chain, p_res_0based, possible_atom) in atom_idx_map:
                            p_atom_name = possible_atom
                            found = True
                            break
                    if not found:
                        raise ValueError(f"Could not infer attachment atom for {p_chain}:{p_res_1based}")

                # Get Global Indices
                if (p_chain, p_res_0based, p_atom_name) not in atom_idx_map:
                    raise ValueError(f"Protein atom {p_chain}:{p_res_1based}:{p_atom_name} not found")
                if (g_chain, g_mono_idx, g_atom_name) not in atom_idx_map:
                    raise ValueError(f"Glycan atom {g_chain}:{g_mono_idx}:{g_atom_name} not found")

                p_c_idx, p_r_idx, p_a_idx = atom_idx_map[(p_chain, p_res_0based, p_atom_name)]
                g_c_idx, g_r_idx, g_a_idx = atom_idx_map[(g_chain, g_mono_idx, g_atom_name)]

                connections_list.append((p_c_idx, g_c_idx, p_r_idx, g_r_idx, p_a_idx, g_a_idx))
                
                # Store site metadata for loss masking
                glycosylation_sites_data.append((p_c_idx, p_res_0based, p_atom_name, g_c_idx, g_mono_idx, g_atom_name))

            except Exception as e:
                raise ValueError(f"Error processing glycosylation site: {e}")

    # --- 5. Construct Final Objects ---
    atoms = np.array(atom_data, dtype=Atom)
    bonds = np.array(bond_data, dtype=Bond)
    residues = np.array(res_data, dtype=Residue)
    chains = np.array(chain_data, dtype=Chain)
    interfaces = np.array([], dtype=Interface)
    connections = np.array(connections_list, dtype=Connection)
    mask = np.ones(len(chain_data), dtype=bool)
    
    # Constraints arrays
    rdkit_bounds_constraints = np.array(rdkit_bounds_constraint_data, dtype=RDKitBoundsConstraint)
    chiral_atom_constraints = np.array(chiral_atom_constraint_data, dtype=ChiralAtomConstraint)
    stereo_bond_constraints = np.array(stereo_bond_constraint_data, dtype=StereoBondConstraint)
    planar_bond_constraints = np.array(planar_bond_constraint_data, dtype=PlanarBondConstraint)
    planar_ring_5_constraints = np.array(planar_ring_5_constraint_data, dtype=PlanarRing5Constraint)
    planar_ring_6_constraints = np.array(planar_ring_6_constraint_data, dtype=PlanarRing6Constraint)

    data = Structure(
        atoms=atoms,
        bonds=bonds,
        residues=residues,
        chains=chains,
        connections=connections,
        interfaces=interfaces,
        mask=mask,
        # New Glycan fields
        glycosylation_sites=np.array(glycosylation_sites_data, dtype=GlycosylationSite) if glycosylation_sites_data else None,
        glycan_feature_map=global_glycan_feature_map,
        atom_to_mono_idx_map=global_atom_to_mono_idx_map
    )

    # Metadata
    struct_info = StructureInfo(num_chains=len(chains))
    chain_infos = []
    for chain in chains:
        chain_infos.append(ChainInfo(
            chain_id=int(chain["asym_id"]),
            chain_name=chain["name"],
            mol_type=int(chain["mol_type"]),
            cluster_id=-1,
            msa_id=chain_to_msa[chain["name"]],
            num_residues=int(chain["res_num"]),
            valid=True,
            entity_id=int(chain["entity_id"]),
        ))

    flat_pocket_residues = [item for sublist in pocket_residues for item in sublist] if pocket_residues else None
    
    options = InferenceOptions(binders=pocket_binders, pocket=flat_pocket_residues)

    record = Record(
        id=name,
        structure=struct_info,
        chains=chain_infos,
        interfaces=[],
        inference_options=options,
    )

    residue_constraints = ResidueConstraints(
        rdkit_bounds_constraints=rdkit_bounds_constraints,
        chiral_atom_constraints=chiral_atom_constraints,
        stereo_bond_constraints=stereo_bond_constraints,
        planar_bond_constraints=planar_bond_constraints,
        planar_ring_5_constraints=planar_ring_5_constraints,
        planar_ring_6_constraints=planar_ring_6_constraints,
    )

    return Target(
        record=record,
        structure=data,
        sequences=entity_to_seq,
        residue_constraints=residue_constraints,
    )
    
def parse_standard_residue_with_bonds(
    name: str,
    ref_mol: Mol,
    res_idx: int,
) -> Optional[ParsedResidue]:
    """
    Parses a standard amino acid using a curated atom list (from const.py)
    but generates its internal bond graph from the full RDKit component.
    This avoids including unwanted atoms like OXT for internal residues.
    """
    ref_mol_no_h = AllChem.RemoveHs(ref_mol, sanitize=False)
    ref_conformer = get_conformer(ref_mol_no_h)
    unk_chirality = const.chirality_type_ids[const.unk_chirality_type]

    # Get the curated list of atom names for an internal residue
    ref_atom_names = const.ref_atoms.get(name)
    if ref_atom_names is None:
        raise ValueError(f"Residue '{name}' not found in const.ref_atoms.")

    # Create a map of all atoms in the full RDKit molecule
    full_atom_map = {a.GetProp("name"): a for a in ref_mol_no_h.GetAtoms()}

    # Parse only the atoms present in our curated list
    atoms: list[ParsedAtom] = []
    # Map from the RDKit atom's original index to its new local index in our list
    rdkit_idx_to_local_idx: Dict[int, int] = {}
    for local_idx, atom_name in enumerate(ref_atom_names):
        if atom_name not in full_atom_map:
            continue # Should not happen if CCD and const.py are consistent
        
        ref_atom = full_atom_map[atom_name]
        rdkit_idx = ref_atom.GetIdx()
        ref_coords = ref_conformer.GetAtomPosition(rdkit_idx)

        atoms.append(
            ParsedAtom(
                name=atom_name,
                element=ref_atom.GetAtomicNum(),
                charge=ref_atom.GetFormalCharge(),
                coords=(0, 0, 0),
                conformer=(ref_coords.x, ref_coords.y, ref_coords.z),
                is_present=True,
                chirality=const.chirality_type_ids.get(str(ref_atom.GetChiralTag()), unk_chirality),
            )
        )
        rdkit_idx_to_local_idx[rdkit_idx] = local_idx

    # Generate bonds ONLY between the atoms we have kept
    bonds: list[ParsedBond] = []
    for bond in ref_mol_no_h.GetBonds():
        start_idx_rdkit, end_idx_rdkit = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if start_idx_rdkit in rdkit_idx_to_local_idx and end_idx_rdkit in rdkit_idx_to_local_idx:
            start_idx_local = rdkit_idx_to_local_idx[start_idx_rdkit]
            end_idx_local = rdkit_idx_to_local_idx[end_idx_rdkit]
            bond_type = const.bond_type_ids.get(bond.GetBondType().name, const.bond_type_ids[const.unk_bond_type])
            bonds.append(ParsedBond(min(start_idx_local, end_idx_local), max(start_idx_local, end_idx_local), bond_type))
            
    return ParsedResidue(
        name=name,
        type=const.token_ids[name],
        idx=res_idx,
        atoms=atoms,
        bonds=bonds,
        orig_idx=None,
        atom_center=const.res_to_center_atom_id[name],
        atom_disto=const.res_to_disto_atom_id[name],
        is_standard=True, # Start with True, will be flipped later
        is_present=True,
    )

def _remove_atom(residue: ParsedResidue, atom_name_to_remove: str) -> ParsedResidue:
    """
    Physically removes an atom from a ParsedResidue and re-indexes bonds and properties.
    Used to enforce stoichiometry (e.g. removing leaving oxygen in glycosidic bonds).
    """
    # 1. Find the index of the atom to remove
    remove_idx = -1
    for i, atom in enumerate(residue.atoms):
        if atom.name.upper() == atom_name_to_remove.upper():
            remove_idx = i
            break
    
    if remove_idx == -1:
        # Atom not found; assume it was already removed or not present. Return as is.
        return residue

    # 2. Build new atom list and index map
    new_atoms = []
    old_to_new_map = {}
    new_counter = 0
    
    for i, atom in enumerate(residue.atoms):
        if i == remove_idx:
            continue
        new_atoms.append(atom)
        old_to_new_map[i] = new_counter
        new_counter += 1

    # 3. Rebuild bonds with new indices
    new_bonds = []
    for bond in residue.bonds:
        # If bond involves removed atom, drop it
        if bond.atom_1 == remove_idx or bond.atom_2 == remove_idx:
            continue
        # Map remaining indices
        n1 = old_to_new_map[bond.atom_1]
        n2 = old_to_new_map[bond.atom_2]
        new_bonds.append(ParsedBond(min(n1, n2), max(n1, n2), bond.type))

    # 4. Remap centers
    # If center was removed, default to 0
    new_center = old_to_new_map.get(residue.atom_center, 0) 
    new_disto = old_to_new_map.get(residue.atom_disto, 0)

    return replace(
        residue,
        atoms=new_atoms,
        bonds=new_bonds,
        atom_center=new_center,
        atom_disto=new_disto
    )


# --- New helper classes and functions for branching ---
class GlycanToken:
    def __init__(self, token_type: str, value: str, bond_spec: Optional[Tuple[str, int, int]] = None):
        """
        token_type: 'residue', 'open', 'close', 'open_curly', 'close_curly'
        value: For residues, the monosaccharide code (e.g. "FRU")
        bond_spec: If the token is a residue and has a bond spec (e.g. from "(a1-4)"),
                   then (alpha_beta_char, donor_num, acceptor_num) e.g. ('a', 1, 4)
        """
        self.type = token_type
        self.value = value
        self.bond_spec = bond_spec
        self.residue_index: Optional[int] = None # will be set after residues are created
        self.is_cyclic_acceptor: bool = False # The special flag

def tokenize_cyclodextrin(iupac: str) -> List[GlycanToken]:
    pattern = re.compile(r'([A-Z0-9]+)(?:\(([abAB])(\d+)-(\d+)\))?|([\[\]{}])')
    tokens = []
    pos = 0
    while pos < len(iupac):
        match = pattern.match(iupac, pos)
        if not match:
            raise ValueError(f"Could not parse IUPAC string starting at: {iupac[pos:]}")

        if match.group(1):
            code = match.group(1)
            bond_spec = None
            if match.group(2):
                alpha_beta = match.group(2).lower()
                donor_num = int(match.group(3))
                acceptor_num = int(match.group(4))
                bond_spec = (alpha_beta, donor_num, acceptor_num)
            tokens.append(GlycanToken('residue', code, bond_spec))
            pos = match.end()
        elif match.group(5):
            symbol = match.group(5)
            type_map = {'[': 'open', ']': 'close', '{': 'open_curly', '}': 'close_curly'}
            tokens.append(GlycanToken(type_map[symbol], symbol))
            pos = match.end()
        else:
             raise ValueError(f"Unexpected parsing state at: {iupac[pos:]}")

    for i, token in enumerate(tokens):
        if token.type == 'open_curly':
            if i == 0:
                raise ValueError("Cyclic notation '{' cannot be at the start of the string.")
            acceptor_token = tokens[i-1]
            if acceptor_token.type != 'residue':
                raise ValueError(f"The character before '{{' must be a residue, but found '{acceptor_token.value}'.")
            acceptor_token.is_cyclic_acceptor = True

    return tokens

def compute_cyclodextrin_bonds(iupac: str, ccd: Mapping[str, Mol]) -> Tuple[List[ParsedResidue], List[Tuple[int, int, Tuple[str, int, int]]]]:
    """
    Parses a cyclodextrin IUPAC string.
    Ensures the cycle is closed correctly from the last residue (Donor) to the flagged acceptor.
    """
    tokens = tokenize_cyclodextrin(iupac)
    residues: List[ParsedResidue] = []
    
    cyclic_acceptor_original_idx: Optional[int] = None
    last_residue_token: Optional[GlycanToken] = None

    # 1. Create Residue Objects
    for token in tokens:
        if token.type == 'residue':
            idx = len(residues)
            token.residue_index = idx
            last_residue_token = token
            
            if token.is_cyclic_acceptor:
                cyclic_acceptor_original_idx = idx
            
            lookup_name = token.value

            # --- IDENTICAL & ANOMERIC MAPPING CORRECTION ---
            config = None
            if token.bond_spec:
                config = token.bond_spec[0]
                
            # 1. Collapse Identical Sugars
            lookup_name = const.IDENTICAL_MAP.get(lookup_name, lookup_name)
                
            # 2. Apply Anomeric Correction
            if config in ['a', 'b'] and lookup_name in const.ANOMER_MAP:
                lookup_name = const.ANOMER_MAP[lookup_name][config]
                
            token.value = lookup_name
            # -----------------------------------
            
            if lookup_name not in ccd:
                raise ValueError(f"CCD code '{lookup_name}' not found.")
            
            res = parse_ccd_residue(token.value, ccd[lookup_name], res_idx=idx)
            residues.append(res)

    if cyclic_acceptor_original_idx is None:
        raise ValueError("Cyclodextrin notation must contain a '{' marking the acceptor.")

    # 2. Parse Linear Connections (Stack-based)
    stack: List[GlycanToken] = []
    linear_connections: List[Tuple[int, int, Tuple[str, int, int]]] = []
    
    linear_tokens = [t for t in tokens if t.type in ('residue', 'open', 'close')]

    for token in reversed(linear_tokens):
        if token.type == 'residue':
            if not stack:
                stack.append(token)
            else:
                if token.bond_spec is not None:
                    if stack and stack[-1].type == 'close':
                        stack.pop() # Remove ']'
                        target = stack[-1]
                        linear_connections.append((target.residue_index, token.residue_index, token.bond_spec))
                        stack.append(target) 
                        stack.append(token)
                    elif stack and stack[-1].type == 'residue':
                        target = stack[-1]
                        linear_connections.append((target.residue_index, token.residue_index, token.bond_spec))
                        stack.pop()
                        stack.append(token)
                else:
                     stack.append(token)
        elif token.type == 'close':
            stack.append(token)
        elif token.type == 'open':
            while stack and stack[-1].type != 'close':
                stack.pop()
            if stack and stack[-1].type == 'close':
                stack.pop()

    # 3. Parse Cyclic Bond Spec
    cyclic_match = re.search(r'(\w+)\(([abAB])(\d+)-(\d+)\)\s*\}$', iupac)
    if not cyclic_match:
        raise ValueError(f"Could not parse cyclic bond specification from: {iupac}")

    alpha_beta = cyclic_match.group(2).lower()
    donor_num = int(cyclic_match.group(3))
    acceptor_num = int(cyclic_match.group(4))
    cyclic_bond_spec = (alpha_beta, donor_num, acceptor_num)

    # 4. Reorder residues (BFS based on LINEAR connections only)
    if not residues:
        return [], []

    child_indices = {c[1] for c in linear_connections}
    root_candidates = [i for i in range(len(residues)) if i not in child_indices]
    if len(root_candidates) != 1:
        raise ValueError(f"Linear glycan parsing error: Found {len(root_candidates)} possible roots. Expected 1.")
    
    root_idx = root_candidates[0]
    
    adj = {i: [] for i in range(len(residues))}
    for p, c, _ in linear_connections:
        adj[p].append(c)

    new_order_indices = []
    queue = [root_idx]
    visited = {root_idx}
    
    while queue:
        parent_idx = queue.pop(0)
        new_order_indices.append(parent_idx)
        for child_idx in sorted(adj.get(parent_idx, [])):
            if child_idx not in visited:
                visited.add(child_idx)
                queue.append(child_idx)

    reordered_residues = [residues[i] for i in new_order_indices]
    old_to_new_idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(new_order_indices)}
    
    final_connections = [
        (old_to_new_idx_map[p], old_to_new_idx_map[c], spec)
        for p, c, spec in linear_connections
    ]

    # 5. Add Cyclic Connection
    if last_residue_token is None:
         raise ValueError("No residues found.")

    cyclic_acceptor_new_idx = old_to_new_idx_map[cyclic_acceptor_original_idx]
    cyclic_donor_new_idx = old_to_new_idx_map[last_residue_token.residue_index]

    final_connections.append((cyclic_acceptor_new_idx, cyclic_donor_new_idx, cyclic_bond_spec))

    return reordered_residues, final_connections
    
def tokenize_glycan(iupac: str) -> List[GlycanToken]:
    """
    Tokenize the glycan IUPAC string into monosaccharide tokens and branch markers.
    
    CRITICAL CHANGE: This regex now strictly enforces that if parentheses exist 
    after a residue code, they MUST contain linkage numbers (e.g. 'a1-4'). 
    It will NOT match 'NAG(a)' or 'NAG(b)'. Those Root cases must be handled 
    by pre-processing in compute_branching_bonds.
    """
    # Regex Breakdown:
    # 1. ([A-Z0-9]+)\(([abAB])(\d+)-(\d+)\) 
    #    -> Matches Code + Anomer + Donor + Acceptor (digits MANDATORY)
    # 2. ([A-Z0-9]+)
    #    -> Matches Code only (e.g. GAL)
    # 3. ([\[\]])
    #    -> Matches Brackets
    
    pattern = re.compile(r'([A-Z0-9]+)\(([abAB])(\d+)-(\d+)\)|([A-Z0-9]+)|([\[\]])')
    
    tokens = []
    pos = 0
    while pos < len(iupac):
        match = pattern.match(iupac, pos)
        if not match:
            # Diagnostic snippet
            snippet = iupac[pos:pos+10]
            raise ValueError(f"Could not parse IUPAC string starting at: {snippet}...")

        # Case 1: Residue WITH Full Bond Spec (e.g. BMA(b1-4))
        if match.group(1):
            code = match.group(1)
            alpha_beta = match.group(2).lower()
            donor_num = int(match.group(3))
            acceptor_num = int(match.group(4))
            
            bond_spec = (alpha_beta, donor_num, acceptor_num)
            tokens.append(GlycanToken('residue', code, bond_spec))
            pos = match.end()

        # Case 2: Residue WITHOUT Bond Spec (e.g. BMA)
        elif match.group(5):
            code = match.group(5)
            tokens.append(GlycanToken('residue', code, None))
            pos = match.end()

        # Case 3: Bracket
        elif match.group(6):
            symbol = match.group(6)
            if symbol == '[':
                tokens.append(GlycanToken('open', symbol))
            elif symbol == ']':
                tokens.append(GlycanToken('close', symbol))
            pos = match.end()
            
        else:
             raise ValueError(f"Unexpected parsing state at: {iupac[pos:]}")

    return tokens

def compute_branching_bonds(iupac: str, ccd: Mapping[str, Mol]) -> Tuple[List[ParsedResidue], List[Tuple[int, int, Tuple[str, int, int]]], Optional[Tuple[str, int, int]]]:
    """
    Implements the branching heuristic with robust Anomer detection.
    """
    # 1. Detect and Strip Root Anomer Config manually
    root_anomer_match = re.search(r'\(([abAB])\)$', iupac.strip())
    
    root_override_spec = None
    clean_iupac = iupac
    
    if root_anomer_match:
        anomer_char = root_anomer_match.group(1).lower()
        root_override_spec = (anomer_char, None, None)
        clean_iupac = iupac[:root_anomer_match.start()]

    # 2. Tokenize the cleaned string
    tokens = tokenize_glycan(clean_iupac)
    residues: List[ParsedResidue] = []
    
    # 3. Create Residue Objects
    for token in tokens:
        if token.type == 'residue':
            idx = len(residues)
            token.residue_index = idx
            
            lookup_name = token.value

            # --- IDENTICAL & ANOMERIC MAPPING CORRECTION ---
            config = None
            if token.bond_spec:
                config = token.bond_spec[0]
            elif root_override_spec:
                config = root_override_spec[0]
                
            # 1. Collapse Identical Sugars
            lookup_name = const.IDENTICAL_MAP.get(lookup_name, lookup_name)
            
            # 2. Apply Anomeric Correction
            if config in ['a', 'b'] and lookup_name in const.ANOMER_MAP:
                lookup_name = const.ANOMER_MAP[lookup_name][config]
                
            token.value = lookup_name
            # -----------------------------------

            if lookup_name not in ccd:
                raise ValueError(f"CCD structure for glycan residue '{lookup_name}' not found!")
            
            # Parse
            res = parse_ccd_residue(token.value, ccd[lookup_name], res_idx=idx)
            if res is None:
                 raise ValueError(f"Failed to parse CCD residue '{token.value}'")
            residues.append(res)

    # 4. Parse Connections (Right-to-Left Stack)
    stack: List[GlycanToken] = []
    connections: List[Tuple[int, int, Tuple[str, int, int]]] = [] 

    for token in reversed(tokens):
        if token.type == 'residue':
            if not stack:
                stack.append(token)
            else:
                if token.bond_spec is not None:
                    if stack and stack[-1].type == 'close':
                        close_token = stack.pop()
                        if not stack or stack[-1].type != 'residue':
                            raise ValueError("Malformed glycan string: expected residue below close bracket.")
                        target = stack[-1]
                        connections.append((target.residue_index, token.residue_index, token.bond_spec))
                        stack.append(target)
                        stack.append(close_token)
                        stack.append(token)
                    elif stack and stack[-1].type == 'residue':
                        target = stack[-1]
                        connections.append((target.residue_index, token.residue_index, token.bond_spec))
                        stack.pop()
                        stack.append(token)
                    else:
                        raise ValueError(f"Malformed glycan string: unexpected token {stack[-1].type if stack else 'empty'} after residue {token.value}")
                else:
                     if stack:
                          if stack[-1].type == 'residue':
                              raise ValueError(f"Residue '{token.value}' (index {token.residue_index}) is missing bond specification but is followed by '{stack[-1].value}'.")
                     stack.append(token)

        elif token.type == 'close':
            stack.append(token)
        elif token.type == 'open':
            while stack and stack[-1].type != 'close':
                stack.pop()
            if stack and stack[-1].type == 'close':
                stack.pop()

    # 5. Reorder residues to be root-first (BFS)
    if not residues:
        return [], [], None
    
    num_residues = len(residues)
    
    if not connections: 
        root_idx = 0 
    else:
        child_indices = {c[1] for c in connections}
        root_candidates = [i for i in range(num_residues) if i not in child_indices]
        if len(root_candidates) != 1:
            raise ValueError(f"Glycan parsing error: Found {len(root_candidates)} possible roots. Expected 1.")
        root_idx = root_candidates[0]

    adj = {i: [] for i in range(num_residues)}
    for p, c, _ in connections:
        adj[p].append(c)

    new_order_indices = []
    queue = [root_idx]
    visited = {root_idx}
    
    while queue:
        parent_idx = queue.pop(0)
        new_order_indices.append(parent_idx)
        for child_idx in sorted(adj.get(parent_idx, [])): 
            if child_idx not in visited:
                visited.add(child_idx)
                queue.append(child_idx)

    reordered_residues = [residues[i] for i in new_order_indices]
    old_to_new_idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(new_order_indices)}
    reordered_connections = [
        (old_to_new_idx_map[p], old_to_new_idx_map[c], spec)
        for p, c, spec in connections
    ]

    # 6. Determine Final Root Bond Spec
    root_bond_spec = root_override_spec
    
    if root_bond_spec is None:
        for t in tokens:
            if t.type == 'residue' and t.residue_index == root_idx:
                root_bond_spec = t.bond_spec
                break

    return reordered_residues, reordered_connections, root_bond_spec

def _build_adjacency_from_parsed(residue: ParsedResidue) -> Dict[int, List[int]]:
    """Builds an adjacency list from a ParsedResidue's bond list."""
    adj = {}
    # Initialize all atom indices
    for i in range(len(residue.atoms)):
        adj[i] = []
    
    for bond in residue.bonds:
        adj[bond.atom_1].append(bond.atom_2)
        adj[bond.atom_2].append(bond.atom_1)
    return adj

def _find_ring_atom_names(residue: ParsedResidue) -> set[str]:
    """
    Finds atoms in the ring using the exact DFS logic from the training script.
    Returns a set of atom names (e.g., {'C1', 'C2', 'C3', 'C4', 'C5', 'O5'}).
    """
    graph = _build_adjacency_from_parsed(residue)
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
                    cycle_path_indices = path[cycle_start_index:]
                    
                    # Basic chemical ring filter (usually 5 or 6 atoms for sugars)
                    if len(cycle_path_indices) >= 3:
                        return {residue.atoms[idx].name.upper() for idx in cycle_path_indices}
                else:
                    stack.append((neighbor, curr, path + [neighbor]))
    
    return set()

def _identify_anomeric_oxygen_target(residue: ParsedResidue) -> Optional[str]:
    """
    Determines if O1 or O2 should be removed based on ring topology.
    Matches logic in preprocess_glycans.py:
    1. If C1 is in ring -> O1
    2. If C2 is in ring (and C1 is not) -> O2
    """
    ring_atom_names = _find_ring_atom_names(residue)
    
    if not ring_atom_names:
        return None

    if 'C1' in ring_atom_names:
        return 'O1'
    elif 'C2' in ring_atom_names:
        # Implicitly, if C1 was there, we matched above. 
        # So this handles the "C2 in ring AND C1 not in ring" case.
        return 'O2'
        
    return None

def parse_glycan(
    iupac: str,
    ccd: Mapping[str, Mol]
) -> Tuple[List[ParsedResidue], List[Tuple[int, int, int, int]], Dict[int, dict], np.ndarray]:
    """
    (OVERHAULED) Parses a glycan IUPAC string into a topology of residues.
    Handles both linear/branched glycans and cyclodextrins.
    
    Features:
    - Stoichiometry: 
        - Non-Root Residues: Anomeric oxygen (O1/O2) is REMOVED.
        - Root Residue: 
            - If config provided (e.g., "GAL(a)"), O1/O2 is KEPT.
            - If no config (e.g., "GAL"), O1/O2 is REMOVED.
    - Connections: Returns atom-level indices based on the filtered atom lists.
    - Correct Indexing: Ensures res.idx matches the topological list index.
    - Linkage Support: Supports O, S, N, and C linked sugars.
    """
    
    root_config_spec = None
    
    # 1. Determine Topology (Residues + Abstract Connections)
    try:
        if '{' in iupac:
             # Cyclodextrins don't strictly have a "root" in the linear sense, usually treated as all connected.
             # Standard logic applies.
             residues, raw_connections = compute_cyclodextrin_bonds(iupac, ccd)
        else:
             residues, raw_connections, root_config_spec = compute_branching_bonds(iupac, ccd)
    except ValueError as e:
        raise ValueError(f"Error parsing glycan IUPAC '{iupac}': {e}") from e

    if not residues:
        return [], [], {}, np.array([])

    # --- RE-INDEXING STEP ---
    residues_reindexed = []
    for new_idx, res in enumerate(residues):
        residues_reindexed.append(replace(res, idx=new_idx))
    
    # Use the re-indexed list for all subsequent operations
    residues_modified = list(residues_reindexed) 
    
    # The Root is always index 0 after reordering in compute_branching_bonds
    root_idx = 0 
    
    # --- STOICHIOMETRY ADJUSTMENT STEP ---
    for i in range(len(residues_modified)):
        # Determine if we should keep the oxygen
        # Only keep if it is the Root AND the Root has an explicit anomeric config
        keep_anomeric_oxygen = (i == root_idx) and (root_config_spec is not None)

        if keep_anomeric_oxygen:
            # Do not delete the oxygen for this residue
            continue

        target_o = _identify_anomeric_oxygen_target(residues_modified[i])
        
        if target_o:
            # Physically remove the atom and re-index the residue
            residues_modified[i] = _remove_atom(residues_modified[i], target_o)

    # 2. Resolve Atomic Connections (Using Post-Deletion Indices)
    connection_indices = []
    
    for parent_idx, child_idx, bond_spec in raw_connections:
        parent_res = residues_modified[parent_idx]
        child_res = residues_modified[child_idx]
        _, donor_num, acceptor_num = bond_spec
        
        # Define connecting atoms (Parent Atom -> Child Carbon)
        # FIX: Support O, S, N, C linkages (e.g. O4, S4, N4)
        donor_c_name = f"C{donor_num}"
        
        potential_acceptor_names = [
            f"O{acceptor_num}", 
            f"S{acceptor_num}", 
            f"N{acceptor_num}", 
            f"C{acceptor_num}"
        ]
        
        p_atom_idx = -1
        found_acceptor_name = "None"

        # Try to find one of the valid acceptor atoms in the parent residue
        for candidate in potential_acceptor_names:
            try:
                p_atom_idx = next(i for i, a in enumerate(parent_res.atoms) if a.name.upper() == candidate.upper())
                found_acceptor_name = candidate
                break
            except StopIteration:
                continue

        try:
            # If we didn't find the parent atom, raise error
            if p_atom_idx == -1:
                 raise StopIteration("Parent linkage atom not found")

            # Find child carbon index
            c_atom_idx = next(i for i, a in enumerate(child_res.atoms) if a.name.upper() == donor_c_name.upper())
            
            connection_indices.append((parent_idx, child_idx, p_atom_idx, c_atom_idx))
                    
        except StopIteration:
            p_atoms = [a.name for a in parent_res.atoms]
            c_atoms = [a.name for a in child_res.atoms]
            raise ValueError(
                f"Topology Error: Could not link {parent_res.name} (idx {parent_idx}) to {child_res.name} (idx {child_idx}).\n"
                f"  Expected Parent Atom: One of {potential_acceptor_names} (Available: {p_atoms})\n"
                f"  Expected Child Atom: {donor_c_name} (Available: {c_atoms})"
            )

    # 3. Build Feature Maps
    prelim_features: Dict[int, dict] = {}
    
    # Map child_idx -> bond_spec to extract anomeric config for features
    connections_by_child = {c: spec for _, c, spec in raw_connections}
    
    for i, res in enumerate(residues_modified):
        anomeric = None
        
        if i == root_idx and root_config_spec is not None:
            # Root specific config
            anomeric = root_config_spec[0]
        else:
            # Internal linkage config
            bond_spec = connections_by_child.get(i)
            anomeric = bond_spec[0] if bond_spec else None
            
        prelim_features[i] = {"ccd_code": res.name, "anomeric_config": anomeric}

    # 4. Build Atom-to-Mono Index Map
    atom_to_mono_idx_list = []
    for i, res in enumerate(residues_modified):
        atom_to_mono_idx_list.extend([i] * len(res.atoms))
    atom_to_mono_idx_array = np.array(atom_to_mono_idx_list, dtype=np.int32)

    return residues_modified, connection_indices, prelim_features, atom_to_mono_idx_array
