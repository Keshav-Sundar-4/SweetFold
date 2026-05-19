import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Optional, Union, Dict, Any, Tuple

import numpy as np
from mashumaro.mixins.dict import DataClassDictMixin
try:
    from boltz.data.parse.schema import MonosaccharideFeatureMapType
except ImportError:
    MonosaccharideFeatureMapType = Dict[Tuple[int, int], Any]

####################################################################################################
# SERIALIZABLE
####################################################################################################


class NumpySerializable:
    """Serializable datatype."""

    @classmethod
    def load(cls: "NumpySerializable", path: Path) -> "NumpySerializable":
        """Load the object from an NPZ file.

        Parameters
        ----------
        path : Path
            The path to the file.

        Returns
        -------
        Serializable
            The loaded object.

        """
        return cls(**np.load(path))

    def dump(self, path: Path) -> None:
        """Dump the object to an NPZ file.

        Parameters
        ----------
        path : Path
            The path to the file.

        """
        np.savez_compressed(str(path), **asdict(self))


class JSONSerializable(DataClassDictMixin):
    """Serializable datatype."""

    @classmethod
    def load(cls: "JSONSerializable", path: Path) -> "JSONSerializable":
        """Load the object from a JSON file.

        Parameters
        ----------
        path : Path
            The path to the file.

        Returns
        -------
        Serializable
            The loaded object.

        """
        with path.open("r") as f:
            return cls.from_dict(json.load(f))

    def dump(self, path: Path) -> None:
        """Dump the object to a JSON file.

        Parameters
        ----------
        path : Path
            The path to the file.

        """
        with path.open("w") as f:
            json.dump(self.to_dict(), f)


####################################################################################################
# STRUCTURE
####################################################################################################

Atom = [
    ("name", np.dtype("4i1")),
    ("element", np.dtype("i1")),
    ("charge", np.dtype("i1")),
    ("coords", np.dtype("3f4")),
    ("conformer", np.dtype("3f4")),
    ("is_present", np.dtype("?")),
    ("chirality", np.dtype("i1")),
]

Bond = [
    ("atom_1", np.dtype("i4")),
    ("atom_2", np.dtype("i4")),
    ("type", np.dtype("i1")),
]

Residue = [
    ("name", np.dtype("<U5")),
    ("res_type", np.dtype("i1")),
    ("res_idx", np.dtype("i4")),
    ("atom_idx", np.dtype("i4")),
    ("atom_num", np.dtype("i4")),
    ("atom_center", np.dtype("i4")),
    ("atom_disto", np.dtype("i4")),
    ("is_standard", np.dtype("?")),
    ("is_present", np.dtype("?")),
]

Chain = [
    ("name", np.dtype("<U5")),
    ("mol_type", np.dtype("i1")),
    ("entity_id", np.dtype("i4")),
    ("sym_id", np.dtype("i4")),
    ("asym_id", np.dtype("i4")),
    ("atom_idx", np.dtype("i4")),
    ("atom_num", np.dtype("i4")),
    ("res_idx", np.dtype("i4")),
    ("res_num", np.dtype("i4")),
    ("cyclic_period", np.dtype("i4")),
]

Connection = [
    ("chain_1", np.dtype("i4")),
    ("chain_2", np.dtype("i4")),
    ("res_1", np.dtype("i4")),
    ("res_2", np.dtype("i4")),
    ("atom_1", np.dtype("i4")),
    ("atom_2", np.dtype("i4")),
]

Interface = [
    ("chain_1", np.dtype("i4")),
    ("chain_2", np.dtype("i4")),
]

GlycosylationSite = [
    ("protein_chain_id", np.dtype("i4")),
    ("protein_res_id", np.dtype("i4")),
    ("protein_atom_name", np.dtype("<U4")),
    ("glycan_chain_id", np.dtype("i4")),    
    ("glycan_res_id", np.dtype("i4")),
    ("glycan_atom_name", np.dtype("<U4")),
]


@dataclass(frozen=True)
class Structure(NumpySerializable):
    """Structure datatype."""

    atoms: np.ndarray
    bonds: np.ndarray
    residues: np.ndarray
    chains: np.ndarray
    connections: np.ndarray
    interfaces: np.ndarray
    mask: np.ndarray
    # ADDED: Glycan-specific fields from Boltz-Glycan
    glycosylation_sites: Optional[np.ndarray] = None
    glycan_feature_map: Optional[MonosaccharideFeatureMapType] = None
    atom_to_mono_idx_map: Optional[Dict[int, np.ndarray]] = None


    @classmethod
    def load(cls: "Structure", path: Path) -> "Structure":
        """
        (Corrected Version) Load a structure from an NPZ file, handling all fields.
        """
        data = np.load(path, allow_pickle=True)
        
        glycan_feature_map_raw = data.get("glycan_feature_map", None)
        glycan_feature_map = glycan_feature_map_raw.item() if glycan_feature_map_raw is not None and glycan_feature_map_raw.shape == () else None

        atom_to_mono_idx_map_raw = data.get("atom_to_mono_idx_map", None)
        atom_to_mono_idx_map = atom_to_mono_idx_map_raw.item() if atom_to_mono_idx_map_raw is not None and atom_to_mono_idx_map_raw.shape == () else None
        
        glycosylation_sites = data.get("glycosylation_sites", None)
        
        # FIX: Robustly handle 0-d arrays containing None (serialized from np.savez)
        if glycosylation_sites is not None:
             if glycosylation_sites.shape == () and glycosylation_sites.item() is None:
                 glycosylation_sites = None
             elif glycosylation_sites.size == 0:
                 glycosylation_sites = None

        return cls(
            atoms=data["atoms"],
            bonds=data["bonds"],
            residues=data["residues"],
            chains=data["chains"],
            connections=data.get("connections", np.array([], dtype=Connection)),
            interfaces=data.get("interfaces", np.array([], dtype=Interface)),
            mask=data["mask"],
            glycosylation_sites=glycosylation_sites,
            glycan_feature_map=glycan_feature_map,
            atom_to_mono_idx_map=atom_to_mono_idx_map,
        )
        
    def remove_invalid_chains(self) -> "Structure":  # noqa: PLR0915
        """
        Filters the structure to include only chains marked as True in self.mask.
        Re-indexes all structures and updates references.
        """
        entity_counter = {}
        atom_idx, res_idx, chain_idx = 0, 0, 0
        atoms, residues, chains = [], [], []
        atom_map, res_map, chain_map = {}, {}, {}

        for i, chain in enumerate(self.chains):
            if not self.mask[i]:
                continue

            entity_id = chain["entity_id"]
            entity_counter[entity_id] = entity_counter.get(entity_id, -1) + 1

            new_chain = chain.copy()
            new_chain["atom_idx"], new_chain["res_idx"], new_chain["asym_id"], new_chain["sym_id"] = \
                atom_idx, res_idx, chain_idx, entity_counter[entity_id]
            chains.append(new_chain)
            
            # Map Old Chain Index (i) -> New Chain Index (chain_idx)
            chain_map[i] = chain_idx
            chain_idx += 1

            res_start, res_end = chain["res_idx"], chain["res_idx"] + chain["res_num"]
            for j, res in enumerate(self.residues[res_start:res_end]):
                new_res = res.copy()
                new_res["atom_idx"] = atom_idx
                new_res["atom_center"] = atom_idx + new_res["atom_center"] - res["atom_idx"]
                new_res["atom_disto"] = atom_idx + new_res["atom_disto"] - res["atom_idx"]
                residues.append(new_res)
                res_map[res_start + j] = res_idx
                res_idx += 1

                start, end = res["atom_idx"], res["atom_idx"] + res["atom_num"]
                atoms.append(self.atoms[start:end])
                atom_map.update({k: atom_idx + k - start for k in range(start, end)})
                atom_idx += res["atom_num"]

        # 1. Update glycosylation sites
        new_glycosylation_sites = []
        if self.glycosylation_sites is not None and self.glycosylation_sites.size > 0:
            for site in self.glycosylation_sites:
                old_p_chain_id, old_g_chain_id = site["protein_chain_id"], site["glycan_chain_id"]
                if old_p_chain_id in chain_map and old_g_chain_id in chain_map:
                    new_site = site.copy()
                    new_site["protein_chain_id"] = chain_map[old_p_chain_id]
                    new_site["glycan_chain_id"] = chain_map[old_g_chain_id]
                    new_glycosylation_sites.append(new_site)

        updated_sites_arr = np.array(new_glycosylation_sites, dtype=GlycosylationSite) if new_glycosylation_sites else np.array([], dtype=GlycosylationSite)

        # 2. Update Glycan Feature Map Keys
        new_glycan_feature_map = {}
        if self.glycan_feature_map is not None:
            for (old_chain_idx, mono_idx), feature_obj in self.glycan_feature_map.items():
                if old_chain_idx in chain_map:
                    new_chain_idx = chain_map[old_chain_idx]
                    # Update the key with the new chain index
                    new_glycan_feature_map[(new_chain_idx, mono_idx)] = feature_obj

        # 3. Update Atom-to-Mono-Idx Map Keys
        new_atom_to_mono_idx_map = {}
        if self.atom_to_mono_idx_map is not None:
            for old_chain_idx, array_data in self.atom_to_mono_idx_map.items():
                if old_chain_idx in chain_map:
                    new_chain_idx = chain_map[old_chain_idx]
                    # Update the key with the new chain index
                    new_atom_to_mono_idx_map[new_chain_idx] = array_data

        # Rebuild final numpy arrays
        atoms = np.concatenate(atoms, dtype=Atom) if atoms else np.array([], dtype=Atom)
        residues = np.array(residues, dtype=Residue)
        chains = np.array(chains, dtype=Chain)

        bonds = [
            (atom_map[b["atom_1"]], atom_map[b["atom_2"]], b["type"])
            for b in self.bonds
            if b["atom_1"] in atom_map and b["atom_2"] in atom_map
        ]

        connections = [
            (chain_map[c["chain_1"]], chain_map[c["chain_2"]], res_map[c["res_1"]], res_map[c["res_2"]], atom_map[c["atom_1"]], atom_map[c["atom_2"]])
            for c in self.connections
            if c["atom_1"] in atom_map and c["atom_2"] in atom_map
        ]

        return Structure(
            atoms=atoms,
            bonds=np.array(bonds, dtype=Bond),
            residues=residues,
            chains=chains,
            connections=np.array(connections, dtype=Connection),
            interfaces=np.array([], dtype=Interface),
            mask=np.ones(len(chains), dtype=bool),
            glycosylation_sites=updated_sites_arr,
            glycan_feature_map=new_glycan_feature_map,     # PASS UPDATED MAP
            atom_to_mono_idx_map=new_atom_to_mono_idx_map, # PASS UPDATED MAP
        )

####################################################################################################
# MSA
####################################################################################################


MSAResidue = [
    ("res_type", np.dtype("i1")),
]

MSADeletion = [
    ("res_idx", np.dtype("i2")),
    ("deletion", np.dtype("i2")),
]

MSASequence = [
    ("seq_idx", np.dtype("i2")),
    ("taxonomy", np.dtype("i4")),
    ("res_start", np.dtype("i4")),
    ("res_end", np.dtype("i4")),
    ("del_start", np.dtype("i4")),
    ("del_end", np.dtype("i4")),
]


@dataclass(frozen=True)
class MSA(NumpySerializable):
    """MSA datatype."""

    sequences: np.ndarray
    deletions: np.ndarray
    residues: np.ndarray


####################################################################################################
# RECORD
####################################################################################################


@dataclass(frozen=True)
class StructureInfo:
    """StructureInfo datatype."""

    resolution: Optional[float] = None
    method: Optional[str] = None
    deposited: Optional[str] = None
    released: Optional[str] = None
    revised: Optional[str] = None
    num_chains: Optional[int] = None
    num_interfaces: Optional[int] = None


@dataclass(frozen=False)
class ChainInfo:
    """ChainInfo datatype."""

    chain_id: int
    chain_name: str
    mol_type: int
    cluster_id: Union[str, int]
    msa_id: Union[str, int]
    num_residues: int
    valid: bool = True
    entity_id: Optional[Union[str, int]] = None


@dataclass(frozen=True)
class InterfaceInfo:
    """InterfaceInfo datatype."""

    chain_1: int
    chain_2: int
    valid: bool = True


@dataclass(frozen=True)
class InferenceOptions:
    """InferenceOptions datatype."""

    binders: list[int]
    pocket: Optional[list[tuple[int, int]]]


@dataclass(frozen=True)
class Record(JSONSerializable):
    """Record datatype."""

    id: str
    structure: StructureInfo
    chains: list[ChainInfo]
    interfaces: list[InterfaceInfo]
    inference_options: Optional[InferenceOptions] = None


####################################################################################################
# RESIDUE CONSTRAINTS
####################################################################################################


RDKitBoundsConstraint = [
    ("atom_idxs", np.dtype("2i4")),
    ("is_bond", np.dtype("?")),
    ("is_angle", np.dtype("?")),
    ("upper_bound", np.dtype("f4")),
    ("lower_bound", np.dtype("f4")),
]

ChiralAtomConstraint = [
    ("atom_idxs", np.dtype("4i4")),
    ("is_reference", np.dtype("?")),
    ("is_r", np.dtype("?")),
]

StereoBondConstraint = [
    ("atom_idxs", np.dtype("4i4")),
    ("is_reference", np.dtype("?")),
    ("is_e", np.dtype("?")),
]

PlanarBondConstraint = [
    ("atom_idxs", np.dtype("6i4")),
]

PlanarRing5Constraint = [
    ("atom_idxs", np.dtype("5i4")),
]

PlanarRing6Constraint = [
    ("atom_idxs", np.dtype("6i4")),
]


@dataclass(frozen=True)
class ResidueConstraints(NumpySerializable):
    """ResidueConstraints datatype."""

    rdkit_bounds_constraints: np.ndarray
    chiral_atom_constraints: np.ndarray
    stereo_bond_constraints: np.ndarray
    planar_bond_constraints: np.ndarray
    planar_ring_5_constraints: np.ndarray
    planar_ring_6_constraints: np.ndarray


####################################################################################################
# TARGET
####################################################################################################


@dataclass(frozen=True)
class Target:
    """Target datatype."""

    record: Record
    structure: Structure
    sequences: Optional[dict[str, str]] = None
    residue_constraints: Optional[ResidueConstraints] = None


@dataclass(frozen=True)
class Manifest(JSONSerializable):
    """Manifest datatype."""

    records: list[Record]

    @classmethod
    def load(cls: "JSONSerializable", path: Path) -> "JSONSerializable":
        """Load the object from a JSON file.

        Parameters
        ----------
        path : Path
            The path to the file.

        Returns
        -------
        Serializable
            The loaded object.

        Raises
        ------
        TypeError
            If the file is not a valid manifest file.

        """
        with path.open("r") as f:
            data = json.load(f)
            if isinstance(data, dict):
                manifest = cls.from_dict(data)
            elif isinstance(data, list):
                records = [Record.from_dict(r) for r in data]
                manifest = cls(records=records)
            else:
                msg = "Invalid manifest file."
                raise TypeError(msg)

        return manifest


####################################################################################################
# INPUT
####################################################################################################


@dataclass(frozen=True)
class Input:
    """Input datatype."""

    structure: Structure
    msa: dict[str, MSA]
    record: Optional[Record] = None
    residue_constraints: Optional[ResidueConstraints] = None


####################################################################################################
# TOKENS
####################################################################################################

Token = [
    ("token_idx", np.dtype("i4")),
    ("atom_idx", np.dtype("i4")),
    ("atom_num", np.dtype("i4")),
    ("res_idx", np.dtype("i4")),
    ("res_type", np.dtype("i1")),
    ("sym_id", np.dtype("i4")),
    ("asym_id", np.dtype("i4")),
    ("entity_id", np.dtype("i4")),
    ("mol_type", np.dtype("i1")),
    ("center_idx", np.dtype("i4")),
    ("disto_idx", np.dtype("i4")),
    ("center_coords", np.dtype("3f4")),
    ("disto_coords", np.dtype("3f4")),
    ("resolved_mask", np.dtype("?")),
    ("disto_mask", np.dtype("?")),
    ("cyclic_period", np.dtype("i4")),
]

TokenBond = [
    ("token_1", np.dtype("i4")),
    ("token_2", np.dtype("i4")),
]


@dataclass(frozen=True)
class Tokenized:
    """Tokenized datatype."""

    tokens: np.ndarray
    bonds: np.ndarray
    structure: Structure
    msa: dict[str, MSA]
    residue_constraints: Optional[ResidueConstraints] = None

