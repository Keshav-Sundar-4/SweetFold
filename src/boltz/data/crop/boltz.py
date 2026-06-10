from dataclasses import replace
from typing import Optional
import sys

import numpy as np
from scipy.spatial.distance import cdist

from boltz.data import const
from boltz.data.crop.cropper import Cropper
from boltz.data.types import Tokenized


def pick_random_token(
    tokens: np.ndarray,
    random: np.random.RandomState,
) -> np.ndarray:
    """Pick a random token from the data.

    Parameters
    ----------
    tokens : np.ndarray
        The token data.
    random : np.ndarray
        The random state for reproducibility.

    Returns
    -------
    np.ndarray
        The selected token.

    """
    return tokens[random.randint(len(tokens))]


def pick_chain_token(
    tokens: np.ndarray,
    chain_id: int,
    random: np.random.RandomState,
) -> np.ndarray:
    """Pick a random token from a chain.

    Parameters
    ----------
    tokens : np.ndarray
        The token data.
    chain_id : int
        The chain ID.
    random : np.ndarray
        The random state for reproducibility.

    Returns
    -------
    np.ndarray
        The selected token.

    """
    # Filter to chain
    chain_tokens = tokens[tokens["asym_id"] == chain_id]

    # Pick from chain, fallback to all tokens
    if chain_tokens.size:
        query = pick_random_token(chain_tokens, random)
    else:
        query = pick_random_token(tokens, random)

    return query


def pick_interface_token(
    tokens: np.ndarray,
    interface: np.ndarray,
    random: np.random.RandomState,
) -> np.ndarray:
    """Pick a random token from an interface.

    Parameters
    ----------
    tokens : np.ndarray
        The token data.
    interface : int
        The interface ID.
    random : np.ndarray
        The random state for reproducibility.

    Returns
    -------
    np.ndarray
        The selected token.

    """
    # Sample random interface
    chain_1 = int(interface["chain_1"])
    chain_2 = int(interface["chain_2"])

    tokens_1 = tokens[tokens["asym_id"] == chain_1]
    tokens_2 = tokens[tokens["asym_id"] == chain_2]

    # If no interface, pick from the chains
    if tokens_1.size and (not tokens_2.size):
        query = pick_random_token(tokens_1, random)
    elif tokens_2.size and (not tokens_1.size):
        query = pick_random_token(tokens_2, random)
    elif (not tokens_1.size) and (not tokens_2.size):
        query = pick_random_token(tokens, random)
    else:
        # If we have tokens, compute distances
        tokens_1_coords = tokens_1["center_coords"]
        tokens_2_coords = tokens_2["center_coords"]

        dists = cdist(tokens_1_coords, tokens_2_coords)
        cuttoff = dists < const.interface_cutoff

        # In rare cases, the interface cuttoff is slightly
        # too small, then we slightly expand it if it happens
        if not np.any(cuttoff):
            cuttoff = dists < (const.interface_cutoff + 5.0)

        tokens_1 = tokens_1[np.any(cuttoff, axis=1)]
        tokens_2 = tokens_2[np.any(cuttoff, axis=0)]

        # Select random token
        candidates = np.concatenate([tokens_1, tokens_2])
        query = pick_random_token(candidates, random)

    return query


class BoltzCropper(Cropper):
    """Interpolate between contiguous and spatial crops."""

    def __init__(self, min_neighborhood: int = 0, max_neighborhood: int = 40) -> None:
        """Initialize the cropper.

        Modulates the type of cropping to be performed.
        Smaller neighborhoods result in more spatial
        cropping. Larger neighborhoods result in more
        continuous cropping. A mix can be achieved by
        providing a range over which to sample.

        Parameters
        ----------
        min_neighborhood : int
            The minimum neighborhood size, by default 0.
        max_neighborhood : int
            The maximum neighborhood size, by default 40.

        """
        sizes = list(range(min_neighborhood, max_neighborhood + 1, 2))
        self.neighborhood_sizes = sizes

    def crop(
        self,
        data: Tokenized,
        max_tokens: int,
        random: np.random.RandomState,
        max_atoms: Optional[int] = None,
        chain_id: Optional[int] = None,  # Kept for compatibility
        interface_id: Optional[int] = None,  # Kept for compatibility
        record_id: str = "Unknown",
    ) -> Tokenized:
        """
        Crops the data using a spatial expansion centered on a glycan, if present.
        Utilizes the robust token/atom dual-recognition expansion loop from base Boltz.
        """
        # --- PRE-CROP DIAGNOSTIC SUMMARY ---
        pre_tokens = data.tokens
        pre_protein_mask = pre_tokens["mol_type"] == const.chain_type_ids["PROTEIN"]
        pre_glycan_mask = pre_tokens["mol_type"] == const.chain_type_ids["NONPOLYMER"]
        
        pre_num_prot_chains = len(np.unique(pre_tokens[pre_protein_mask]["asym_id"])) if np.any(pre_protein_mask) else 0
        pre_num_glyc_chains = len(np.unique(pre_tokens[pre_glycan_mask]["asym_id"])) if np.any(pre_glycan_mask) else 0
        pre_num_sites = len(data.structure.glycosylation_sites) if data.structure.glycosylation_sites is not None else 0
        
        print(f"[{record_id}] Pre-Crop Summary: {pre_num_prot_chains} P, {pre_num_glyc_chains} G, {pre_num_sites} S.", flush=True)

        token_data = data.tokens

        # Early Exit Safeguard: Ensure BOTH tokens and atoms fit within bounds before skipping
        total_atoms_in_struct = np.sum(token_data["atom_num"])
        if len(token_data) <= max_tokens and (max_atoms is None or total_atoms_in_struct <= max_atoms):
            print(f"[{record_id}] Post-Crop Summary (No Crop): {pre_num_prot_chains} P, {pre_num_glyc_chains} G, {pre_num_sites} S.", flush=True)
            return data

        valid_tokens = token_data[token_data["resolved_mask"]]
        if not valid_tokens.size:
            msg = "No valid (resolved) tokens in structure to perform cropping."
            raise ValueError(msg)

        # --- Step 1: Select a Query Token (Epicenter), Prioritizing Glycans ---
        resolved_glycan_tokens = valid_tokens[
            valid_tokens["mol_type"] == const.chain_type_ids["NONPOLYMER"]
        ]

        if resolved_glycan_tokens.size > 0:
            glycan_chain_ids = np.unique(resolved_glycan_tokens["asym_id"])
            chosen_chain_id = random.choice(glycan_chain_ids)
            tokens_from_chosen_glycan = resolved_glycan_tokens[
                resolved_glycan_tokens["asym_id"] == chosen_chain_id
            ]
            query_token = pick_random_token(tokens_from_chosen_glycan, random)
        else:
            query_token = pick_random_token(valid_tokens, random)

        # --- Step 2: Spatially Sort All Other Tokens ---
        dists = valid_tokens["center_coords"] - query_token["center_coords"]
        spatially_sorted_indices = np.argsort(np.linalg.norm(dists, axis=1))

        # --- Step 3: Granular Spatial Expansion (Dual Recognition) ---
        cropped_indices: set[int] = set()
        total_atoms = 0
        neighborhood_size = random.choice(self.neighborhood_sizes)

        for idx in spatially_sorted_indices:
            token = valid_tokens[idx]
            
            if token["token_idx"] in cropped_indices:
                continue

            chain_tokens = token_data[token_data["asym_id"] == token["asym_id"]]

            # Pick the whole chain if possible, otherwise expand outward symmetrically
            if len(chain_tokens) <= neighborhood_size:
                new_tokens = chain_tokens
            else:
                min_idx = token["res_idx"] - neighborhood_size
                max_idx = token["res_idx"] + neighborhood_size

                max_token_set = chain_tokens
                max_token_set = max_token_set[max_token_set["res_idx"] >= min_idx]
                max_token_set = max_token_set[max_token_set["res_idx"] <= max_idx]

                # Start with just the token itself
                new_tokens = max_token_set[max_token_set["res_idx"] == token["res_idx"]]

                # Expand the neighborhood one-by-one
                min_idx = max_idx = token["res_idx"]
                while new_tokens.size < neighborhood_size:
                    min_idx = min_idx - 1
                    max_idx = max_idx + 1
                    new_tokens = max_token_set
                    new_tokens = new_tokens[new_tokens["res_idx"] >= min_idx]
                    new_tokens = new_tokens[new_tokens["res_idx"] <= max_idx]

            # Compute new tokens and new atoms
            new_indices_to_add = set(new_tokens["token_idx"]) - cropped_indices
            
            if not new_indices_to_add:
                continue

            new_tokens_data = token_data[list(new_indices_to_add)]
            new_atoms_to_add = np.sum(new_tokens_data["atom_num"])

            # DUAL RECOGNITION SAFEGUARD: Break if we exceed max_tokens OR max_atoms
            if (len(new_indices_to_add) > (max_tokens - len(cropped_indices))) or (
                (max_atoms is not None) and ((total_atoms + new_atoms_to_add) > max_atoms)
            ):
                break

            # Add to crop safely
            cropped_indices.update(new_indices_to_add)
            total_atoms += new_atoms_to_add

        # --- Step 4: Finalize the Cropped Data ---
        if not cropped_indices:
            msg = "Cropping resulted in zero tokens. This should not happen."
            raise ValueError(msg)

        # Get the cropped tokens sorted by original index to maintain order
        final_token_indices = sorted(list(cropped_indices))
        final_cropped_token_data = token_data[
            np.isin(token_data["token_idx"], final_token_indices)
        ]

        sorter = np.argsort(final_cropped_token_data["token_idx"])
        final_cropped_token_data = final_cropped_token_data[
            sorter[
                np.searchsorted(
                    final_cropped_token_data["token_idx"][sorter], final_token_indices
                )
            ]
        ]

        # Only keep bonds within the cropped tokens
        token_bonds = data.bonds
        indices_map = final_cropped_token_data["token_idx"]
        token_bonds = token_bonds[np.isin(token_bonds["token_1"], indices_map)]
        token_bonds = token_bonds[np.isin(token_bonds["token_2"], indices_map)]

        # --- POST-CROP DIAGNOSTIC SUMMARY ---
        final_protein_mask = final_cropped_token_data["mol_type"] == const.chain_type_ids["PROTEIN"]
        final_glycan_mask = final_cropped_token_data["mol_type"] == const.chain_type_ids["NONPOLYMER"]
        
        num_prot_chains = len(np.unique(final_cropped_token_data[final_protein_mask]["asym_id"])) if np.any(final_protein_mask) else 0
        num_glyc_chains = len(np.unique(final_cropped_token_data[final_glycan_mask]["asym_id"])) if np.any(final_glycan_mask) else 0
        
        num_sites = 0
        if data.structure.glycosylation_sites is not None and np.any(final_protein_mask) and np.any(final_glycan_mask):
            protein_tokens_in_crop = final_cropped_token_data[final_protein_mask]
            final_protein_residues = set(zip(protein_tokens_in_crop["asym_id"], protein_tokens_in_crop["res_idx"]))
            final_glycan_chain_ids = set(np.unique(final_cropped_token_data[final_glycan_mask]["asym_id"]))

            num_sites = sum(1 for site in data.structure.glycosylation_sites 
                            if (site["protein_chain_id"], site["protein_res_id"]) in final_protein_residues 
                            and site["glycan_chain_id"] in final_glycan_chain_ids)

        print(f"[{record_id}] Post-Crop Summary: {num_prot_chains} P, {num_glyc_chains} G, {num_sites} S.", flush=True)

        return replace(data, tokens=final_cropped_token_data, bonds=token_bonds)
