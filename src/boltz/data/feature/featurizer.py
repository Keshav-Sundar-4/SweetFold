import math
import random
from typing import Any, Dict, List, Optional, Tuple
import sys
import pkg_resources
import json

import numba
import numpy as np
import numpy.typing as npt
import torch
from numba import types
from torch import Tensor, from_numpy
from torch.nn.functional import one_hot
import torch.nn.functional as F

from boltz.data import const
from boltz.data.feature.pad import pad_dim
from boltz.data.feature.symmetry import (
    get_amino_acids_symmetries,
    get_chain_symmetries,
    get_ligand_symmetries,
)
from boltz.data.types import (
    MSA,
    MSADeletion,
    MSAResidue,
    MSASequence,
    Tokenized,
)
from boltz.model.modules.utils import center_random_augmentation

####################################################################################################
# HELPERS
####################################################################################################

def compute_frames_nonpolymer(
    data: Tokenized,
    coords,
    resolved_mask,
    atom_to_token,
    frame_data: list,
    resolved_frame_data: list,
) -> tuple[list, list]:
    """Get the frames for non-polymer tokens.

    Parameters
    ----------
    data : Tokenized
        The tokenized data.
    frame_data : list
        The frame data.
    resolved_frame_data : list
        The resolved frame data.

    Returns
    -------
    tuple[list, list]
        The frame data and resolved frame data.

    """
    frame_data = np.array(frame_data)
    resolved_frame_data = np.array(resolved_frame_data)
    asym_id_token = data.tokens["asym_id"]
    asym_id_atom = data.tokens["asym_id"][atom_to_token]
    token_idx = 0
    atom_idx = 0
    for id in np.unique(data.tokens["asym_id"]):
        mask_chain_token = asym_id_token == id
        mask_chain_atom = asym_id_atom == id
        num_tokens = mask_chain_token.sum()
        num_atoms = mask_chain_atom.sum()
        if (
            data.tokens[token_idx]["mol_type"] != const.chain_type_ids["NONPOLYMER"]
            or num_atoms < 3
        ):
            token_idx += num_tokens
            atom_idx += num_atoms
            continue
        dist_mat = (
            (
                coords.reshape(-1, 3)[mask_chain_atom][:, None, :]
                - coords.reshape(-1, 3)[mask_chain_atom][None, :, :]
            )
            ** 2
        ).sum(-1) ** 0.5
        resolved_pair = 1 - (
            resolved_mask[mask_chain_atom][None, :]
            * resolved_mask[mask_chain_atom][:, None]
        ).astype(np.float32)
        resolved_pair[resolved_pair == 1] = math.inf
        indices = np.argsort(dist_mat + resolved_pair, axis=1)
        frames = (
            np.concatenate(
                [
                    indices[:, 1:2],
                    indices[:, 0:1],
                    indices[:, 2:3],
                ],
                axis=1,
            )
            + atom_idx
        )
        frame_data[token_idx : token_idx + num_atoms, :] = frames
        resolved_frame_data[token_idx : token_idx + num_atoms] = resolved_mask[
            frames
        ].all(axis=1)
        token_idx += num_tokens
        atom_idx += num_atoms
    frames_expanded = coords.reshape(-1, 3)[frame_data]

    mask_collinear = compute_collinear_mask(
        frames_expanded[:, 1] - frames_expanded[:, 0],
        frames_expanded[:, 1] - frames_expanded[:, 2],
    )
    return frame_data, resolved_frame_data & mask_collinear


def compute_collinear_mask(v1, v2):
    norm1 = np.linalg.norm(v1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(v2, axis=1, keepdims=True)
    v1 = v1 / (norm1 + 1e-6)
    v2 = v2 / (norm2 + 1e-6)
    mask_angle = np.abs(np.sum(v1 * v2, axis=1)) < 0.9063
    mask_overlap1 = norm1.reshape(-1) > 1e-2
    mask_overlap2 = norm2.reshape(-1) > 1e-2
    return mask_angle & mask_overlap1 & mask_overlap2


def dummy_msa(residues: np.ndarray) -> MSA:
    """Create a dummy MSA for a chain.

    Parameters
    ----------
    residues : np.ndarray
        The residues for the chain.

    Returns
    -------
    MSA
        The dummy MSA.

    """
    residues = [res["res_type"] for res in residues]
    deletions = []
    sequences = [(0, -1, 0, len(residues), 0, 0)]
    return MSA(
        residues=np.array(residues, dtype=MSAResidue),
        deletions=np.array(deletions, dtype=MSADeletion),
        sequences=np.array(sequences, dtype=MSASequence),
    )


def construct_paired_msa(  # noqa: C901, PLR0915, PLR0912
    data: Tokenized,
    max_seqs: int,
    max_pairs: int = 8192,
    max_total: int = 16384,
    random_subset: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """Pair the MSA data.

    Parameters
    ----------
    data : Input
        The input data.

    Returns
    -------
    Tensor
        The MSA data.
    Tensor
        The deletion data.
    Tensor
        Mask indicating paired sequences.

    """
    # Get unique chains (ensuring monotonicity in the order)
    assert np.all(np.diff(data.tokens["asym_id"], n=1) >= 0)
    chain_ids = np.unique(data.tokens["asym_id"])

    # Get relevant MSA, and create a dummy for chains without
    msa = {k: data.msa[k] for k in chain_ids if k in data.msa}
    for chain_id in chain_ids:
        if chain_id not in msa:
            chain = data.structure.chains[chain_id]
            res_start = chain["res_idx"]
            res_end = res_start + chain["res_num"]
            residues = data.structure.residues[res_start:res_end]
            msa[chain_id] = dummy_msa(residues)

    # Map taxonomies to (chain_id, seq_idx)
    taxonomy_map: dict[str, list] = {}
    for chain_id, chain_msa in msa.items():
        sequences = chain_msa.sequences
        sequences = sequences[sequences["taxonomy"] != -1]
        for sequence in sequences:
            seq_idx = sequence["seq_idx"]
            taxon = sequence["taxonomy"]
            taxonomy_map.setdefault(taxon, []).append((chain_id, seq_idx))

    # Remove taxonomies with only one sequence and sort by the
    # number of chain_id present in each of the taxonomies
    taxonomy_map = {k: v for k, v in taxonomy_map.items() if len(v) > 1}
    taxonomy_map = sorted(
        taxonomy_map.items(),
        key=lambda x: len({c for c, _ in x[1]}),
        reverse=True,
    )

    # Keep track of the sequences available per chain, keeping the original
    # order of the sequences in the MSA to favor the best matching sequences
    visited = {(c, s) for c, items in taxonomy_map for s in items}
    available = {}
    for c in chain_ids:
        available[c] = [
            i for i in range(1, len(msa[c].sequences)) if (c, i) not in visited
        ]

    # Create sequence pairs
    is_paired = []
    pairing = []

    # Start with the first sequence for each chain
    is_paired.append({c: 1 for c in chain_ids})
    pairing.append({c: 0 for c in chain_ids})

    # Then add up to 8191 paired rows
    for _, pairs in taxonomy_map:
        # Group occurences by chain_id in case we have multiple
        # sequences from the same chain and same taxonomy
        chain_occurences = {}
        for chain_id, seq_idx in pairs:
            chain_occurences.setdefault(chain_id, []).append(seq_idx)

        # We create as many pairings as the maximum number of occurences
        max_occurences = max(len(v) for v in chain_occurences.values())
        for i in range(max_occurences):
            row_pairing = {}
            row_is_paired = {}

            # Add the chains present in the taxonomy
            for chain_id, seq_idxs in chain_occurences.items():
                # Roll over the sequence index to maximize diversity
                idx = i % len(seq_idxs)
                seq_idx = seq_idxs[idx]

                # Add the sequence to the pairing
                row_pairing[chain_id] = seq_idx
                row_is_paired[chain_id] = 1

            # Add any missing chains
            for chain_id in chain_ids:
                if chain_id not in row_pairing:
                    row_is_paired[chain_id] = 0
                    if available[chain_id]:
                        # Add the next available sequence
                        seq_idx = available[chain_id].pop(0)
                        row_pairing[chain_id] = seq_idx
                    else:
                        # No more sequences available, we place a gap
                        row_pairing[chain_id] = -1

            pairing.append(row_pairing)
            is_paired.append(row_is_paired)

            # Break if we have enough pairs
            if len(pairing) >= max_pairs:
                break

        # Break if we have enough pairs
        if len(pairing) >= max_pairs:
            break

    # Now add up to 16384 unpaired rows total
    max_left = max(len(v) for v in available.values())
    for _ in range(min(max_total - len(pairing), max_left)):
        row_pairing = {}
        row_is_paired = {}
        for chain_id in chain_ids:
            row_is_paired[chain_id] = 0
            if available[chain_id]:
                # Add the next available sequence
                seq_idx = available[chain_id].pop(0)
                row_pairing[chain_id] = seq_idx
            else:
                # No more sequences available, we place a gap
                row_pairing[chain_id] = -1

        pairing.append(row_pairing)
        is_paired.append(row_is_paired)

        # Break if we have enough sequences
        if len(pairing) >= max_total:
            break

    # Randomly sample a subset of the pairs
    # ensuring the first row is always present
    if random_subset:
        num_seqs = len(pairing)
        if num_seqs > max_seqs:
            indices = np.random.choice(
                list(range(1, num_seqs)), size=max_seqs - 1, replace=False
            )  # noqa: NPY002
            pairing = [pairing[0]] + [pairing[i] for i in indices]
            is_paired = [is_paired[0]] + [is_paired[i] for i in indices]
    else:
        # Deterministic downsample to max_seqs
        pairing = pairing[:max_seqs]
        is_paired = is_paired[:max_seqs]

    # Map (chain_id, seq_idx, res_idx) to deletion
    deletions = {}
    for chain_id, chain_msa in msa.items():
        chain_deletions = chain_msa.deletions
        for sequence in chain_msa.sequences:
            del_start = sequence["del_start"]
            del_end = sequence["del_end"]
            chain_deletions = chain_msa.deletions[del_start:del_end]
            for deletion_data in chain_deletions:
                seq_idx = sequence["seq_idx"]
                res_idx = deletion_data["res_idx"]
                deletion = deletion_data["deletion"]
                deletions[(chain_id, seq_idx, res_idx)] = deletion

    # Add all the token MSA data
    msa_data, del_data, paired_data = prepare_msa_arrays(
        data.tokens, pairing, is_paired, deletions, msa
    )

    msa_data = torch.tensor(msa_data, dtype=torch.long)
    del_data = torch.tensor(del_data, dtype=torch.float)
    paired_data = torch.tensor(paired_data, dtype=torch.float)

    return msa_data, del_data, paired_data



def prepare_msa_arrays(
    tokens,
    pairing: list[dict[int, int]],
    is_paired: list[dict[int, int]],
    deletions: dict[tuple[int, int, int], int],
    msa: dict[int, MSA],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Reshape data to play nicely with numba jit."""
    token_asym_ids_arr = np.array([t["asym_id"] for t in tokens], dtype=np.int64)
    token_res_idxs_arr = np.array([t["res_idx"] for t in tokens], dtype=np.int64)

    chain_ids = sorted(msa.keys())

    # chain_ids are not necessarily contiguous (e.g. they might be 0, 24, 25).
    # This allows us to look up a chain_id by it's index in the chain_ids list.
    chain_id_to_idx = {chain_id: i for i, chain_id in enumerate(chain_ids)}
    token_asym_ids_idx_arr = np.array(
        [chain_id_to_idx[asym_id] for asym_id in token_asym_ids_arr], dtype=np.int64
    )

    pairing_arr = np.zeros((len(pairing), len(chain_ids)), dtype=np.int64)
    is_paired_arr = np.zeros((len(is_paired), len(chain_ids)), dtype=np.int64)

    for i, row_pairing in enumerate(pairing):
        for chain_id in chain_ids:
            pairing_arr[i, chain_id_to_idx[chain_id]] = row_pairing[chain_id]

    for i, row_is_paired in enumerate(is_paired):
        for chain_id in chain_ids:
            is_paired_arr[i, chain_id_to_idx[chain_id]] = row_is_paired[chain_id]

    max_seq_len = max(len(msa[chain_id].sequences) for chain_id in chain_ids)

    # we want res_start from sequences
    msa_sequences = np.full((len(chain_ids), max_seq_len), -1, dtype=np.int64)
    for chain_id in chain_ids:
        for i, seq in enumerate(msa[chain_id].sequences):
            msa_sequences[chain_id_to_idx[chain_id], i] = seq["res_start"]

    max_residues_len = max(len(msa[chain_id].residues) for chain_id in chain_ids)
    msa_residues = np.full((len(chain_ids), max_residues_len), -1, dtype=np.int64)
    for chain_id in chain_ids:
        residues = msa[chain_id].residues.astype(np.int64)
        idxs = np.arange(len(residues))
        chain_idx = chain_id_to_idx[chain_id]
        msa_residues[chain_idx, idxs] = residues

    deletions_dict = numba.typed.Dict.empty(
        key_type=numba.types.Tuple(
            [numba.types.int64, numba.types.int64, numba.types.int64]
        ),
        value_type=numba.types.int64,
    )
    deletions_dict.update(deletions)

    return _prepare_msa_arrays_inner(
        token_asym_ids_arr,
        token_res_idxs_arr,
        token_asym_ids_idx_arr,
        pairing_arr,
        is_paired_arr,
        deletions_dict,
        msa_sequences,
        msa_residues,
        const.token_ids["-"],
    )


deletions_dict_type = types.DictType(types.UniTuple(types.int64, 3), types.int64)

@numba.njit(
    [
        types.Tuple(
            (
                types.int64[:, ::1],
                types.int64[:, ::1],
                types.int64[:, ::1],
            )
        )(
            types.int64[::1],
            types.int64[::1],
            types.int64[::1],
            types.int64[:, ::1],
            types.int64[:, ::1],
            deletions_dict_type,
            types.int64[:, ::1],
            types.int64[:, ::1],
            types.int64,
        )
    ],
    cache=True,
)
def _prepare_msa_arrays_inner(
    token_asym_ids: npt.NDArray[np.int64],
    token_res_idxs: npt.NDArray[np.int64],
    token_asym_ids_idx: npt.NDArray[np.int64],
    pairing: npt.NDArray[np.int64],
    is_paired: npt.NDArray[np.int64],
    deletions: dict[tuple[int, int, int], int],
    msa_sequences: npt.NDArray[np.int64],
    msa_residues: npt.NDArray[np.int64],
    gap_token: int,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    n_tokens = len(token_asym_ids)
    n_pairs = len(pairing)
    msa_data = np.full((n_tokens, n_pairs), gap_token, dtype=np.int64)
    paired_data = np.zeros((n_tokens, n_pairs), dtype=np.int64)
    del_data = np.zeros((n_tokens, n_pairs), dtype=np.int64)

    # Add all the token MSA data
    for token_idx in range(n_tokens):
        chain_id_idx = token_asym_ids_idx[token_idx]
        chain_id = token_asym_ids[token_idx]
        res_idx = token_res_idxs[token_idx]

        for pair_idx in range(n_pairs):
            seq_idx = pairing[pair_idx, chain_id_idx]
            paired_data[token_idx, pair_idx] = is_paired[pair_idx, chain_id_idx]

            # Add residue type
            if seq_idx != -1:
                res_start = msa_sequences[chain_id_idx, seq_idx]
                res_type = msa_residues[chain_id_idx, res_start + res_idx]
                k = (chain_id, seq_idx, res_idx)
                if k in deletions:
                    del_data[token_idx, pair_idx] = deletions[k]
                msa_data[token_idx, pair_idx] = res_type

    return msa_data, del_data, paired_data




####################################################################################################
# FEATURES
####################################################################################################


def select_subset_from_mask(mask, p):
    num_true = np.sum(mask)
    v = np.random.geometric(p) + 1
    k = min(v, num_true)

    true_indices = np.where(mask)[0]

    # Randomly select k indices from the true_indices
    selected_indices = np.random.choice(true_indices, size=k, replace=False)

    new_mask = np.zeros_like(mask)
    new_mask[selected_indices] = 1

    return new_mask

def process_token_features(
    data: Tokenized,
    max_tokens: Optional[int] = None,
    binder_pocket_conditioned_prop: Optional[float] = 0.0,
    binder_pocket_cutoff: Optional[float] = 6.0,
    binder_pocket_sampling_geometric_p: Optional[float] = 0.0,
    only_ligand_binder_pocket: Optional[bool] = False,
    inference_binder: Optional[list[int]] = None,
    inference_pocket: Optional[list[tuple[int, int]]] = None,
) -> dict[str, Tensor]:
    """Get the token features.

    Parameters
    ----------
    data : Tokenized
        The tokenized data.
    max_tokens : int
        The maximum number of tokens.

    Returns
    -------
    dict[str, Tensor]
        The token features.

    """
    # Token data
    token_data = data.tokens
    token_bonds = data.bonds

    # Token core features
    token_index = torch.arange(len(token_data), dtype=torch.long)
    residue_index = from_numpy(token_data["res_idx"].copy()).long()
    asym_id = from_numpy(token_data["asym_id"].copy()).long()
    entity_id = from_numpy(token_data["entity_id"].copy()).long()
    sym_id = from_numpy(token_data["sym_id"].copy()).long()
    mol_type = from_numpy(token_data["mol_type"].copy()).long()
    res_type = from_numpy(token_data["res_type"].copy()).long()
    res_type = one_hot(res_type, num_classes=const.num_tokens)
    disto_center = from_numpy(token_data["disto_coords"].copy())

    # Token mask features
    pad_mask = torch.ones(len(token_data), dtype=torch.float)
    resolved_mask = from_numpy(token_data["resolved_mask"].copy()).float()
    disto_mask = from_numpy(token_data["disto_mask"].copy()).float()
    cyclic_period = from_numpy(token_data["cyclic_period"].copy())

    # Token bond features
    if max_tokens is not None:
        pad_len = max_tokens - len(token_data)
        num_tokens = max_tokens if pad_len > 0 else len(token_data)
    else:
        num_tokens = len(token_data)

    tok_to_idx = {tok["token_idx"]: idx for idx, tok in enumerate(token_data)}
    bonds = torch.zeros(num_tokens, num_tokens, dtype=torch.float)
    for token_bond in token_bonds:
        token_1 = tok_to_idx[token_bond["token_1"]]
        token_2 = tok_to_idx[token_bond["token_2"]]
        bonds[token_1, token_2] = 1
        bonds[token_2, token_1] = 1

    bonds = bonds.unsqueeze(-1)

    # Pocket conditioned feature
    pocket_feature = (
        np.zeros(len(token_data)) + const.pocket_contact_info["UNSPECIFIED"]
    )
    
    # FIX: Changed `if inference_binder is not None:` to `if inference_binder:`
    # This ensures we skip this block if inference_binder is an empty list [].
    if inference_binder:
        assert inference_pocket is not None
        pocket_residues = set(inference_pocket)
        for idx, token in enumerate(token_data):
            if token["asym_id"] in inference_binder:
                pocket_feature[idx] = const.pocket_contact_info["BINDER"]
            elif (token["asym_id"], token["res_idx"]) in pocket_residues:
                pocket_feature[idx] = const.pocket_contact_info["POCKET"]
            else:
                pocket_feature[idx] = const.pocket_contact_info["UNSELECTED"]
    elif (
        binder_pocket_conditioned_prop > 0.0
        and random.random() < binder_pocket_conditioned_prop
    ):
        # choose as binder a random ligand in the crop, if there are no ligands select a protein chain
        binder_asym_ids = np.unique(
            token_data["asym_id"][
                token_data["mol_type"] == const.chain_type_ids["NONPOLYMER"]
            ]
        )

        if len(binder_asym_ids) == 0:
            if not only_ligand_binder_pocket:
                binder_asym_ids = np.unique(token_data["asym_id"])

        if len(binder_asym_ids) > 0:
            pocket_asym_id = random.choice(binder_asym_ids)
            binder_mask = token_data["asym_id"] == pocket_asym_id

            binder_coords = []
            for token in token_data:
                if token["asym_id"] == pocket_asym_id:
                    binder_coords.append(
                        data.structure.atoms["coords"][
                            token["atom_idx"] : token["atom_idx"] + token["atom_num"]
                        ]
                    )
            binder_coords = np.concatenate(binder_coords, axis=0)

            # find the tokens in the pocket
            token_dist = np.zeros(len(token_data)) + 1000
            for i, token in enumerate(token_data):
                if (
                    token["mol_type"] != const.chain_type_ids["NONPOLYMER"]
                    and token["asym_id"] != pocket_asym_id
                    and token["resolved_mask"] == 1
                ):
                    token_coords = data.structure.atoms["coords"][
                        token["atom_idx"] : token["atom_idx"] + token["atom_num"]
                    ]

                    # find chain and apply chain transformation
                    for chain in data.structure.chains:
                        if chain["asym_id"] == token["asym_id"]:
                            break

                    token_dist[i] = np.min(
                        np.linalg.norm(
                            token_coords[:, None, :] - binder_coords[None, :, :],
                            axis=-1,
                        )
                    )

            pocket_mask = token_dist < binder_pocket_cutoff

            if np.sum(pocket_mask) > 0:
                pocket_feature = (
                    np.zeros(len(token_data)) + const.pocket_contact_info["UNSELECTED"]
                )
                pocket_feature[binder_mask] = const.pocket_contact_info["BINDER"]

                if binder_pocket_sampling_geometric_p > 0.0:
                    # select a subset of the pocket, according
                    # to a geometric distribution with one as minimum
                    pocket_mask = select_subset_from_mask(
                        pocket_mask, binder_pocket_sampling_geometric_p
                    )

                pocket_feature[pocket_mask] = const.pocket_contact_info["POCKET"]
    pocket_feature = from_numpy(pocket_feature).long()
    pocket_feature = one_hot(pocket_feature, num_classes=len(const.pocket_contact_info))

    # Pad to max tokens if given
    if max_tokens is not None:
        pad_len = max_tokens - len(token_data)
        if pad_len > 0:
            token_index = pad_dim(token_index, 0, pad_len)
            residue_index = pad_dim(residue_index, 0, pad_len)
            asym_id = pad_dim(asym_id, 0, pad_len)
            entity_id = pad_dim(entity_id, 0, pad_len)
            sym_id = pad_dim(sym_id, 0, pad_len)
            mol_type = pad_dim(mol_type, 0, pad_len)
            res_type = pad_dim(res_type, 0, pad_len)
            disto_center = pad_dim(disto_center, 0, pad_len)
            pad_mask = pad_dim(pad_mask, 0, pad_len)
            resolved_mask = pad_dim(resolved_mask, 0, pad_len)
            disto_mask = pad_dim(disto_mask, 0, pad_len)
            pocket_feature = pad_dim(pocket_feature, 0, pad_len)

    token_features = {
        "token_index": token_index,
        "residue_index": residue_index,
        "asym_id": asym_id,
        "entity_id": entity_id,
        "sym_id": sym_id,
        "mol_type": mol_type,
        "res_type": res_type,
        "disto_center": disto_center,
        "token_bonds": bonds,
        "token_pad_mask": pad_mask,
        "token_resolved_mask": resolved_mask,
        "token_disto_mask": disto_mask,
        "pocket_feature": pocket_feature,
        "cyclic_period": cyclic_period,
    }
    return token_features
    
def process_atom_features(
    data: Tokenized,
    atoms_per_window_queries: int = 32,
    min_dist: float = 2.0,
    max_dist: float = 22.0,
    num_bins: int = 64,
    max_atoms: Optional[int] = None,
    max_tokens: Optional[int] = None,
) -> dict[str, Tensor]:
    """Get the atom features with v1's corrected mono-idx reconstruction + v2 bedrock."""
    # Collect per-atom / per-token data
    atom_data = []
    ref_space_uid = []
    coord_data = []
    frame_data = []
    resolved_frame_data = []
    atom_to_token = []
    token_to_rep_atom = []     # index on cropped atom table
    r_set_to_rep_atom = []
    disto_coords = []
    atom_idx = 0

    # Track residue identity across tokens (keep outside loop; v2 behavior)
    chain_res_ids = {}

    # We'll fill this as we enumerate atoms in token order.
    new_atom_mono_idx_list = []

    for token_id, token in enumerate(data.tokens):
        # Map residue ids to a compact "space uid"
        chain_idx, res_id = token["asym_id"], token["res_idx"]
        chain = data.structure.chains[chain_idx]

        if (chain_idx, res_id) not in chain_res_ids:
            new_uid = len(chain_res_ids)
            chain_res_ids[(chain_idx, res_id)] = new_uid
        else:
            new_uid = chain_res_ids[(chain_idx, res_id)]

        ref_space_uid.extend([new_uid] * token["atom_num"])
        atom_to_token.extend([token_id] * token["atom_num"])

        # Pull token's atoms
        start = token["atom_idx"]
        end = token["atom_idx"] + token["atom_num"]
        token_atoms = data.structure.atoms[start:end]

        # Compute per-atom mono indices
        atom_to_mono_idx_map = data.structure.atom_to_mono_idx_map
        is_glycan_chain = (atom_to_mono_idx_map is not None) and (token["asym_id"] in atom_to_mono_idx_map)

        if is_glycan_chain:
            mono_idx_map_for_chain = atom_to_mono_idx_map[token["asym_id"]]
            chain_start_atom_idx = data.structure.chains[token["asym_id"]]["atom_idx"]

            # Iterate atoms in this token by local index to recover original global index
            for local_idx_in_token, _atom in enumerate(token_atoms):
                original_global_atom_idx = token["atom_idx"] + local_idx_in_token
                local_atom_idx_in_glycan = original_global_atom_idx - chain_start_atom_idx
                if 0 <= local_atom_idx_in_glycan < len(mono_idx_map_for_chain):
                    mono_idx = mono_idx_map_for_chain[local_atom_idx_in_glycan]
                    new_atom_mono_idx_list.append(mono_idx)
                else:
                    new_atom_mono_idx_list.append(-1)
        else:
            # Non-glycan chains get sentinel -1
            new_atom_mono_idx_list.extend([-1] * len(token_atoms))

        # Representative atoms for distogram / residue-center
        token_to_rep_atom.append(atom_idx + token["disto_idx"] - start)
        
        # FIX: Include glycan atoms in the Reference Set so pLDDT loss unmasks
        if (chain["mol_type"] != const.chain_type_ids["NONPOLYMER"] or is_glycan_chain) and token["resolved_mask"]:
            r_set_to_rep_atom.append(atom_idx + token["center_idx"] - start)

        # Coordinates for this token
        token_coords = np.array([token_atoms["coords"]])
        coord_data.append(token_coords)

        # Frame indices and masks
        res_type = const.tokens[token["res_type"]]
        if token["atom_num"] < 3 or res_type in ["PAD", "UNK", "-"]:
            idx_frame_a, idx_frame_b, idx_frame_c = 0, 0, 0
            mask_frame = False
        elif (token["mol_type"] == const.chain_type_ids["PROTEIN"]) and (res_type in const.ref_atoms):
            idx_frame_a, idx_frame_b, idx_frame_c = (
                const.ref_atoms[res_type].index("N"),
                const.ref_atoms[res_type].index("CA"),
                const.ref_atoms[res_type].index("C"),
            )
            mask_frame = (
                token_atoms["is_present"][idx_frame_a]
                and token_atoms["is_present"][idx_frame_b]
                and token_atoms["is_present"][idx_frame_c]
            )
        elif (token["mol_type"] == const.chain_type_ids["DNA"] or token["mol_type"] == const.chain_type_ids["RNA"]) and (res_type in const.ref_atoms):
            idx_frame_a, idx_frame_b, idx_frame_c = (
                const.ref_atoms[res_type].index("C1'"),
                const.ref_atoms[res_type].index("C3'"),
                const.ref_atoms[res_type].index("C4'"),
            )
            mask_frame = (
                token_atoms["is_present"][idx_frame_a]
                and token_atoms["is_present"][idx_frame_b]
                and token_atoms["is_present"][idx_frame_c]
            )
        else:
            idx_frame_a, idx_frame_b, idx_frame_c = 0, 0, 0
            mask_frame = False

        frame_data.append([idx_frame_a + atom_idx, idx_frame_b + atom_idx, idx_frame_c + atom_idx])
        resolved_frame_data.append(mask_frame)

        # Distogram centers (take the distogram rep atom in the original table)
        disto_coords_tok = data.structure.atoms[token["disto_idx"]]["coords"]
        disto_coords.append(disto_coords_tok)

        # Append atom records and advance cursor
        token_atoms = token_atoms.copy()
        token_atoms["coords"] = token_coords[0]
        atom_data.append(token_atoms)
        atom_idx += len(token_atoms)

    # --- Distogram over representative atoms ---
    disto_coords = np.array(disto_coords)
    t_center = torch.Tensor(disto_coords)
    t_dists = torch.cdist(t_center, t_center)
    boundaries = torch.linspace(min_dist, max_dist, num_bins - 1)
    distogram = (t_dists.unsqueeze(-1) > boundaries).sum(dim=-1).long()
    disto_target = one_hot(distogram, num_classes=num_bins)

    # Concatenate per-token arrays
    atom_data = np.concatenate(atom_data)
    coord_data = np.concatenate(coord_data, axis=1)
    ref_space_uid = np.array(ref_space_uid)

    # --- Featurization ---
    ref_atom_name_chars = from_numpy(atom_data["name"]).long()
    ref_atom_name_raw = from_numpy(atom_data["name"].copy()) # For diagnostics
    ref_element = from_numpy(atom_data["element"]).long()
    ref_charge = from_numpy(atom_data["charge"])
    ref_pos = from_numpy(atom_data["conformer"].copy())
    ref_space_uid = from_numpy(ref_space_uid)
    coords = from_numpy(coord_data.copy())
    resolved_mask = from_numpy(atom_data["is_present"])
    pad_mask = torch.ones(len(atom_data), dtype=torch.float)
    atom_to_token = torch.tensor(atom_to_token, dtype=torch.long)
    token_to_rep_atom = torch.tensor(token_to_rep_atom, dtype=torch.long)
    r_set_to_rep_atom = torch.tensor(r_set_to_rep_atom, dtype=torch.long)

    # Frames recomputation to cover non-polymers (v2 bedrock)
    frame_data, resolved_frame_data = compute_frames_nonpolymer(
        data, coord_data, atom_data["is_present"], atom_to_token, frame_data, resolved_frame_data
    )
    frames = from_numpy(frame_data.copy())
    frame_resolved_mask = from_numpy(resolved_frame_data.copy())

    # One-hot encodings
    ref_atom_name_chars = one_hot(ref_atom_name_chars % num_bins, num_classes=num_bins)
    ref_element = one_hot(ref_element, num_classes=const.num_elements)
    atom_to_token = one_hot(atom_to_token, num_classes=len(data.tokens))
    token_to_rep_atom = one_hot(token_to_rep_atom, num_classes=len(atom_data))
    r_set_to_rep_atom = one_hot(r_set_to_rep_atom, num_classes=len(atom_data))

    # Center coords by resolved atoms; augment inputs
    center = (coords * resolved_mask[None, :, None]).sum(dim=1)
    center = center / resolved_mask.sum().clamp(min=1)
    coords = coords - center[:, None]
    ref_pos = center_random_augmentation(ref_pos[None], resolved_mask[None], centering=False)[0]

    # --- Padding: atoms-first path (window multiple) ---
    if max_atoms is not None:
        pad_len_atoms = max_atoms - len(atom_data)
    else:
        pad_len_atoms = ((len(atom_data) - 1) // atoms_per_window_queries + 1) * atoms_per_window_queries - len(atom_data)

    if pad_len_atoms > 0:
        pad_mask = pad_dim(pad_mask, 0, pad_len_atoms)
        ref_pos = pad_dim(ref_pos, 0, pad_len_atoms)
        resolved_mask = pad_dim(resolved_mask, 0, pad_len_atoms)
        ref_element = pad_dim(ref_element, 0, pad_len_atoms)
        ref_charge = pad_dim(ref_charge, 0, pad_len_atoms)
        ref_atom_name_chars = pad_dim(ref_atom_name_chars, 0, pad_len_atoms)
        ref_atom_name_raw = pad_dim(ref_atom_name_raw, 0, pad_len_atoms) # Pad raw names
        ref_space_uid = pad_dim(ref_space_uid, 0, pad_len_atoms)
        coords = pad_dim(coords, 1, pad_len_atoms)
        atom_to_token = pad_dim(atom_to_token, 0, pad_len_atoms)
        token_to_rep_atom = pad_dim(token_to_rep_atom, 1, pad_len_atoms)
        r_set_to_rep_atom = pad_dim(r_set_to_rep_atom, 1, pad_len_atoms)

    # Finalize mono-idx + CONSISTENT padding
    atom_mono_idx = from_numpy(np.array(new_atom_mono_idx_list, dtype=np.int64))
    if pad_len_atoms > 0:
        # Always pad to the same atom length, regardless of max_atoms vs window multiple
        atom_mono_idx = pad_dim(atom_mono_idx, 0, pad_len_atoms, value=-1)

    # --- Padding: tokens dimension (optional) ---
    if max_tokens is not None:
        pad_len_tokens = max_tokens - token_to_rep_atom.shape[0]
        if pad_len_tokens > 0:
            atom_to_token = pad_dim(atom_to_token, 1, pad_len_tokens)
            token_to_rep_atom = pad_dim(token_to_rep_atom, 0, pad_len_tokens)
            r_set_to_rep_atom = pad_dim(r_set_to_rep_atom, 0, pad_len_tokens)
            disto_target = pad_dim(pad_dim(disto_target, 0, pad_len_tokens), 1, pad_len_tokens)
            frames = pad_dim(frames, 0, pad_len_tokens)
            frame_resolved_mask = pad_dim(frame_resolved_mask, 0, pad_len_tokens)

    return {
        "ref_pos": ref_pos,
        "atom_resolved_mask": resolved_mask,
        "ref_element": ref_element,
        "ref_charge": ref_charge,
        "ref_atom_name_chars": ref_atom_name_chars,
        "ref_atom_name_raw": ref_atom_name_raw, # For diagnostics
        "ref_space_uid": ref_space_uid,
        "coords": coords,
        "atom_mono_idx": atom_mono_idx,         
        "atom_pad_mask": pad_mask,
        "atom_to_token": atom_to_token,
        "token_to_rep_atom": token_to_rep_atom,
        "r_set_to_rep_atom": r_set_to_rep_atom,
        "disto_target": disto_target,
        "frames_idx": frames,
        "frame_resolved_mask": frame_resolved_mask,
    }

def process_msa_features(
    data: Tokenized,
    max_seqs_batch: int,
    max_seqs: int,
    max_tokens: Optional[int] = None,
    pad_to_max_seqs: bool = False,
) -> dict[str, Tensor]:
    """Get the MSA features.

    Parameters
    ----------
    data : Tokenized
        The tokenized data.
    max_seqs : int
        The maximum number of MSA sequences.
    max_tokens : int
        The maximum number of tokens.
    pad_to_max_seqs : bool
        Whether to pad to the maximum number of sequences.

    Returns
    -------
    dict[str, Tensor]
        The MSA features.

    """
    # Created paired MSA
    msa, deletion, paired = construct_paired_msa(data, max_seqs_batch)
    msa, deletion, paired = (
        msa.transpose(1, 0),
        deletion.transpose(1, 0),
        paired.transpose(1, 0),
    )  # (N_MSA, N_RES, N_AA)

    # Prepare features
    msa = torch.nn.functional.one_hot(msa, num_classes=const.num_tokens)
    msa_mask = torch.ones_like(msa[:, :, 0])
    profile = msa.float().mean(dim=0)
    has_deletion = deletion > 0
    deletion = np.pi / 2 * np.arctan(deletion / 3)
    deletion_mean = deletion.mean(axis=0)

    # Pad in the MSA dimension (dim=0)
    if pad_to_max_seqs:
        pad_len = max_seqs - msa.shape[0]
        if pad_len > 0:
            msa = pad_dim(msa, 0, pad_len, const.token_ids["-"])
            paired = pad_dim(paired, 0, pad_len)
            msa_mask = pad_dim(msa_mask, 0, pad_len)
            has_deletion = pad_dim(has_deletion, 0, pad_len)
            deletion = pad_dim(deletion, 0, pad_len)

    # Pad in the token dimension (dim=1)
    if max_tokens is not None:
        pad_len = max_tokens - msa.shape[1]
        if pad_len > 0:
            msa = pad_dim(msa, 1, pad_len, const.token_ids["-"])
            paired = pad_dim(paired, 1, pad_len)
            msa_mask = pad_dim(msa_mask, 1, pad_len)
            has_deletion = pad_dim(has_deletion, 1, pad_len)
            deletion = pad_dim(deletion, 1, pad_len)
            profile = pad_dim(profile, 0, pad_len)
            deletion_mean = pad_dim(deletion_mean, 0, pad_len)

    return {
        "msa": msa,
        "msa_paired": paired,
        "deletion_value": deletion,
        "has_deletion": has_deletion,
        "deletion_mean": deletion_mean,
        "profile": profile,
        "msa_mask": msa_mask,
    }


def process_symmetry_features(
    cropped: Tokenized, symmetries: dict
) -> dict[str, Tensor]:
    """Get the symmetry features.

    Parameters
    ----------
    data : Tokenized
        The tokenized data.

    Returns
    -------
    dict[str, Tensor]
        The symmetry features.

    """
    features = get_chain_symmetries(cropped)
    features.update(get_amino_acids_symmetries(cropped))
    features.update(get_ligand_symmetries(cropped, symmetries))

    return features


def process_residue_constraint_features(
    data: Tokenized,
) -> dict[str, Tensor]:
    residue_constraints = data.residue_constraints
    if residue_constraints is not None:
        rdkit_bounds_constraints = residue_constraints.rdkit_bounds_constraints
        chiral_atom_constraints = residue_constraints.chiral_atom_constraints
        stereo_bond_constraints = residue_constraints.stereo_bond_constraints
        planar_bond_constraints = residue_constraints.planar_bond_constraints
        planar_ring_5_constraints = residue_constraints.planar_ring_5_constraints
        planar_ring_6_constraints = residue_constraints.planar_ring_6_constraints

        rdkit_bounds_index = torch.tensor(
            rdkit_bounds_constraints["atom_idxs"].copy(), dtype=torch.long
        ).T
        rdkit_bounds_bond_mask = torch.tensor(
            rdkit_bounds_constraints["is_bond"].copy(), dtype=torch.bool
        )
        rdkit_bounds_angle_mask = torch.tensor(
            rdkit_bounds_constraints["is_angle"].copy(), dtype=torch.bool
        )
        rdkit_upper_bounds = torch.tensor(
            rdkit_bounds_constraints["upper_bound"].copy(), dtype=torch.float
        )
        rdkit_lower_bounds = torch.tensor(
            rdkit_bounds_constraints["lower_bound"].copy(), dtype=torch.float
        )

        chiral_atom_index = torch.tensor(
            chiral_atom_constraints["atom_idxs"].copy(), dtype=torch.long
        ).T
        chiral_reference_mask = torch.tensor(
            chiral_atom_constraints["is_reference"].copy(), dtype=torch.bool
        )
        chiral_atom_orientations = torch.tensor(
            chiral_atom_constraints["is_r"].copy(), dtype=torch.bool
        )

        stereo_bond_index = torch.tensor(
            stereo_bond_constraints["atom_idxs"].copy(), dtype=torch.long
        ).T
        stereo_reference_mask = torch.tensor(
            stereo_bond_constraints["is_reference"].copy(), dtype=torch.bool
        )
        stereo_bond_orientations = torch.tensor(
            stereo_bond_constraints["is_e"].copy(), dtype=torch.bool
        )

        planar_bond_index = torch.tensor(
            planar_bond_constraints["atom_idxs"].copy(), dtype=torch.long
        ).T
        planar_ring_5_index = torch.tensor(
            planar_ring_5_constraints["atom_idxs"].copy(), dtype=torch.long
        ).T
        planar_ring_6_index = torch.tensor(
            planar_ring_6_constraints["atom_idxs"].copy(), dtype=torch.long
        ).T
    else:
        rdkit_bounds_index = torch.empty((2, 0), dtype=torch.long)
        rdkit_bounds_bond_mask = torch.empty((0,), dtype=torch.bool)
        rdkit_bounds_angle_mask = torch.empty((0,), dtype=torch.bool)
        rdkit_upper_bounds = torch.empty((0,), dtype=torch.float)
        rdkit_lower_bounds = torch.empty((0,), dtype=torch.float)
        chiral_atom_index = torch.empty(
            (
                4,
                0,
            ),
            dtype=torch.long,
        )
        chiral_reference_mask = torch.empty((0,), dtype=torch.bool)
        chiral_atom_orientations = torch.empty((0,), dtype=torch.bool)
        stereo_bond_index = torch.empty((4, 0), dtype=torch.long)
        stereo_reference_mask = torch.empty((0,), dtype=torch.bool)
        stereo_bond_orientations = torch.empty((0,), dtype=torch.bool)
        planar_bond_index = torch.empty((6, 0), dtype=torch.long)
        planar_ring_5_index = torch.empty((5, 0), dtype=torch.long)
        planar_ring_6_index = torch.empty((6, 0), dtype=torch.long)

    return {
        "rdkit_bounds_index": rdkit_bounds_index,
        "rdkit_bounds_bond_mask": rdkit_bounds_bond_mask,
        "rdkit_bounds_angle_mask": rdkit_bounds_angle_mask,
        "rdkit_upper_bounds": rdkit_upper_bounds,
        "rdkit_lower_bounds": rdkit_lower_bounds,
        "chiral_atom_index": chiral_atom_index,
        "chiral_reference_mask": chiral_reference_mask,
        "chiral_atom_orientations": chiral_atom_orientations,
        "stereo_bond_index": stereo_bond_index,
        "stereo_reference_mask": stereo_reference_mask,
        "stereo_bond_orientations": stereo_bond_orientations,
        "planar_bond_index": planar_bond_index,
        "planar_ring_5_index": planar_ring_5_index,
        "planar_ring_6_index": planar_ring_6_index,
    }


def process_chain_feature_constraints(
    data: Tokenized,
) -> dict[str, Tensor]:
    structure = data.structure
    if structure.connections.shape[0] > 0:
        connected_chain_index, connected_atom_index = [], []
        for connection in structure.connections:
            connected_chain_index.append([connection["chain_1"], connection["chain_2"]])
            connected_atom_index.append([connection["atom_1"], connection["atom_2"]])
        connected_chain_index = torch.tensor(connected_chain_index, dtype=torch.long).T
        connected_atom_index = torch.tensor(connected_atom_index, dtype=torch.long).T
    else:
        connected_chain_index = torch.empty((2, 0), dtype=torch.long)
        connected_atom_index = torch.empty((2, 0), dtype=torch.long)

    symmetric_chain_index = []
    for i, chain_i in enumerate(structure.chains):
        for j, chain_j in enumerate(structure.chains):
            if j <= i:
                continue
            if chain_i["entity_id"] == chain_j["entity_id"]:
                symmetric_chain_index.append([i, j])
    if len(symmetric_chain_index) > 0:
        symmetric_chain_index = torch.tensor(symmetric_chain_index, dtype=torch.long).T
    else:
        symmetric_chain_index = torch.empty((2, 0), dtype=torch.long)
    return {
        "connected_chain_index": connected_chain_index,
        "connected_atom_index": connected_atom_index,
        "symmetric_chain_index": symmetric_chain_index,
    }

class BoltzFeaturizer:
    """Boltz featurizer."""

    def process(
            self,
            data: Tokenized,
            training: bool,
            max_seqs: int = 4096,
            atoms_per_window_queries: int = 32,
            min_dist: float = 2.0,
            max_dist: float = 22.0,
            num_bins: int = 64,
            max_tokens: Optional[int] = None,
            max_atoms: Optional[int] = None,
            pad_to_max_seqs: bool = False,
            compute_symmetries: bool = False,
            symmetries: Optional[dict] = None,
            binder_pocket_conditioned_prop: Optional[float] = 0.0,
            binder_pocket_cutoff: Optional[float] = 6.0,
            binder_pocket_sampling_geometric_p: Optional[float] = 0.0,
            only_ligand_binder_pocket: Optional[bool] = False,
            inference_binder: Optional[int] = None,
            inference_pocket: Optional[list[tuple[int, int]]] = None,
            compute_constraint_features: bool = False,
            max_mono_chains: int = 128,
            compute_glycan_features: bool = True,
        ) -> dict[str, Tensor]:
            """Compute features."""
            sites = data.structure.glycosylation_sites
            if training and max_seqs is not None and max_seqs > 1:
                max_seqs_batch = np.random.randint(1, max_seqs + 1)
            else:
                max_seqs_batch = max_seqs

            token_features = process_token_features(
                data,
                max_tokens,
                binder_pocket_conditioned_prop,
                binder_pocket_cutoff,
                binder_pocket_sampling_geometric_p,
                only_ligand_binder_pocket,
                inference_binder=inference_binder,
                inference_pocket=inference_pocket,
            )

            atom_features = process_atom_features(
                data,
                atoms_per_window_queries,
                min_dist,
                max_dist,
                num_bins,
                max_atoms,
                max_tokens,
            )

            msa_features = process_msa_features(
                data,
                max_seqs_batch,
                max_seqs,
                max_tokens,
                pad_to_max_seqs,
            )

            symmetry_features = {}
            if compute_symmetries:
                symmetry_features = process_symmetry_features(data, symmetries)

            residue_constraint_features = {}
            chain_constraint_features = {}
            if compute_constraint_features:
                residue_constraint_features = process_residue_constraint_features(data)
                chain_constraint_features = process_chain_feature_constraints(data)

            monosaccharide_features = {}
            if compute_glycan_features:
                monosaccharide_features = process_monosaccharide_features(
                    data=data,
                    atom_features=atom_features,
                    token_features=token_features,
                    max_tokens=max_tokens,
                    max_mono_chains=max_mono_chains,
                )
        
            features =  {
                **token_features,
                **atom_features,
                **msa_features,
                **symmetry_features,
                **residue_constraint_features,
                **chain_constraint_features,
                **monosaccharide_features,
            }

            features['raw_glycosylation_sites'] = _structured_sites_to_tensor(
                data.structure.glycosylation_sites
            )

            return features

def _get_default_mono_features(num_tokens: int, max_tokens: Optional[int], max_mono_chains: int) -> Dict[str, Tensor]:
    """ Helper to return dictionary of zero tensors for monosaccharide features. """
    # Assume necessary constants like NUM_MONO_TYPES are defined/imported globally
    target_num_tokens = max_tokens if max_tokens is not None else num_tokens
    if target_num_tokens < 0: target_num_tokens = 0 # Ensure non-negative

    default_shape_token = (target_num_tokens,)
    default_shape_adj = (max_mono_chains, max_mono_chains)

    return {
        "mono_type": torch.zeros(*default_shape_token, NUM_MONO_TYPES, dtype=torch.float32),
        "mono_anomeric": torch.zeros(*default_shape_token, NUM_ANOMERIC_TYPES, dtype=torch.float32),
        "is_monosaccharide": torch.zeros(*default_shape_token, 1, dtype=torch.float32),
        "inter_glycan_mask": torch.zeros(default_shape_adj, dtype=torch.float32),
        "token_to_mono_idx": torch.full(default_shape_token, -1, dtype=torch.long),
    }

def _structured_sites_to_tensor(sites_array: np.ndarray) -> torch.Tensor:
    """
    (Corrected Version 2)
    Converts a structured NumPy array of glycosylation sites into a homogeneous
    integer torch.Tensor. Accesses the glycan monosaccharide index using the
    correct field name 'glycan_res_id'.

    Args:
        sites_array: A structured NumPy array with the GlycosylationSite dtype.

    Returns:
        A torch.Tensor of shape (num_sites, 12) and dtype torch.long.
        The 12 columns are:
        [p_chain, p_res, p_name(4), g_chain, g_mono, g_name(4)]
    """
    if sites_array is None or sites_array.size == 0:
        return torch.empty((0, 12), dtype=torch.long)

    numerical_sites_list = []
    for site in sites_array:
        p_name_str = str(site["protein_atom_name"]).strip()
        g_name_str = str(site["glycan_atom_name"]).strip()

        p_name_padded = p_name_str.ljust(4)
        g_name_padded = g_name_str.ljust(4)

        p_name_encoded = [ord(c) - 32 for c in p_name_padded[:4]]
        g_name_encoded = [ord(c) - 32 for c in g_name_padded[:4]]

        row = [
            site["protein_chain_id"],
            site["protein_res_id"],
            *p_name_encoded,
            site["glycan_chain_id"],
            site["glycan_res_id"],
            *g_name_encoded,
        ]
        numerical_sites_list.append(row)

    return torch.tensor(numerical_sites_list, dtype=torch.long)


def process_monosaccharide_features(
    data: Tokenized,
    atom_features: Dict[str, Tensor],
    token_features: Dict[str, Tensor],
    max_tokens: Optional[int] = None,
    max_mono_chains: int = 128,
) -> Dict[str, Tensor]:
    """
    (High-Performance Version)
    Generates monosaccharide feature tensors based on the single-chain glycan model.
    This vectorized implementation avoids Python loops for maximum efficiency.
    """
    num_input_tokens = len(data.tokens)
    glycan_feature_map = data.structure.glycan_feature_map
    
    # Early exit if there are no tokens or no glycan data in the structure
    if num_input_tokens == 0 or not glycan_feature_map:
        return _get_default_mono_features(num_input_tokens, max_tokens, max_mono_chains)

    # --- Step 1: Map atoms to local mono_idx, then to tokens. (Already vectorized) ---
    atom_mono_idx = atom_features['atom_mono_idx']
    atom_to_token_dense = atom_features['atom_to_token'].argmax(dim=1)
    token_asym_id = token_features['asym_id']

    token_mono_idx_local = torch.full((num_input_tokens,), -1, dtype=torch.long, device=atom_mono_idx.device)
    valid_atom_mask = atom_mono_idx != -1
    token_indices_to_update = atom_to_token_dense[valid_atom_mask]
    mono_indices_to_assign = atom_mono_idx[valid_atom_mask]
    token_mono_idx_local.scatter_(0, token_indices_to_update, mono_indices_to_assign)

    # --- Step 2: Create a global index for all unique monosaccharides. (Fast) ---
    unique_monos = sorted(list(set(
        (asym.item(), mono_idx.item())
        for asym, mono_idx in zip(token_asym_id, token_mono_idx_local) if mono_idx.item() != -1
    )))

    if not unique_monos:
        return _get_default_mono_features(num_input_tokens, max_tokens, max_mono_chains)

    mono_map = {mono: i for i, mono in enumerate(unique_monos)}
    num_global_monos = len(unique_monos)

    token_to_mono_idx = torch.tensor([
        mono_map.get((asym.item(), mono_idx.item()), -1)
        for asym, mono_idx in zip(token_asym_id, token_mono_idx_local)
    ], dtype=torch.long, device=token_asym_id.device)

    # --- Step 3: Create the inter-glycan mask. (Already vectorized) ---
    mono_asym_ids = torch.tensor([mono[0] for mono in unique_monos], device=token_asym_id.device)
    inter_glycan_mask = (mono_asym_ids.unsqueeze(1) == mono_asym_ids.unsqueeze(0)).float()

    # --- Step 4: Build "Source of Truth" feature tensors for all unique monosaccharides. ---
    mono_type_ids = []
    anomeric_ids = []
    
    default_mono_type = MONO_TYPE_MAP.get("OTHER", 0)
    default_anomeric = ANOMERIC_MAP.get(None, 2)

    for asym_id, local_idx in unique_monos:
        features = glycan_feature_map.get((asym_id, local_idx))
        if features:
            # FIX: Handle both dicts (from npz load) and dataclass objects
            if isinstance(features, dict):
                ccd_code = features.get('ccd_code', 'UNK').upper()
                anomeric_config = features.get('anomeric_config', None)
            else:
                ccd_code = getattr(features, 'ccd_code', 'UNK').upper()
                anomeric_config = getattr(features, 'anomeric_config', None)
                
            mono_type_ids.append(MONO_TYPE_MAP.get(ccd_code, default_mono_type))
            anomeric_ids.append(ANOMERIC_MAP.get(anomeric_config, default_anomeric))
        else:
            mono_type_ids.append(default_mono_type)
            anomeric_ids.append(default_anomeric)

    source_mono_type_ids = torch.tensor(mono_type_ids, dtype=torch.long, device=token_asym_id.device)
    source_anomeric_ids = torch.tensor(anomeric_ids, dtype=torch.long, device=token_asym_id.device)
    
    # --- Step 5: Vectorized Feature Generation using advanced indexing (gather). ---
    mono_type = torch.zeros(num_input_tokens, NUM_MONO_TYPES, dtype=torch.float32, device=token_asym_id.device)
    mono_anomeric = torch.zeros(num_input_tokens, NUM_ANOMERIC_TYPES, dtype=torch.float32, device=token_asym_id.device)
    
    is_mono_mask = token_to_mono_idx != -1
    valid_global_indices = token_to_mono_idx[is_mono_mask]
    
    gathered_type_ids = source_mono_type_ids[valid_global_indices]
    gathered_anomeric_ids = source_anomeric_ids[valid_global_indices]
    
    one_hot_types = F.one_hot(gathered_type_ids, num_classes=NUM_MONO_TYPES).float()
    one_hot_anomerics = F.one_hot(gathered_anomeric_ids, num_classes=NUM_ANOMERIC_TYPES).float()
    
    mono_type[is_mono_mask] = one_hot_types
    mono_anomeric[is_mono_mask] = one_hot_anomerics
    
    is_monosaccharide = is_mono_mask.float().unsqueeze(-1)
    
    # --- Step 6: Pad tensors and finalize the output dictionary. ---
    final_features = {
        "token_to_mono_idx": token_to_mono_idx,
        "mono_type": mono_type,
        "mono_anomeric": mono_anomeric,
        "is_monosaccharide": is_monosaccharide,
    }

    if max_tokens is not None:
        pad_len = max_tokens - num_input_tokens
        if pad_len > 0:
            for key, tensor in final_features.items():
                pad_val = -1 if key == "token_to_mono_idx" else 0
                final_features[key] = pad_dim(tensor, 0, pad_len, value=pad_val)
        elif pad_len < 0:
             for key, tensor in final_features.items():
                final_features[key] = tensor[:max_tokens]

    pad_len_mono = max_mono_chains - num_global_monos
    if pad_len_mono > 0:
        inter_glycan_mask = pad_dim(pad_dim(inter_glycan_mask, 0, pad_len_mono), 1, pad_len_mono)
    elif pad_len_mono < 0:
        inter_glycan_mask = inter_glycan_mask[:max_mono_chains, :max_mono_chains]
    
    final_features["inter_glycan_mask"] = inter_glycan_mask
    
    return final_features


#######################################################################################################
#######################################################################################################

# MONOSACCHARIDE MAPPINGS

#######################################################################################################
#######################################################################################################

#MAX_MONO_CHAINS = 128 # Example: Maximum number of monosaccharide chains expected in a glycan


# 2. Anomeric Configuration
ANOMERIC_MAP: Dict[Optional[str], int] = {'a': 0, 'b': 1, None: 2}
NUM_ANOMERIC_TYPES: int = len(ANOMERIC_MAP)


# 1. Monosaccharide Type (Initial set: FRU, MAN, GLC + OTHER)
MONO_TYPE_MAP: Dict[str, int] = {
    "05L": 0,   "07E": 1,   "0HX": 2,   "0LP": 3,   "0MK": 4,   "0NZ": 5,   "0UB": 6,   "0WK": 7,   "0XY": 8,   "0YT": 9,   "12E": 10,  "145": 11,  "147": 12,  "149": 13,  "14T": 14,  "15L": 15,  "16F": 16,  "16G": 17,  "16O": 18,  "17T": 19,  "18D": 20,  "18O": 21,  "1CF": 22,  "1GL": 23,  "1GN": 24,  "1S3": 25,  "1S4": 26,  "1SD": 27,  "1X4": 28,  "20S": 29,  "20X": 30,
    "22O": 31,  "22S": 32,  "23V": 33,  "24S": 34,  "25E": 35,  "26O": 36,  "27C": 37,  "289": 38,  "291": 39,  "293": 40,  "2DG": 41,  "2DR": 42,  "2F8": 43,  "2FG": 44,  "2FL": 45,  "2GL": 46,  "2GS": 47,  "2H5": 48,  "2M5": 49,  "2M8": 50,  "2WP": 51,  "32O": 52,  "34V": 53,  "38J": 54,  "3DO": 55,  "3FM": 56,  "3HD": 57,  "3J3": 58,  "3J4": 59,  "3LJ": 60,  "3MG": 61,
    "3MK": 62,  "3R3": 63,  "3S6": 64,  "3YW": 65,  "42D": 66,  "445": 67,  "44S": 68,  "46Z": 69,  "475": 70,  "491": 71,  "49A": 72,  "49S": 73,  "49T": 74,  "49V": 75,  "4AM": 76,  "4CQ": 77,  "4GL": 78,  "4GP": 79,  "4JA": 80,  "4N2": 81,  "4NN": 82,  "4QY": 83,  "4R1": 84,  "4SG": 85,  "4U0": 86,  "4U1": 87,  "4U2": 88,  "4UZ": 89,  "4V5": 90,  "50A": 91,
    "510": 92,  "51N": 93,  "56N": 94,  "57S": 95,  "5DI": 96,  "5GF": 97,  "5GO": 98,  "5KQ": 99,  "5KV": 100, "5L2": 101, "5L3": 102, "5LS": 103, "5LT": 104, "5N6": 105, "5QP": 106, "5TH": 107, "5TJ": 108, "5TK": 109, "5TM": 110, "604": 111, "61J": 112, "62I": 113, "64K": 114, "66O": 115, "6BG": 116,  "6C2": 117,  "6GB": 118,  "6GP": 119,  "6GR": 120,
    "6K3": 121, "6KH": 122, "6KL": 123, "6KS": 124, "6KU": 125, "6KW": 126, "6LS": 127, "6LW": 128, "6MJ": 129, "6MN": 130, "6PY": 131, "6PZ": 132, "6S2": 133, "6UD": 134, "6Y6": 135, "6YR": 136, "6ZC": 137, "73E": 138, "79J": 139, "7CV": 140, "7D1": 141, "7GP": 142, "7JZ": 143, "7K2": 144, "7K3": 145, "7NU": 146, "83Y": 147, "89Y": 148, "8B7": 149, "8B9": 150, "8EX": 151, "8GA": 152,
    "8GG": 153, "8GP": 154, "8LM": 155, "8LR": 156, "8OQ": 157, "8PK": 158, "8S0": 159, "95Z": 160, "96O": 161, "9AM": 162, "9C1": 163, "9CD": 164, "9GP": 165, "9KJ": 166, "9MR": 167, "9OK": 168, "9PG": 169, "9QG": 170, "9QZ": 171, "9S7": 172, "9SG": 173, "9SJ": 174, "9SM": 175, "9SP": 176, "9T1": 177, "9T7": 178, "9VP": 179, "9WJ": 180, "9WN": 181, "9WZ": 182, "9YW": 183,
    "A0K": 184, "A1Q": 185, "A2G": 186, "A5C": 187, "A6P": 188, "AAL": 189, "ABD": 190, "ABE": 191, "ABF": 192, "ABL": 193, "AC1": 194, "ACR": 195, "ACX": 196, "ADA": 197, "AF1": 198, "AFD": 199, "AFO": 200, "AFP": 201, "AFR": 202, "AGL": 203, "AGR": 204, "AH2": 205, "AH8": 206, "AHG": 207, "AHM": 208, "AHR": 209, "AIG": 210, "ALL": 211, "ALX": 212, "AMG": 213, "AMN": 214, "AMU": 215,
    "AMV": 216, "ANA": 217, "AOG": 218, "AQA": 219, "ARA": 220, "ARB": 221, "ARI": 222, "ARW": 223, "ASC": 224, "ASG": 225, "ASO": 226, "AXP": 227, "AXR": 228, "AY9": 229, "AZC": 230, "B0D": 231, "B16": 232, "B1H": 233, "B1N": 234, "B6D": 235, "B7G": 236, "B8D": 237, "B9D": 238, "BBK": 239, "BBV": 240, "BCD": 241, "BCW": 242, "BDF": 243, "BDG": 244, "BDP": 245, "BDR": 246, "BDZ": 247,
    "BEM": 248, "BFN": 249, "BG6": 250, "BG8": 251, "BGC": 252, "BGL": 253, "BGN": 254, "BGP": 255, "BGS": 256, "BHG": 257, "BM3": 258, "BM7": 259, "BMA": 260, "BMX": 261, "BND": 262, "BNG": 263, "BNX": 264, "BO1": 265, "BOG": 266, "BQY": 267, "BS7": 268, "BTG": 269, "BTU": 270, "BWG": 271, "BXF": 272, "BXX": 273, "BXY": 274, "BZD": 275,
    "C3B": 276, "C3G": 277, "C3X": 278, "C4B": 279, "C4W": 280, "C5X": 281, "CBF": 282, "CBI": 283, "CBK": 284, "CDR": 285, "CE5": 286, "CE6": 287, "CE8": 288, "CEG": 289, "CEX": 290, "CEY": 291, "CEZ": 292, "CGF": 293, "CJB": 294, "CKB": 295, "CKP": 296, "CNP": 297, "CR1": 298, "CR6": 299, "CRA": 300, "CT3": 301, "CTO": 302, "CTR": 303, "CTT": 304,
    "D0N": 305, "D1M": 306, "D5E": 307, "D6G": 308, "DAF": 309, "DAG": 310, "DAN": 311, "DDA": 312, "DDL": 313, "DEG": 314, "DEL": 315, "DFR": 316, "DFX": 317, "DGO": 318, "DGS": 319, "DJB": 320, "DJE": 321, "DK4": 322, "DKX": 323, "DKZ": 324, "DL6": 325, "DLD": 326, "DLF": 327, "DLG": 328, "DO8": 329, "DOM": 330, "DPC": 331, "DQR": 332, "DR2": 333, "DR3": 334, "DR5": 335,
    "DRI": 336, "DSR": 337, "DT6": 338, "DVC": 339, "DYM": 340, "E3M": 341, "E5G": 342, "EAG": 343, "EBG": 344, "EBQ": 345, "EEN": 346, "EEQ": 347, "EGA": 348, "EMP": 349, "EMZ": 350, "EPG": 351, "EQP": 352, "EQV": 353, "ERE": 354, "ERI": 355, "ETT": 356, "F1P": 357, "F1X": 358, "F55": 359, "F58": 360, "F6P": 361, "FBP": 362, "FCA": 363, "FCB": 364, "FCT": 365, "FDP": 366,
    "FDQ": 367, "FFC": 368, "FFX": 369, "FIF": 370, "FK9": 371, "FKD": 372, "FMF": 373, "FMO": 374, "FNG": 375, "FNY": 376, "FRU": 377, "FSA": 378, "FSI": 379, "FSM": 380, "FSR": 381, "FSW": 382, "FUB": 383, "FUC": 384, "FUF": 385, "FUL": 386, "FUY": 387, "FVQ": 388, "FX1": 389, "FYJ": 390, "G0S": 391, "G16": 392, "G1P": 393, "G20": 394, "G28": 395, "G2F": 396,
    "G3F": 397, "G4D": 398, "G4S": 399, "G6D": 400, "G6P": 401, "G6S": 402, "G7P": 403, "G8Z": 404, "GAA": 405, "GAC": 406, "GAD": 407, "GAF": 408, "GAL": 409, "GAT": 410, "GBH": 411, "GC1": 412, "GC4": 413, "GC9": 414, "GCB": 415, "GCD": 416, "GCN": 417, "GCO": 418, "GCS": 419, "GCT": 420, "GCU": 421, "GCV": 422, "GCW": 423, "GDA": 424, "GDL": 425,
    "GE1": 426, "GE3": 427, "GFP": 428, "GIV": 429, "GL0": 430, "GL1": 431, "GL2": 432, "GL4": 433, "GL5": 434, "GL6": 435, "GL7": 436, "GL9": 437, "GLA": 438, "GLC": 439, "GLD": 440, "GLF": 441, "GLG": 442, "GLO": 443, "GLP": 444, "GLS": 445, "GLT": 446, "GM0": 447, "GMB": 448, "GMH": 449, "GMT": 450, "GMZ": 451, "GN1": 452, "GN4": 453, "GNS": 454, "GNX": 455,
    "GP0": 456, "GP1": 457, "GP4": 458, "GPH": 459, "GPK": 460, "GPM": 461, "GPO": 462, "GPQ": 463, "GPU": 464, "GPV": 465, "GPW": 466, "GQ1": 467, "GRF": 468, "GRX": 469, "GS1": 470, "GS9": 471, "GTK": 472, "GTM": 473, "GTR": 474, "GU0": 475, "GU1": 476, "GU2": 477, "GU3": 478, "GU4": 479, "GU5": 480, "GU6": 481, "GU8": 482, "GU9": 483, "GUF": 484, "GUL": 485, "GUP": 486,
    "GUZ": 487, "GXL": 488, "GYE": 489, "GYG": 490, "GYP": 491, "GYU": 492, "GYV": 493, "GZL": 494, "H1M": 495, "H1S": 496, "H2P": 497, "H53": 498, "H6Q": 499, "H6Z": 500, "HBZ": 501, "HD4": 502, "HNV": 503, "HNW": 504, "HSG": 505, "HSH": 506, "HSJ": 507, "HSQ": 508, "HSX": 509, "HSY": 510, "HTG": 511, "HTM": 512, "I57": 513, "IAB": 514, "IDC": 515, "IDF": 516, "IDG": 517, "IDR": 518, 
    "IDS": 519, "IDU": 520, "IDX": 521, "IDY": 522, "IEM": 523, "IN1": 524, "IPT": 525, "ISD": 526, "ISL": 527, "ISX": 528, "IXD": 529, "J5B": 530, "JFZ": 531, "JHM": 532, "JLT": 533, "JS2": 534, "JV4": 535, "JVA": 536, "JVS": 537, "JZR": 538, "K5B": 539, "K99": 540, "KBA": 541, "KBG": 542, "KD5": 543, "KDA": 544, "KDB": 545, "KDD": 546, "KDE": 547, "KDF": 548, "KDM": 549, "KDN": 550, 
    "KDO": 551, "KDR": 552, "KFN": 553, "KG1": 554, "KGM": 555, "KHP": 556, "KME": 557, "KO1": 558, "KO2": 559, "KOT": 560, "KTU": 561,
    "L1L": 562, "L6S": 563, "LAH": 564, "LAK": 565, "LAO": 566, "LAT": 567, "LB2": 568, "LBS": 569, "LBT": 570, "LCN": 571, "LDY": 572, "LEC": 573, "LFR": 574, "LGC": 575, "LGU": 576, "LKA": 577, "LKS": 578, "LNV": 579, "LOG": 580, "LOX": 581, "LRH": 582, "LVO": 583, "LVZ": 584, "LXB": 585, "LXC": 586, "LXZ": 587, "LZ0": 588, "M1F": 589, "M1P": 590, "M2F": 591, "M3N": 592, "M55": 593, "M6D": 594, 
    "M6P": 595, "M7B": 596, "M7P": 597, "M8C": 598, "MA1": 599, "MA2": 600, "MA3": 601, "MA8": 602, "MAF": 603, "MAG": 604, "MAL": 605, "MAN": 606, "MAT": 607, "MAV": 608, "MAW": 609, "MBE": 610, "MBF": 611, "MBG": 612, "MCU": 613, "MDA": 614, "MDP": 615, "MFB": 616, "MFU": 617, "MG5": 618, "MGC": 619, "MGL": 620, "MGS": 621, "MJJ": 622, "MLB": 623, "MLR": 624, "MMA": 625, "MN0": 626, 
    "MNA": 627, "MQG": 628, "MQT": 629, "MRH": 630, "MRP": 631,"MSX": 632, "MTT": 633, "MUB": 634, "MUR": 635, "MVP": 636, "MXY": 637, "MXZ": 638, "MYG": 639, "N1L": 640, "N9S": 641, "NA1": 642, "NAA": 643, "NAG": 644, "NBG": 645, "NBX": 646, "NBY": 647, "NDG": 648, "NFG": 649, "NG1": 650, "NG6": 651, "NGA": 652, "NGC": 653, "NGE": 654, "NGK": 655, "NGR": 656, "NGS": 657, "NGY": 658, "NGZ": 659, "NHF": 660, 
    "NLC": 661, "NM6": 662, "NM9": 663, "NNG": 664, "NPF": 665, "NSQ": 666, "NT1": 667, "NTF": 668, "NTO": 669, "NTP": 670, "NXD": 671, "NYT": 672,
    "O1G": 673, "OAK": 674, "OEL": 675, "OI7": 676, "OPM": 677, "OSU": 678, "OTG": 679, "OTN": 680, "OTU": 681, "OX2": 682, "P53": 683, "P6P": 684, "PA1": 685, "PAV": 686, "PDX": 687, "PH5": 688, "PKM": 689, "PNA": 690, "PNG": 691, "PNJ": 692, "PNW": 693, "PPC": 694, "PRP": 695, "PSG": 696, "PSV": 697, "PUF": 698, "PZU": 699, "QIF": 700, "QKH": 701, "QPS": 702, "R1P": 703, "R1X": 704, "R2B": 705, "R2G": 706,
    "RAE": 707, "RAF": 708, "RAM": 709, "RAO": 710, "RCD": 711, "RER": 712, "RF5": 713, "RGG": 714, "RHA": 715, "RHC": 716, "RI2": 717, "RIB": 718, "RIP": 719, "RM4": 720, "RP3": 721, "RP5": 722, "RP6": 723, "RR7": 724, "RRJ": 725, "RRY": 726, "RST": 727, "RTG": 728, "RTV": 729, "RUG": 730, "RUU": 731, "RV7": 732, "RVG": 733, "RVM": 734, "RWI": 735, "RY7": 736, "RZM": 737, "S7P": 738, "S81": 739,
     "SA0": 740, "SCG": 741, "SCR": 742, "SDY": 743, "SEJ": 744, "SF6": 745, "SF9": 746, 
    "SFJ": 747, "SFU": 748, "SG4": 749, "SG5": 750, "SG6": 751, "SG7": 752, "SGA": 753, "SGC": 754, "SGD": 755, "SGN": 756, "SHB": 757, "SHD": 758, "SHG": 759, "SIA": 760, "SID": 761, "SIO": 762, "SIZ": 763, "SLB": 764, "SLM": 765, "SLT": 766, "SMD": 767, "SN5": 768, "SNG": 769, "SOE": 770, "SOG": 771, 
    "SOR": 772, "SR1": 773, "SSG": 774, "STZ": 775, "SUC": 776, "SUP": 777, "SUS": 778, "SWE": 779, "SZZ": 780, "T68": 781, "T6P": 782, "T6T": 783, "TA6": 784, "TCB": 785, "TCG": 786, "TDG": 787, "TEU": 788, "TF0": 789, "TFU": 790, "TGA": 791, "TGK": 792, "TGR": 793, "TGY": 794, "TH1": 795, "TMR": 796, 
    "TMX": 797, "TNX": 798, "TOA": 799, "TOC": 800, "TQY": 801, "TRE": 802, "TRV": 803, "TS8": 804, "TT7": 805, "TTV": 806, "TTZ": 807, "TU4": 808, "TUG": 809, "TUJ": 810, "TUP": 811, "TUR": 812, "TVD": 813, "TVG": 814, "TVM": 815, "TVS": 816, "TVV": 817, "TVY": 818, "TW7": 819, "TWA": 820, "TWD": 821, "TWG": 822, "TWJ": 823, "TWY": 824, "TXB": 825, "TYV": 826,
    "U1Y": 827, "U2A": 828, "U2D": 829, "U63": 830, "U8V": 831, "U97": 832, "U9A": 833, "U9D": 834, "U9G": 835, "U9J": 836, "U9M": 837, "UAP": 838, "UCD": 839, "UDC": 840, "UEA": 841, "V3M": 842, "V3P": 843, "V71": 844, "VG1": 845, "VTB": 846, "W9T": 847, "WIA": 848, "WOO": 849, "WUN": 850, "X0X": 851, "X1P": 852, "X1X": 853, "X2F": 854, "X6X": 855, "XDX": 856, "XGP": 857, 
    "XIL": 858, "XLF": 859, "XLS": 860, "XMM": 861, "XXM": 862, "XXR": 863, "XXX": 864, "XYF": 865, "XYL": 866, "XYP": 867, "XYS": 868, "XYT": 869, "XYZ": 870, "YIO": 871, "YJM": 872, "YKR": 873, "YO5": 874, "YX0": 875, "YX1": 876, "YYB": 877, "YYH": 878, "YYJ": 879, "YYK": 880, "YYM": 881, "YYQ": 882, "YZ0": 883, "Z0F": 884, "Z15": 885, "Z16": 886, "Z2D": 887, "Z2T": 888, "Z3K": 889, "Z3L": 890, "Z3Q": 891, "Z3U": 892, 
    "Z4K": 893, "Z4R": 894, "Z4S": 895, "Z4U": 896, "Z4V": 897, "Z4W": 898, "Z4Y": 899, "Z57": 900, "Z5J": 901, "Z5L": 902, "Z61": 903, "Z6H": 904, "Z6J": 905, "Z6W": 906, "Z8H": 907, "Z8T": 908, "Z9D": 909, "Z9E": 910, "Z9H": 911, "Z9K": 912, "Z9L": 913, "Z9M": 914, "Z9N": 915, "Z9W": 916, "ZB0": 917, "ZB1": 918, "ZB2": 919, "ZB3": 920, "ZCD": 921, "ZCZ": 922, "ZD0": 923, "ZDC": 924, "ZDO": 925, "ZEE": 926, "ZEL": 927, "ZGE": 928, "ZMR": 929,
    "OTHER": 930
}


NUM_MONO_TYPES: int = len(MONO_TYPE_MAP)
