import torch
import re
from collections import defaultdict
import torch.nn.functional as F
import time
import sys
from typing import Dict, Optional, List, Tuple
from boltz.data import const
from boltz.model.loss.confidence import (
    compute_frame_pred,
    express_coordinate_in_frame,
    lddt_dist,
)
from boltz.model.loss.diffusion import weighted_rigid_align


def factored_lddt_loss(
    true_atom_coords,
    pred_atom_coords,
    feats,
    atom_mask,
    multiplicity=1,
    cardinality_weighted=False,
):
    """Compute the lddt factorized into the different modalities.

    Parameters
    ----------
    true_atom_coords : torch.Tensor
        Ground truth atom coordinates after symmetry correction
    pred_atom_coords : torch.Tensor
        Predicted atom coordinates
    feats : Dict[str, torch.Tensor]
        Input features
    atom_mask : torch.Tensor
        Atom mask
    multiplicity : int
        Diffusion batch size, by default 1

    Returns
    -------
    Dict[str, torch.Tensor]
        The lddt for each modality
    Dict[str, torch.Tensor]
        The total number of pairs for each modality

    """
    # extract necessary features
    atom_type = (
        torch.bmm(
            feats["atom_to_token"].float(), feats["mol_type"].unsqueeze(-1).float()
        )
        .squeeze(-1)
        .long()
    )
    atom_type = atom_type.repeat_interleave(multiplicity, 0)

    ligand_mask = (atom_type == const.chain_type_ids["NONPOLYMER"]).float()
    dna_mask = (atom_type == const.chain_type_ids["DNA"]).float()
    rna_mask = (atom_type == const.chain_type_ids["RNA"]).float()
    protein_mask = (atom_type == const.chain_type_ids["PROTEIN"]).float()

    nucleotide_mask = dna_mask + rna_mask

    true_d = torch.cdist(true_atom_coords, true_atom_coords)
    pred_d = torch.cdist(pred_atom_coords, pred_atom_coords)

    pair_mask = atom_mask[:, :, None] * atom_mask[:, None, :]
    pair_mask = (
        pair_mask
        * (1 - torch.eye(pair_mask.shape[1], device=pair_mask.device))[None, :, :]
    )

    cutoff = 15 + 15 * (
        1 - (1 - nucleotide_mask[:, :, None]) * (1 - nucleotide_mask[:, None, :])
    )

    # compute different lddts
    dna_protein_mask = pair_mask * (
        dna_mask[:, :, None] * protein_mask[:, None, :]
        + protein_mask[:, :, None] * dna_mask[:, None, :]
    )
    dna_protein_lddt, dna_protein_total = lddt_dist(
        pred_d, true_d, dna_protein_mask, cutoff
    )
    del dna_protein_mask

    rna_protein_mask = pair_mask * (
        rna_mask[:, :, None] * protein_mask[:, None, :]
        + protein_mask[:, :, None] * rna_mask[:, None, :]
    )
    rna_protein_lddt, rna_protein_total = lddt_dist(
        pred_d, true_d, rna_protein_mask, cutoff
    )
    del rna_protein_mask

    ligand_protein_mask = pair_mask * (
        ligand_mask[:, :, None] * protein_mask[:, None, :]
        + protein_mask[:, :, None] * ligand_mask[:, None, :]
    )
    ligand_protein_lddt, ligand_protein_total = lddt_dist(
        pred_d, true_d, ligand_protein_mask, cutoff
    )
    del ligand_protein_mask

    dna_ligand_mask = pair_mask * (
        dna_mask[:, :, None] * ligand_mask[:, None, :]
        + ligand_mask[:, :, None] * dna_mask[:, None, :]
    )
    dna_ligand_lddt, dna_ligand_total = lddt_dist(
        pred_d, true_d, dna_ligand_mask, cutoff
    )
    del dna_ligand_mask

    rna_ligand_mask = pair_mask * (
        rna_mask[:, :, None] * ligand_mask[:, None, :]
        + ligand_mask[:, :, None] * rna_mask[:, None, :]
    )
    rna_ligand_lddt, rna_ligand_total = lddt_dist(
        pred_d, true_d, rna_ligand_mask, cutoff
    )
    del rna_ligand_mask

    intra_dna_mask = pair_mask * (dna_mask[:, :, None] * dna_mask[:, None, :])
    intra_dna_lddt, intra_dna_total = lddt_dist(pred_d, true_d, intra_dna_mask, cutoff)
    del intra_dna_mask

    intra_rna_mask = pair_mask * (rna_mask[:, :, None] * rna_mask[:, None, :])
    intra_rna_lddt, intra_rna_total = lddt_dist(pred_d, true_d, intra_rna_mask, cutoff)
    del intra_rna_mask

    chain_id = feats["asym_id"]
    atom_chain_id = (
        torch.bmm(feats["atom_to_token"].float(), chain_id.unsqueeze(-1).float())
        .squeeze(-1)
        .long()
    )
    atom_chain_id = atom_chain_id.repeat_interleave(multiplicity, 0)
    same_chain_mask = (atom_chain_id[:, :, None] == atom_chain_id[:, None, :]).float()

    intra_ligand_mask = (
        pair_mask
        * same_chain_mask
        * (ligand_mask[:, :, None] * ligand_mask[:, None, :])
    )
    intra_ligand_lddt, intra_ligand_total = lddt_dist(
        pred_d, true_d, intra_ligand_mask, cutoff
    )
    del intra_ligand_mask

    intra_protein_mask = (
        pair_mask
        * same_chain_mask
        * (protein_mask[:, :, None] * protein_mask[:, None, :])
    )
    intra_protein_lddt, intra_protein_total = lddt_dist(
        pred_d, true_d, intra_protein_mask, cutoff
    )
    del intra_protein_mask

    protein_protein_mask = (
        pair_mask
        * (1 - same_chain_mask)
        * (protein_mask[:, :, None] * protein_mask[:, None, :])
    )
    protein_protein_lddt, protein_protein_total = lddt_dist(
        pred_d, true_d, protein_protein_mask, cutoff
    )
    del protein_protein_mask

    lddt_dict = {
        "dna_protein": dna_protein_lddt,
        "rna_protein": rna_protein_lddt,
        "ligand_protein": ligand_protein_lddt,
        "dna_ligand": dna_ligand_lddt,
        "rna_ligand": rna_ligand_lddt,
        "intra_ligand": intra_ligand_lddt,
        "intra_dna": intra_dna_lddt,
        "intra_rna": intra_rna_lddt,
        "intra_protein": intra_protein_lddt,
        "protein_protein": protein_protein_lddt,
    }

    total_dict = {
        "dna_protein": dna_protein_total,
        "rna_protein": rna_protein_total,
        "ligand_protein": ligand_protein_total,
        "dna_ligand": dna_ligand_total,
        "rna_ligand": rna_ligand_total,
        "intra_ligand": intra_ligand_total,
        "intra_dna": intra_dna_total,
        "intra_rna": intra_rna_total,
        "intra_protein": intra_protein_total,
        "protein_protein": protein_protein_total,
    }
    if not cardinality_weighted:
        for key in total_dict:
            total_dict[key] = (total_dict[key] > 0.0).float()

    return lddt_dict, total_dict


def factored_token_lddt_dist_loss(true_d, pred_d, feats, cardinality_weighted=False):
    """Compute the distogram lddt factorized into the different modalities.

    Parameters
    ----------
    true_d : torch.Tensor
        Ground truth atom distogram
    pred_d : torch.Tensor
        Predicted atom distogram
    feats : Dict[str, torch.Tensor]
        Input features

    Returns
    -------
    Tensor
        The lddt for each modality
    Tensor
        The total number of pairs for each modality

    """
    # extract necessary features
    token_type = feats["mol_type"]

    ligand_mask = (token_type == const.chain_type_ids["NONPOLYMER"]).float()
    dna_mask = (token_type == const.chain_type_ids["DNA"]).float()
    rna_mask = (token_type == const.chain_type_ids["RNA"]).float()
    protein_mask = (token_type == const.chain_type_ids["PROTEIN"]).float()
    nucleotide_mask = dna_mask + rna_mask

    token_mask = feats["token_disto_mask"]
    token_mask = token_mask[:, :, None] * token_mask[:, None, :]
    token_mask = token_mask * (1 - torch.eye(token_mask.shape[1])[None]).to(token_mask)

    cutoff = 15 + 15 * (
        1 - (1 - nucleotide_mask[:, :, None]) * (1 - nucleotide_mask[:, None, :])
    )

    # compute different lddts
    dna_protein_mask = token_mask * (
        dna_mask[:, :, None] * protein_mask[:, None, :]
        + protein_mask[:, :, None] * dna_mask[:, None, :]
    )
    dna_protein_lddt, dna_protein_total = lddt_dist(
        pred_d, true_d, dna_protein_mask, cutoff
    )

    rna_protein_mask = token_mask * (
        rna_mask[:, :, None] * protein_mask[:, None, :]
        + protein_mask[:, :, None] * rna_mask[:, None, :]
    )
    rna_protein_lddt, rna_protein_total = lddt_dist(
        pred_d, true_d, rna_protein_mask, cutoff
    )

    ligand_protein_mask = token_mask * (
        ligand_mask[:, :, None] * protein_mask[:, None, :]
        + protein_mask[:, :, None] * ligand_mask[:, None, :]
    )
    ligand_protein_lddt, ligand_protein_total = lddt_dist(
        pred_d, true_d, ligand_protein_mask, cutoff
    )

    dna_ligand_mask = token_mask * (
        dna_mask[:, :, None] * ligand_mask[:, None, :]
        + ligand_mask[:, :, None] * dna_mask[:, None, :]
    )
    dna_ligand_lddt, dna_ligand_total = lddt_dist(
        pred_d, true_d, dna_ligand_mask, cutoff
    )

    rna_ligand_mask = token_mask * (
        rna_mask[:, :, None] * ligand_mask[:, None, :]
        + ligand_mask[:, :, None] * rna_mask[:, None, :]
    )
    rna_ligand_lddt, rna_ligand_total = lddt_dist(
        pred_d, true_d, rna_ligand_mask, cutoff
    )

    chain_id = feats["asym_id"]
    same_chain_mask = (chain_id[:, :, None] == chain_id[:, None, :]).float()
    intra_ligand_mask = (
        token_mask
        * same_chain_mask
        * (ligand_mask[:, :, None] * ligand_mask[:, None, :])
    )
    intra_ligand_lddt, intra_ligand_total = lddt_dist(
        pred_d, true_d, intra_ligand_mask, cutoff
    )

    intra_dna_mask = token_mask * (dna_mask[:, :, None] * dna_mask[:, None, :])
    intra_dna_lddt, intra_dna_total = lddt_dist(pred_d, true_d, intra_dna_mask, cutoff)

    intra_rna_mask = token_mask * (rna_mask[:, :, None] * rna_mask[:, None, :])
    intra_rna_lddt, intra_rna_total = lddt_dist(pred_d, true_d, intra_rna_mask, cutoff)

    chain_id = feats["asym_id"]
    same_chain_mask = (chain_id[:, :, None] == chain_id[:, None, :]).float()

    intra_protein_mask = (
        token_mask
        * same_chain_mask
        * (protein_mask[:, :, None] * protein_mask[:, None, :])
    )
    intra_protein_lddt, intra_protein_total = lddt_dist(
        pred_d, true_d, intra_protein_mask, cutoff
    )

    protein_protein_mask = (
        token_mask
        * (1 - same_chain_mask)
        * (protein_mask[:, :, None] * protein_mask[:, None, :])
    )
    protein_protein_lddt, protein_protein_total = lddt_dist(
        pred_d, true_d, protein_protein_mask, cutoff
    )

    lddt_dict = {
        "dna_protein": dna_protein_lddt,
        "rna_protein": rna_protein_lddt,
        "ligand_protein": ligand_protein_lddt,
        "dna_ligand": dna_ligand_lddt,
        "rna_ligand": rna_ligand_lddt,
        "intra_ligand": intra_ligand_lddt,
        "intra_dna": intra_dna_lddt,
        "intra_rna": intra_rna_lddt,
        "intra_protein": intra_protein_lddt,
        "protein_protein": protein_protein_lddt,
    }

    total_dict = {
        "dna_protein": dna_protein_total,
        "rna_protein": rna_protein_total,
        "ligand_protein": ligand_protein_total,
        "dna_ligand": dna_ligand_total,
        "rna_ligand": rna_ligand_total,
        "intra_ligand": intra_ligand_total,
        "intra_dna": intra_dna_total,
        "intra_rna": intra_rna_total,
        "intra_protein": intra_protein_total,
        "protein_protein": protein_protein_total,
    }

    if not cardinality_weighted:
        for key in total_dict:
            total_dict[key] = (total_dict[key] > 0.0).float()

    return lddt_dict, total_dict


def compute_plddt_mae(
    pred_atom_coords,
    feats,
    true_atom_coords,
    pred_lddt,
    true_coords_resolved_mask,
    multiplicity=1,
):
    """Compute the plddt mean absolute error.

    Parameters
    ----------
    pred_atom_coords : torch.Tensor
        Predicted atom coordinates
    feats : torch.Tensor
        Input features
    true_atom_coords : torch.Tensor
        Ground truth atom coordinates
    pred_lddt : torch.Tensor
        Predicted lddt
    true_coords_resolved_mask : torch.Tensor
        Resolved atom mask
    multiplicity : int
        Diffusion batch size, by default 1

    Returns
    -------
    Tensor
        The mae for each modality
    Tensor
        The total number of pairs for each modality

    """
    # extract necessary features
    atom_mask = true_coords_resolved_mask
    R_set_to_rep_atom = feats["r_set_to_rep_atom"]
    R_set_to_rep_atom = R_set_to_rep_atom.repeat_interleave(multiplicity, 0).float()

    token_type = feats["mol_type"]
    token_type = token_type.repeat_interleave(multiplicity, 0)
    is_nucleotide_token = (token_type == const.chain_type_ids["DNA"]).float() + (
        token_type == const.chain_type_ids["RNA"]
    ).float()

    B = true_atom_coords.shape[0]

    atom_to_token = feats["atom_to_token"].float()
    atom_to_token = atom_to_token.repeat_interleave(multiplicity, 0)

    token_to_rep_atom = feats["token_to_rep_atom"].float()
    token_to_rep_atom = token_to_rep_atom.repeat_interleave(multiplicity, 0)

    true_token_coords = torch.bmm(token_to_rep_atom, true_atom_coords)
    pred_token_coords = torch.bmm(token_to_rep_atom, pred_atom_coords)

    # compute true lddt
    true_d = torch.cdist(
        true_token_coords,
        torch.bmm(R_set_to_rep_atom, true_atom_coords),
    )
    pred_d = torch.cdist(
        pred_token_coords,
        torch.bmm(R_set_to_rep_atom, pred_atom_coords),
    )

    pair_mask = atom_mask.unsqueeze(-1) * atom_mask.unsqueeze(-2)
    pair_mask = (
        pair_mask
        * (1 - torch.eye(pair_mask.shape[1], device=pair_mask.device))[None, :, :]
    )
    pair_mask = torch.einsum("bnm,bkm->bnk", pair_mask, R_set_to_rep_atom)

    pair_mask = torch.bmm(token_to_rep_atom, pair_mask)
    atom_mask = torch.bmm(token_to_rep_atom, atom_mask.unsqueeze(-1).float()).squeeze(
        -1
    )
    is_nucleotide_R_element = torch.bmm(
        R_set_to_rep_atom, torch.bmm(atom_to_token, is_nucleotide_token.unsqueeze(-1))
    ).squeeze(-1)
    cutoff = 15 + 15 * is_nucleotide_R_element.reshape(B, 1, -1).repeat(
        1, true_d.shape[1], 1
    )

    target_lddt, mask_no_match = lddt_dist(
        pred_d, true_d, pair_mask, cutoff, per_atom=True
    )

    protein_mask = (
        (token_type == const.chain_type_ids["PROTEIN"]).float()
        * atom_mask
        * mask_no_match
    )
    ligand_mask = (
        (token_type == const.chain_type_ids["NONPOLYMER"]).float()
        * atom_mask
        * mask_no_match
    )
    dna_mask = (
        (token_type == const.chain_type_ids["DNA"]).float() * atom_mask * mask_no_match
    )
    rna_mask = (
        (token_type == const.chain_type_ids["RNA"]).float() * atom_mask * mask_no_match
    )

    protein_mae = torch.sum(torch.abs(target_lddt - pred_lddt) * protein_mask) / (
        torch.sum(protein_mask) + 1e-5
    )
    protein_total = torch.sum(protein_mask)
    ligand_mae = torch.sum(torch.abs(target_lddt - pred_lddt) * ligand_mask) / (
        torch.sum(ligand_mask) + 1e-5
    )
    ligand_total = torch.sum(ligand_mask)
    dna_mae = torch.sum(torch.abs(target_lddt - pred_lddt) * dna_mask) / (
        torch.sum(dna_mask) + 1e-5
    )
    dna_total = torch.sum(dna_mask)
    rna_mae = torch.sum(torch.abs(target_lddt - pred_lddt) * rna_mask) / (
        torch.sum(rna_mask) + 1e-5
    )
    rna_total = torch.sum(rna_mask)

    mae_plddt_dict = {
        "protein": protein_mae,
        "ligand": ligand_mae,
        "dna": dna_mae,
        "rna": rna_mae,
    }
    total_dict = {
        "protein": protein_total,
        "ligand": ligand_total,
        "dna": dna_total,
        "rna": rna_total,
    }

    return mae_plddt_dict, total_dict


def compute_pde_mae(
    pred_atom_coords,
    feats,
    true_atom_coords,
    pred_pde,
    true_coords_resolved_mask,
    multiplicity=1,
):
    """Compute the plddt mean absolute error.

    Parameters
    ----------
    pred_atom_coords : torch.Tensor
        Predicted atom coordinates
    feats : torch.Tensor
        Input features
    true_atom_coords : torch.Tensor
        Ground truth atom coordinates
    pred_pde : torch.Tensor
        Predicted pde
    true_coords_resolved_mask : torch.Tensor
        Resolved atom mask
    multiplicity : int
        Diffusion batch size, by default 1

    Returns
    -------
    Tensor
        The mae for each modality
    Tensor
        The total number of pairs for each modality

    """
    # extract necessary features
    token_to_rep_atom = feats["token_to_rep_atom"].float()
    token_to_rep_atom = token_to_rep_atom.repeat_interleave(multiplicity, 0)

    token_mask = torch.bmm(
        token_to_rep_atom, true_coords_resolved_mask.unsqueeze(-1).float()
    ).squeeze(-1)

    token_type = feats["mol_type"]
    token_type = token_type.repeat_interleave(multiplicity, 0)

    true_token_coords = torch.bmm(token_to_rep_atom, true_atom_coords)
    pred_token_coords = torch.bmm(token_to_rep_atom, pred_atom_coords)

    # compute true pde
    true_d = torch.cdist(true_token_coords, true_token_coords)
    pred_d = torch.cdist(pred_token_coords, pred_token_coords)
    target_pde = (
        torch.clamp(
            torch.floor(torch.abs(true_d - pred_d) * 64 / 32).long(), max=63
        ).float()
        * 0.5
        + 0.25
    )

    pair_mask = token_mask.unsqueeze(-1) * token_mask.unsqueeze(-2)
    pair_mask = (
        pair_mask
        * (1 - torch.eye(pair_mask.shape[1], device=pair_mask.device))[None, :, :]
    )

    protein_mask = (token_type == const.chain_type_ids["PROTEIN"]).float()
    ligand_mask = (token_type == const.chain_type_ids["NONPOLYMER"]).float()
    dna_mask = (token_type == const.chain_type_ids["DNA"]).float()
    rna_mask = (token_type == const.chain_type_ids["RNA"]).float()

    # compute different pdes
    dna_protein_mask = pair_mask * (
        dna_mask[:, :, None] * protein_mask[:, None, :]
        + protein_mask[:, :, None] * dna_mask[:, None, :]
    )
    dna_protein_mae = torch.sum(torch.abs(target_pde - pred_pde) * dna_protein_mask) / (
        torch.sum(dna_protein_mask) + 1e-5
    )
    dna_protein_total = torch.sum(dna_protein_mask)

    rna_protein_mask = pair_mask * (
        rna_mask[:, :, None] * protein_mask[:, None, :]
        + protein_mask[:, :, None] * rna_mask[:, None, :]
    )
    rna_protein_mae = torch.sum(torch.abs(target_pde - pred_pde) * rna_protein_mask) / (
        torch.sum(rna_protein_mask) + 1e-5
    )
    rna_protein_total = torch.sum(rna_protein_mask)

    ligand_protein_mask = pair_mask * (
        ligand_mask[:, :, None] * protein_mask[:, None, :]
        + protein_mask[:, :, None] * ligand_mask[:, None, :]
    )
    ligand_protein_mae = torch.sum(
        torch.abs(target_pde - pred_pde) * ligand_protein_mask
    ) / (torch.sum(ligand_protein_mask) + 1e-5)
    ligand_protein_total = torch.sum(ligand_protein_mask)

    dna_ligand_mask = pair_mask * (
        dna_mask[:, :, None] * ligand_mask[:, None, :]
        + ligand_mask[:, :, None] * dna_mask[:, None, :]
    )
    dna_ligand_mae = torch.sum(torch.abs(target_pde - pred_pde) * dna_ligand_mask) / (
        torch.sum(dna_ligand_mask) + 1e-5
    )
    dna_ligand_total = torch.sum(dna_ligand_mask)

    rna_ligand_mask = pair_mask * (
        rna_mask[:, :, None] * ligand_mask[:, None, :]
        + ligand_mask[:, :, None] * rna_mask[:, None, :]
    )
    rna_ligand_mae = torch.sum(torch.abs(target_pde - pred_pde) * rna_ligand_mask) / (
        torch.sum(rna_ligand_mask) + 1e-5
    )
    rna_ligand_total = torch.sum(rna_ligand_mask)

    intra_ligand_mask = pair_mask * (ligand_mask[:, :, None] * ligand_mask[:, None, :])
    intra_ligand_mae = torch.sum(
        torch.abs(target_pde - pred_pde) * intra_ligand_mask
    ) / (torch.sum(intra_ligand_mask) + 1e-5)
    intra_ligand_total = torch.sum(intra_ligand_mask)

    intra_dna_mask = pair_mask * (dna_mask[:, :, None] * dna_mask[:, None, :])
    intra_dna_mae = torch.sum(torch.abs(target_pde - pred_pde) * intra_dna_mask) / (
        torch.sum(intra_dna_mask) + 1e-5
    )
    intra_dna_total = torch.sum(intra_dna_mask)

    intra_rna_mask = pair_mask * (rna_mask[:, :, None] * rna_mask[:, None, :])
    intra_rna_mae = torch.sum(torch.abs(target_pde - pred_pde) * intra_rna_mask) / (
        torch.sum(intra_rna_mask) + 1e-5
    )
    intra_rna_total = torch.sum(intra_rna_mask)

    chain_id = feats["asym_id"].repeat_interleave(multiplicity, 0)
    same_chain_mask = (chain_id[:, :, None] == chain_id[:, None, :]).float()

    intra_protein_mask = (
        pair_mask
        * same_chain_mask
        * (protein_mask[:, :, None] * protein_mask[:, None, :])
    )
    intra_protein_mae = torch.sum(
        torch.abs(target_pde - pred_pde) * intra_protein_mask
    ) / (torch.sum(intra_protein_mask) + 1e-5)
    intra_protein_total = torch.sum(intra_protein_mask)

    protein_protein_mask = (
        pair_mask
        * (1 - same_chain_mask)
        * (protein_mask[:, :, None] * protein_mask[:, None, :])
    )
    protein_protein_mae = torch.sum(
        torch.abs(target_pde - pred_pde) * protein_protein_mask
    ) / (torch.sum(protein_protein_mask) + 1e-5)
    protein_protein_total = torch.sum(protein_protein_mask)

    mae_pde_dict = {
        "dna_protein": dna_protein_mae,
        "rna_protein": rna_protein_mae,
        "ligand_protein": ligand_protein_mae,
        "dna_ligand": dna_ligand_mae,
        "rna_ligand": rna_ligand_mae,
        "intra_ligand": intra_ligand_mae,
        "intra_dna": intra_dna_mae,
        "intra_rna": intra_rna_mae,
        "intra_protein": intra_protein_mae,
        "protein_protein": protein_protein_mae,
    }
    total_pde_dict = {
        "dna_protein": dna_protein_total,
        "rna_protein": rna_protein_total,
        "ligand_protein": ligand_protein_total,
        "dna_ligand": dna_ligand_total,
        "rna_ligand": rna_ligand_total,
        "intra_ligand": intra_ligand_total,
        "intra_dna": intra_dna_total,
        "intra_rna": intra_rna_total,
        "intra_protein": intra_protein_total,
        "protein_protein": protein_protein_total,
    }

    return mae_pde_dict, total_pde_dict


def compute_pae_mae(
    pred_atom_coords,
    feats,
    true_atom_coords,
    pred_pae,
    true_coords_resolved_mask,
    multiplicity=1,
):
    """Compute the pae mean absolute error.

    Parameters
    ----------
    pred_atom_coords : torch.Tensor
        Predicted atom coordinates
    feats : torch.Tensor
        Input features
    true_atom_coords : torch.Tensor
        Ground truth atom coordinates
    pred_pae : torch.Tensor
        Predicted pae
    true_coords_resolved_mask : torch.Tensor
        Resolved atom mask
    multiplicity : int
        Diffusion batch size, by default 1

    Returns
    -------
    Tensor
        The mae for each modality
    Tensor
        The total number of pairs for each modality

    """
    # Retrieve frames and resolved masks
    frames_idx_original = feats["frames_idx"]
    mask_frame_true = feats["frame_resolved_mask"]

    # Adjust the frames for nonpolymers after symmetry correction!
    # NOTE: frames of polymers do not change under symmetry!
    frames_idx_true, mask_collinear_true = compute_frame_pred(
        true_atom_coords,
        frames_idx_original,
        feats,
        multiplicity,
        resolved_mask=true_coords_resolved_mask,
    )

    frame_true_atom_a, frame_true_atom_b, frame_true_atom_c = (
        frames_idx_true[:, :, :, 0],
        frames_idx_true[:, :, :, 1],
        frames_idx_true[:, :, :, 2],
    )
    # Compute token coords in true frames
    B, N, _ = true_atom_coords.shape
    true_atom_coords = true_atom_coords.reshape(B // multiplicity, multiplicity, -1, 3)
    true_coords_transformed = express_coordinate_in_frame(
        true_atom_coords, frame_true_atom_a, frame_true_atom_b, frame_true_atom_c
    )

    # Compute pred frames and mask
    frames_idx_pred, mask_collinear_pred = compute_frame_pred(
        pred_atom_coords, frames_idx_original, feats, multiplicity
    )
    frame_pred_atom_a, frame_pred_atom_b, frame_pred_atom_c = (
        frames_idx_pred[:, :, :, 0],
        frames_idx_pred[:, :, :, 1],
        frames_idx_pred[:, :, :, 2],
    )
    # Compute token coords in pred frames
    B, N, _ = pred_atom_coords.shape
    pred_atom_coords = pred_atom_coords.reshape(B // multiplicity, multiplicity, -1, 3)
    pred_coords_transformed = express_coordinate_in_frame(
        pred_atom_coords, frame_pred_atom_a, frame_pred_atom_b, frame_pred_atom_c
    )

    target_pae_continuous = torch.sqrt(
        ((true_coords_transformed - pred_coords_transformed) ** 2).sum(-1) + 1e-8
    )
    target_pae = (
        torch.clamp(torch.floor(target_pae_continuous * 64 / 32).long(), max=63).float()
        * 0.5
        + 0.25
    )

    # Compute mask for the pae loss
    b_true_resolved_mask = true_coords_resolved_mask[
        torch.arange(B // multiplicity)[:, None, None].to(
            pred_coords_transformed.device
        ),
        frame_true_atom_b,
    ]

    pair_mask = (
        mask_frame_true[:, None, :, None]  # if true frame is invalid
        * mask_collinear_true[:, :, :, None]  # if true frame is invalid
        * mask_collinear_pred[:, :, :, None]  # if pred frame is invalid
        * b_true_resolved_mask[:, :, None, :]  # If atom j is not resolved
        * feats["token_pad_mask"][:, None, :, None]
        * feats["token_pad_mask"][:, None, None, :]
    )

    token_type = feats["mol_type"]
    token_type = token_type.repeat_interleave(multiplicity, 0)

    protein_mask = (token_type == const.chain_type_ids["PROTEIN"]).float()
    ligand_mask = (token_type == const.chain_type_ids["NONPOLYMER"]).float()
    dna_mask = (token_type == const.chain_type_ids["DNA"]).float()
    rna_mask = (token_type == const.chain_type_ids["RNA"]).float()

    # compute different paes
    dna_protein_mask = pair_mask * (
        dna_mask[:, :, None] * protein_mask[:, None, :]
        + protein_mask[:, :, None] * dna_mask[:, None, :]
    )
    dna_protein_mae = torch.sum(torch.abs(target_pae - pred_pae) * dna_protein_mask) / (
        torch.sum(dna_protein_mask) + 1e-5
    )
    dna_protein_total = torch.sum(dna_protein_mask)

    rna_protein_mask = pair_mask * (
        rna_mask[:, :, None] * protein_mask[:, None, :]
        + protein_mask[:, :, None] * rna_mask[:, None, :]
    )
    rna_protein_mae = torch.sum(torch.abs(target_pae - pred_pae) * rna_protein_mask) / (
        torch.sum(rna_protein_mask) + 1e-5
    )
    rna_protein_total = torch.sum(rna_protein_mask)

    ligand_protein_mask = pair_mask * (
        ligand_mask[:, :, None] * protein_mask[:, None, :]
        + protein_mask[:, :, None] * ligand_mask[:, None, :]
    )
    ligand_protein_mae = torch.sum(
        torch.abs(target_pae - pred_pae) * ligand_protein_mask
    ) / (torch.sum(ligand_protein_mask) + 1e-5)
    ligand_protein_total = torch.sum(ligand_protein_mask)

    dna_ligand_mask = pair_mask * (
        dna_mask[:, :, None] * ligand_mask[:, None, :]
        + ligand_mask[:, :, None] * dna_mask[:, None, :]
    )
    dna_ligand_mae = torch.sum(torch.abs(target_pae - pred_pae) * dna_ligand_mask) / (
        torch.sum(dna_ligand_mask) + 1e-5
    )
    dna_ligand_total = torch.sum(dna_ligand_mask)

    rna_ligand_mask = pair_mask * (
        rna_mask[:, :, None] * ligand_mask[:, None, :]
        + ligand_mask[:, :, None] * rna_mask[:, None, :]
    )
    rna_ligand_mae = torch.sum(torch.abs(target_pae - pred_pae) * rna_ligand_mask) / (
        torch.sum(rna_ligand_mask) + 1e-5
    )
    rna_ligand_total = torch.sum(rna_ligand_mask)

    intra_ligand_mask = pair_mask * (ligand_mask[:, :, None] * ligand_mask[:, None, :])
    intra_ligand_mae = torch.sum(
        torch.abs(target_pae - pred_pae) * intra_ligand_mask
    ) / (torch.sum(intra_ligand_mask) + 1e-5)
    intra_ligand_total = torch.sum(intra_ligand_mask)

    intra_dna_mask = pair_mask * (dna_mask[:, :, None] * dna_mask[:, None, :])
    intra_dna_mae = torch.sum(torch.abs(target_pae - pred_pae) * intra_dna_mask) / (
        torch.sum(intra_dna_mask) + 1e-5
    )
    intra_dna_total = torch.sum(intra_dna_mask)

    intra_rna_mask = pair_mask * (rna_mask[:, :, None] * rna_mask[:, None, :])
    intra_rna_mae = torch.sum(torch.abs(target_pae - pred_pae) * intra_rna_mask) / (
        torch.sum(intra_rna_mask) + 1e-5
    )
    intra_rna_total = torch.sum(intra_rna_mask)

    chain_id = feats["asym_id"].repeat_interleave(multiplicity, 0)
    same_chain_mask = (chain_id[:, :, None] == chain_id[:, None, :]).float()

    intra_protein_mask = (
        pair_mask
        * same_chain_mask
        * (protein_mask[:, :, None] * protein_mask[:, None, :])
    )
    intra_protein_mae = torch.sum(
        torch.abs(target_pae - pred_pae) * intra_protein_mask
    ) / (torch.sum(intra_protein_mask) + 1e-5)
    intra_protein_total = torch.sum(intra_protein_mask)

    protein_protein_mask = (
        pair_mask
        * (1 - same_chain_mask)
        * (protein_mask[:, :, None] * protein_mask[:, None, :])
    )
    protein_protein_mae = torch.sum(
        torch.abs(target_pae - pred_pae) * protein_protein_mask
    ) / (torch.sum(protein_protein_mask) + 1e-5)
    protein_protein_total = torch.sum(protein_protein_mask)

    mae_pae_dict = {
        "dna_protein": dna_protein_mae,
        "rna_protein": rna_protein_mae,
        "ligand_protein": ligand_protein_mae,
        "dna_ligand": dna_ligand_mae,
        "rna_ligand": rna_ligand_mae,
        "intra_ligand": intra_ligand_mae,
        "intra_dna": intra_dna_mae,
        "intra_rna": intra_rna_mae,
        "intra_protein": intra_protein_mae,
        "protein_protein": protein_protein_mae,
    }
    total_pae_dict = {
        "dna_protein": dna_protein_total,
        "rna_protein": rna_protein_total,
        "ligand_protein": ligand_protein_total,
        "dna_ligand": dna_ligand_total,
        "rna_ligand": rna_ligand_total,
        "intra_ligand": intra_ligand_total,
        "intra_dna": intra_dna_total,
        "intra_rna": intra_rna_total,
        "intra_protein": intra_protein_total,
        "protein_protein": protein_protein_total,
    }

    return mae_pae_dict, total_pae_dict


def weighted_minimum_rmsd(
    pred_atom_coords,
    feats,
    multiplicity=1,
    nucleotide_weight=5.0,
    ligand_weight=10.0,
):
    """Compute rmsd of the aligned atom coordinates.

    Parameters
    ----------
    pred_atom_coords : torch.Tensor
        Predicted atom coordinates
    feats : torch.Tensor
        Input features
    multiplicity : int
        Diffusion batch size, by default 1

    Returns
    -------
    Tensor
        The rmsds
    Tensor
        The best rmsd

    """
    atom_coords = feats["coords"]
    atom_coords = atom_coords.repeat_interleave(multiplicity, 0)
    atom_coords = atom_coords[:, 0]

    atom_mask = feats["atom_resolved_mask"]
    atom_mask = atom_mask.repeat_interleave(multiplicity, 0)

    align_weights = atom_coords.new_ones(atom_coords.shape[:2])
    atom_type = (
        torch.bmm(
            feats["atom_to_token"].float(), feats["mol_type"].unsqueeze(-1).float()
        )
        .squeeze(-1)
        .long()
    )
    atom_type = atom_type.repeat_interleave(multiplicity, 0)

    align_weights = align_weights * (
        1
        + nucleotide_weight
        * (
            torch.eq(atom_type, const.chain_type_ids["DNA"]).float()
            + torch.eq(atom_type, const.chain_type_ids["RNA"]).float()
        )
        + ligand_weight
        * torch.eq(atom_type, const.chain_type_ids["NONPOLYMER"]).float()
    )

    with torch.no_grad():
        atom_coords_aligned_ground_truth = weighted_rigid_align(
            atom_coords, pred_atom_coords, align_weights, mask=atom_mask
        )

    # weighted MSE loss of denoised atom positions
    mse_loss = ((pred_atom_coords - atom_coords_aligned_ground_truth) ** 2).sum(dim=-1)
    rmsd = torch.sqrt(
        torch.sum(mse_loss * align_weights * atom_mask, dim=-1)
        / torch.sum(align_weights * atom_mask, dim=-1)
    )
    best_rmsd = torch.min(rmsd.reshape(-1, multiplicity), dim=1).values

    return rmsd, best_rmsd


def weighted_minimum_rmsd_single(
    pred_atom_coords,
    atom_coords,
    atom_mask,
    atom_to_token,
    mol_type,
    nucleotide_weight=5.0,
    ligand_weight=10.0,
):
    """Compute rmsd of the aligned atom coordinates.

    Parameters
    ----------
    pred_atom_coords : torch.Tensor
        Predicted atom coordinates
    atom_coords: torch.Tensor
        Ground truth atom coordinates
    atom_mask : torch.Tensor
        Resolved atom mask
    atom_to_token : torch.Tensor
        Atom to token mapping
    mol_type : torch.Tensor
        Atom type

    Returns
    -------
    Tensor
        The rmsd
    Tensor
        The aligned coordinates
    Tensor
        The aligned weights

    """
    align_weights = atom_coords.new_ones(atom_coords.shape[:2])
    atom_type = (
        torch.bmm(atom_to_token.float(), mol_type.unsqueeze(-1).float())
        .squeeze(-1)
        .long()
    )

    align_weights = align_weights * (
        1
        + nucleotide_weight
        * (
            torch.eq(atom_type, const.chain_type_ids["DNA"]).float()
            + torch.eq(atom_type, const.chain_type_ids["RNA"]).float()
        )
        + ligand_weight
        * torch.eq(atom_type, const.chain_type_ids["NONPOLYMER"]).float()
    )

    with torch.no_grad():
        atom_coords_aligned_ground_truth = weighted_rigid_align(
            atom_coords, pred_atom_coords, align_weights, mask=atom_mask
        )

    # weighted MSE loss of denoised atom positions
    mse_loss = ((pred_atom_coords - atom_coords_aligned_ground_truth) ** 2).sum(dim=-1)
    rmsd = torch.sqrt(
        torch.sum(mse_loss * align_weights * atom_mask, dim=-1)
        / torch.sum(align_weights * atom_mask, dim=-1)
    )
    return rmsd, atom_coords_aligned_ground_truth, align_weights

def compute_rmsd_glycan(
    feats: Dict[str, torch.Tensor],
    pred_atom_coords: torch.Tensor,
    true_atom_coords: torch.Tensor,
    multiplicity: int,
) -> Tuple[torch.Tensor, int]:
    """
    Computes RMSD for each glycan chain individually and averages the results.

    This function identifies each glycan chain within the batch, isolates the atoms
    belonging to it, performs an independent rigid alignment of the predicted
    coordinates to the ground truth, calculates the RMSD for that glycan, and
    then averages the RMSD values over all glycans found.

    Parameters
    ----------
    feats : Dict[str, torch.Tensor]
        Input features dictionary.
    pred_atom_coords : torch.Tensor
        Predicted atom coordinates, shape [B * multiplicity, N_atoms, 3].
    true_atom_coords : torch.Tensor
        Ground truth atom coordinates, shape [B * multiplicity, N_atoms, 3].
    multiplicity : int
        The number of samples generated per input example.

    Returns
    -------
    Tuple[torch.Tensor, int]
        A tuple containing:
        - The mean RMSD over all glycan chains in the batch.
        - The total number of glycan chains processed.
    """
    device = pred_atom_coords.device
    b_orig, n_atoms = feats["atom_pad_mask"].shape

    # Identify all atoms that are part of any glycan
    is_glycan_atom_mask = feats["atom_mono_idx"] != -1  # Shape [B_orig, N_atoms]

    # Get the chain ID for each atom
    atom_to_token_idx = feats["atom_to_token"].argmax(dim=-1)  # Shape [B_orig, N_atoms]
    asym_id_token = feats["asym_id"]  # Shape [B_orig, N_tokens]

    # Create a batch index that matches the shape of atom_to_token_idx
    batch_idx_atom_orig = (
        torch.arange(b_orig, device=device).unsqueeze(1).expand_as(atom_to_token_idx)
    )
    # Use the correctly shaped batch index for advanced indexing
    atom_asym_id = asym_id_token[
        batch_idx_atom_orig, atom_to_token_idx
    ]  # Shape [B_orig, N_atoms]

    all_glycan_rmsds = []
    num_glycans_processed = 0

    # Iterate over each item in the original batch
    for b_idx in range(b_orig):
        # Find unique glycan chains for this specific batch item
        glycan_atoms_in_batch_item = is_glycan_atom_mask[b_idx]
        if not glycan_atoms_in_batch_item.any():
            continue

        asym_ids_for_glycans = torch.unique(
            atom_asym_id[b_idx][glycan_atoms_in_batch_item]
        )

        # Iterate over each unique glycan chain in the batch item
        for asym_id in asym_ids_for_glycans:
            # Mask for atoms in this specific glycan chain
            glycan_chain_mask = (
                atom_asym_id[b_idx] == asym_id
            ) & is_glycan_atom_mask[b_idx]

            # Also consider only resolved atoms for RMSD calculation
            resolved_mask = feats["atom_resolved_mask"][b_idx]
            final_mask = glycan_chain_mask & resolved_mask

            num_resolved_atoms = final_mask.sum()
            if num_resolved_atoms < 3:
                continue

            # Now iterate through the multiplicity samples for this glycan
            for m in range(multiplicity):
                sample_idx = b_idx * multiplicity + m

                # Isolate coordinates for this specific glycan sample
                pred_coords_glycan = pred_atom_coords[sample_idx][final_mask]
                true_coords_glycan = true_atom_coords[sample_idx][final_mask]

                # Reshape for alignment function (expects [1, N_glycan_atoms, 3])
                pred_coords_glycan = pred_coords_glycan.unsqueeze(0)
                true_coords_glycan = true_coords_glycan.unsqueeze(0)

                # Use unweighted alignment for standard RMSD
                align_weights = torch.ones_like(pred_coords_glycan[..., 0])

                # Since coordinates are already filtered, the mask is all True.
                alignment_mask = torch.ones_like(
                    pred_coords_glycan[..., 0], dtype=torch.bool, device=device
                )

                # Align predicted coords to ground truth for this glycan only
                true_coords_aligned = weighted_rigid_align(
                    true_coords_glycan,
                    pred_coords_glycan,
                    align_weights,
                    mask=alignment_mask,
                )

                # Calculate RMSD for this glycan sample
                mse = ((pred_coords_glycan - true_coords_aligned) ** 2).sum(dim=-1).mean()
                rmsd = torch.sqrt(mse)
                all_glycan_rmsds.append(rmsd)

            num_glycans_processed += 1

    if not all_glycan_rmsds:
        return torch.tensor(0.0, device=device), 0

    # Average the RMSDs over all samples of all glycans
    mean_rmsd = torch.stack(all_glycan_rmsds).mean()

    # The metric expects a single value and a weight. The weight should be the number of unique glycans.
    return mean_rmsd, num_glycans_processed


# =====================================================================================
# ======================== DIHEDRAL VALIDATION FUNCTIONS ==================
# =====================================================================================

def _build_couplet_pair_mask(feats: dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Locate glycosidic couplets (inter-monosaccharide bonds) and return a boolean mask
    [B, L, L] where mask[b, i, j] = True iff token i is a nucleophile (O or N), token j is carbon,
    and they are bonded across different monosaccharides.
    """
    token_bonds    = feats["token_bonds"].squeeze(-1).bool()
    token_to_mono = feats["token_to_mono_idx"]

    ref_elem_oh = feats["ref_element"].float()
    selector    = feats["token_to_rep_atom"].float()
    elem_oh     = torch.einsum("bla,bae->ble", selector, ref_elem_oh)
    atom_num    = elem_oh.argmax(dim=-1)
    is_O        = atom_num == 8
    is_N        = atom_num == 7
    is_C        = atom_num == 6
    
    is_nucleophile = is_O | is_N

    mono_i      = token_to_mono.unsqueeze(2)
    mono_j      = token_to_mono.unsqueeze(1)
    inter_mono  = token_bonds & (mono_i != mono_j) & (mono_i != -1) & (mono_j != -1)

    mask_nuc_to_C = inter_mono & is_nucleophile.unsqueeze(2) & is_C.unsqueeze(1)
    return mask_nuc_to_C

def _calculate_dihedral_torch(p0, p1, p2, p3):
    """
    Calculates dihedral angles for a batch of 4-point sets in PyTorch.
    """
    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1_norm = torch.linalg.norm(b1, dim=-1, keepdim=True)
    b1 = b1 / (b1_norm + 1e-7)
    v = b0 - torch.sum(b0 * b1, dim=-1, keepdim=True) * b1
    w = b2 - torch.sum(b2 * b1, dim=-1, keepdim=True) * b1
    x = torch.sum(v * w, dim=-1)
    y = torch.sum(torch.cross(b1, v, dim=-1) * w, dim=-1)
    return torch.atan2(y, x)

def _decode_one_hot_to_str(one_hot_encoded_name: torch.Tensor) -> str:
    """Decodes a one-hot encoded name from ref_atom_name_chars."""
    integer_indices = torch.argmax(one_hot_encoded_name, dim=-1)
    char_codes = [idx.item() + 32 for idx in integer_indices]
    return "".join([chr(c) for c in char_codes if c > 32]).strip()

def _get_canonical_ring_order_by_name(atom_names: list[str]) -> list[str]:
    """Sorts atom names canonically for a ring (e.g., C1, C2, ..., O5)."""
    def sort_key(name):
        match = re.search(r'\d+', name)
        num = int(match.group(0)) if match else 99
        return (num, name[0])
    return sorted(atom_names, key=sort_key)

def _find_anomeric_dihedral(
    canonical_ring_indices: list[int],
    anomeric_carbon_idx: int,
    adj_list: dict[int, list[int]],
    feats_b: dict[str, torch.Tensor],
    global_to_local_map: dict[int, int],
    all_mono_atoms_set: set[int]
) -> list[tuple[int, int, int, int]]:
    """
    Finds the anomeric dihedral using a topologically-identified anomeric carbon.
    """
    name_to_global_idx = {
        _decode_one_hot_to_str(feats_b['ref_atom_name_chars'][idx]): idx
        for idx in canonical_ring_indices
    }
    ring_oxygen_idx = next((idx for name, idx in name_to_global_idx.items() if name.startswith('O')), None)
    if not ring_oxygen_idx: return []

    anomeric_carbon_local_idx = global_to_local_map.get(anomeric_carbon_idx)
    if anomeric_carbon_local_idx is None: return []

    exocyclic_nucleophile_idx = -1
    for neighbor_local_idx in adj_list.get(anomeric_carbon_local_idx, []):
        neighbor_global_idx = list(global_to_local_map.keys())[neighbor_local_idx]
        is_nucleophile = feats_b['ref_element'][neighbor_global_idx].argmax().item() in [7, 8]
        if is_nucleophile and neighbor_global_idx != ring_oxygen_idx:
            exocyclic_nucleophile_idx = neighbor_global_idx
            break
    if exocyclic_nucleophile_idx == -1: return []

    ring_oxygen_local_idx = global_to_local_map.get(ring_oxygen_idx)
    if ring_oxygen_local_idx is None: return []

    next_carbon_in_ring_idx = -1
    for neighbor_local_idx in adj_list.get(ring_oxygen_local_idx, []):
        neighbor_global_idx = list(global_to_local_map.keys())[neighbor_local_idx]
        is_carbon = feats_b['ref_element'][neighbor_global_idx].argmax().item() == 6
        if is_carbon and neighbor_global_idx != anomeric_carbon_idx and neighbor_global_idx in all_mono_atoms_set:
            next_carbon_in_ring_idx = neighbor_global_idx
            break
    if next_carbon_in_ring_idx == -1: return []

    return [(exocyclic_nucleophile_idx, anomeric_carbon_idx, ring_oxygen_idx, next_carbon_in_ring_idx)]

def _discover_dihedral_indices(
    mono_atom_indices,
    original_mono_atom_set,
    anomeric_carbon_idx,
    mono_coords,
    feats_b,
    bond_distance_cutoff
) -> Tuple[List[Tuple[int, ...]], List[Tuple[int, ...]]]:
    """
    Discovers topology for dihedrals, separating ring substituents from anomeric dihedrals.
    This version includes a planarity filter to exclude aromatic-like rings and a broader
    definition of substituents to match ccd_info.py.
    """
    start_time = time.time()
    TIMEOUT_SECONDS = 2.0

    if len(mono_atom_indices) < 5: return [], []

    global_to_local_map = {global_idx.item(): local_idx for local_idx, global_idx in enumerate(mono_atom_indices)}
    dist_matrix = torch.cdist(mono_coords, mono_coords)
    adj_matrix = (dist_matrix > 0) & (dist_matrix < bond_distance_cutoff)
    adj_list = defaultdict(list)
    rows, cols = torch.where(adj_matrix)
    for i, j in zip(rows, cols):
        adj_list[i.item()].append(j.item())

    original_local_indices = [global_to_local_map[idx] for idx in original_mono_atom_set]
    adj_list_original_only = {k: [v for v in vs if v in original_local_indices] for k, vs in adj_list.items() if k in original_local_indices}
    
    potential_rings = []
    found_rings_canonical = set()

    for start_node in adj_list_original_only.keys():
        q = [(start_node, [start_node])]
        while q:
            if (time.time() - start_time) > TIMEOUT_SECONDS:
                raise TimeoutError(f"Dihedral discovery timed out after {TIMEOUT_SECONDS}s.")
            
            curr, path = q.pop(0)
            if len(path) > 9: continue

            for neighbor in adj_list_original_only.get(curr, []):
                if len(path) > 2 and neighbor == start_node and len(path) in [5, 6, 7, 8, 9]:
                    canonical_form = tuple(sorted(path))
                    if canonical_form not in found_rings_canonical:
                        potential_rings.append(path)
                        found_rings_canonical.add(canonical_form)
                if neighbor not in path:
                    q.append((neighbor, path + [neighbor]))

    # --- NEW: Planarity Filter to exclude aromatic-like rings ---
    non_planar_rings = []
    PLANARITY_RMSD_THRESHOLD = 0.1 # In Angstroms. Exclude rings flatter than this.
    
    for ring_path in potential_rings:
        # It's safer to use tensors for indexing on device
        ring_local_indices = torch.tensor(ring_path, device=mono_coords.device, dtype=torch.long)
        ring_coords = mono_coords[ring_local_indices]
        
        centroid = ring_coords.mean(dim=0)
        centered_coords = ring_coords - centroid
        
        try:
            # S contains singular values. The smallest one relates to deviation from the best-fit plane.
            _, S, _ = torch.linalg.svd(centered_coords)
            # RMSD from plane is S_min / sqrt(N)
            rmsd_from_plane = S[-1] / torch.sqrt(torch.tensor(len(ring_path), device=S.device, dtype=S.dtype))

            # We want NON-planar rings, so we keep rings where the deviation is ABOVE the threshold.
            if rmsd_from_plane > PLANARITY_RMSD_THRESHOLD:
                non_planar_rings.append(ring_path)
        except torch.linalg.LinAlgError:
            # SVD can fail for collinear points, which are perfectly planar. Skip them.
            continue
            
    if not non_planar_rings: return [], []
    
    def ring_sort_key(ring_path):
        global_indices = [mono_atom_indices[i].item() for i in ring_path]
        elements = feats_b['ref_element'][global_indices].argmax(dim=-1)
        has_oxygen = (elements == 8).any().item()
        return (has_oxygen, len(ring_path))

    non_planar_rings.sort(key=ring_sort_key, reverse=True)
    ring_atom_local_indices = non_planar_rings[0]

    ring_atom_global_indices = [mono_atom_indices[i].item() for i in ring_atom_local_indices]
    ring_atom_names = [_decode_one_hot_to_str(feats_b['ref_atom_name_chars'][i]) for i in ring_atom_global_indices]
    
    try: canonical_ring_names = _get_canonical_ring_order_by_name(ring_atom_names)
    except (ValueError, IndexError): return [], []

    name_to_global_idx = {name: idx for name, idx in zip(ring_atom_names, ring_atom_global_indices)}
    canonical_ring_indices = [name_to_global_idx[name] for name in canonical_ring_names]
    num_ring_atoms, ring_atom_set = len(canonical_ring_indices), set(canonical_ring_indices)
    ring_substituent_dihedrals = []

    # --- CORRECTED: Expanded logic for finding all substituents ---
    for i, ring_atom_idx in enumerate(canonical_ring_indices):
        # We only define these dihedrals for ring carbons.
        if feats_b['ref_element'][ring_atom_idx].argmax().item() != 6: 
            continue
        
        local_ring_atom_idx = global_to_local_map[ring_atom_idx]

        # Iterate through all bonded neighbors of the current ring carbon.
        for neighbor_local_idx in adj_list.get(local_ring_atom_idx, []):
            neighbor_global_idx = mono_atom_indices[neighbor_local_idx].item()

            # A substituent is any bonded atom that is NOT part of the ring itself.
            if neighbor_global_idx not in ring_atom_set:
                p0 = neighbor_global_idx  # The substituent atom
                p1 = ring_atom_idx        # The ring carbon it's attached to
                p2 = canonical_ring_indices[(i - 1 + num_ring_atoms) % num_ring_atoms]
                p3 = canonical_ring_indices[(i - 2 + num_ring_atoms) % num_ring_atoms]
                ring_substituent_dihedrals.append((p0, p1, p2, p3))

    anomeric_dihedrals = []
    if anomeric_carbon_idx is not None:
        anomeric_dihedrals = _find_anomeric_dihedral(
            canonical_ring_indices, anomeric_carbon_idx, adj_list, feats_b, global_to_local_map, original_mono_atom_set
        )
    
    return list(set(ring_substituent_dihedrals)), list(set(anomeric_dihedrals))

def compute_glycan_dihedral_validation(
    feats: dict[str, torch.Tensor],
    pred_atom_coords: torch.Tensor,
    true_atom_coords: torch.Tensor,
    multiplicity: int,
    bond_distance_cutoff: float = 2.0,
) -> Dict[str, Tuple[torch.Tensor, int]]:
    """
    Computes separate validation metrics for anomeric and ring substituent glycan dihedrals.
    Returns a dictionary containing the mean loss and count for each dihedral type.
    """
    sample_id = "ID_NOT_FOUND"
    id_val = feats.get('record_id', feats.get('id'))
    if id_val is not None:
        if isinstance(id_val, (list, tuple)): id_val = id_val[0]
        if hasattr(id_val, 'item'): id_val = id_val.item()
        if isinstance(id_val, bytes): sample_id = id_val.decode('utf-8', errors='ignore')
        else: sample_id = str(id_val)

    device = pred_atom_coords.device
    b_orig, _ = feats["token_pad_mask"].shape
    b_mult = b_orig * multiplicity

    glycosidic_couplet_mask = _build_couplet_pair_mask(feats)
    (bond_b_idx, bond_nuc_token_idx, bond_c_token_idx) = torch.where(
        glycosidic_couplet_mask
    )

    child_id_to_linkage_atoms = {}
    token_to_rep_atom_idx_map = feats["token_to_rep_atom"].argmax(-1)

    if bond_b_idx.numel() > 0:
        parent_nuc_atom_idx = token_to_rep_atom_idx_map[bond_b_idx, bond_nuc_token_idx]
        child_c_atom_idx = token_to_rep_atom_idx_map[bond_b_idx, bond_c_token_idx]

        child_asym_id = feats["asym_id"][bond_b_idx, bond_c_token_idx]
        child_mono_id = feats["atom_mono_idx"][bond_b_idx, child_c_atom_idx]

        for i in range(len(bond_b_idx)):
            b, asym, mono = (
                bond_b_idx[i].item(),
                child_asym_id[i].item(),
                child_mono_id[i].item(),
            )
            parent_nuc = parent_nuc_atom_idx[i].item()
            child_c = child_c_atom_idx[i].item()
            child_id_to_linkage_atoms[(b, asym, mono)] = (parent_nuc, child_c)

    atom_mono_idx_eff = feats["atom_mono_idx"].repeat_interleave(multiplicity, 0)
    atom_pad_mask_eff = feats["atom_pad_mask"].repeat_interleave(multiplicity, 0).bool()
    valid_mono_atom_mask = (atom_mono_idx_eff != -1) & atom_pad_mask_eff
    if not valid_mono_atom_mask.any():
        return {
            "anomeric": (torch.tensor(0.0, device=device), 0),
            "ring": (torch.tensor(0.0, device=device), 0),
        }

    atom_to_token_idx = feats["atom_to_token"].argmax(dim=-1).repeat_interleave(
        multiplicity, 0
    )
    asym_id_token = feats["asym_id"].repeat_interleave(multiplicity, 0)
    batch_idx_atom = (
        torch.arange(b_mult, device=device).unsqueeze(1).expand_as(atom_to_token_idx)
    )
    atom_asym_id_eff = asym_id_token[batch_idx_atom, atom_to_token_idx]

    filtered_identifiers = torch.stack(
        [batch_idx_atom, atom_asym_id_eff, atom_mono_idx_eff], dim=-1
    )[valid_mono_atom_mask]
    unique_instances, instance_map = torch.unique(
        filtered_identifiers, dim=0, return_inverse=True
    )
    _, valid_atom_coords_n = torch.where(valid_mono_atom_mask)

    all_ring_dihedral_indices, all_ring_batch_indices = [], []
    all_anomeric_dihedral_indices, all_anomeric_batch_indices = [], []
    ref_element_eff = feats["ref_element"].repeat_interleave(multiplicity, 0)
    ref_atom_name_chars_eff = feats["ref_atom_name_chars"].repeat_interleave(
        multiplicity, 0
    )

    for i in range(len(unique_instances)):
        b_mult_idx, asym_val, mono_val = unique_instances[i]
        b_orig_idx = b_mult_idx % b_orig

        instance_atom_indices = valid_atom_coords_n[instance_map == i]

        augmented_indices = instance_atom_indices
        child_key = (b_orig_idx.item(), asym_val.item(), mono_val.item())
        linkage_info = child_id_to_linkage_atoms.get(child_key)
        anomeric_carbon_idx = None
        if linkage_info is not None:
            parent_nuc_idx, anomeric_carbon_idx = linkage_info
            augmented_indices = torch.cat(
                [instance_atom_indices, torch.tensor([parent_nuc_idx], device=device)]
            )

        feats_b_cpu = {
            "ref_element": ref_element_eff[b_mult_idx].cpu(),
            "ref_atom_name_chars": ref_atom_name_chars_eff[b_mult_idx].cpu(),
        }
        
        try:
            ring_dihedrals, anomeric_dihedrals = _discover_dihedral_indices(
                augmented_indices.cpu(),
                set(instance_atom_indices.cpu().tolist()),
                anomeric_carbon_idx,
                true_atom_coords[b_mult_idx, augmented_indices].cpu(),
                feats_b_cpu,
                bond_distance_cutoff,
            )
        except TimeoutError:
            print(f"Timeout in dihedral validation for sample: {sample_id}")
            continue

        if ring_dihedrals:
            global_dihedrals = torch.tensor(ring_dihedrals, dtype=torch.long, device=device)
            all_ring_dihedral_indices.append(global_dihedrals)
            all_ring_batch_indices.append(torch.full((global_dihedrals.shape[0],), b_mult_idx.item(), device=device, dtype=torch.long))

        if anomeric_dihedrals:
            global_dihedrals = torch.tensor(anomeric_dihedrals, dtype=torch.long, device=device)
            all_anomeric_dihedral_indices.append(global_dihedrals)
            all_anomeric_batch_indices.append(torch.full((global_dihedrals.shape[0],), b_mult_idx.item(), device=device, dtype=torch.long))

    # --- Ring Dihedral Validation ---
    if not all_ring_batch_indices:
        mean_ring_loss, total_ring_dihedrals = torch.tensor(0.0, device=device), 0
    else:
        final_batch_indices = torch.cat(all_ring_batch_indices)
        final_dihedral_indices = torch.cat(all_ring_dihedral_indices)
        p0_idx, p1_idx, p2_idx, p3_idx = final_dihedral_indices.T
        p0_pred, p0_true = pred_atom_coords[final_batch_indices, p0_idx], true_atom_coords[final_batch_indices, p0_idx]
        p1_pred, p1_true = pred_atom_coords[final_batch_indices, p1_idx], true_atom_coords[final_batch_indices, p1_idx]
        p2_pred, p2_true = pred_atom_coords[final_batch_indices, p2_idx], true_atom_coords[final_batch_indices, p2_idx]
        p3_pred, p3_true = pred_atom_coords[final_batch_indices, p3_idx], true_atom_coords[final_batch_indices, p3_idx]
        pred_rad = _calculate_dihedral_torch(p0_pred, p1_pred, p2_pred, p3_pred)
        true_rad = _calculate_dihedral_torch(p0_true, p1_true, p2_true, p3_true)
        loss_per_dihedral = 1.0 - torch.cos(pred_rad - true_rad)
        total_ring_dihedrals = len(loss_per_dihedral)
        mean_ring_loss = loss_per_dihedral.mean() if total_ring_dihedrals > 0 else torch.tensor(0.0, device=device)

    # --- Anomeric Dihedral Validation ---
    if not all_anomeric_batch_indices:
        mean_anomeric_loss, total_anomeric_dihedrals = torch.tensor(0.0, device=device), 0
    else:
        final_batch_indices = torch.cat(all_anomeric_batch_indices)
        final_dihedral_indices = torch.cat(all_anomeric_dihedral_indices)
        p0_idx, p1_idx, p2_idx, p3_idx = final_dihedral_indices.T
        p0_pred, p0_true = pred_atom_coords[final_batch_indices, p0_idx], true_atom_coords[final_batch_indices, p0_idx]
        p1_pred, p1_true = pred_atom_coords[final_batch_indices, p1_idx], true_atom_coords[final_batch_indices, p1_idx]
        p2_pred, p2_true = pred_atom_coords[final_batch_indices, p2_idx], true_atom_coords[final_batch_indices, p2_idx]
        p3_pred, p3_true = pred_atom_coords[final_batch_indices, p3_idx], true_atom_coords[final_batch_indices, p3_idx]
        pred_rad = _calculate_dihedral_torch(p0_pred, p1_pred, p2_pred, p3_pred)
        true_rad = _calculate_dihedral_torch(p0_true, p1_true, p2_true, p3_true)
        loss_per_dihedral = 1.0 - torch.cos(pred_rad - true_rad)
        total_anomeric_dihedrals = len(loss_per_dihedral)
        mean_anomeric_loss = loss_per_dihedral.mean() if total_anomeric_dihedrals > 0 else torch.tensor(0.0, device=device)
    
    return {
        "anomeric": (mean_anomeric_loss, total_anomeric_dihedrals),
        "ring": (mean_ring_loss, total_ring_dihedrals),
    }
