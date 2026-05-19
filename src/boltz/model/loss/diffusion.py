# started from code from https://github.com/lucidrains/alphafold3-pytorch, MIT License, Copyright (c) 2024 Phil Wang

from einops import einsum
import torch
import torch.nn.functional as F
import sys
from typing import Dict, List, Any, Tuple, Optional
import time
import re
from collections import defaultdict

def weighted_rigid_align(
    true_coords,
    pred_coords,
    weights,
    mask,
):
    """Compute weighted alignment.

    Parameters
    ----------
    true_coords: torch.Tensor
        The ground truth atom coordinates
    pred_coords: torch.Tensor
        The predicted atom coordinates
    weights: torch.Tensor
        The weights for alignment
    mask: torch.Tensor
        The atoms mask

    Returns
    -------
    torch.Tensor
        Aligned coordinates

    """

    batch_size, num_points, dim = true_coords.shape
    weights = (mask * weights).unsqueeze(-1)

    # Compute weighted centroids
    true_centroid = (true_coords * weights).sum(dim=1, keepdim=True) / weights.sum(
        dim=1, keepdim=True
    )
    pred_centroid = (pred_coords * weights).sum(dim=1, keepdim=True) / weights.sum(
        dim=1, keepdim=True
    )

    # Center the coordinates
    true_coords_centered = true_coords - true_centroid
    pred_coords_centered = pred_coords - pred_centroid

    if num_points < (dim + 1):
        print(
            "Warning: The size of one of the point clouds is <= dim+1. "
            + "`WeightedRigidAlign` cannot return a unique rotation."
        )

    # Compute the weighted covariance matrix
    cov_matrix = einsum(
        weights * pred_coords_centered, true_coords_centered, "b n i, b n j -> b i j"
    )

    # Compute the SVD of the covariance matrix, required float32 for svd and determinant
    original_dtype = cov_matrix.dtype
    cov_matrix_32 = cov_matrix.to(dtype=torch.float32)
    U, S, V = torch.linalg.svd(
        cov_matrix_32, driver="gesvd" if cov_matrix_32.is_cuda else None
    )
    V = V.mH

    # Catch ambiguous rotation by checking the magnitude of singular values
    if (S.abs() <= 1e-15).any() and not (num_points < (dim + 1)):
        print(
            "Warning: Excessively low rank of "
            + "cross-correlation between aligned point clouds. "
            + "`WeightedRigidAlign` cannot return a unique rotation."
        )

    # Compute the rotation matrix
    rot_matrix = torch.einsum("b i j, b k j -> b i k", U, V).to(dtype=torch.float32)

    # Ensure proper rotation matrix with determinant 1
    F = torch.eye(dim, dtype=cov_matrix_32.dtype, device=cov_matrix.device)[
        None
    ].repeat(batch_size, 1, 1)
    F[:, -1, -1] = torch.det(rot_matrix)
    rot_matrix = einsum(U, F, V, "b i j, b j k, b l k -> b i l")
    rot_matrix = rot_matrix.to(dtype=original_dtype)

    # Apply the rotation and translation
    aligned_coords = (
        einsum(true_coords_centered, rot_matrix, "b n i, b j i -> b n j")
        + pred_centroid
    )
    aligned_coords.detach_()

    return aligned_coords


def smooth_lddt_loss(
    pred_coords,
    true_coords,
    is_nucleotide,
    coords_mask,
    nucleic_acid_cutoff: float = 30.0,
    other_cutoff: float = 15.0,
    multiplicity: int = 1,
):
    """Compute weighted alignment.

    Parameters
    ----------
    pred_coords: torch.Tensor
        The predicted atom coordinates
    true_coords: torch.Tensor
        The ground truth atom coordinates
    is_nucleotide: torch.Tensor
        The weights for alignment
    coords_mask: torch.Tensor
        The atoms mask
    nucleic_acid_cutoff: float
        The nucleic acid cutoff
    other_cutoff: float
        The non nucleic acid cutoff
    multiplicity: int
        The multiplicity
    Returns
    -------
    torch.Tensor
        Aligned coordinates

    """
    B, N, _ = true_coords.shape
    true_dists = torch.cdist(true_coords, true_coords)
    is_nucleotide = is_nucleotide.repeat_interleave(multiplicity, 0)

    coords_mask = coords_mask.repeat_interleave(multiplicity, 0)
    is_nucleotide_pair = is_nucleotide.unsqueeze(-1).expand(
        -1, -1, is_nucleotide.shape[-1]
    )

    mask = (
        is_nucleotide_pair * (true_dists < nucleic_acid_cutoff).float()
        + (1 - is_nucleotide_pair) * (true_dists < other_cutoff).float()
    )
    mask = mask * (1 - torch.eye(pred_coords.shape[1], device=pred_coords.device))
    mask = mask * (coords_mask.unsqueeze(-1) * coords_mask.unsqueeze(-2))

    # Compute distances between all pairs of atoms
    pred_dists = torch.cdist(pred_coords, pred_coords)
    dist_diff = torch.abs(true_dists - pred_dists)

    # Compute epsilon values
    eps = (
        (
            (
                F.sigmoid(0.5 - dist_diff)
                + F.sigmoid(1.0 - dist_diff)
                + F.sigmoid(2.0 - dist_diff)
                + F.sigmoid(4.0 - dist_diff)
            )
            / 4.0
        )
        .view(multiplicity, B // multiplicity, N, N)
        .mean(dim=0)
    )

    # Calculate masked averaging
    eps = eps.repeat_interleave(multiplicity, 0)
    num = (eps * mask).sum(dim=(-1, -2))
    den = mask.sum(dim=(-1, -2)).clamp(min=1)
    lddt = num / den

    return 1.0 - lddt.mean()

def Linkage_Loss(
    feats: Dict[str, torch.Tensor],
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor, # This should be the UN-ALIGNED true coordinates
    loss_weights: torch.Tensor,
    multiplicity: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """
    Calculates a distance-based loss for glycosylation bonds.
    Returns the final scalar loss tensor if any glycosylation sites are present
    in the batch, otherwise returns None.
    """
    from boltz.model.modules.sugar_trunk import _get_glycosylation_features
    glyco_feats = _get_glycosylation_features(feats)

    # 1. Get the TOKEN bond pairs and their batch indices from the helper function.
    token_bond_pairs = glyco_feats.get("t_glycosylation_indices")
    batch_indices = glyco_feats.get("t_batch_idx")

    # If no glycosylation sites are found in the entire batch, return None.
    # The caller (compute_loss) is responsible for handling this.
    if token_bond_pairs is None or token_bond_pairs.numel() == 0:
        return None

    # --- Section A: Convert Token Indices to Atom Indices ---
    B_orig = feats['atom_pad_mask'].shape[0]
    token_to_rep_atom_idx = feats["token_to_rep_atom"].argmax(-1)
    p_token_indices = token_bond_pairs[:, 0]
    g_token_indices = token_bond_pairs[:, 1]
    p_atom_indices = token_to_rep_atom_idx[batch_indices, p_token_indices]
    g_atom_indices = token_to_rep_atom_idx[batch_indices, g_token_indices]

    # --- Section B: Index Preparation for Multiplicity ---
    num_bonds_in_batch = token_bond_pairs.shape[0]
    offset = torch.arange(0, multiplicity, device=device) * B_orig
    offset_expanded = offset.repeat_interleave(num_bonds_in_batch)
    final_batch_indices = batch_indices.repeat(multiplicity) + offset_expanded
    final_protein_atom_indices = p_atom_indices.repeat(multiplicity)
    final_glycan_atom_indices = g_atom_indices.repeat(multiplicity)

    # --- Section C: Dense Tensor Gathering ---
    p_true_coords = true_coords[final_batch_indices, final_protein_atom_indices]
    g_true_coords = true_coords[final_batch_indices, final_glycan_atom_indices]
    p_pred_coords = pred_coords[final_batch_indices, final_protein_atom_indices]
    g_pred_coords = pred_coords[final_batch_indices, final_glycan_atom_indices]

    # --- Section D: Loss Calculation on Dense Tensors ---
    true_dist = torch.linalg.norm(p_true_coords - g_true_coords, dim=-1)
    pred_dist = torch.linalg.norm(p_pred_coords - g_pred_coords, dim=-1)
    per_bond_loss = (pred_dist - true_dist) ** 2

    # --- Section E: Aggregation ---
    B_mult = pred_coords.shape[0]
    dist_loss_per_item = torch.zeros(B_mult, device=device).scatter_add_(0, final_batch_indices, per_bond_loss)
    ones = torch.ones_like(per_bond_loss)
    counts = torch.zeros(B_mult, device=device).scatter_add_(0, final_batch_indices, ones)
    avg_dist_loss_per_item = dist_loss_per_item / counts.clamp(min=1)

    # --- Section F: Final Sigma-Weighted Mean ---
    valid_items_mask = counts > 0
    # This check is now redundant due to the check at the top, but kept for safety.
    if not valid_items_mask.any():
        return None

    final_loss = (avg_dist_loss_per_item[valid_items_mask] * loss_weights[valid_items_mask]).mean()

    # Handle potential NaN from division by zero if all weights are zero for valid items
    return final_loss if not torch.isnan(final_loss) else torch.tensor(0.0, device=device)

def Glyco_AA_MSE_Loss(
    feats: Dict[str, torch.Tensor],
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    loss_weights: torch.Tensor,
    multiplicity: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """
    Calculates the Mean Squared Error (MSE) of the glycosylated amino acid and the 
    attached glycan atom after performing an independent rigid alignment.
    
    This enforces the correct internal geometry of the entire amino acid and the 
    linkage bond simultaneously, preventing backbone warping.
    """
    # Robustly determine original batch size (feats tensors are [B, N, ...])
    B_orig = feats['atom_to_token'].shape[0]
    
    raw_sites_tensors = feats.get('raw_glycosylation_sites')
    # Pre-fetch resolved mask: [B, N]
    atom_resolved_mask_orig = feats["atom_resolved_mask"]

    if raw_sites_tensors is None:
        return None

    total_mse_loss = torch.tensor(0.0, device=device)
    site_count = torch.tensor(0.0, device=device)

    # Iterate through each structure in the batch 'b'
    for b in range(B_orig):
        sites_tensor_b = raw_sites_tensors[b]
        if sites_tensor_b is None or sites_tensor_b.numel() == 0:
            continue

        # Get atom mapping data for this batch item
        # atom_to_token: [N, L] -> argmax -> [N]
        atom_to_token = feats["atom_to_token"][b].argmax(-1)
        token_asym_ids = feats["asym_id"][b]
        token_res_indices = feats["residue_index"][b]
        ref_name_chars = feats["ref_atom_name_chars"][b]

        # Map atoms to Chain/Residue
        atom_asym_ids = torch.gather(token_asym_ids, 0, atom_to_token)
        atom_res_indices = torch.gather(token_res_indices, 0, atom_to_token)

        for site_data_tensor in sites_tensor_b:
            p_chain_id = site_data_tensor[0].item()
            p_res_id = site_data_tensor[1].item()
            
            g_chain_id = site_data_tensor[6].item()
            g_res_id = site_data_tensor[7].item()
            g_name = _decode_int_to_str(site_data_tensor[8:12]).upper()

            # 1. Get all atoms of the Glycosylated Amino Acid
            p_res_mask = (atom_asym_ids == p_chain_id) & (atom_res_indices == p_res_id)
            p_atoms_indices = torch.where(p_res_mask)[0]

            if p_atoms_indices.numel() == 0:
                continue

            # 2. Get the singular covalently bonded Glycan Atom (e.g., C1)
            g_res_mask = (atom_asym_ids == g_chain_id) & (atom_res_indices == g_res_id)
            g_atoms_in_res = torch.where(g_res_mask)[0]
            
            g_atom_index = -1
            for g_idx in g_atoms_in_res:
                decoded_name = _decode_one_hot_to_str(ref_name_chars[g_idx]).upper()
                if decoded_name == g_name:
                    g_atom_index = g_idx.item()
                    break
            
            if g_atom_index == -1:
                continue

            # 3. Combine Indices: [All AA Atoms] + [Linkage Glycan Atom]
            g_idx_tensor = torch.tensor([g_atom_index], device=device, dtype=torch.long)
            subset_indices = torch.cat([p_atoms_indices, g_idx_tensor])

            # 4. Handle Multiplicity and Extraction
            # The coordinate tensors have shape [B*M, N, 3]
            # The rows corresponding to batch 'b' are [b*M, ..., (b+1)*M - 1]
            batch_start = b * multiplicity
            batch_end = (b + 1) * multiplicity

            # Extract Coords: Shape [M, N_subset, 3]
            # We slice the batch dim [batch_start:batch_end] and select specific atoms [subset_indices]
            curr_pred = pred_coords[batch_start:batch_end][:, subset_indices, :]
            curr_true = true_coords[batch_start:batch_end][:, subset_indices, :]
            
            # Extract Masks: Shape [N_subset] -> Expand to [M, N_subset]
            curr_mask = atom_resolved_mask_orig[b, subset_indices]
            curr_mask = curr_mask.unsqueeze(0).expand(multiplicity, -1)
            
            # Extract Sigma Weights: Shape [M]
            curr_weights = loss_weights[batch_start:batch_end]
            
            # 5. Independent Rigid Alignment
            # Align 'True' onto 'Pred' using the validity mask as weights
            # Shape: [M, N_subset, 3]
            aligned_true = weighted_rigid_align(
                curr_true, 
                curr_pred, 
                curr_mask, 
                curr_mask
            )
            
            # 6. Calculate MSE on Aligned Subsets
            diff = aligned_true - curr_pred
            mse_per_atom = (diff ** 2).sum(dim=-1) # [M, N_subset]
            
            # Weighted average over atoms (ignoring unresolved atoms)
            sum_sq_error = (mse_per_atom * curr_mask).sum(dim=-1) # [M]
            num_valid_atoms = curr_mask.sum(dim=-1).clamp(min=1e-6) # [M]
            
            masked_mse_per_sample = sum_sq_error / num_valid_atoms
            
            # 7. Apply Sigma Weighting and Accumulate
            # Average over the M samples for this site
            weighted_site_loss = (masked_mse_per_sample * curr_weights).mean()
            
            total_mse_loss += weighted_site_loss
            site_count += 1

    if site_count == 0:
        return None

    return total_mse_loss / site_count

def _decode_int_to_str(encoded_name: torch.Tensor) -> str:
    """Decodes a tensor of 4 integers back into a string atom name."""
    # Add 32 to convert back to ASCII character codes
    char_codes = [c.item() + 32 for c in encoded_name]
    # Convert codes to characters and join, stripping trailing whitespace
    return "".join([chr(c) for c in char_codes]).strip()

def _decode_one_hot_to_str(one_hot_encoded_name: torch.Tensor) -> str:
    """Decodes a one-hot encoded name from ref_atom_name_chars."""
    # Find the integer index for each of the 4 character positions
    integer_indices = torch.argmax(one_hot_encoded_name, dim=-1)
    # Add 32 to convert back to ASCII character codes
    char_codes = [idx.item() + 32 for idx in integer_indices]
    # Filter out null characters (code 32) and join
    return "".join([chr(c) for c in char_codes if c > 32]).strip()

def _build_couplet_pair_mask(feats: dict[str, torch.Tensor]) -> torch.Tensor:
    token_bonds    = feats["token_bonds"].squeeze(-1).bool()
    token_to_mono  = feats["token_to_mono_idx"]

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
    """Calculates dihedral angle with epsilons to prevent NaN gradients."""
    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    
    b1_norm = torch.linalg.norm(b1, dim=-1, keepdim=True)
    b1 = b1 / (b1_norm + 1e-7)
    
    v = b0 - torch.sum(b0 * b1, dim=-1, keepdim=True) * b1
    w = b2 - torch.sum(b2 * b1, dim=-1, keepdim=True) * b1
    
    x = torch.sum(v * w, dim=-1)
    y = torch.sum(torch.cross(b1, v, dim=-1) * w, dim=-1)
    
    # Add a tiny epsilon to x and y to prevent torch.atan2(0, 0) returning NaN gradients 
    # if noise collapses the points into perfect collinearity.
    return torch.atan2(y + 1e-8, x + 1e-8)

def _get_canonical_ring_order_by_name(atom_names: list[str]) -> list[str]:
    def sort_key(name):
        match = re.search(r'\d+', name)
        num = int(match.group(0)) if match else 99
        return (num, name[0])
    return sorted(atom_names, key=sort_key)

def _discover_dihedral_indices(
    mono_atom_indices,
    original_mono_atom_set,
    mono_coords,
    feats_b,
    bond_distance_cutoff
) -> List[Tuple[int, int, int, int]]:
    start_time = time.time()
    TIMEOUT_SECONDS = 2.0

    if len(mono_atom_indices) < 5: return []

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

    non_planar_rings = []
    PLANARITY_RMSD_THRESHOLD = 0.1 
    
    for ring_path in potential_rings:
        ring_local_indices = torch.tensor(ring_path, device=mono_coords.device, dtype=torch.long)
        ring_coords = mono_coords[ring_local_indices]
        
        centroid = ring_coords.mean(dim=0)
        centered_coords = ring_coords - centroid
        
        try:
            _, S, _ = torch.linalg.svd(centered_coords)
            rmsd_from_plane = S[-1] / torch.sqrt(torch.tensor(len(ring_path), device=S.device, dtype=S.dtype))
            if rmsd_from_plane > PLANARITY_RMSD_THRESHOLD:
                non_planar_rings.append(ring_path)
        except torch.linalg.LinAlgError:
            continue
            
    if not non_planar_rings: return []
    
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
    except (ValueError, IndexError): return []

    name_to_global_idx = {name: idx for name, idx in zip(ring_atom_names, ring_atom_global_indices)}
    canonical_ring_indices = [name_to_global_idx[name] for name in canonical_ring_names]
    num_ring_atoms = len(canonical_ring_indices)
    ring_atom_set = set(canonical_ring_indices)
    
    substituent_dihedrals = []

    for i, ring_atom_idx in enumerate(canonical_ring_indices):
        if feats_b['ref_element'][ring_atom_idx].argmax().item() != 6: 
            continue
        
        local_ring_atom_idx = global_to_local_map[ring_atom_idx]
        for neighbor_local_idx in adj_list.get(local_ring_atom_idx, []):
            neighbor_global_idx = mono_atom_indices[neighbor_local_idx].item()
            
            # If the neighbor is a substituent (OH, sidechain, linkage)
            if neighbor_global_idx not in ring_atom_set:
                p0 = neighbor_global_idx 
                p1 = ring_atom_idx       
                
                # Direction 1: Backwards around the ring
                p2_rev = canonical_ring_indices[(i - 1 + num_ring_atoms) % num_ring_atoms]
                p3_rev = canonical_ring_indices[(i - 2 + num_ring_atoms) % num_ring_atoms]
                substituent_dihedrals.append((p0, p1, p2_rev, p3_rev))

                # Direction 2: Forwards around the ring
                p2_fwd = canonical_ring_indices[(i + 1) % num_ring_atoms]
                p3_fwd = canonical_ring_indices[(i + 2) % num_ring_atoms]
                substituent_dihedrals.append((p0, p1, p2_fwd, p3_fwd))
    
    return list(set(substituent_dihedrals))
        
def Glycan_Dihedral_Loss(
    feats: Dict[str, torch.Tensor],
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    loss_weights: torch.Tensor,
    multiplicity: int,
    bond_distance_cutoff: float = 2.0,
) -> Optional[torch.Tensor]:
    """
    Computes unified diffusion training losses for all glycan ring substituents
    using a smooth, bounded cosine penalty.
    """
    device = pred_coords.device
    b_orig, _ = feats["token_pad_mask"].shape
    b_mult = b_orig * multiplicity

    glycosidic_couplet_mask = _build_couplet_pair_mask(feats)
    (bond_b_idx, bond_nuc_token_idx, bond_c_token_idx) = torch.where(glycosidic_couplet_mask)

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
        return None

    atom_to_token_idx = feats["atom_to_token"].argmax(dim=-1).repeat_interleave(multiplicity, 0)
    asym_id_token = feats["asym_id"].repeat_interleave(multiplicity, 0)
    batch_idx_atom = torch.arange(b_mult, device=device).unsqueeze(1).expand_as(atom_to_token_idx)
    atom_asym_id_eff = asym_id_token[batch_idx_atom, atom_to_token_idx]

    filtered_identifiers = torch.stack(
        [batch_idx_atom, atom_asym_id_eff, atom_mono_idx_eff], dim=-1
    )[valid_mono_atom_mask]
    unique_instances, instance_map = torch.unique(filtered_identifiers, dim=0, return_inverse=True)
    _, valid_atom_coords_n = torch.where(valid_mono_atom_mask)

    all_dihedral_indices, all_batch_indices = [], []
    ref_element_eff = feats["ref_element"].repeat_interleave(multiplicity, 0)
    ref_atom_name_chars_eff = feats["ref_atom_name_chars"].repeat_interleave(multiplicity, 0)

    for i in range(len(unique_instances)):
        b_mult_idx, asym_val, mono_val = unique_instances[i]
        b_orig_idx = b_mult_idx % b_orig

        instance_atom_indices = valid_atom_coords_n[instance_map == i]
        augmented_indices = instance_atom_indices
        child_key = (b_orig_idx.item(), asym_val.item(), mono_val.item())
        linkage_info = child_id_to_linkage_atoms.get(child_key)
        
        if linkage_info is not None:
            parent_nuc_idx, _ = linkage_info
            augmented_indices = torch.cat(
                [instance_atom_indices, torch.tensor([parent_nuc_idx], device=device)]
            )

        feats_b_cpu = {
            "ref_element": ref_element_eff[b_mult_idx].cpu(),
            "ref_atom_name_chars": ref_atom_name_chars_eff[b_mult_idx].cpu(),
        }
        
        try:
            dihedrals = _discover_dihedral_indices(
                augmented_indices.cpu(),
                set(instance_atom_indices.cpu().tolist()),
                true_coords[b_mult_idx, augmented_indices].cpu(),
                feats_b_cpu,
                bond_distance_cutoff,
            )
        except TimeoutError:
            continue

        if dihedrals:
            global_dihedrals = torch.tensor(dihedrals, dtype=torch.long, device=device)
            all_dihedral_indices.append(global_dihedrals)
            all_batch_indices.append(torch.full((global_dihedrals.shape[0],), b_mult_idx.item(), device=device, dtype=torch.long))

    mean_dihedral_loss = None
    if all_batch_indices:
        final_batch_indices = torch.cat(all_batch_indices)
        final_dihedral_indices = torch.cat(all_dihedral_indices)
        p0_idx, p1_idx, p2_idx, p3_idx = final_dihedral_indices.T
        
        p0_pred, p0_true = pred_coords[final_batch_indices, p0_idx], true_coords[final_batch_indices, p0_idx]
        p1_pred, p1_true = pred_coords[final_batch_indices, p1_idx], true_coords[final_batch_indices, p1_idx]
        p2_pred, p2_true = pred_coords[final_batch_indices, p2_idx], true_coords[final_batch_indices, p2_idx]
        p3_pred, p3_true = pred_coords[final_batch_indices, p3_idx], true_coords[final_batch_indices, p3_idx]
        
        # --- Safety Mask: Ignore collapsed bonds at high noise ---
        b1_pred = p2_pred - p1_pred
        b1_norm = torch.linalg.norm(b1_pred, dim=-1)
        valid_geometry_mask = b1_norm > 0.5 

        if valid_geometry_mask.any():
            pred_rad = _calculate_dihedral_torch(
                p0_pred[valid_geometry_mask], p1_pred[valid_geometry_mask], 
                p2_pred[valid_geometry_mask], p3_pred[valid_geometry_mask]
            )
            true_rad = _calculate_dihedral_torch(
                p0_true[valid_geometry_mask], p1_true[valid_geometry_mask], 
                p2_true[valid_geometry_mask], p3_true[valid_geometry_mask]
            )
            
            # --- Smooth Cosine Loss ---
            # Automatically handles periodicity, no wrapping/atan2 needed here.
            # Max penalty is 2.0 (when completely flipped). Minimum is 0.0.
            loss_per_dihedral = 1.0 - torch.cos(pred_rad - true_rad)
            
            batch_weights = loss_weights[final_batch_indices[valid_geometry_mask]]
            weighted_loss = loss_per_dihedral * batch_weights
            
            if len(weighted_loss) > 0:
                mean_dihedral_loss = weighted_loss.mean()
            
    return mean_dihedral_loss

    
