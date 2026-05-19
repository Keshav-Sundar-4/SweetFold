from __future__ import annotations
import torch
from torch import Tensor
import torch.nn as nn
from typing import Dict, Any, Tuple
import sys
import pkg_resources
import json

from boltz.data import const
import torch.nn.functional as F

from boltz.model.layers.triangular_mult import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from boltz.model.layers.triangular_attention.attention import (
    TriangleAttentionEndingNode,
    TriangleAttentionStartingNode,
)
from boltz.model.layers.attention import AttentionPairBias
from boltz.model.layers.transition import Transition
from boltz.model.layers.dropout import get_dropout_mask
from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper
from boltz.data.feature.featurizer import MONO_TYPE_MAP

#############################################################################################################
#############################################################################################################
#CONSTANTS
#############################################################################################################
#############################################################################################################

NUM_MONO_TYPES_PLACEHOLDER = 931 # As derived from the provided map example (+1 for OTHER)
NUM_ANOMERIC_TYPES = 3
#NUM_MONO_TYPES = len(NUM_MONO_TYPES_PLACEHOLDER)
D_MONO_EMB = 64
NUM_AMINO_ACIDS = 22

#############################################################################################################
#############################################################################################################
#HELPERS
#############################################################################################################
#############################################################################################################

def compute_glycan_stereobias(
    z: torch.Tensor,
    feats: Dict[str, Any],
    stereo_proj: nn.Module
) -> torch.Tensor:
    """
    Computes and applies stereochemical bias to the Z-track.
    Includes advanced glycosidic linkage aliasing and strict heteroatom ring discovery.
    """
    if "mono_type" not in feats or "token_to_mono_idx" not in feats:
        return torch.zeros_like(z)

    B, N = feats["token_pad_mask"].shape
    token_mono_idx = feats["token_to_mono_idx"]
    mono_types = feats["mono_type"].argmax(dim=-1)
    adj_matrix = feats["token_bonds"].squeeze(-1)
    
    rep_atom_idx = feats["token_to_rep_atom"].argmax(dim=-1)
    ref_atom_name_chars = feats["ref_atom_name_chars"] 
    
    atom_name_first_char_ints = ref_atom_name_chars.argmax(dim=-1)[:, :, 0]
    token_first_chars = torch.gather(atom_name_first_char_ints, 1, rep_atom_idx)

    def decode_name(b_idx, t_idx):
        a_idx = rep_atom_idx[b_idx, t_idx]
        ints = ref_atom_name_chars[b_idx, a_idx].argmax(dim=-1)
        return "".join([chr(c.item() + 32) for c in ints if c.item() > 0]).strip()

    reverse_mono_map = {v: k for k, v in MONO_TYPE_MAP.items()}

    z_bias = torch.zeros_like(z)

    for b in range(B):
        valid_mask = (token_mono_idx[b] != -1)
        if not valid_mask.any():
            continue
            
        unique_monos = torch.unique(token_mono_idx[b][valid_mask])
        
        for m_val in unique_monos:
            m_indices = torch.where(token_mono_idx[b] == m_val)[0]
            m_type_int = mono_types[b, m_indices[0]].item()
            m_type_str = reverse_mono_map.get(m_type_int, "UNK")
            
            # 1. Local Ring Discovery
            local_bonds = adj_matrix[b][m_indices][:, m_indices]
            adj_list = [torch.where(local_bonds[i])[0].tolist() for i in range(len(m_indices))]
            
            def find_cycles(nodes, adj):
                cycles = []
                for start_node in range(len(nodes)):
                    stack = [(start_node, [start_node])]
                    while stack:
                        node, path = stack.pop()
                        for neighbor in adj[node]:
                            if neighbor == start_node and len(path) in [5, 6]:
                                sorted_path = sorted(path)
                                if sorted_path not in cycles:
                                    cycles.append(sorted_path)
                            elif neighbor not in path and len(path) < 6:
                                stack.append((neighbor, path + [neighbor]))
                return cycles
            
            all_cycles = find_cycles(range(len(m_indices)), adj_list)
            if not all_cycles:
                continue

            # Strict Heteroatom Ring Filtering
            valid_rings = []
            for cycle in all_cycles:
                global_cycle = [m_indices[i].item() for i in cycle]
                non_carbon_count = 0
                for a_idx in global_cycle:
                    name = decode_name(b, a_idx)
                    if name and name[0] != 'C':
                        non_carbon_count += 1
                
                if non_carbon_count == 1:
                    valid_rings.append(global_cycle)

            if len(valid_rings) == 0:
                continue
            elif len(valid_rings) > 1:
                continue
            
            main_ring_global = valid_rings[0]
            ring_set = set(main_ring_global)
            
            # 2. Iterate Ring Atoms to find ALL substituents
            for ring_pos, r_idx in enumerate(main_ring_global):
                neighbors = torch.where(adj_matrix[b, r_idx])[0].tolist()
                
                for n_idx in neighbors:
                    if n_idx in ring_set:
                        continue 
                        
                    opp_pos = (ring_pos + len(main_ring_global) // 2) % len(main_ring_global)
                    anchor_idx = main_ring_global[opp_pos]
                    
                    r_name = decode_name(b, r_idx)
                    n_name = decode_name(b, n_idx)
                    anc_name = decode_name(b, anchor_idx)
                    
                    is_inter = token_mono_idx[b, n_idx] != m_val
                    is_glycan_neighbor = token_mono_idx[b, n_idx] != -1
                    alias_first_char_int = token_first_chars[b, n_idx].item()

                    if is_inter:
                        if is_glycan_neighbor:
                            n_char = n_name[0] if len(n_name) > 0 else ""
                            if n_char in ['O', 'N', 'S']:
                                c_num = "".join(filter(str.isdigit, r_name))
                                if not c_num:
                                    c_num = "1" 
                                
                                alias_name = f"O{c_num}"
                                alias_first_char_int = ord('O') - 32 

                    # 3. Compute and Apply Bias to Z-Track
                    m_oh = torch.nn.functional.one_hot(
                        torch.tensor(m_type_int, device=z.device), num_classes=931
                    ).float()
                    anc_char_int = token_first_chars[b, anchor_idx].item()
                    anc_oh = torch.nn.functional.one_hot(
                        torch.tensor(anc_char_int, device=z.device), num_classes=64
                    ).float()
                    sub_oh = torch.nn.functional.one_hot(
                        torch.tensor(alias_first_char_int, device=z.device), num_classes=64
                    ).float()
                    
                    cat_feats = torch.cat([m_oh, anc_oh, sub_oh], dim=-1)
                    bias = stereo_proj(cat_feats)
                    
                    z_bias[b, n_idx, anchor_idx] += bias
                    z_bias[b, anchor_idx, n_idx] += bias
                    
    return z_bias
            
def get_anomeric_pair_features(feats: Dict[str, Tensor]) -> Tensor:
    """
    Returns a [B, L, L, NUM_ANOMERIC_TYPES] tensor containing the one-hot anomeric
    configuration for valid intra-glycan bonds. 
    This extracts the explicit edge bias needed after we have collapsed the tokens.
    """
    device = feats['token_pad_mask'].device
    B, L = feats['token_pad_mask'].shape

    # Grab the true anomeric state instead of the mono type
    dummy_anomeric = feats.get("mono_anomeric", torch.zeros(B, L, NUM_ANOMERIC_TYPES, device=device, dtype=torch.float32))

    if "mono_anomeric" not in feats:
        return torch.zeros((B, L, L, NUM_ANOMERIC_TYPES), device=device, dtype=torch.float32)

    # 1. Identify glycan tokens
    is_mono_feat = feats.get("is_monosaccharide", torch.zeros((B, L, 1), device=device))
    is_glycan = is_mono_feat.squeeze(-1) > 0.5 
    
    # 2. Identify tokens within the same chain
    b_same_chain = torch.eq(feats.get("asym_id", torch.zeros((B, L), device=device))[:, :, None], 
                            feats.get("asym_id", torch.zeros((B, L), device=device))[:, None, :])
    
    # 3. Create mask for ALL Intra-Chain Glycan pairs
    intra_glycan_graph_mask = is_glycan[:, :, None] & is_glycan[:, None, :] & b_same_chain

    # 4. Get valid edges using the ground-truth bond graph
    raw_bonds = feats.get("token_bonds", torch.zeros((B, L, L, 1), device=device)).squeeze(-1).bool()
    glycan_edges = raw_bonds & intra_glycan_graph_mask

    # 5. Determine Carbon atoms for directionality (Inter-residue bonds)
    ref_elem = feats.get('ref_element', torch.zeros((B, L, 1), device=device)).float()
    selector = feats.get('token_to_rep_atom', torch.zeros((B, L, L), device=device)).float()
    token_elem_oh = torch.einsum('bla,bae->ble', selector, ref_elem)
    token_elem = token_elem_oh.argmax(dim=-1)
    is_C = (token_elem == 6)

    anomeric_i = dummy_anomeric.unsqueeze(2).expand(-1, -1, L, -1)
    anomeric_j = dummy_anomeric.unsqueeze(1).expand(-1, L, -1, -1)
    
    is_C_i = is_C.view(B, L, 1, 1)
    is_C_j = is_C.view(B, 1, L, 1)
    
    # For glycosidic bonds, the anomeric feature defaults to the Carbon's state.
    # If token j is Carbon and token i is not, use j's state. Otherwise, use i's state.
    selected_anomeric = torch.where(
        is_C_j & ~is_C_i,
        anomeric_j,
        anomeric_i
    )
    
    return selected_anomeric * glycan_edges.unsqueeze(-1).float()

def stereo_discovery(
    mono_indices: torch.Tensor,
    bond_matrix: torch.Tensor,
) -> list[dict]:
    """
    Simplified singular-anchor topological discovery.
    """
    num_atoms = mono_indices.shape[0]
    local_bonds = bond_matrix[mono_indices][:, mono_indices]
    adj_list = [torch.where(local_bonds[i])[0].tolist() for i in range(num_atoms)]

    # Find cycles (rings)
    def find_cycles(nodes, adj):
        cycles = []
        for start_node in range(len(nodes)):
            stack = [(start_node, [start_node])]
            while stack:
                node, path = stack.pop()
                for neighbor in adj[node]:
                    if neighbor == start_node and len(path) in [5, 6]:
                        cycles.append(path)
                    elif neighbor not in path and len(path) < 6:
                        stack.append((neighbor, path + [neighbor]))
        return cycles

    all_cycles = find_cycles(range(num_atoms), adj_list)
    if not all_cycles:
        return []
    
    main_ring = all_cycles[0]
    ring_set = set(main_ring)
    discovery_results = []
    
    for r_idx in main_ring:
        for n_idx in adj_list[r_idx]:
            if n_idx not in ring_set:
                # n_idx is substituent. Find singular anchor.
                # In a 6-ring, the 'opposite' atom is index + 3 (modulo ring size)
                ring_pos = main_ring.index(r_idx)
                # We pick the atom roughly 'across' the ring
                opp_pos = (ring_pos + len(main_ring) // 2) % len(main_ring)
                anchor_idx = main_ring[opp_pos]
                
                discovery_results.append({
                    'sub_idx': n_idx,
                    'anchor_indices': [anchor_idx] # Now a singular anchor
                })
    return discovery_results
            
def build_couplet_pair_mask(feats: Dict[str, Tensor]) -> Tensor:
    """
    Locate glycosidic couplets and return a boolean mask [B, L, L] where
    mask[b, i, j] = True iff token i is a nucleophile (O or N), token j is carbon,
    they are bonded, and they belong to different monosaccharides.
    """
    token_bonds    = feats["token_bonds"].squeeze(-1).bool()      # [B, L, L]
    token_to_mono = feats["token_to_mono_idx"]                  # [B, L]

    # derive atomic numbers
    ref_elem_oh = feats["ref_element"].float()                  # [B, A, E]
    selector    = feats["token_to_rep_atom"].float()            # [B, L, A]
    elem_oh     = torch.einsum("bla,bae->ble", selector, ref_elem_oh)
    atom_num    = elem_oh.argmax(dim=-1)                        # [B, L]
    is_O        = atom_num == 8
    is_N        = atom_num == 7
    is_C        = atom_num == 6

    # A glycosidic nucleophile can be Oxygen or Nitrogen
    is_nucleophile = is_O | is_N

    # inter‑monosaccharide bonds only
    mono_i      = token_to_mono.unsqueeze(2)                    # [B, L, 1]
    mono_j      = token_to_mono.unsqueeze(1)                    # [B, 1, L]
    inter_mono  = token_bonds & (mono_i != mono_j)

    # mask nucleophile→carbon bonds
    mask_nuc_to_C = inter_mono & is_nucleophile.unsqueeze(2) & is_C.unsqueeze(1) # True if row_idx is O/N, col_idx is C
    # also identify carbon->nucleophile bonds for symmetry
    mask_C_to_nuc = inter_mono & is_C.unsqueeze(2) & is_nucleophile.unsqueeze(1) # True if row_idx is C, col_idx is O/N
    
    return mask_nuc_to_C | mask_C_to_nuc

def _decode_atom_name(one_hot_encoded_name: torch.Tensor) -> str:
    integer_indices = torch.argmax(one_hot_encoded_name, dim=-1)
    chars = []
    for idx_tensor in integer_indices:
        num = idx_tensor.item()
        char_code = num + 32
        if char_code > 32:
            chars.append(chr(char_code))
    return "".join(chars).strip()

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


def _get_glycosylation_features(feats: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    (Streamlined & Instrumented Version)
    Extracts the token indices of the specific protein-glycan covalent attachment points.
    This version is simplified to only compute what is used by the debug print function,
    removing extraneous feature generation.
    """
    device = feats["token_pad_mask"].device
    B, L = feats["token_pad_mask"].shape

    raw_sites_tensors = feats.get('raw_glycosylation_sites')
    if raw_sites_tensors is None:
        raw_sites_tensors = [torch.empty((0, 12), device=device, dtype=torch.long)] * B

    empty_result = {
        "t_glycosylation_indices": torch.empty((0, 2), dtype=torch.long, device=device),
        "t_batch_idx": torch.empty((0,), dtype=torch.long, device=device),
    }

    all_t_indices, all_t_batch_idx = [], []
    any_sites_found = False

    for b in range(B):
        sites_tensor_b = raw_sites_tensors[b]
        if sites_tensor_b is None or sites_tensor_b.numel() == 0:
            continue
        
        any_sites_found = True
        
        atom_to_token = feats["atom_to_token"][b].argmax(-1)
        token_asym_ids = feats["asym_id"][b]
        token_res_indices = feats["residue_index"][b]
        atom_asym_ids = torch.gather(token_asym_ids, 0, atom_to_token)
        atom_res_indices = torch.gather(token_res_indices, 0, atom_to_token)
        ref_name_chars = feats["ref_atom_name_chars"][b]
        
        for site_idx, site_data_tensor in enumerate(sites_tensor_b):
            p_chain_id, p_res_id = site_data_tensor[0].item(), site_data_tensor[1].item()
            g_chain_id = site_data_tensor[6].item()
            
            tgt_p_name = _decode_int_to_str(site_data_tensor[2:6]).upper()
            tgt_g_name = _decode_int_to_str(site_data_tensor[8:12]).upper()

            p_res_mask = (atom_asym_ids == p_chain_id) & (atom_res_indices == p_res_id)
            p_atoms_in_res = torch.where(p_res_mask)[0]
            
            glycan_chain_atoms = torch.where(atom_asym_ids == g_chain_id)[0]
            
            trg_p_atom = -1
            for p_atom_idx in p_atoms_in_res:
                decoded_name = _decode_one_hot_to_str(ref_name_chars[p_atom_idx]).upper()
                if decoded_name == tgt_p_name:
                    trg_p_atom = p_atom_idx.item()
                    break
            
            trg_g_atom = -1
            for g_atom_idx in glycan_chain_atoms:
                 decoded_name = _decode_one_hot_to_str(ref_name_chars[g_atom_idx]).upper()
                 if decoded_name == tgt_g_name:
                    trg_g_atom = g_atom_idx.item()
                    break
            
            if trg_p_atom != -1 and trg_g_atom != -1:
                p_tok_idx = atom_to_token[trg_p_atom].item()
                g_tok_idx = atom_to_token[trg_g_atom].item()
                
                all_t_indices.append([p_tok_idx, g_tok_idx])
                all_t_batch_idx.append(b)

    if not any_sites_found:
        return empty_result

    final_result = {
        "t_glycosylation_indices": torch.tensor(all_t_indices, dtype=torch.long, device=device) if all_t_indices else torch.empty((0, 2), dtype=torch.long, device=device),
        "t_batch_idx": torch.tensor(all_t_batch_idx, dtype=torch.long, device=device),
    }

    return final_result

def _get_glycosylation_linkage_mask(feats: Dict[str, Any], device: torch.device) -> Tensor:
    """
    Creates a boolean mask [B, L, L] that is True for token pairs
    forming a protein-glycan covalent bond, using the ground-truth feature extractor.
    """
    B, L = feats["token_pad_mask"].shape
    linkage_mask = torch.zeros((B, L, L), dtype=torch.bool, device=device)

    glyco_features = _get_glycosylation_features(feats)
    batch_indices = glyco_features["t_batch_idx"]
    token_pairs = glyco_features["t_glycosylation_indices"]

    if token_pairs.numel() > 0:
        p_tokens = token_pairs[:, 0]
        g_tokens = token_pairs[:, 1]
        linkage_mask[batch_indices, p_tokens, g_tokens] = True
        linkage_mask[batch_indices, g_tokens, p_tokens] = True

    return linkage_mask


#############################################################################################################
#############################################################################################################
#CLASSES
#############################################################################################################
#############################################################################################################

class SugarPairformerLayer(nn.Module):
    """
    A single layer of the SugarPairformer refinery. This version is simplified
    and assumes its inputs (s, z) correspond to a single, isolated glycan.
    Therefore, no internal masking is required.
    """
    def __init__(
        self,
        token_s: int,
        token_z: int,
        num_heads: int,
        pairwise_num_heads: int,
        dropout: float = 0.25,
        pairwise_head_width: int = 32,
    ) -> None:
        super().__init__()

        self.dropout = dropout
        self.attention = AttentionPairBias(token_s, token_z, num_heads)
        self.transition_s = Transition(token_s, token_s * 4)

        self.tri_mul_out = TriangleMultiplicationOutgoing(token_z)
        self.tri_mul_in = TriangleMultiplicationIncoming(token_z)
        self.tri_att_start = TriangleAttentionStartingNode(
            token_z, pairwise_head_width, pairwise_num_heads, inf=1e9
        )
        self.tri_att_end = TriangleAttentionEndingNode(
            token_z, pairwise_head_width, pairwise_num_heads, inf=1e9
        )
        self.transition_z = Transition(token_z, token_z * 4)
        
    def forward(
        self,
        s: Tensor,
        z: Tensor,
        mask: Tensor, 
        chunk_size_tri_attn: int | None = None,
        s_bias_continuous: Tensor | None = None,
        z_bias_continuous: Tensor | None = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for an isolated glycan.
        """
        padding_pair_mask = mask.unsqueeze(2) * mask.unsqueeze(1)
        
        # --- Z (Pairwise) Updates ---
        z = z + get_dropout_mask(self.dropout, z, self.training) * self.tri_mul_out(z, mask=padding_pair_mask)
        z = z + get_dropout_mask(self.dropout, z, self.training) * self.tri_mul_in(z, mask=padding_pair_mask)
        z = z + get_dropout_mask(self.dropout, z, self.training) * self.tri_att_start(z, mask=padding_pair_mask, chunk_size=chunk_size_tri_attn)
        z = z + get_dropout_mask(self.dropout, z, self.training, columnwise=True) * self.tri_att_end(z, mask=padding_pair_mask, chunk_size=chunk_size_tri_attn)
        z = z + self.transition_z(z)
        if z_bias_continuous is not None:
            z = z + z_bias_continuous

        # --- S (Single) Updates ---
        s = s + self.attention(s, z, mask=mask)
        s = s + self.transition_s(s)
        if s_bias_continuous is not None:
            s = s + s_bias_continuous
            
        return s, z

class SugarPairformerModule(nn.Module):
    """
    A stack of SugarPairformer layers. This module processes a single,
    isolated glycan tensor.
    """
    def __init__(
        self,
        token_s: int,
        token_z: int,
        num_blocks: int,
        num_heads: int,
        pairwise_num_heads: int,
        dropout: float,
        pairwise_head_width: int,
        activation_checkpointing: bool = True,
        offload_to_cpu: bool = False,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_blocks):
            layer = SugarPairformerLayer(
                token_s,
                token_z,
                num_heads,
                pairwise_num_heads,
                dropout,
                pairwise_head_width,
            )
            if activation_checkpointing:
                layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
            self.layers.append(layer)

    def forward(
        self,
        s: Tensor,
        z: Tensor,
        mask: Tensor, # Pass the standard padding mask down.
        s_bias_continuous: Tensor | None = None,
        z_bias_continuous: Tensor | None = None,
    ) -> Tuple[Tensor, Tensor]:

        if not self.training:
            if z.shape[1] > const.chunk_size_threshold:
                chunk_size_tri_attn = 128
            else:
                chunk_size_tri_attn = 512
        else:
            chunk_size_tri_attn = None

        for layer in self.layers:
            s, z = layer(
                s, 
                z, 
                mask=mask, 
                chunk_size_tri_attn=chunk_size_tri_attn,
                s_bias_continuous=s_bias_continuous,
                z_bias_continuous=z_bias_continuous
            )

        return s, z

class SugarPairformer(nn.Module):
    """
    (Corrected for DDP)
    A specialist module that refines glycan representations.
    This version uses a gather-batch-process-scatter approach to guarantee
    mathematical isolation for each glycan chain while remaining compatible
    with DDP and activation checkpointing by making only a single call
    to its core processing stack.
    """
    def __init__(
        self,
        token_s: int,
        token_z: int,
        num_blocks: int = 4,
        num_heads: int = 16,
        pairwise_num_heads: int = 32,
        pairwise_head_width: int = 4,
        dropout: float = 0.25,
        activation_checkpointing: bool = True,
        offload_to_cpu: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.sugar_pairformer_stack = SugarPairformerModule(
            token_s=token_s,
            token_z=token_z,
            num_blocks=num_blocks,
            num_heads=num_heads,
            pairwise_num_heads=pairwise_num_heads,
            pairwise_head_width=pairwise_head_width,
            dropout=dropout,
            activation_checkpointing=activation_checkpointing,
            offload_to_cpu=offload_to_cpu,
        )

    def forward(self, s_input: Tensor, z_input: Tensor, feats: Dict[str, Any], s_bias_continuous: Tensor | None = None, z_bias_continuous: Tensor | None = None) -> Tuple[Tensor, Tensor]:
        """
        Extracts glycan representations, processes them in a single batched call,
        and scatters the results back. This module is designed to operate only on
        tokens identified as part of a monosaccharide, leaving all other token
        representations mathematically unchanged. If no glycans are present in a batch,
        it uses a parameter-summing mechanism to ensure compatibility with Distributed
        Data Parallel (DDP) training without performing any data processing.
        """
        B, N, _ = s_input.shape
        
        is_glycan_token = feats['is_monosaccharide'].squeeze(-1).bool()
        
        # Initialize lists to gather glycan-specific data
        glycan_s_list, glycan_z_list, glycan_mask_list = [], [], []
        glycan_s_bias_list, glycan_z_bias_list = [], []
        scatter_map = []

        # 1. GATHER (Only if there are any glycans in the batch)
        if torch.any(is_glycan_token):
            s_out = s_input.clone()
            z_out = z_input.clone()
            asym_id_all: Tensor = feats['asym_id']
            pad_mask_all: Tensor = feats['token_pad_mask']

            for b in range(B):
                is_glycan_token_b = is_glycan_token[b]
                if not torch.any(is_glycan_token_b): continue
                token_asym_id = asym_id_all[b]
                unique_chain_ids = torch.unique(token_asym_id[is_glycan_token_b])
                for chain_id in unique_chain_ids:
                    glycan_indices = torch.where((token_asym_id == chain_id) & is_glycan_token_b)[0]
                    if glycan_indices.numel() == 0: continue
                    
                    glycan_s_list.append(s_input[b, glycan_indices, :])
                    glycan_z_list.append(z_input[b, glycan_indices][:, glycan_indices])
                    glycan_mask_list.append(pad_mask_all[b, glycan_indices])
                    
                    if s_bias_continuous is not None:
                        glycan_s_bias_list.append(s_bias_continuous[b, glycan_indices, :])
                    if z_bias_continuous is not None:
                        glycan_z_bias_list.append(z_bias_continuous[b, glycan_indices][:, glycan_indices])
                        
                    scatter_map.append({'batch_idx': b, 'indices': glycan_indices})

        # If no glycan chains were gathered across the entire batch, apply the DDP-safe exit.
        if not glycan_s_list:
            # DDP-safe exit: "touch" all parameters in the stack by summing them.
            # This adds them to the computation graph without running a forward pass.
            # The result is multiplied by 0.0, ensuring no mathematical impact.
            dummy_loss = 0.0
            for p in self.sugar_pairformer_stack.parameters():
                dummy_loss += p.sum()
            
            return s_input + (dummy_loss * 0.0), z_input + (dummy_loss * 0.0)

        # 2. PAD & BATCH
        max_glycan_len = max(s.shape[0] for s in glycan_s_list)
        s_padded_list, z_padded_list, mask_padded_list = [], [], []
        s_bias_padded_list, z_bias_padded_list = [], []
        
        for i in range(len(glycan_s_list)):
            s = glycan_s_list[i]
            z = glycan_z_list[i]
            m = glycan_mask_list[i]
            pad_len = max_glycan_len - s.shape[0]
            
            s_padded = F.pad(s, (0, 0, 0, pad_len)) if pad_len > 0 else s
            z_padded = F.pad(z, (0, 0, 0, pad_len, 0, pad_len)) if pad_len > 0 else z
            m_padded = F.pad(m, (0, pad_len)) if pad_len > 0 else m
            
            s_padded_list.append(s_padded)
            z_padded_list.append(z_padded)
            mask_padded_list.append(m_padded)
            
            if s_bias_continuous is not None:
                sb = glycan_s_bias_list[i]
                s_bias_padded_list.append(F.pad(sb, (0, 0, 0, pad_len)) if pad_len > 0 else sb)
            if z_bias_continuous is not None:
                zb = glycan_z_bias_list[i]
                z_bias_padded_list.append(F.pad(zb, (0, 0, 0, pad_len, 0, pad_len)) if pad_len > 0 else zb)

        s_batch = torch.stack(s_padded_list, dim=0)
        z_batch = torch.stack(z_padded_list, dim=0)
        mask_batch = torch.stack(mask_padded_list, dim=0)
        s_bias_batch = torch.stack(s_bias_padded_list, dim=0) if s_bias_continuous is not None else None
        z_bias_batch = torch.stack(z_bias_padded_list, dim=0) if z_bias_continuous is not None else None

        # 3. SINGLE PROCESS CALL
        s_refined_batch, z_refined_batch = self.sugar_pairformer_stack(
            s=s_batch, z=z_batch, mask=mask_batch.float(),
            s_bias_continuous=s_bias_batch, z_bias_continuous=z_bias_batch
        )

        # 4. UNBATCH & SCATTER
        for i, meta in enumerate(scatter_map):
            b, original_indices = meta['batch_idx'], meta['indices']
            original_len = len(original_indices)
            s_refined = s_refined_batch[i, :original_len, :]
            z_refined = z_refined_batch[i, :original_len, :original_len, :]
            s_out[b, original_indices, :] = s_refined
            rows, cols = torch.meshgrid(original_indices, original_indices, indexing='ij')
            z_out[b, rows, cols, :] = z_refined
            
        return s_out, z_out
