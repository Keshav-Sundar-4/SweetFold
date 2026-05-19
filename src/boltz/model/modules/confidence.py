import torch
from torch import nn
import torch.nn.functional as F

from boltz.data import const
import boltz.model.layers.initialize as init
from boltz.model.modules.confidence_utils import (
    compute_aggregated_metric,
    compute_ptms,
)
from boltz.model.modules.encoders import RelativePositionEncoder
from boltz.model.modules.trunk import (
    InputEmbedder,
    MSAModule,
    PairformerModule,
)
from boltz.model.modules.sugar_trunk import (
    SugarPairformer, 
    stereo_discovery,
)
from boltz.model.modules.sugar_trunk import get_anomeric_pair_features
from boltz.model.modules.utils import LinearNoBias

class ConfidenceModule(nn.Module):
    """Confidence module."""

    def __init__(
        self,
        token_s,
        token_z,
        pairformer_args: dict,
        glycan_bias_args: dict | None = None,
        num_dist_bins=64,
        max_dist=22,
        add_s_to_z_prod=False,
        add_s_input_to_s=False,
        use_s_diffusion=False,
        add_z_input_to_z=False,
        confidence_args: dict = None,
        compute_pae: bool = False,
        imitate_trunk=False,
        full_embedder_args: dict = None,
        msa_args: dict = None,
        compile_pairformer=False,
        stereo_proj: nn.Module = None,
    ):
        super().__init__()
        
        self.max_num_atoms_per_token = 23
        self.no_update_s = pairformer_args.get("no_update_s", False)
        boundaries = torch.linspace(2, max_dist, num_dist_bins - 1)
        self.register_buffer("boundaries", boundaries)
        self.dist_bin_pairwise_embed = nn.Embedding(num_dist_bins, token_z)
        init.gating_init_(self.dist_bin_pairwise_embed.weight)
        
        s_input_dim = (token_s + 2 * const.num_tokens + 1 + len(const.pocket_contact_info))

        self.use_s_diffusion = use_s_diffusion
        if use_s_diffusion:
            self.s_diffusion_norm = nn.LayerNorm(2 * token_s)
            self.s_diffusion_to_s = LinearNoBias(2 * token_s, token_s)
            init.gating_init_(self.s_diffusion_to_s.weight)

        self.s_to_z = LinearNoBias(s_input_dim, token_z)
        self.s_to_z_transpose = LinearNoBias(s_input_dim, token_z)
        init.gating_init_(self.s_to_z.weight)
        init.gating_init_(self.s_to_z_transpose.weight)

        self.add_s_to_z_prod = add_s_to_z_prod
        if add_s_to_z_prod:
            self.s_to_z_prod_in1 = LinearNoBias(s_input_dim, token_z)
            self.s_to_z_prod_in2 = LinearNoBias(s_input_dim, token_z)
            self.s_to_z_prod_out = LinearNoBias(token_z, token_z)
            init.gating_init_(self.s_to_z_prod_out.weight)

        # ROLE-AWARE STEREO EMBEDDING
        if stereo_proj is not None:
            self.stereo_proj = stereo_proj
        else:
            self.stereo_proj = nn.Sequential(
                nn.Linear(931 + 64 + 64, 256, bias=True),
                nn.GELU(),
                nn.Linear(256, token_z, bias=False)
            )
            torch.nn.init.normal_(self.stereo_proj[2].weight, mean=0.0, std=0.02)

        self.imitate_trunk = imitate_trunk
        if self.imitate_trunk:
            self.s_init = nn.Linear(s_input_dim, token_s, bias=False)
            self.z_init_1 = nn.Linear(s_input_dim, token_z, bias=False)
            self.z_init_2 = nn.Linear(s_input_dim, token_z, bias=False)

            self.input_embedder = InputEmbedder(**full_embedder_args)
            self.rel_pos = RelativePositionEncoder(token_z)
            self.token_bonds = nn.Linear(1, token_z, bias=False)
            
            self.s_norm = nn.LayerNorm(token_s)
            self.z_norm = nn.LayerNorm(token_z)

            self.s_recycle = nn.Linear(token_s, token_s, bias=False)
            self.z_recycle = nn.Linear(token_z, token_z, bias=False)
            init.gating_init_(self.s_recycle.weight)
            init.gating_init_(self.z_recycle.weight)

            self.msa_module = MSAModule(token_z=token_z, s_input_dim=s_input_dim, **msa_args)
            
            # Omit **pairformer_args to use the defaults!
            self.sugar_pairformer_module = SugarPairformer(
                token_s=token_s, 
                token_z=token_z,
                activation_checkpointing=pairformer_args.get("activation_checkpointing", False),
                offload_to_cpu=pairformer_args.get("offload_to_cpu", False)
            )
            self.pairformer_module = PairformerModule(token_s, token_z, **pairformer_args)
            
            if compile_pairformer:
                self.is_pairformer_compiled = True
                self.is_sugar_pairformer_compiled = True
                self.sugar_pairformer_module = torch.compile(self.sugar_pairformer_module, dynamic=False, fullgraph=False)
                self.pairformer_module = torch.compile(self.pairformer_module, dynamic=False, fullgraph=False)

            self.final_s_norm = nn.LayerNorm(token_s)
            self.final_z_norm = nn.LayerNorm(token_z)

        else:
            self.s_inputs_norm = nn.LayerNorm(s_input_dim)
            if not self.no_update_s:
                self.s_norm = nn.LayerNorm(token_s)
                self.z_norm = nn.LayerNorm(token_z)

            self.add_s_input_to_s = add_s_input_to_s
            if add_s_input_to_s:
                self.s_input_to_s = LinearNoBias(s_input_dim, token_s)
                init.gating_init_(self.s_input_to_s.weight)

            self.add_z_input_to_z = add_z_input_to_z
            if add_z_input_to_z:
                self.rel_pos = RelativePositionEncoder(token_z)
                self.token_bonds = nn.Linear(1, token_z, bias=False)

            # Omit **pairformer_args to use the defaults!
            self.sugar_pairformer_stack = SugarPairformer(
                token_s=token_s, 
                token_z=token_z,
                activation_checkpointing=pairformer_args.get("activation_checkpointing", False),
                offload_to_cpu=pairformer_args.get("offload_to_cpu", False)
            )
            self.pairformer_stack = PairformerModule(token_s, token_z, **pairformer_args)

        self.confidence_heads = ConfidenceHeads(
            token_s, token_z, compute_pae=compute_pae, **confidence_args
        )

    def forward(
        self,
        s_inputs,
        s,
        z,
        x_pred,
        feats,
        pred_distogram_logits,
        multiplicity=1,
        s_diffusion=None,
        run_sequentially=False,
    ):
        # --- DDP-PROOFING: Always touch stereo_proj parameters ---
        stereo_dummy = sum(p.sum() for p in self.stereo_proj.parameters())
        # Note: s and z are updated later in the logic, so we inject the touch into the base s/z
        s = s + (0.0 * stereo_dummy)
        z = z + (0.0 * stereo_dummy)

        if run_sequentially and multiplicity > 1:
            assert z.shape[0] == 1, "Not supported with batch size > 1"
            out_dicts = []
            for sample_idx in range(multiplicity):
                out_dicts.append(
                    self.forward(
                        s_inputs, s, z, x_pred[sample_idx : sample_idx + 1], feats,
                        pred_distogram_logits, multiplicity=1,
                        s_diffusion=s_diffusion[sample_idx : sample_idx + 1] if s_diffusion is not None else None,
                        run_sequentially=False,
                    )
                )
            # ... (rest of sequential logic)
            out_dict = {}
            for key in out_dicts[0]:
                if key != "pair_chains_iptm":
                    out_dict[key] = torch.cat([out[key] for out in out_dicts], dim=0)
                else:
                    pair_chains_iptm = {}
                    for c1 in out_dicts[0][key].keys():
                        pair_chains_iptm[c1] = {
                            c2: torch.cat([out[key][c1][c2] for out in out_dicts], dim=0)
                            for c2 in out_dicts[0][key][c1].keys()
                        }
                    out_dict[key] = pair_chains_iptm
            return out_dict

        B, N = s_inputs.shape[:2]

        if self.imitate_trunk:
            s_inputs_trunk = self.input_embedder(feats)
            s_init = self.s_init(s_inputs_trunk)
            z_init = self.z_init_1(s_inputs_trunk)[:, :, None] + self.z_init_2(s_inputs_trunk)[:, None, :]
            z_init = z_init + self.rel_pos(feats)
            z_init = z_init + self.token_bonds(feats["token_bonds"].float())
            
            # Singular Stereo Bias Injection
            if "mono_type" in feats and "token_to_mono_idx" in feats:
                adj = feats["token_bonds"].squeeze(-1)
                m_types = feats["mono_type"].argmax(dim=-1)
                t_idx = feats["token_to_mono_idx"]
                
                rep_atom_idx = feats["token_to_rep_atom"].argmax(dim=-1) 
                atom_name_ints = feats["ref_atom_name_chars"].argmax(dim=-1)[:, :, 0] 
                token_names = torch.gather(atom_name_ints, 1, rep_atom_idx) 

                for b in range(B):
                    valid = t_idx[b] != -1
                    if not valid.any(): continue
                    for m_val in torch.unique(t_idx[b][valid]):
                        indices = torch.where(t_idx[b] == m_val)[0]
                        res = stereo_discovery(indices, adj[b])
                        for e in res:
                            si, ai = indices[e['sub_idx']], indices[e['anchor_indices'][0]]
                            m_type = m_types[b, si]
                            sub_n = token_names[b, si]
                            anc_n = token_names[b, ai]

                            m_oh = torch.nn.functional.one_hot(m_type, num_classes=931).float()
                            anc_oh = torch.nn.functional.one_hot(anc_n, num_classes=64).float()
                            sub_oh = torch.nn.functional.one_hot(sub_n, num_classes=64).float()
                            
                            cat_feats = torch.cat([m_oh, anc_oh, sub_oh], dim=-1)
                            bias = self.stereo_proj(cat_feats)
                            
                            z_init[b, si, ai] += bias
                            z_init[b, ai, si] += bias

            s = s_init + self.s_recycle(self.s_norm(s))
            z = z_init + self.z_recycle(self.z_norm(z))

        else:
            # Standard Confidence Branch Logic
            s_inputs = self.s_inputs_norm(s_inputs).repeat_interleave(multiplicity, 0)
            if not self.no_update_s: s = self.s_norm(s)
            if self.add_s_input_to_s: s = s + self.s_input_to_s(s_inputs)

            z = self.z_norm(z)
            if self.add_z_input_to_z:
                z = z + self.rel_pos(feats)
                z = z + self.token_bonds(feats["token_bonds"].float())
                
                if "mono_type" in feats and "token_to_mono_idx" in feats:
                    adj = feats["token_bonds"].squeeze(-1).repeat_interleave(multiplicity, 0)
                    m_types = feats["mono_type"].argmax(dim=-1).repeat_interleave(multiplicity, 0)
                    t_idx = feats["token_to_mono_idx"].repeat_interleave(multiplicity, 0)
                    
                    rep_atom_idx = feats["token_to_rep_atom"].argmax(dim=-1).repeat_interleave(multiplicity, 0)
                    atom_name_ints = feats["ref_atom_name_chars"].argmax(dim=-1)[:, :, 0].repeat_interleave(multiplicity, 0)
                    token_names = torch.gather(atom_name_ints, 1, rep_atom_idx)

                    for b in range(z.shape[0]):
                        valid = t_idx[b] != -1
                        if not valid.any(): continue
                        for m_val in torch.unique(t_idx[b][valid]):
                            indices = torch.where(t_idx[b] == m_val)[0]
                            res = stereo_discovery(indices, adj[b])
                            for e in res:
                                si, ai = indices[e['sub_idx']], indices[e['anchor_indices'][0]]
                                m_oh = torch.nn.functional.one_hot(m_types[b, si], num_classes=931).float()
                                anc_oh = torch.nn.functional.one_hot(token_names[b, ai], num_classes=64).float()
                                sub_oh = torch.nn.functional.one_hot(token_names[b, si], num_classes=64).float()
                                cat_feats = torch.cat([m_oh, anc_oh, sub_oh], dim=-1)
                                bias = self.stereo_proj(cat_feats)
                                z[b, si, ai] += bias
                                z[b, ai, si] += bias

        # (Rest of confidence forward remains same...)
        s = s.repeat_interleave(multiplicity, 0)
        if self.use_s_diffusion:
            s = s + self.s_diffusion_to_s(self.s_diffusion_norm(s_diffusion))

        z = z.repeat_interleave(multiplicity, 0)
        z = z + self.s_to_z(s_inputs)[:, :, None, :] + self.s_to_z_transpose(s_inputs)[:, None, :, :]

        if self.add_s_to_z_prod:
            z = z + self.s_to_z_prod_out(self.s_to_z_prod_in1(s_inputs)[:, :, None, :] * self.s_to_z_prod_in2(s_inputs)[:, None, :, :])

        token_to_rep_atom = feats["token_to_rep_atom"].repeat_interleave(multiplicity, 0)
        if len(x_pred.shape) == 4:
            x_pred = x_pred.reshape(B * multiplicity, -1, 3)
        x_pred_repr = torch.bmm(token_to_rep_atom.float(), x_pred)
        d = torch.cdist(x_pred_repr, x_pred_repr)

        z = z + self.dist_bin_pairwise_embed((d.unsqueeze(-1) > self.boundaries).sum(dim=-1).long())

        mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)
        pair_mask = mask[:, :, None] * mask[:, None, :]

        if self.imitate_trunk:
            z = z + self.msa_module(z, s_inputs, feats)
            sugar_pairformer_module = (
                self.sugar_pairformer_module._orig_mod 
                if (getattr(self, "is_sugar_pairformer_compiled", False) and not self.training) 
                else self.sugar_pairformer_module
            )
            s, z = sugar_pairformer_module(s, z, feats)
            s, z = self.pairformer_module(s, z, mask=mask, pair_mask=pair_mask)
            s, z = self.final_s_norm(s), self.final_z_norm(z)
        else:
            s, z = self.sugar_pairformer_stack(s, z, feats)
            s, z = self.pairformer_stack(s, z, mask=mask, pair_mask=pair_mask)

        out_dict = {}
        out_dict.update(self.confidence_heads(
            s=s, z=z, x_pred=x_pred, d=d, feats=feats,
            multiplicity=multiplicity, pred_distogram_logits=pred_distogram_logits,
        ))
        return out_dict
            
class ConfidenceHeads(nn.Module):
    """Confidence heads."""

    def __init__(
        self,
        token_s,
        token_z,
        num_plddt_bins=50,
        num_pde_bins=64,
        num_pae_bins=64,
        compute_pae: bool = True,
    ):
        """Initialize the confidence head.

        Parameters
        ----------
        token_s : int
            The single representation dimension.
        token_z : int
            The pair representation dimension.
        num_plddt_bins : int
            The number of plddt bins, by default 50.
        num_pde_bins : int
            The number of pde bins, by default 64.
        num_pae_bins : int
            The number of pae bins, by default 64.
        compute_pae : bool
            Whether to compute pae, by default False
        """

        super().__init__()
        self.max_num_atoms_per_token = 23
        self.to_pde_logits = LinearNoBias(token_z, num_pde_bins)
        self.to_plddt_logits = LinearNoBias(token_s, num_plddt_bins)
        self.to_resolved_logits = LinearNoBias(token_s, 2)
        self.compute_pae = compute_pae
        if self.compute_pae:
            self.to_pae_logits = LinearNoBias(token_z, num_pae_bins)

    def forward(
        self,
        s,
        z,
        x_pred,
        d,
        feats,
        pred_distogram_logits,
        multiplicity=1,
    ):
        # Compute the pLDDT, PDE, PAE, and resolved logits
        plddt_logits = self.to_plddt_logits(s)
        pde_logits = self.to_pde_logits(z + z.transpose(1, 2))
        resolved_logits = self.to_resolved_logits(s)
        if self.compute_pae:
            pae_logits = self.to_pae_logits(z)

        # Weights used to compute the interface pLDDT
        ligand_weight = 2
        interface_weight = 1

        # Retrieve relevant features
        token_type = feats["mol_type"]
        token_type = token_type.repeat_interleave(multiplicity, 0)
        is_ligand_token = (token_type == const.chain_type_ids["NONPOLYMER"]).float()

        # Compute the aggregated pLDDT and iPLDDT
        plddt = compute_aggregated_metric(plddt_logits)
        token_pad_mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)
        complex_plddt = (plddt * token_pad_mask).sum(dim=-1) / token_pad_mask.sum(
            dim=-1
        )

        is_contact = (d < 8).float()
        is_different_chain = (
            feats["asym_id"].unsqueeze(-1) != feats["asym_id"].unsqueeze(-2)
        ).float()
        is_different_chain = is_different_chain.repeat_interleave(multiplicity, 0)
        token_interface_mask = torch.max(
            is_contact * is_different_chain * (1 - is_ligand_token).unsqueeze(-1),
            dim=-1,
        ).values
        iplddt_weight = (
            is_ligand_token * ligand_weight + token_interface_mask * interface_weight
        )
        complex_iplddt = (plddt * token_pad_mask * iplddt_weight).sum(dim=-1) / (
            torch.sum(token_pad_mask * iplddt_weight, dim=-1) + 1e-5
        )

        # Compute the aggregated PDE and iPDE
        pde = compute_aggregated_metric(pde_logits, end=32)
        pred_distogram_prob = nn.functional.softmax(
            pred_distogram_logits, dim=-1
        ).repeat_interleave(multiplicity, 0)
        contacts = torch.zeros((1, 1, 1, 64), dtype=pred_distogram_prob.dtype).to(
            pred_distogram_prob.device
        )
        contacts[:, :, :, :20] = 1.0
        prob_contact = (pred_distogram_prob * contacts).sum(-1)
        token_pad_mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)
        token_pad_pair_mask = (
            token_pad_mask.unsqueeze(-1)
            * token_pad_mask.unsqueeze(-2)
            * (
                1
                - torch.eye(
                    token_pad_mask.shape[1], device=token_pad_mask.device
                ).unsqueeze(0)
            )
        )
        token_pair_mask = token_pad_pair_mask * prob_contact
        complex_pde = (pde * token_pair_mask).sum(dim=(1, 2)) / token_pair_mask.sum(
            dim=(1, 2)
        )
        asym_id = feats["asym_id"].repeat_interleave(multiplicity, 0)
        token_interface_pair_mask = token_pair_mask * (
            asym_id.unsqueeze(-1) != asym_id.unsqueeze(-2)
        )
        complex_ipde = (pde * token_interface_pair_mask).sum(dim=(1, 2)) / (
            token_interface_pair_mask.sum(dim=(1, 2)) + 1e-5
        )

        out_dict = dict(
            pde_logits=pde_logits,
            plddt_logits=plddt_logits,
            resolved_logits=resolved_logits,
            pde=pde,
            plddt=plddt,
            complex_plddt=complex_plddt,
            complex_iplddt=complex_iplddt,
            complex_pde=complex_pde,
            complex_ipde=complex_ipde,
        )
        if self.compute_pae:
            out_dict["pae_logits"] = pae_logits
            out_dict["pae"] = compute_aggregated_metric(pae_logits, end=32)
            ptm, iptm, ligand_iptm, protein_iptm, pair_chains_iptm = compute_ptms(
                pae_logits, x_pred, feats, multiplicity
            )
            out_dict["ptm"] = ptm
            out_dict["iptm"] = iptm
            out_dict["ligand_iptm"] = ligand_iptm
            out_dict["protein_iptm"] = protein_iptm
            out_dict["pair_chains_iptm"] = pair_chains_iptm

        return out_dict
