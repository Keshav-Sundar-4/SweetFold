from typing import Dict, Iterator, List, Optional
import sys
import numpy as np
from numpy.random import RandomState
from pathlib import Path

from boltz.data import const
from boltz.data.types import ChainInfo, InterfaceInfo, Record
from boltz.data.sample.sampler import Sample, Sampler


def get_chain_cluster(chain: ChainInfo, record: Record) -> str:  # noqa: ARG001
    """Get the cluster id for a chain.

    Parameters
    ----------
    chain : ChainInfo
        The chain id to get the cluster id for.
    record : Record
        The record the interface is part of.

    Returns
    -------
    str
        The cluster id of the chain.

    """
    return chain.cluster_id


def get_interface_cluster(interface: InterfaceInfo, record: Record) -> str:
    """Get the cluster id for an interface.

    Parameters
    ----------
    interface : InterfaceInfo
        The interface to get the cluster id for.
    record : Record
        The record the interface is part of.

    Returns
    -------
    str
        The cluster id of the interface.

    """
    chain1 = record.chains[interface.chain_1]
    chain2 = record.chains[interface.chain_2]

    cluster_1 = str(chain1.cluster_id)
    cluster_2 = str(chain2.cluster_id)

    cluster_id = (cluster_1, cluster_2)
    cluster_id = tuple(sorted(cluster_id))

    return cluster_id


def get_chain_weight(
    chain: ChainInfo,
    record: Record,  # noqa: ARG001
    clusters: Dict[str, int],
    beta_chain: float,
    alpha_prot: float,
    alpha_nucl: float,
    alpha_ligand: float,
) -> float:
    """Get the weight of a chain.

    Parameters
    ----------
    chain : ChainInfo
        The chain to get the weight for.
    record : Record
        The record the chain is part of.
    clusters : Dict[str, int]
        The cluster sizes.
    beta_chain : float
        The beta value for chains.
    alpha_prot : float
        The alpha value for proteins.
    alpha_nucl : float
        The alpha value for nucleic acids.
    alpha_ligand : float
        The alpha value for ligands.

    Returns
    -------
    float
        The weight of the chain.

    """
    prot_id = const.chain_type_ids["PROTEIN"]
    rna_id = const.chain_type_ids["RNA"]
    dna_id = const.chain_type_ids["DNA"]
    ligand_id = const.chain_type_ids["NONPOLYMER"]

    weight = beta_chain / clusters[chain.cluster_id]
    if chain.mol_type == prot_id:
        weight *= alpha_prot
    elif chain.mol_type in [rna_id, dna_id]:
        weight *= alpha_nucl
    elif chain.mol_type == ligand_id:
        weight *= alpha_ligand

    return weight


def get_interface_weight(
    interface: InterfaceInfo,
    record: Record,
    clusters: Dict[str, int],
    beta_interface: float,
    alpha_prot: float,
    alpha_nucl: float,
    alpha_ligand: float,
) -> float:
    """Get the weight of an interface.

    Parameters
    ----------
    interface : InterfaceInfo
        The interface to get the weight for.
    record : Record
        The record the interface is part of.
    clusters : Dict[str, int]
        The cluster sizes.
    beta_interface : float
        The beta value for interfaces.
    alpha_prot : float
        The alpha value for proteins.
    alpha_nucl : float
        The alpha value for nucleic acids.
    alpha_ligand : float
        The alpha value for ligands.

    Returns
    -------
    float
        The weight of the interface.

    """
    prot_id = const.chain_type_ids["PROTEIN"]
    rna_id = const.chain_type_ids["RNA"]
    dna_id = const.chain_type_ids["DNA"]
    ligand_id = const.chain_type_ids["NONPOLYMER"]

    chain1 = record.chains[interface.chain_1]
    chain2 = record.chains[interface.chain_2]

    n_prot = (chain1.mol_type) == prot_id
    n_nuc = chain1.mol_type in [rna_id, dna_id]
    n_ligand = chain1.mol_type == ligand_id

    n_prot += chain2.mol_type == prot_id
    n_nuc += chain2.mol_type in [rna_id, dna_id]
    n_ligand += chain2.mol_type == ligand_id

    weight = beta_interface / clusters[get_interface_cluster(interface, record)]
    weight *= alpha_prot * n_prot + alpha_nucl * n_nuc + alpha_ligand * n_ligand
    return weight

class ClusterSampler(Sampler):
    """The weighted sampling approach, as described in AF3.

    Each chain / interface is given a weight according
    to the following formula, and sampled accordingly:

    w = b / n_clust *(a_prot * n_prot + a_nuc * n_nuc
        + a_ligand * n_ligand)

    This sampler includes a modification for equitable glycan modeling, 
    separating data into 5 specific pools (SER, THR, Other Glyco, Lectin, 
    Free Glycan) and enforcing exact sampling ratios per chunk.
    """

    def __init__(
        self,
        alpha_prot: float = 3.0,
        alpha_nucl: float = 3.0,
        alpha_ligand: float = 1.0,
        beta_chain: float = 0.5,
        beta_interface: float = 1.0,
        glycan_context_prob: Optional[float] = 0.70,
        equity_glycan_training: bool = False,
        target_dir: Optional[str] = None,
    ) -> None:
        """Initialize the sampler."""
        self.alpha_prot = alpha_prot
        self.alpha_nucl = alpha_nucl
        self.alpha_ligand = alpha_ligand
        self.beta_chain = beta_chain
        self.beta_interface = beta_interface
        self.glycan_context_prob = glycan_context_prob
        self.equity_glycan_training = equity_glycan_training
        self.target_dir = target_dir

    def sample(self, records: List[Record], random: RandomState) -> Iterator[Sample]:  # noqa: C901, PLR0912
        """Sample a structure from the dataset infinitely."""
        # Import the map to act as the source of truth for glycan residue names
        from boltz.data.feature.featurizer import MONO_TYPE_MAP
        GLYCAN_CODES = set(MONO_TYPE_MAP.keys())

        # 1. Compute chain cluster sizes (Standard Boltz/AF3 logic)
        chain_clusters: Dict[str, int] = {}
        for record in records:
            for chain in record.chains:
                if not chain.valid:
                    continue
                cluster_id = get_chain_cluster(chain, record)
                chain_clusters[cluster_id] = chain_clusters.get(cluster_id, 0) + 1

        # 2. Compute interface clusters sizes
        interface_clusters: Dict[str, int] = {}
        for record in records:
            for interface in record.interfaces:
                if not interface.valid:
                    continue
                cluster_id = get_interface_cluster(interface, record)
                interface_clusters[cluster_id] = interface_clusters.get(cluster_id, 0) + 1

        # Pool initialization
        ser_items, ser_weights = [], []
        thr_items, thr_weights = [], []
        other_glyco_items, other_glyco_weights = [], []
        lectin_items, lectin_weights = [], []
        eq_free_glycan_items, eq_free_glycan_weights = [], []
        eq_monosaccharide_items, eq_monosaccharide_weights = [], []
        
        contextual_items, contextual_weights = [], []
        free_glycan_items, free_glycan_weights = [], []
        standard_other_items, standard_other_weights = [], []

        prot_id = const.chain_type_ids["PROTEIN"]
        ligand_id = const.chain_type_ids["NONPOLYMER"]

        record_category_cache = {}

        for record in records:
            has_protein = any(c.mol_type == prot_id for c in record.chains)
            has_nonpolymer = any(c.mol_type == ligand_id for c in record.chains)

            category = "standard_other"

            # Check for actual glycan residues if non-polymers are present
            if self.target_dir is not None and has_nonpolymer:
                has_actual_glycan = False
                has_sites = False
                is_ser, is_thr = False, False
                glycan_residue_count = 0

                try:
                    npz_path = Path(self.target_dir) / "structures" / f"{record.id}.npz"
                    # Optimization: only load if it's a potential glycan candidate
                    with np.load(npz_path, allow_pickle=True) as data:
                        residues = data["residues"]
                        
                        # Verify every sample has at least one recognized glycan residue
                        for res in residues:
                            res_name = str(res["name"]).strip().upper()
                            if res_name in GLYCAN_CODES:
                                has_actual_glycan = True
                                glycan_residue_count += 1
                        
                        if has_actual_glycan:
                            # Now check covalent attachment points
                            sites = data.get("glycosylation_sites")
                            if sites is not None and (sites.shape != () or sites.item() is not None) and sites.size > 0:
                                has_sites = True
                                chains = data["chains"]
                                for site in np.atleast_1d(sites):
                                    if site is None: continue
                                    global_res_idx = chains[site["protein_chain_id"]]["res_idx"] + site["protein_res_id"]
                                    p_res_name = str(residues[global_res_idx]["name"]).strip().upper()
                                    if p_res_name == "SER": is_ser = True
                                    elif p_res_name == "THR": is_thr = True
                except Exception:
                    pass # Keep as standard_other if structure load fails

                if has_actual_glycan:
                    if has_protein:
                        if not has_sites:
                            category = "lectin" # Now confirmed Protein + Glycan
                        elif is_ser: category = "ser"
                        elif is_thr: category = "thr"
                        else: category = "other_glyco"
                    else:
                        # Differentiate between isolated monomers and free glycans
                        category = "monosaccharide" if glycan_residue_count == 1 else "free_glycan"

            record_category_cache[record.id] = category
            
            # Map Pointer to the correct pools based on training mode
            if self.equity_glycan_training:
                if category == "ser": target_items, target_weights = ser_items, ser_weights
                elif category == "thr": target_items, target_weights = thr_items, thr_weights
                elif category == "other_glyco": target_items, target_weights = other_glyco_items, other_glyco_weights
                elif category == "lectin": target_items, target_weights = lectin_items, lectin_weights
                elif category == "free_glycan": target_items, target_weights = eq_free_glycan_items, eq_free_glycan_weights
                elif category == "monosaccharide": target_items, target_weights = eq_monosaccharide_items, eq_monosaccharide_weights
                else: target_items, target_weights = standard_other_items, standard_other_weights
            else:
                if category in ["ser", "thr", "other_glyco", "lectin"]:
                    target_items, target_weights = contextual_items, contextual_weights
                elif category in ["free_glycan", "monosaccharide"]:
                    target_items, target_weights = free_glycan_items, free_glycan_weights
                else:
                    target_items, target_weights = standard_other_items, standard_other_weights

            # Compute and add weights for chains
            for chain_id, chain in enumerate(record.chains):
                if not chain.valid: continue
                weight = get_chain_weight(chain, record, chain_clusters, self.beta_chain, self.alpha_prot, self.alpha_nucl, self.alpha_ligand)
                target_items.append((record, 0, chain_id))
                target_weights.append(weight)

            # Compute and add weights for interfaces
            for int_id, interface in enumerate(record.interfaces):
                if not interface.valid: continue
                weight = get_interface_weight(interface, record, interface_clusters, self.beta_interface, self.alpha_prot, self.alpha_nucl, self.alpha_ligand)
                target_items.append((record, 1, int_id))
                target_weights.append(weight)

        # 3. Sampling Logic
        if self.equity_glycan_training:
            def sample_group(pool_items, pool_weights, size):
                if not pool_items: return []
                weights = np.array(pool_weights)
                p = weights / np.sum(weights)
                indices = random.choice(len(pool_items), size=size, p=p, replace=True)
                return [pool_items[i] for i in indices]

            while True:
                chunk = []
                chunk.extend(sample_group(ser_items, ser_weights, 750))
                chunk.extend(sample_group(thr_items, thr_weights, 750))
                chunk.extend(sample_group(other_glyco_items, other_glyco_weights, 750))
                chunk.extend(sample_group(lectin_items, lectin_weights, 1000))
                chunk.extend(sample_group(eq_free_glycan_items, eq_free_glycan_weights, 250))
                chunk.extend(sample_group(eq_monosaccharide_items, eq_monosaccharide_weights, 1000))
                if not chunk: return
                random.shuffle(chunk)
                for rec, kind, idx in chunk:
                    yield Sample(record=rec, chain_id=idx if kind == 0 else None, interface_id=idx if kind == 1 else None)

        elif (self.glycan_context_prob is not None) and contextual_items and free_glycan_items:
            cw, fw = np.array(contextual_weights), np.array(free_glycan_weights)
            cp, fp = cw / np.sum(cw), fw / np.sum(fw)
            while True:
                pool = contextual_items if random.rand() < self.glycan_context_prob else free_glycan_items
                probs = cp if pool is contextual_items else fp
                rec, kind, idx = pool[random.choice(len(pool), p=probs)]
                yield Sample(record=rec, chain_id=idx if kind == 0 else None, interface_id=idx if kind == 1 else None)
        else:
            all_items = contextual_items + free_glycan_items + standard_other_items
            all_weights = np.array(contextual_weights + free_glycan_weights + standard_other_weights)
            p = all_weights / np.sum(all_weights)
            while True:
                rec, kind, idx = all_items[random.choice(len(all_items), p=p)]
                yield Sample(record=rec, chain_id=idx if kind == 0 else None, interface_id=idx if kind == 1 else None)
