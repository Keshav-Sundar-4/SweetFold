from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from boltz.data import const
from boltz.data.feature.featurizer import BoltzFeaturizer
from boltz.data.feature.pad import pad_to_max
from boltz.data.tokenize.boltz import BoltzTokenizer
from boltz.data.types import (
    MSA,
    Connection,
    Input,
    Manifest,
    Record,
    ResidueConstraints,
    Structure,
    GlycosylationSite,
)

def load_input(
    record: Record,
    target_dir: Path,
    msa_dir: Path,
    constraints_dir: Optional[Path] = None,
) -> Input:
    # Load the structure NPZ, allowing pickle for dictionary-like objects
    structure_data = np.load(target_dir / f"{record.id}.npz", allow_pickle=True)

    # Extract glycan maps if they exist
    glycan_feature_map_raw = structure_data.get('glycan_feature_map')
    glycan_feature_map = None
    if glycan_feature_map_raw is not None and glycan_feature_map_raw.shape == ():
        glycan_feature_map = glycan_feature_map_raw.item()

    atom_to_mono_idx_map_raw = structure_data.get('atom_to_mono_idx_map')
    atom_to_mono_idx_map = None
    if atom_to_mono_idx_map_raw is not None and atom_to_mono_idx_map_raw.shape == ():
        atom_to_mono_idx_map = atom_to_mono_idx_map_raw.item()

    # Handle glycosylation sites
    glyco_raw = structure_data.get('glycosylation_sites', None)
    glycosylation_sites = None

    if glyco_raw is not None:
        # Check if it is a 0-d array containing None (result of np.savez with None)
        if glyco_raw.shape == () and glyco_raw.item() is None:
            glycosylation_sites = None
        # Check if it's explicitly empty
        elif glyco_raw.size == 0:
            glycosylation_sites = None
        else:
            glycosylation_sites = glyco_raw

    # Reconstruct the Structure object
    structure = Structure(
        atoms=structure_data["atoms"],
        bonds=structure_data["bonds"],
        residues=structure_data["residues"],
        chains=structure_data["chains"],
        connections=structure_data["connections"].astype(Connection),
        interfaces=structure_data["interfaces"],
        mask=structure_data["mask"],
        glycan_feature_map=glycan_feature_map,
        atom_to_mono_idx_map=atom_to_mono_idx_map,
        glycosylation_sites=glycosylation_sites,  
    )

    msas = {}
    for chain in record.chains:
        if (msa_id := chain.msa_id) != -1:
            try:
                msa = np.load(msa_dir / f"{msa_id}.npz")
                msas[chain.chain_id] = MSA(**msa)
            except FileNotFoundError:
                pass

    residue_constraints = None
    if constraints_dir is not None:
        try:
            residue_constraints = ResidueConstraints.load(
                constraints_dir / f"{record.id}.npz"
            )
        except FileNotFoundError:
            pass

    return Input(structure, msas, residue_constraints=residue_constraints)

def collate(data: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Collate the data.

    Parameters
    ----------
    data : List[Dict[str, Tensor]]
        The data to collate.

    Returns
    -------
    Dict[str, Tensor]
        The collated data.

    """
    # Get the keys
    keys = data[0].keys()

    # Collate the data
    collated = {}
    for key in keys:
        values = [d[key] for d in data]

        if key not in [
            "all_coords",
            "all_resolved_mask",
            "crop_to_all_atom_map",
            "chain_symmetries",
            "amino_acids_symmetries",
            "ligand_symmetries",
            "record",
        ]:
            # Check if all have the same shape
            shape = values[0].shape
            if not all(v.shape == shape for v in values):
                values, _ = pad_to_max(values, 0)
            else:
                values = torch.stack(values, dim=0)

        # Stack the values
        collated[key] = values

    return collated


class PredictionDataset(torch.utils.data.Dataset):
    """Base iterable dataset for prediction."""

    def __init__(
        self,
        manifest: Manifest,
        target_dir: Path,
        msa_dir: Path,
        constraints_dir: Optional[Path] = None,
    ) -> None:
        """Initialize the prediction dataset.

        Parameters
        ----------
        manifest : Manifest
            The manifest to load data from.
        target_dir : Path
            The path to the target directory.
        msa_dir : Path
            The path to the msa directory.
        constraints_dir : Optional[Path], optional
            The path to the residue constraints directory, by default None.
        """
        super().__init__()
        self.manifest = manifest
        self.target_dir = target_dir
        self.msa_dir = msa_dir
        self.constraints_dir = constraints_dir
        self.tokenizer = BoltzTokenizer()
        self.featurizer = BoltzFeaturizer()

    def __getitem__(self, idx: int) -> dict:
        """Get an item from the dataset.

        Returns
        -------
        Dict[str, Tensor]
            The sampled data features.
        """
        record = self.manifest.records[idx]
        try:
            input_data = load_input(
                record,
                self.target_dir,
                self.msa_dir,
                self.constraints_dir,
            )
            tokenized = self.tokenizer.tokenize(input_data)

            options = record.inference_options
            binders, pocket = (options.binders, options.pocket) if options else (None, None)

            features = self.featurizer.process(
                tokenized,
                training=False,
                max_atoms=None,
                max_tokens=None,
                max_seqs=const.max_msa_seqs,
                pad_to_max_seqs=False,
                symmetries={},
                compute_symmetries=False,
                inference_binder=binders,
                inference_pocket=pocket,
                compute_constraint_features=True, # Preserved from Boltz-1x
                compute_glycan_features=True,     # Added for glycan logic
            )
        except Exception as e:
            print(f"Processing failed for {record.id} with error: {e}. Skipping.")
            # Safely skip to the next item on error
            return self.__getitem__((idx + 1) % len(self))

        features["record"] = record
        return features

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.manifest.records)



class BoltzInferenceDataModule(pl.LightningDataModule):
    """DataModule for Boltz inference."""

    def __init__(
        self,
        manifest: Manifest,
        target_dir: Path,
        msa_dir: Path,
        num_workers: int,
        constraints_dir: Optional[Path] = None,
    ) -> None:
        """Initialize the DataModule.

        Parameters
        ----------
        config : DataConfig
            The data configuration.

        """
        super().__init__()
        self.num_workers = num_workers
        self.manifest = manifest
        self.target_dir = target_dir
        self.msa_dir = msa_dir
        self.constraints_dir = constraints_dir

    def predict_dataloader(self) -> DataLoader:
        """Get the training dataloader.

        Returns
        -------
        DataLoader
            The training dataloader.

        """
        dataset = PredictionDataset(
            manifest=self.manifest,
            target_dir=self.target_dir,
            msa_dir=self.msa_dir,
            constraints_dir=self.constraints_dir,
        )
        return DataLoader(
            dataset,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=collate,
        )

    def transfer_batch_to_device(
        self,
        batch: dict,
        device: torch.device,
        dataloader_idx: int,  # noqa: ARG002
    ) -> dict:
        """Transfer a batch to the given device.

        Parameters
        ----------
        batch : Dict
            The batch to transfer.
        device : torch.device
            The device to transfer to.
        dataloader_idx : int
            The dataloader index.

        Returns
        -------
        np.Any
            The transferred batch.

        """
        for key in batch:
            if key not in [
                "all_coords",
                "all_resolved_mask",
                "crop_to_all_atom_map",
                "chain_symmetries",
                "amino_acids_symmetries",
                "ligand_symmetries",
                "record",
            ]:
                batch[key] = batch[key].to(device)
        return batch
