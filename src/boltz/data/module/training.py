from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from boltz.data.crop.cropper import Cropper
from boltz.data.feature.featurizer import BoltzFeaturizer
from boltz.data.feature.pad import pad_to_max
from boltz.data.feature.symmetry import get_symmetries
from boltz.data.filter.dynamic.filter import DynamicFilter
from boltz.data.sample.sampler import Sample, Sampler
from boltz.data.tokenize.tokenizer import Tokenizer
from boltz.data.types import MSA, Connection, Input, Manifest, Record, Structure


@dataclass
class DatasetConfig:
    """Dataset configuration."""

    target_dir: str
    msa_dir: str
    prob: float
    sampler: Sampler
    cropper: Cropper
    filters: Optional[list] = None
    split: Optional[str] = None
    manifest_path: Optional[str] = None


@dataclass
class DataConfig:
    """Data configuration."""

    datasets: list[DatasetConfig]
    filters: list[DynamicFilter]
    featurizer: BoltzFeaturizer
    tokenizer: Tokenizer
    max_atoms: int
    max_tokens: int
    max_seqs: int
    samples_per_epoch: int
    batch_size: int
    num_workers: int
    random_seed: int
    pin_memory: bool
    symmetries: str
    atoms_per_window_queries: int
    min_dist: float
    max_dist: float
    num_bins: int
    overfit: Optional[int] = None
    pad_to_max_tokens: bool = False
    pad_to_max_atoms: bool = False
    pad_to_max_seqs: bool = False
    crop_validation: bool = False
    return_train_symmetries: bool = False
    return_val_symmetries: bool = True
    train_binder_pocket_conditioned_prop: float = 0.0
    val_binder_pocket_conditioned_prop: float = 0.0
    binder_pocket_cutoff: float = 6.0
    binder_pocket_sampling_geometric_p: float = 0.0
    val_batch_size: int = 1


@dataclass
class Dataset:
    """Data holder."""

    target_dir: Path
    msa_dir: Path
    manifest: Manifest
    prob: float
    sampler: Sampler
    cropper: Cropper
    tokenizer: Tokenizer
    featurizer: BoltzFeaturizer


def load_input(record: Record, target_dir: Path, msa_dir: Path) -> Input:
    """Load the given input data, including glycan features if present."""
    # Load the structure data, allowing for pickled glycan map objects
    npz_path = target_dir / "structures" / f"{record.id}.npz"
    structure_data = np.load(npz_path, allow_pickle=True)

    # Extract glycan maps if they exist. They are stored as 0-d arrays of objects.
    glycan_feature_map_raw = structure_data.get("glycan_feature_map", None)
    glycan_feature_map = (
        glycan_feature_map_raw.item()
        if glycan_feature_map_raw is not None and glycan_feature_map_raw.shape == ()
        else None
    )

    atom_to_mono_idx_map_raw = structure_data.get("atom_to_mono_idx_map", None)
    atom_to_mono_idx_map = (
        atom_to_mono_idx_map_raw.item()
        if atom_to_mono_idx_map_raw is not None and atom_to_mono_idx_map_raw.shape == ()
        else None
    )

    glycosylation_sites_raw = structure_data.get("glycosylation_sites", None)
    glycosylation_sites = (
        None
        if glycosylation_sites_raw is None or (
            hasattr(glycosylation_sites_raw, 'shape') and glycosylation_sites_raw.shape == ()
        )
        else glycosylation_sites_raw
    )

    # Create the Structure object.
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

    # Load MSAs
    msas = {}
    for chain in record.chains:
        msa_id = chain.msa_id
        if msa_id != -1 and msa_id != "":
            msa = np.load(msa_dir / f"{msa_id}.npz")
            msas[chain.chain_id] = MSA(**msa)

    return Input(structure, msas)


def collate(data: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Collate the data, safely filtering out failed (None) samples."""
    
    # Filter out any samples that failed in __getitem__ and returned None.
    valid_data = [d for d in data if d is not None]

    if not valid_data:
        return {}

    all_keys = set().union(*[d.keys() for d in valid_data])
    keys = sorted(list(all_keys))

    collated = {}
    for key in keys:
        values = [d.get(key) for d in valid_data]

        if key not in [
            "raw_glycosylation_sites",
            "record_id",
            "all_coords",
            "all_resolved_mask",
            "crop_to_all_atom_map",
            "chain_symmetries",
            "amino_acids_symmetries",
            "ligand_symmetries",
        ]:
            tensor_values = [v for v in values if v is not None]
            if not tensor_values: continue
            
            # Check if all tensors in the batch have the same shape.
            first_shape = tensor_values[0].shape
            is_ragged = any(v.shape != first_shape for v in tensor_values[1:])

            if is_ragged:
                try:
                    # FIX: pad_to_max returns the stacked tensor directly.
                    # Do not call torch.stack() on the result.
                    values, _ = pad_to_max(tensor_values, 0)
                except Exception:
                    # Fallback to simple stacking (will likely fail if ragged, but preserves original error flow)
                    values = torch.stack(tensor_values, dim=0)
            else:
                values = torch.stack(tensor_values, dim=0)
        
        collated[key] = values
    
    return collated
    
class TrainingDataset(torch.utils.data.Dataset):
    """Base iterable dataset."""

    def __init__(
        self,
        datasets: list[Dataset],
        samples_per_epoch: int,
        symmetries: dict,
        max_atoms: int,
        max_tokens: int,
        max_seqs: int,
        pad_to_max_atoms: bool = False,
        pad_to_max_tokens: bool = False,
        pad_to_max_seqs: bool = False,
        atoms_per_window_queries: int = 32,
        min_dist: float = 2.0,
        max_dist: float = 22.0,
        num_bins: int = 64,
        overfit: Optional[int] = None,
        binder_pocket_conditioned_prop: Optional[float] = 0.0,
        binder_pocket_cutoff: Optional[float] = 6.0,
        binder_pocket_sampling_geometric_p: Optional[float] = 0.0,
        return_symmetries: Optional[bool] = False,
    ) -> None:
        """Initialize the training dataset."""
        super().__init__()
        self.datasets = datasets
        self.probs = [d.prob for d in datasets]
        self.samples_per_epoch = samples_per_epoch
        self.symmetries = symmetries
        self.max_tokens = max_tokens
        self.max_seqs = max_seqs
        self.max_atoms = max_atoms
        self.pad_to_max_tokens = pad_to_max_tokens
        self.pad_to_max_atoms = pad_to_max_atoms
        self.pad_to_max_seqs = pad_to_max_seqs
        self.atoms_per_window_queries = atoms_per_window_queries
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.num_bins = num_bins
        self.binder_pocket_conditioned_prop = binder_pocket_conditioned_prop
        self.binder_pocket_cutoff = binder_pocket_cutoff
        self.binder_pocket_sampling_geometric_p = binder_pocket_sampling_geometric_p
        self.return_symmetries = return_symmetries
        self.samples = []
        for dataset in datasets:
            records = dataset.manifest.records
            if overfit is not None:
                records = records[:overfit]
            iterator = dataset.sampler.sample(records, np.random)
            self.samples.append(iterator)

    def __getitem__(self, idx: int) -> Optional[dict[str, Tensor]]:
        """Get an item from the dataset, now with robust error handling and safeguards."""
        try:
            # Pick a random dataset
            dataset_idx = np.random.choice(
                len(self.datasets),
                p=self.probs,
            )
            dataset = self.datasets[dataset_idx]

            # Get a sample from the dataset
            sample: Sample = next(self.samples[dataset_idx])

            # Get the structure
            input_data = load_input(sample.record, dataset.target_dir, dataset.msa_dir)

            # Tokenize structure
            tokenized = dataset.tokenizer.tokenize(input_data)

            # Compute crop
            if self.max_tokens is not None:
                tokenized = dataset.cropper.crop(
                    tokenized,
                    max_atoms=self.max_atoms,
                    max_tokens=self.max_tokens,
                    random=np.random,
                    chain_id=sample.chain_id,
                    interface_id=sample.interface_id,
                    record_id=sample.record.id,
                )

            # Check if there are tokens
            if len(tokenized.tokens) == 0:
                # This is a valid skip, not an error.
                return self.__getitem__(idx) # Retry instead of returning None

            # Compute features
            features = dataset.featurizer.process(
                tokenized,
                training=True,
                max_atoms=self.max_atoms if self.pad_to_max_atoms else None,
                max_tokens=self.max_tokens if self.pad_to_max_tokens else None,
                max_seqs=self.max_seqs,
                pad_to_max_seqs=self.pad_to_max_seqs,
                symmetries=self.symmetries,
                atoms_per_window_queries=self.atoms_per_window_queries,
                min_dist=self.min_dist,
                max_dist=self.max_dist,
                num_bins=self.num_bins,
                compute_symmetries=self.return_symmetries,
                binder_pocket_conditioned_prop=self.binder_pocket_conditioned_prop,
                binder_pocket_cutoff=self.binder_pocket_cutoff,
                binder_pocket_sampling_geometric_p=self.binder_pocket_sampling_geometric_p,
            )
            
            features["record_id"] = sample.record.id

            # ==========================================
            # GPU SHAPE SAFEGUARD (Prevents Model Crash)
            # ==========================================
            total_atoms = features["atom_pad_mask"].shape[0]
            if total_atoms % self.atoms_per_window_queries != 0:
                raise ValueError(f"Atom count {total_atoms} is not divisible by window size {self.atoms_per_window_queries}!")
            if self.max_atoms is not None and total_atoms > self.max_atoms:
                raise ValueError(f"Atom count {total_atoms} exceeds max_atoms {self.max_atoms}!")

            return features

        except MemoryError:
            record_id = "Unknown (OOM before sample selection)"
            try:
                record_id = sample.record.id
            except NameError:
                pass
            
            print("\n" + "="*80, flush=True)
            print(f"CRITICAL DATALOADER WARNING: CPU OUT OF MEMORY", flush=True)
            print(f"  - Record ID causing failure: {record_id}", flush=True)
            print(f"  - This is likely due to a large, uncropped structure.", flush=True)
            print(f"  - ACTION: Skipping sample and fetching a new one.", flush=True)
            print("="*80 + "\n", flush=True)
            
            return self.__getitem__(idx) # Retry

        except Exception as e:
            record_id = "Unknown (error before sample selection)"
            try:
                record_id = sample.record.id
            except NameError:
                pass
            
            print(f"\n--- DATALOADER WORKER ERROR ---", flush=True)
            print(f"WARNING: An unexpected error occurred while processing record '{record_id}'.", flush=True)
            print(f"  - Error Type: {type(e).__name__}", flush=True)
            print(f"  - Error Message: {e}", flush=True)
            print(f"  - ACTION: Skipping sample and fetching a new one to prevent crash.", flush=True)
            print(f"-----------------------------\n", flush=True)
            
            return self.__getitem__(idx) # Retry

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.

        """
        return self.samples_per_epoch


class ValidationDataset(torch.utils.data.Dataset):
    """Base iterable dataset."""

    def __init__(
        self,
        datasets: list[Dataset],
        seed: int,
        symmetries: dict,
        max_atoms: Optional[int] = None,
        max_tokens: Optional[int] = None,
        max_seqs: Optional[int] = None,
        pad_to_max_atoms: bool = False,
        pad_to_max_tokens: bool = False,
        pad_to_max_seqs: bool = False,
        atoms_per_window_queries: int = 32,
        min_dist: float = 2.0,
        max_dist: float = 22.0,
        num_bins: int = 64,
        overfit: Optional[int] = None,
        crop_validation: bool = False,
        return_symmetries: Optional[bool] = False,
        binder_pocket_conditioned_prop: Optional[float] = 0.0,
        binder_pocket_cutoff: Optional[float] = 6.0,
    ) -> None:
        """Initialize the validation dataset."""
        super().__init__()
        self.datasets = datasets
        self.max_atoms = max_atoms
        self.max_tokens = max_tokens
        self.max_seqs = max_seqs
        self.seed = seed
        self.symmetries = symmetries
        self.random = np.random if overfit else np.random.RandomState(self.seed)
        self.pad_to_max_tokens = pad_to_max_tokens
        self.pad_to_max_atoms = pad_to_max_atoms
        self.pad_to_max_seqs = pad_to_max_seqs
        self.overfit = overfit
        self.crop_validation = crop_validation
        self.atoms_per_window_queries = atoms_per_window_queries
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.num_bins = num_bins
        self.return_symmetries = return_symmetries
        self.binder_pocket_conditioned_prop = binder_pocket_conditioned_prop
        self.binder_pocket_cutoff = binder_pocket_cutoff

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Get an item from the dataset."""
        # Pick dataset based on idx
        for dataset in self.datasets:
            size = len(dataset.manifest.records)
            if self.overfit is not None:
                size = min(size, self.overfit)
            if idx < size:
                break
            idx -= size

        # Get a sample from the dataset
        record = dataset.manifest.records[idx]

        # Get the structure
        try:
            input_data = load_input(record, dataset.target_dir, dataset.msa_dir)
        except Exception as e:
            print(f"Failed to load input for {record.id} with error {e}. Skipping.")
            return self.__getitem__(0)

        # Tokenize structure
        try:
            tokenized = dataset.tokenizer.tokenize(input_data)
        except Exception as e:
            print(f"Tokenizer failed on {record.id} with error {e}. Skipping.")
            return self.__getitem__(0)

        # Compute crop
        try:
            if self.crop_validation and (self.max_tokens is not None):
                tokenized = dataset.cropper.crop(
                    tokenized,
                    max_tokens=self.max_tokens,
                    random=self.random,
                    max_atoms=self.max_atoms,
                )
        except Exception as e:
            print(f"Cropper failed on {record.id} with error {e}. Skipping.")
            return self.__getitem__(0)

        # Check if there are tokens
        if len(tokenized.tokens) == 0:
            print("\n" + "="*80, flush=True)
            print(f"[DEBUG] ZERO-TOKEN ERROR FOR VALIDATION RECORD: {record.id}", flush=True)
            print(f"[DEBUG] File Path Hint: {dataset.target_dir / 'structures' / f'{record.id}.npz'}", flush=True)
            print(f"[DEBUG] 'crop_validation' setting in use: {self.crop_validation}", flush=True)
            print(f"[DEBUG] The 'mask' loaded from the npz file is: {input_data.structure.mask}", flush=True)
            print(f"[DEBUG] Number of chains before masking: {len(input_data.structure.chains)}", flush=True)
            print("="*80 + "\n", flush=True)
            return self.__getitem__(0) # Gracefully retry instead of crashing

        # Compute features
        try:
            pad_atoms = self.crop_validation and self.pad_to_max_atoms
            pad_tokens = self.crop_validation and self.pad_to_max_tokens

            features = dataset.featurizer.process(
                tokenized,
                training=False,
                max_atoms=self.max_atoms if pad_atoms else None,
                max_tokens=self.max_tokens if pad_tokens else None,
                max_seqs=self.max_seqs,
                pad_to_max_seqs=self.pad_to_max_seqs,
                symmetries=self.symmetries,
                atoms_per_window_queries=self.atoms_per_window_queries,
                min_dist=self.min_dist,
                max_dist=self.max_dist,
                num_bins=self.num_bins,
                compute_symmetries=self.return_symmetries,
                binder_pocket_conditioned_prop=self.binder_pocket_conditioned_prop,
                binder_pocket_cutoff=self.binder_pocket_cutoff,
                binder_pocket_sampling_geometric_p=1.0,  # this will only sample a single pocket token
                only_ligand_binder_pocket=True,
            )
            
            # ==========================================
            # GPU SHAPE SAFEGUARD (Prevents Model Crash)
            # ==========================================
            total_atoms = features["atom_pad_mask"].shape[0]
            if total_atoms % self.atoms_per_window_queries != 0:
                raise ValueError(f"Atom count {total_atoms} is not divisible by window size {self.atoms_per_window_queries}!")
            if self.max_atoms is not None and self.crop_validation and total_atoms > self.max_atoms:
                raise ValueError(f"Atom count {total_atoms} exceeds max_atoms {self.max_atoms}!")

        except Exception as e:
            print(f"Featurizer or Shape Verification failed on {record.id} with error {e}. Skipping.")
            return self.__getitem__(0)

        return features
        
    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.

        """
        if self.overfit is not None:
            length = sum(len(d.manifest.records[: self.overfit]) for d in self.datasets)
        else:
            length = sum(len(d.manifest.records) for d in self.datasets)

        return length


class BoltzTrainingDataModule(pl.LightningDataModule):
    """DataModule for boltz."""

    def __init__(self, cfg: DataConfig) -> None:
        """Initialize the DataModule.

        Parameters
        ----------
        config : DataConfig
            The data configuration.

        """
        super().__init__()
        self.cfg = cfg

        assert self.cfg.val_batch_size == 1, "Validation only works with batch size=1."

        # Load symmetries
        symmetries = get_symmetries(cfg.symmetries)

        # Load datasets
        train: list[Dataset] = []
        val: list[Dataset] = []

        for data_config in cfg.datasets:
            # Set target_dir
            target_dir = Path(data_config.target_dir)
            msa_dir = Path(data_config.msa_dir)

            # Load manifest
            if data_config.manifest_path is not None:
                path = Path(data_config.manifest_path)
            else:
                path = target_dir / "manifest.json"
            manifest: Manifest = Manifest.load(path)

            # Split records if given
            if data_config.split is not None:
                with Path(data_config.split).open("r") as f:
                    split = {x.lower() for x in f.read().splitlines()}

                train_records = []
                val_records = []
                for record in manifest.records:
                    if record.id.lower() in split:
                        val_records.append(record)
                    else:
                        train_records.append(record)
            else:
                train_records = manifest.records
                val_records = []

            # Filter training records
            train_records = [
                record
                for record in train_records
                if all(f.filter(record) for f in cfg.filters)
            ]
            # Filter training records
            if data_config.filters is not None:
                train_records = [
                    record
                    for record in train_records
                    if all(f.filter(record) for f in data_config.filters)
                ]

            # Create train dataset
            train_manifest = Manifest(train_records)
            train.append(
                Dataset(
                    target_dir,
                    msa_dir,
                    train_manifest,
                    data_config.prob,
                    data_config.sampler,
                    data_config.cropper,
                    cfg.tokenizer,
                    cfg.featurizer,
                )
            )

            # Create validation dataset
            if val_records:
                val_manifest = Manifest(val_records)
                val.append(
                    Dataset(
                        target_dir,
                        msa_dir,
                        val_manifest,
                        data_config.prob,
                        data_config.sampler,
                        data_config.cropper,
                        cfg.tokenizer,
                        cfg.featurizer,
                    )
                )

        # Print dataset sizes
        for dataset in train:
            dataset: Dataset
            print(f"Training dataset size: {len(dataset.manifest.records)}")

        for dataset in val:
            dataset: Dataset
            print(f"Validation dataset size: {len(dataset.manifest.records)}")

        # Create wrapper datasets
        self._train_set = TrainingDataset(
            datasets=train,
            samples_per_epoch=cfg.samples_per_epoch,
            max_atoms=cfg.max_atoms,
            max_tokens=cfg.max_tokens,
            max_seqs=cfg.max_seqs,
            pad_to_max_atoms=cfg.pad_to_max_atoms,
            pad_to_max_tokens=cfg.pad_to_max_tokens,
            pad_to_max_seqs=cfg.pad_to_max_seqs,
            symmetries=symmetries,
            atoms_per_window_queries=cfg.atoms_per_window_queries,
            min_dist=cfg.min_dist,
            max_dist=cfg.max_dist,
            num_bins=cfg.num_bins,
            overfit=cfg.overfit,
            binder_pocket_conditioned_prop=cfg.train_binder_pocket_conditioned_prop,
            binder_pocket_cutoff=cfg.binder_pocket_cutoff,
            binder_pocket_sampling_geometric_p=cfg.binder_pocket_sampling_geometric_p,
            return_symmetries=cfg.return_train_symmetries,
        )
        self._val_set = ValidationDataset(
            datasets=train if cfg.overfit is not None else val,
            seed=cfg.random_seed,
            max_atoms=cfg.max_atoms,
            max_tokens=cfg.max_tokens,
            max_seqs=cfg.max_seqs,
            pad_to_max_atoms=cfg.pad_to_max_atoms,
            pad_to_max_tokens=cfg.pad_to_max_tokens,
            pad_to_max_seqs=cfg.pad_to_max_seqs,
            symmetries=symmetries,
            atoms_per_window_queries=cfg.atoms_per_window_queries,
            min_dist=cfg.min_dist,
            max_dist=cfg.max_dist,
            num_bins=cfg.num_bins,
            overfit=cfg.overfit,
            crop_validation=cfg.crop_validation,
            return_symmetries=cfg.return_val_symmetries,
            binder_pocket_conditioned_prop=cfg.val_binder_pocket_conditioned_prop,
            binder_pocket_cutoff=cfg.binder_pocket_cutoff,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Run the setup for the DataModule.

        Parameters
        ----------
        stage : str, optional
            The stage, one of 'fit', 'validate', 'test'.

        """
        return

    def train_dataloader(self) -> DataLoader:
        """Get the training dataloader.

        Returns
        -------
        DataLoader
            The training dataloader.

        """
        return DataLoader(
            self._train_set,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            shuffle=False,
            collate_fn=collate,
        )

    def val_dataloader(self) -> DataLoader:
        """Get the validation dataloader.

        Returns
        -------
        DataLoader
            The validation dataloader.

        """
        return DataLoader(
            self._val_set,
            batch_size=self.cfg.val_batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            shuffle=False,
            collate_fn=collate,
        )
