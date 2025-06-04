from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any # Added Any
import logging # Added logging

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
# Added NOESY imports
from boltz.data.noesy.parser import NOESYParser
from boltz.data.noesy.feature import NOESYFeature


logger = logging.getLogger(__name__) # Added logger


@dataclass
class DatasetConfig:
    """Dataset configuration."""

    target_dir: str
    msa_dir: str
    prob: float
    sampler: Sampler
    cropper: Cropper
    filters: Optional[list[Any]] = None # Assuming list contains various filter types
    split: Optional[str] = None
    manifest_path: Optional[str] = None
    noesy_dir: Optional[str] = None  # Per-dataset NOESY directory


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
    # NOESY related global config
    noesy_dir: Optional[str] = None
    noesy_num_bins: int = 64
    noesy_min_dist: float = 0.0
    noesy_max_dist: float = 5.0
    noesy_noise_threshold: Optional[float] = None
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
    noesy_dir: Optional[Path] = None # Effective NOESY dir for this dataset instance


def load_input(record: Record, target_dir: Path, msa_dir: Path) -> Input:
    """Load the given input data.

    Parameters
    ----------
    record : Record
        The record to load.
    target_dir : Path
        The path to the data directory.
    msa_dir : Path
        The path to msa directory.

    Returns
    -------
    Input
        The loaded input.

    """
    # Load the structure
    structure = np.load(target_dir / "structures" / f"{record.id}.npz")

    # In order to add cyclic_period to chains if it does not exist
    # Extract the chains array
    chains = structure["chains"]
    # Check if the field exists
    if "cyclic_period" not in chains.dtype.names:
        # Create a new dtype with the additional field
        new_dtype = chains.dtype.descr + [("cyclic_period", "i4")]
        # Create a new array with the new dtype
        new_chains = np.empty(chains.shape, dtype=new_dtype)
        # Copy over existing fields
        for name in chains.dtype.names:
            new_chains[name] = chains[name]
        # Set the new field to 0
        new_chains["cyclic_period"] = 0
        # Replace old chains array with new one
        chains = new_chains

    structure = Structure(
        atoms=structure["atoms"],
        bonds=structure["bonds"],
        residues=structure["residues"],
        chains=chains, # chains var accounting for missing cyclic_period
        connections=structure["connections"].astype(Connection),
        interfaces=structure["interfaces"],
        mask=structure["mask"],
    )

    msas = {}
    for chain in record.chains:
        msa_id = chain.msa_id
        # Load the MSA for this chain, if any
        if msa_id != -1 and msa_id != "":
            msa = np.load(msa_dir / f"{msa_id}.npz")
            msas[chain.chain_id] = MSA(**msa)

    return Input(structure, msas)


def collate(data: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Collate the data.

    Parameters
    ----------
    data : list[dict[str, Tensor]]
        The data to collate.

    Returns
    -------
    dict[str, Tensor]
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


class TrainingDataset(torch.utils.data.Dataset):
    """Base iterable dataset."""

    def __init__(
        self,
        datasets: list[Dataset], # This is BoltzTrainingDataModule.Dataset
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
        compute_constraint_features: bool = False,
        # NOESY related args passed from BoltzTrainingDataModule
        noesy_parser: Optional[NOESYParser] = None,
        noesy_featurizer: Optional[NOESYFeature] = None,
        noesy_num_bins: int = 64,
    ) -> None:
        """Initialize the training dataset."""
        super().__init__()
        self.datasets = datasets # List of BoltzTrainingDataModule.Dataset objects
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
        self.compute_constraint_features = compute_constraint_features

        # Store NOESY related objects
        self.noesy_parser = noesy_parser
        self.noesy_featurizer = noesy_featurizer
        # noesy_num_bins is used for placeholder if featurizer is None but NOESY is globally expected
        self.noesy_num_bins = noesy_num_bins

        self.samples = []
        for dataset_obj in datasets: # dataset_obj is an instance of BoltzTrainingDataModule.Dataset
            records = dataset_obj.manifest.records
            if overfit is not None:
                records = records[:overfit]
            iterator = dataset_obj.sampler.sample(records, np.random) # Use dataset_obj
            self.samples.append(iterator)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Get an item from the dataset.

        Parameters
        ----------
        idx : int
            The data index.

        Returns
        -------
        dict[str, Tensor]
            The sampled data features.

        """
        # Pick a random dataset
        dataset_idx = np.random.choice(
            len(self.datasets),
            p=self.probs,
        )
        current_dataset = self.datasets[dataset_idx] # current_dataset is a BoltzTrainingDataModule.Dataset instance

        # Get a sample from the dataset
        sample: Sample = next(self.samples[dataset_idx])

        # Get the structure
        try:
            input_data = load_input(sample.record, current_dataset.target_dir, current_dataset.msa_dir)
        except Exception as e:
            logger.error( # Use logger
                f"Failed to load input for {sample.record.id} with error {e}. Skipping."
            )
            return self.__getitem__(idx)

        # Tokenize structure
        try:
            tokenized = current_dataset.tokenizer.tokenize(input_data)
        except Exception as e:
            logger.error(f"Tokenizer failed on {sample.record.id} with error {e}. Skipping.") # Use logger
            return self.__getitem__(idx)

        # Compute crop
        try:
            if self.max_tokens is not None:
                tokenized = current_dataset.cropper.crop( # Use current_dataset
                    tokenized,
                    max_atoms=self.max_atoms,
                    max_tokens=self.max_tokens,
                    random=np.random,
                    chain_id=sample.chain_id,
                    interface_id=sample.interface_id,
                )
        except Exception as e:
            logger.error(f"Cropper failed on {sample.record.id} with error {e}. Skipping.") # Use logger
            return self.__getitem__(idx)

        # Check if there are tokens
        if len(tokenized.tokens) == 0:
            # This case should ideally be filtered out by a prior filter step if possible
            logger.warning(f"No tokens in cropped structure for {sample.record.id}. Skipping.")
            return self.__getitem__(idx) # Skip sample

        # Compute features
        try:
            features = current_dataset.featurizer.process( # Use current_dataset
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
                compute_constraint_features=self.compute_constraint_features,
            )
        except Exception as e:
            logger.error(f"Featurizer failed on {sample.record.id} with error {e}. Skipping.") # Use logger
            return self.__getitem__(idx)

        # Add NOESY features
        # Determine sequence length from existing features (e.g., after cropping by main featurizer)
        # 'coords_mask' is a good candidate if it represents per-residue presence (L, )
        if "coords_mask" not in features:
            logger.error(f"Key 'coords_mask' not found in features for {sample.record.id}. Cannot determine sequence length for NOESY. Skipping NOESY.")
            # Ensure 'noesy_feat' still exists if expected by collate fn
            # Fallback sequence length or handle error appropriately
            # For now, if this happens, we can't create a meaningful placeholder of correct L.
            # This indicates a bigger issue if coords_mask is always expected.
            # However, the collate function might handle missing keys or None values.
            # For safety in batching, a zero-length or predefined small placeholder might be added,
            # or an error raised. Let's assume coords_mask will be present.
            # If not, the code below will fail.
            pass # Let it fail if coords_mask is missing, to highlight the issue.

        sequence_length = features["coords_mask"].shape[0]

        noesy_feat_tensor = torch.zeros((sequence_length, sequence_length, self.noesy_num_bins), dtype=torch.float32)

        if self.noesy_parser and self.noesy_featurizer: # Check if NOESY processing is globally enabled
            if current_dataset.noesy_dir: # Check if the current dataset has a NOESY dir configured
                target_id = sample.record.id
                # Assume NOESY filename convention: <pdb_id>_noesy.txt (e.g. 1abc_noesy.txt)
                # target_id might be "1ABC_A", so take the first part.
                noesy_filename_candidate = target_id.split('_')[0].lower() + "_noesy.txt"
                noesy_file_path = current_dataset.noesy_dir / noesy_filename_candidate

                if noesy_file_path.exists():
                    try:
                        parsed_noesy = self.noesy_parser.parse(str(noesy_file_path))
                        if parsed_noesy and parsed_noesy.get('peaks'): # Check if parsing yielded any peaks
                            noesy_feat_tensor = self.noesy_featurizer.featurize(parsed_noesy, sequence_length)
                        else:
                            logger.warning(f"NOESY file {noesy_file_path} parsed but no peaks found. Using placeholder.")
                    except Exception as e:
                        logger.error(f"NOESY processing failed for {noesy_file_path}: {e}. Using placeholder.")
                else:
                    logger.warning(f"NOESY file not found: {noesy_file_path}. Using placeholder.")
            # else: No NOESY dir for this specific dataset, use placeholder.
            # logger.debug implicitly handled by not entering the 'if current_dataset.noesy_dir:'
        # else: NOESY not configured at module level (parser/featurizer are None), so use placeholder.

        features['noesy_feat'] = noesy_feat_tensor
        return features

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
        datasets: list[Dataset], # This is BoltzTrainingDataModule.Dataset
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
        compute_constraint_features: bool = False,
        # NOESY related args passed from BoltzTrainingDataModule
        noesy_parser: Optional[NOESYParser] = None,
        noesy_featurizer: Optional[NOESYFeature] = None,
        noesy_num_bins: int = 64,
    ) -> None:
        """Initialize the validation dataset."""
        super().__init__()
        self.datasets = datasets # List of BoltzTrainingDataModule.Dataset objects
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
        self.compute_constraint_features = compute_constraint_features

        # Store NOESY related objects
        self.noesy_parser = noesy_parser
        self.noesy_featurizer = noesy_featurizer
        self.noesy_num_bins = noesy_num_bins

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Get an item from the dataset.

        Parameters
        ----------
        idx : int
            The data index.

        Returns
        -------
        dict[str, Tensor]
            The sampled data features.

        """
        # Pick dataset based on idx
        current_dataset_val = None
        temp_idx = idx # Use a temp variable for idx manipulation
        for ds_val_loop in self.datasets: # ds_val_loop is BoltzTrainingDataModule.Dataset
            size = len(ds_val_loop.manifest.records)
            if self.overfit is not None:
                size = min(size, self.overfit)
            if temp_idx < size:
                current_dataset_val = ds_val_loop
                idx = temp_idx # Update idx to be relative to the chosen dataset
                break
            temp_idx -= size

        if current_dataset_val is None:
            raise IndexError(f"ValidationDataset index {idx} out of bounds.")

        # Get a sample from the dataset
        record = current_dataset_val.manifest.records[idx]

        # Get the structure
        try:
            input_data = load_input(record, current_dataset_val.target_dir, current_dataset_val.msa_dir)
        except Exception as e:
            logger.error(f"Failed to load input for {record.id} with error {e}. Skipping.") # Use logger
            return self.__getitem__(0) # Risk of recursion depth, but follows existing pattern

        # Tokenize structure
        try:
            tokenized = current_dataset_val.tokenizer.tokenize(input_data)
        except Exception as e:
            logger.error(f"Tokenizer failed on {record.id} with error {e}. Skipping.") # Use logger
            return self.__getitem__(0)

        # Compute crop
        try:
            if self.crop_validation and (self.max_tokens is not None):
                tokenized = current_dataset_val.cropper.crop( # Use current_dataset_val
                    tokenized,
                    max_tokens=self.max_tokens,
                    random=self.random, # self.random should be used for val
                    max_atoms=self.max_atoms,
                )
        except Exception as e:
            logger.error(f"Cropper failed on {record.id} with error {e}. Skipping.") # Use logger
            return self.__getitem__(0)

        # Check if there are tokens
        if len(tokenized.tokens) == 0:
            logger.warning(f"No tokens in cropped structure for {record.id} (validation). Skipping.")
            return self.__getitem__(0) # Skip sample

        # Compute features
        try:
            pad_atoms = self.crop_validation and self.pad_to_max_atoms
            pad_tokens = self.crop_validation and self.pad_to_max_tokens

            features = current_dataset_val.featurizer.process( # Use current_dataset_val
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
                compute_constraint_features=self.compute_constraint_features,
            )
        except Exception as e:
            logger.error(f"Featurizer failed on {record.id} with error {e}. Skipping.") # Use logger
            return self.__getitem__(0)

        # Add NOESY features (similar to TrainingDataset)
        if "coords_mask" not in features:
             logger.error(f"Key 'coords_mask' not found in features for {record.id} (validation). Cannot determine sequence length for NOESY. Skipping NOESY.")
             pass # Let it fail if coords_mask is missing or handle as in training

        sequence_length = features["coords_mask"].shape[0]
        noesy_feat_tensor = torch.zeros((sequence_length, sequence_length, self.noesy_num_bins), dtype=torch.float32)

        if self.noesy_parser and self.noesy_featurizer:
            if current_dataset_val.noesy_dir:
                target_id = record.id
                noesy_filename_candidate = target_id.split('_')[0].lower() + "_noesy.txt"
                noesy_file_path = current_dataset_val.noesy_dir / noesy_filename_candidate

                if noesy_file_path.exists():
                    try:
                        parsed_noesy = self.noesy_parser.parse(str(noesy_file_path))
                        if parsed_noesy and parsed_noesy.get('peaks'):
                             noesy_feat_tensor = self.noesy_featurizer.featurize(parsed_noesy, sequence_length)
                        else:
                            logger.warning(f"NOESY file {noesy_file_path} parsed (validation) but no peaks found. Using placeholder.")
                    except Exception as e:
                        logger.error(f"NOESY processing failed for {noesy_file_path} (validation): {e}. Using placeholder.")
                else:
                    logger.warning(f"NOESY file not found: {noesy_file_path} (validation). Using placeholder.")

        features['noesy_feat'] = noesy_feat_tensor
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

        # Initialize NOESY parser and featurizer if configured
        self.noesy_parser: Optional[NOESYParser] = None
        self.noesy_featurizer: Optional[NOESYFeature] = None

        any_noesy_configured = False
        if self.cfg.noesy_dir: # Global config
            any_noesy_configured = True
        else:
            for dc_cfg in self.cfg.datasets: # Per-dataset config
                if dc_cfg.noesy_dir:
                    any_noesy_configured = True
                    break

        if any_noesy_configured:
            self.noesy_parser = NOESYParser() # Uses default __init__ args
            self.noesy_featurizer = NOESYFeature(
                num_bins=self.cfg.noesy_num_bins,
                min_dist=self.cfg.noesy_min_dist,
                max_dist=self.cfg.noesy_max_dist,
                noise_threshold=self.cfg.noesy_noise_threshold
            )
            logger.info("NOESY Parser and Featurizer initialized with num_bins=%d, dist_range=(%.1f, %.1f)",
                        self.cfg.noesy_num_bins, self.cfg.noesy_min_dist, self.cfg.noesy_max_dist)


        # Load datasets
        train: list[Dataset] = [] # This Dataset is BoltzTrainingDataModule.Dataset
        val: list[Dataset] = []   # This Dataset is BoltzTrainingDataModule.Dataset

        for dataset_conf in cfg.datasets: # dataset_conf is DatasetConfig from cfg
            # Set target_dir
            target_dir = Path(dataset_conf.target_dir)
            msa_dir = Path(dataset_conf.msa_dir)

            # Determine effective NOESY directory for this dataset
            effective_noesy_dir: Optional[Path] = None
            if dataset_conf.noesy_dir: # Per-dataset path takes precedence
                effective_noesy_dir = Path(dataset_conf.noesy_dir)
            elif cfg.noesy_dir: # Fallback to global path
                effective_noesy_dir = Path(cfg.noesy_dir)

            if effective_noesy_dir and not any_noesy_configured:
                # This case should ideally not be hit if any_noesy_configured is derived correctly
                logger.warning(f"Effective NOESY dir {effective_noesy_dir} found, but parser/featurizer not initialized.")


            # Load manifest
            if dataset_conf.manifest_path is not None:
                path = Path(dataset_conf.manifest_path)
            else:
                path = target_dir / "manifest.json"
            manifest: Manifest = Manifest.load(path)

            # Split records if given
            if dataset_conf.split is not None:
                with Path(dataset_conf.split).open("r") as f:
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
            # Filter training records based on per-dataset filters
            if dataset_conf.filters is not None:
                train_records = [
                    record
                    for record in train_records
                    if all(f.filter(record) for f in dataset_conf.filters)
                ]

            # Create train dataset
            train_manifest = Manifest(train_records)
            train.append(
                Dataset( # This is BoltzTrainingDataModule.Dataset dataclass
                    target_dir=target_dir,
                    msa_dir=msa_dir,
                    manifest=train_manifest,
                    prob=dataset_conf.prob,
                    sampler=dataset_conf.sampler,
                    cropper=dataset_conf.cropper,
                    tokenizer=cfg.tokenizer,
                    featurizer=cfg.featurizer,
                    noesy_dir=effective_noesy_dir # Pass effective NOESY dir
                )
            )

            # Create validation dataset
            if val_records:
                val_manifest = Manifest(val_records)
                val.append(
                    Dataset( # This is BoltzTrainingDataModule.Dataset dataclass
                        target_dir=target_dir,
                        msa_dir=msa_dir,
                        manifest=val_manifest,
                        # prob here might not be used if val set is just all val_records
                        prob=dataset_conf.prob,
                        sampler=dataset_conf.sampler, # Sampler might differ for val
                        cropper=dataset_conf.cropper, # Cropper might differ for val
                        tokenizer=cfg.tokenizer,
                        featurizer=cfg.featurizer,
                        noesy_dir=effective_noesy_dir # Pass effective NOESY dir
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
            datasets=train, # This is list of BoltzTrainingDataModule.Dataset
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
            # Pass NOESY related objects and parameters from BoltzTrainingDataModule
            noesy_parser=self.noesy_parser,
            noesy_featurizer=self.noesy_featurizer,
            noesy_num_bins=self.cfg.noesy_num_bins
        )
        self._val_set = ValidationDataset(
            datasets=train if cfg.overfit is not None else val, # This is list of BoltzTrainingDataModule.Dataset
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
            # Pass NOESY related objects and parameters from BoltzTrainingDataModule
            noesy_parser=self.noesy_parser,
            noesy_featurizer=self.noesy_featurizer,
            noesy_num_bins=self.cfg.noesy_num_bins
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
