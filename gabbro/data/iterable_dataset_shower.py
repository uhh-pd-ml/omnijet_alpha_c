import gc
import glob
import random
from typing import Optional

import awkward as ak
import lightning as L
import numpy as np
import torch
import torch.distributed as dist
import vector
from torch.distributed import get_rank, get_world_size
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from gabbro.data.loading import read_shower_file, read_tokenized_shower_file
from gabbro.utils.arrays import ak_pad, ak_padding, ak_preprocess, ak_to_np_stack
from gabbro.utils.pylogger import get_pylogger

vector.register_awkward()


class CustomIterableDataset(IterableDataset):
    """Custom IterableDataset that loads data from multiple files."""

    def __init__(
        self,
        files_dict: dict,
        n_files_at_once: int = None,
        n_shower_per_file: int = None,
        max_n_files_per_type: int = None,
        shuffle_files: bool = True,
        shuffle_data: bool = True,
        seed: int = 4697,
        seed_shuffle_data: int = 3838,
        pad_length: int = 1700,
        logger_name: str = "CustomIterableDataset",
        feature_dict: dict = None,
        labels_to_load: list = None,
        token_reco_cfg: dict = None,
        token_id_cfg: dict = None,
        load_only_once: bool = False,
        shuffle_only_once: bool = False,
        random_seed_for_per_file_shuffling: int = 4350,
        h5file: bool = False,
        energy_threshold: float = 0,
        energy_sorting: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        files_dict : dict
            Dict with the file names for each type. Can be e.g. a dict like
            {"tbqq": ["tbqq_0.root", ...], "qcd": ["qcd_0.root", ...], ...}.
        n_files_at_once : int, optional
            Number of files to load at once. If None, one file per files_dict key
            is loaded.
        n_shower_per_file : int, optional
            Number of showers loaded from each individual file. Defaults to None, which
            means that all showers are loaded.
        max_n_files_per_type : int, optional
            Maximum number of files to use per type. If None, all files are used.
            Can be used to use e.g. always the first file from the sorted list of files
            in validation.
        shuffle_files : bool, optional
            Whether to shuffle the list of files.
        shuffle_data : bool, optional
            Whether to shuffle the data after loading.
        seed : int, optional
            Random seed.
        seed_shuffle_data : int, optional
            Random seed for shuffling the data. This is useful if you want to shuffle
            the data in the same way for different datasets (e.g. train and val).
            The default value is 3838.
        pad_length : int, optional
            Maximum number of particles per shower. If a shower has more particles, the
            first pad_length particles are used, the rest is discarded.
        logger_name : str, optional
            Name of the logger.
        feature_dict : dict, optional
            Dictionary with the features to load. The keys are the names of the features
            and the values are the preprocessing parameters passed to the
            `ak_select_and_preprocess` function.
        labels_to_load : list, optional
            List with the shower_type labels to load.
        token_reco_cfg : dict, optional
            Dictionary with the configuration to reconstruct the tokenized showerclass files.
            If None, this is not used.
        token_id_cfg : dict, optional
            Dictionary with the tokenization configuration, this is to be used when the
            token-id data is to be loaded. If None, this is ignored.
        load_only_once : bool, optional
            If True, the data is loaded only once and then returned in the same order
            in each iteration. NOTE: this is only useful if the whole dataset fits into
            memory. If the dataset is too large, this will lead to a memory error.
        shuffle_only_once : bool, optional
            If True, the data is shuffled only once and then returned in the same order
            in each iteration. NOTE: this should only be used for val/test.
        random_seed_for_per_file_shuffling : int, optional
            Random seed for shuffling the showers within a file. This is useful if you want
            to only load a subset of the showers from a file and want to choose different
            showers in different training runs.
            If load_only_once is False, this is ignored.
        h5file : bool, optional
            If True, the data is loaded from an h5 file. If False, the data is loaded from a root file.
        energy_threshhold : float, optional
            Is the minimum energy for which shower events are considered as non-zero.
        energy_sorting : bool, optional
            If True, the showers are sorted by energy in descending order.
        **kwargs
            Additional keyword arguments.

        """
        if feature_dict is None:
            raise ValueError("feature_dict must be provided.")
        if labels_to_load is None:
            raise ValueError("labels_to_load must be provided.")

        worker_info = get_worker_info()
        rank = get_rank() if dist.is_initialized() else 0
        world_size = get_world_size() if dist.is_initialized() else 1

        self.multi_gpu_info = {
            "num_gpus": torch.cuda.device_count(),
            "process_rank": rank,
            "world_size": world_size,
            "device": f"cuda:{rank}" if torch.cuda.is_available() else "cpu",
            "worker_id": worker_info.id if worker_info is not None else 0,
            "num_workers": worker_info.num_workers if worker_info is not None else 1,
        }

        self.logger_name = logger_name
        self.setup_logger(rank=None)

        self.logger.info(f"{[f'{key}={value}' for key, value in self.multi_gpu_info.items()]}")

        self.logger.info(f"Using seed {seed}")
        self.pad_length = pad_length
        self.shuffle_data = shuffle_data
        self.shuffle_files = shuffle_files
        self.processed_files_counter = 0
        self.max_n_files_per_type = max_n_files_per_type
        self.n_shower_per_file = n_shower_per_file
        self.feature_dict = feature_dict
        self.labels_to_load = labels_to_load
        self.particle_features_list = [feat for feat in self.feature_dict.keys()]
        self.seed_shuffle_data = seed_shuffle_data
        self.load_only_once = load_only_once
        self.shuffle_only_once = shuffle_only_once
        self.data_shuffled = False
        self.random_seed_for_per_file_shuffling = random_seed_for_per_file_shuffling
        self.h5file = h5file
        self.energy_threshold = energy_threshold
        self.energy_sorting = energy_sorting

        if self.random_seed_for_per_file_shuffling is not None:
            if not self.load_only_once:
                self.logger.warning(
                    "random_seed_for_per_file_shuffling is only used if load_only_once is True."
                )
                self.random_seed_for_per_file_shuffling = None
            else:
                self.logger.info(
                    f"Using random seed {self.random_seed_for_per_file_shuffling} for per-file shuffling."
                )

        self.logger.info(f"Using the following labels: {self.labels_to_load}")
        self.logger.info(f"Using the following particle features: {self.particle_features_list}")
        self.logger.info(f"pad_length {self.pad_length} for the number of hits per shower.")
        self.logger.info(f"energy_threshold {self.energy_threshold}")
        self.logger.info(f"shuffle_data={self.shuffle_data}")
        self.logger.info(f"shuffle_files={self.shuffle_files}")
        self.logger.info(
            "Number of showers loaded per file: "
            f"{self.n_shower_per_file if self.n_shower_per_file is not None else 'all'}"
        )
        self.logger.info("Using the following features:")
        for feat, params in self.feature_dict.items():
            self.logger.info(f"- {feat}: {params}")
        self.files_dict = {}
        for shower_type, files in files_dict.items():
            expanded_files = []
            for file in files:
                expanded_files.extend(sorted(list(glob.glob(file))))
            self.files_dict[shower_type] = (
                expanded_files
                if max_n_files_per_type is None
                else expanded_files[:max_n_files_per_type]
            )

            self.logger.info(f"Files for shower_type {shower_type}:")
            for file in self.files_dict[shower_type]:
                self.logger.info(f" - {file}")

        if self.load_only_once:
            self.logger.warning(
                "load_only_once is True. This means that there will only be the initial data loading."
            )

        # add all files from the dict to a list (the values are lists of files)
        self.file_list = []
        for files in self.files_dict.values():
            self.file_list.extend(files)

        # if not specified how many files to use at once, use one file per shower_type
        if n_files_at_once is None:
            self.n_files_at_once = len(self.files_dict)
        else:
            if n_files_at_once > len(self.file_list):
                self.logger.warning(
                    f"n_files_at_once={n_files_at_once} is larger than the number of files in the"
                    f" dataset ({len(self.file_list)})."
                )
                self.logger.warning(f"Setting n_files_at_once to {len(self.file_list)}.")
                self.n_files_at_once = len(self.file_list)
            else:
                self.n_files_at_once = n_files_at_once

        self.logger.info(f"Will load {self.n_files_at_once} files at a time and combine them.")

        self.file_indices = np.array([0, self.n_files_at_once])
        self.file_iterations = len(self.file_list) // self.n_files_at_once
        if self.load_only_once:
            self.file_iterations = 1

        self.current_part_data = None
        self.current_part_mask = None
        self.token_reco_cfg = token_reco_cfg
        self.token_id_cfg = token_id_cfg

    def setup_logger(self, rank: int = None) -> None:
        self.logger = get_pylogger(f"{__name__}-{self.logger_name}", rank=rank)
        self.logger.info("Logger set up (potentially with new rank information).")

    def get_data(self):
        """Returns a generator (i.e. iterator) that goes over the current files list and returns
        batches of the corresponding data."""
        # Iterate over shower_type
        self.logger.debug("\n>>> __iter__ called\n")
        self.file_indices = np.array([0, self.n_files_at_once])

        # shuffle the file list
        if self.shuffle_files:
            self.logger.info(">>> Shuffling files")
            random.shuffle(self.file_list)
            # self.logger.info(">>> self.file_list:")
            # for filename in self.file_list:
            #     self.logger.info(f" - {filename}")

        # Iterate over files
        for j in range(self.file_iterations):
            self.logger.debug(20 * "-")
            # Increment file index if not first iteration
            if j > 0:
                self.logger.info(">>> Incrementing file index")
                self.file_indices += self.n_files_at_once

            # stop the iteration if self.file_indices[1] is larger than the number of files
            # FIXME: this means that the last batch of files (in case the number of files is not
            # divisible by self.n_files_at_once) is not used --> fix this
            # but if shuffling is used, this should not be a problem
            if self.file_indices[1] <= len(self.file_list):
                self.load_next_files()

                # loop over the current data
                for i in range(self.start_idx_this_gpu, self.end_idx_this_gpu):
                    yield {
                        "part_features": self.current_part_data[i],
                        "part_mask": self.current_part_mask[i],
                    }

    def __iter__(self):
        """Returns an iterable which represents an iterator that iterates over the dataset."""
        # get current global rank to make sure the logger is set up correctly and displays
        # the rank in the logs
        self.multi_gpu_info["process_rank"] = get_rank() if dist.is_initialized() else 0
        self.setup_logger(rank=self.multi_gpu_info["process_rank"])
        self.logger.info(">>> __iter__(self.get_data()) called")
        return iter(self.get_data())

    def set_indices_for_this_rank(self):
        """Set the start and end indices to load for this rank."""
        # set the indices to load for each gpu
        if self.multi_gpu_info["world_size"] > 1:
            # split the self.current_part_data over the gpus
            n_shower = len(self.current_part_data)
            n_shower_per_gpu = n_shower // self.multi_gpu_info["world_size"]
            self.start_idx_this_gpu = n_shower_per_gpu * self.multi_gpu_info["process_rank"]
            self.end_idx_this_gpu = n_shower_per_gpu * (self.multi_gpu_info["process_rank"] + 1)
        else:
            self.start_idx_this_gpu = 0
            self.end_idx_this_gpu = len(self.current_part_data)

        self.logger.info(
            f"Rank {self.multi_gpu_info['process_rank']} will load data from index "
            f"{self.start_idx_this_gpu} to {self.end_idx_this_gpu}"
        )

    def load_next_files(self):
        if self.load_only_once:
            if self.current_part_data is not None:
                self.logger.warning("Data has already been loaded. Will not load again.")
                self.shuffle_current_data()
                return
        if self.processed_files_counter > 0:
            self.logger.info(
                f"self.processed_files_counter={self.processed_files_counter} is larger than 0 "
                f"and smaller than the total number of files in the dataset ({len(self.file_list)})."
                " This means that the files list was not fully traversed in the previous "
                "iteration. Will continue with the current files list."
            )
        self.part_data_list = []
        self.mask_data_list = []
        self.shower_type_labels_list = []

        self.current_files = self.file_list[self.file_indices[0] : self.file_indices[1]]
        self.logger.info(f">>> Loading next files - self.file_indices={self.file_indices}")
        if self.load_only_once:
            self.logger.warning(
                "Loading data only once. Will not load again.\n"
                "--> This will be the data for all iterations."
            )
        for i_file, filename in enumerate(self.current_files):
            self.logger.info(f"{i_file+1} / {len(self.current_files)} : {filename}")
            self.logger.info(f"Loading data from file: {filename}")
            self.logger.info(f"Is the Data loaded from h5 file?: {self.h5file}")

            # This Part will be used if you want to load an h5 file:
            if self.h5file:
                # this part will be used if you want to load a parquet file of tokens
                if self.token_id_cfg is not None:
                    self.logger.info("Loading tokenized shower file")
                    tokens = read_tokenized_shower_file(
                        filename,
                        particle_features=["part_token_id"],
                        remove_start_token=self.token_id_cfg.get("remove_start_token", False),
                        remove_end_token=self.token_id_cfg.get("remove_end_token", False),
                        shift_tokens_minus_one=self.token_id_cfg.get(
                            "shift_tokens_minus_one", False
                        ),
                        n_load=self.n_shower_per_file,
                        random_seed=self.random_seed_for_per_file_shuffling,
                    )
                    self.logger.info(f"the tokens are: {tokens}")
                    ak_x_particles = ak.Array(
                        {
                            "part_token_id": tokens["part_token_id"],
                            "part_token_id_without_last": tokens["part_token_id"][:, :-1],
                            "part_token_id_without_first": tokens["part_token_id"][:, 1:],
                        }
                    )
                    self.logger.info(f"the ak_x_particles is: {ak_x_particles}")
                    ak_x_particles = ak_preprocess(ak_x_particles, self.feature_dict)
                    self.logger.info("the Data was successfully preprocessed")
                    ak_x_particles_padded, ak_mask_particles = ak_pad(
                        ak_x_particles, self.pad_length, return_mask=True
                    )
                    self.logger.info("the Data was successfully padded")
                    np_x_particles_padded = ak_to_np_stack(
                        ak_x_particles_padded, names=self.particle_features_list
                    )
                    self.logger.info("the Data was successfully stacked to numpy")
                    # mask to numpy
                    np_mask_particles = ak.to_numpy(ak_mask_particles)
                    # add the data to the lists
                    self.part_data_list.append(torch.tensor(np_x_particles_padded))
                    self.mask_data_list.append(torch.tensor(np_mask_particles, dtype=torch.bool))

                else:
                    # this part is for loading an h5 file in the shower_format
                    # load data (only the amount of showers defined if n_shower_per_file is set)
                    data_showers = read_shower_file(
                        filename,
                        n_load=self.n_shower_per_file,
                    )
                    if self.energy_sorting:
                        # Sort the showers by energy
                        sorted_energy = ak.argsort(data_showers.energy, axis=1, ascending=False)
                        # Update data_showers with sorted energy
                        data_showers = data_showers[sorted_energy]

                    # Applying the Padding and getting the mask
                    ak_x_particles_padded, ak_mask_particles = ak_padding(
                        data_showers, self.pad_length, self.energy_threshold
                    )
                    # Apply the mask to the energy field using element-wise multiplication
                    ak_x_particles_padded["energy"] = ak.where(
                        ~ak_mask_particles,
                        1,
                        ak_x_particles_padded["energy"],
                    )
                    # shape = ak.to_numpy(ak_x_particles_padded["energy"]).shape
                    # shape_mask = ak.to_numpy(ak_mask_particles).shape
                    # self.logger.info(
                    #     f"Shape {shape} of the padded file and mask {shape_mask} with the pad_length of {self.pad_length}"
                    # )

                    # preprocessing the data
                    ak_x_particles = ak_preprocess(ak_x_particles_padded, self.feature_dict)
                    self.logger.info("ak_preprocess ran successfully")

                    # Define a constant for the large negative value

                    self.logger.info("ak.where ran successfully")

                    np_x_particles_padded = ak_to_np_stack(
                        ak_x_particles, names=self.particle_features_list
                    )
                    # mask to numpy
                    np_mask_particles = ak.to_numpy(ak_mask_particles)
                    # add the data to the lists
                    self.part_data_list.append(torch.tensor(np_x_particles_padded))
                    self.mask_data_list.append(torch.tensor(np_mask_particles, dtype=torch.bool))
                    gc.collect()
                    # self.shower_type_labels_list.append(torch.tensor(np_shower_type_labels))

                    # mask = torch.any(torch.tensor(data_showers) != 0, dim=-1)
                    # self.logger.info("this is the mask shape:", mask.shape)
                    # Add the data to the lists (adjust based on data structure)
                    # self.part_data_list.append(torch.tensor(data_showers))
                    # self.mask_data_list.append(torch.tensor(mask))

            # This part would be used if you want to load a root file:
            else:
                pass

        # concatenate the data from all files
        self.current_part_data = torch.cat(self.part_data_list, dim=0)
        self.current_part_mask = torch.cat(self.mask_data_list, dim=0)
        if not self.h5file:
            self.current_shower_type_labels_one_hot = torch.cat(
                self.shower_type_labels_list, dim=0
            )

        self.shuffle_current_data()

        self.logger.info(
            f">>> Data loaded. (self.current_part_data.shape = {self.current_part_data.shape})"
        )
        self.set_indices_for_this_rank()

        self.processed_files_counter += self.n_files_at_once

        self.logger.info(
            "Updating self.processed_files_counter. The new value is "
            f"self.processed_files_counter = {self.processed_files_counter}."
        )
        self.logger.info(
            "Checking if all files in the current files list have been processed. "
            "If so, the file list will be shuffled (unless `shuffle_files=False`)"
            "such that the next iteration will proceed with a new file list."
        )

    def shuffle_current_data(self):
        # shuffle the data
        if self.shuffle_only_once and self.data_shuffled:
            self.logger.info("Data has already been shuffled. Will not shuffle again.")
            return
        if self.shuffle_data:
            if self.seed_shuffle_data is not None:
                self.logger.info(f"Shuffling data with seed {self.seed_shuffle_data}")
                rng = np.random.default_rng(self.seed_shuffle_data)
            else:
                self.logger.info("Shuffling data without seed")
                rng = np.random.default_rng()
            perm = rng.permutation(len(self.current_part_data))
            self.current_part_data = self.current_part_data[perm]
            self.current_part_mask = self.current_part_mask[perm]
            if not self.h5file:
                self.current_shower_type_labels_one_hot = self.current_shower_type_labels_one_hot[
                    perm
                ]
            self.data_shuffled = True
            self.logger.info("Data shuffled.")


class IterableCaloDatamodule(L.LightningDataModule):
    def __init__(
        self,
        dataset_kwargs_train: dict,
        dataset_kwargs_val: dict,
        dataset_kwargs_test: dict,
        dataset_kwargs_common: dict,
        batch_size: int = 256,
        **kwargs,
    ):
        super().__init__()

        # save the parameters as attributes
        self.save_hyperparameters()

    def prepare_data(self) -> None:
        """Prepare the data."""
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit":
            self.train_dataset = CustomIterableDataset(
                **self.hparams.dataset_kwargs_train,
                **self.hparams.dataset_kwargs_common,
            )
            self.val_dataset = CustomIterableDataset(
                **self.hparams.dataset_kwargs_val,
                **self.hparams.dataset_kwargs_common,
            )
        elif stage == "test":
            self.test_dataset = CustomIterableDataset(
                **self.hparams.dataset_kwargs_test,
                **self.hparams.dataset_kwargs_common,
            )

    def train_dataloader(self):
        # Use a DistributedSampler for multi-GPU training
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            # pin_memory=True,  # Pre-transfer data to pinned memory
            # num_workers=2,
        )

    def val_dataloader(self):
        # Optionally use a sampler for validation
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            # pin_memory=True,
            # num_workers=2,
        )

    def test_dataloader(self):
        # Optionally use a sampler for testing
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            # pin_memory=True,
            # num_workers=2,
        )
