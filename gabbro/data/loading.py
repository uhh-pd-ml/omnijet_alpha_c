import logging

import awkward as ak
import h5py
import numpy as np
import vector

logger = logging.getLogger(__name__)

vector.register_awkward()


def read_tokenized_shower_file(
    filepath,
    particle_features=["part_token_id"],
    remove_start_token=False,
    remove_end_token=False,
    shift_tokens_minus_one=False,
    n_load=None,
    random_seed=None,
):
    """Reads a file containing a list of file paths.

    Parameters:
    ----------
    filepath : str
        Path to the file.
    particle_features : List[str], optional
        A list of particle-level features to be loaded. Should only contain "part_token_id".
    labels : List[str], optional
        A list of truth labels to be loaded.
    remove_start_token : bool, optional
        Whether to remove the start token from the tokenized sequence.
    remove_end_token : bool, optional
        Whether to remove the end token from the tokenized sequence.
    shift_tokens_minus_one : bool, optional
        Whether to shift the token values by -1.
    n_load : int, optional
        Number of events to load. If None, all events are loaded.
    random_seed : int, optional
        Random seed for shuffling the data. If None, no shuffling is performed.


    Returns:
    -------
    tokens : List[str]
        A list of file paths.
    """

    ak_tokens = ak.from_parquet(filepath)

    if random_seed is not None:
        print(f"shuffling with random seed {random_seed}")
        rng = np.random.default_rng(random_seed)
        permutation = rng.permutation(len(ak_tokens))
        print("ak_tokens", ak_tokens)
        ak_tokens = ak_tokens[permutation]
        print("ak_tokens after permutation", ak_tokens)

    if n_load is not None:
        print(f"will only load tokens of {n_load} events")
        ak_tokens = ak_tokens[:n_load]

    # one-hot encode the shower type

    if remove_start_token:
        ak_tokens = ak_tokens[:, 1:]
    if remove_end_token:
        ak_tokens = ak_tokens[:, :-1]
    if shift_tokens_minus_one:
        ak_tokens = ak_tokens - 1

    x_ak = ak.Array({"part_token_id": ak_tokens})

    return x_ak


def read_shower_file(filepath, n_load=None, chunk_size=1000):
    """Loads a single file from the showerClass dataset.

    Parameters:
    ----------
    filepath : str
        Path to the h5 data file.
    n_load : int, optional
        Number of showers to load. If None, load all data.
    chunk_size : int, optional
        Size of chunks to load at a time. Default is 1000.

    Returns:
    -------
    shower : ak.Array
        An awkward array of the shower features.
    """
    with h5py.File(filepath, "r") as h5file:
        dataset_showers = h5file["showers"]
        total_showers = dataset_showers.shape[0]

        if n_load is None:
            n_load = total_showers

        data_dict = {
            "x": [],
            "y": [],
            "z": [],
            "energy": [],
        }

        for start in range(0, n_load, chunk_size):
            end = min(start + chunk_size, n_load)
            table = dataset_showers[start:end]

            data_dict["x"].append(table[:, :, 0])
            data_dict["y"].append(table[:, :, 1])
            data_dict["z"].append(table[:, :, 2])
            data_dict["energy"].append(table[:, :, 3])

    # Concatenate chunks
    data_dict = {key: ak.concatenate(value) for key, value in data_dict.items()}
    ak_array = ak.Array(data_dict)
    return ak_array
