import logging
import os
from pathlib import Path

import awkward as ak
import vector
from omegaconf import OmegaConf

from gabbro.data.loading import read_shower_file
from gabbro.models.vqvae import VQVAELightning
from gabbro.utils.mapping import merge_duplicates_numpy

# import gabbro.plotting as jplt

vector.register_awkward()

logger = logging.getLogger(__name__)


def tokenize_shower_file(
    filename_in: str,
    model_ckpt_path: str,
    filename_out: str = None,
    add_start_end_tokens: bool = False,
    print_model: bool = False,
    energy_sorting: bool = False,
    layer_sorting: bool = False,
    n_load: int = None,
):
    """Tokenize a single file using a trained model.

    Parameters
    ----------
    filename : str
        Path to the file to be tokenized.
    model_ckpt_path : str
        Path to the model checkpoint.
    filename_out : str, optional
        Path to the output file.
    add_start_end_tokens : bool, optional
        Whether to add start and end tokens to the tokenized sequence.
    print_model : bool, optional
        Whether to print the model architecture.
    n_load : int, optional
        Number of events to load from the file. If None, all events are loaded.

    Returns
    -------
    tokens_int : ak.Array
        Array of tokens.
    p4s_original : ak.Array
        Momentum4D array of the original particles.
    x_ak_original : ak.Array
        Array of the original particles.
    """
    data_showers = read_shower_file(
        filename_in,
        n_load=n_load,
    )
    if energy_sorting:
        print("Sorting by energy")
        # Sort the showers by energy
        sorted_energy = ak.argsort(data_showers.energy, axis=1, ascending=False)
        # Update data_showers with sorted energy
        data_showers = data_showers[sorted_energy]
    if layer_sorting:
        print("Sorting by layer")
        # Sort the showers by layer
        sorted_layer = ak.argsort(data_showers.z, axis=1, ascending=True)
        # Update data_showers with sorted layer
        data_showers = data_showers[sorted_layer]

    # --- Model and config loading ---
    ckpt_path = Path(model_ckpt_path)
    config_path = ckpt_path.parent.parent / "config.yaml"
    cfg = OmegaConf.load(config_path)
    logger.info(f"Loaded config from {config_path}")
    model = VQVAELightning.load_from_checkpoint(ckpt_path)
    if print_model:
        print(model)
    pp_dict = cfg.data.dataset_kwargs_common["feature_dict"]
    logger.info("Preprocessing dictionary:")
    for key, value in pp_dict.items():
        logger.info(f" | {key}: {value}")

    model = model.to("cuda")
    model.eval()

    # --------------------------------

    p4s_original = ak.zip(
        {
            "x": data_showers["x"],
            "y": data_showers["y"],
            "z": data_showers["z"],
            "energy": data_showers["energy"],
        },
        with_name="Momentum4D",
    )
    # ak_data_shower = np_to_akward(data_showers, pp_dict)

    tokens = model.tokenize_shower_ak_array(data_showers, pp_dict)
    print("tokens", tokens)

    if add_start_end_tokens:
        n_tokens = model.model.vqlayer.num_codes
        tokens = ak.concatenate(
            [
                ak.zeros_like(tokens[:, :1]),  # start token is 0
                tokens + 1,
                ak.ones_like(tokens[:, :1]) + n_tokens,  # end token is n_tokens + 1
            ],
            axis=1,
        )

    tokens_int = ak.values_astype(tokens, int)

    if filename_out is not None:
        os.makedirs(os.path.dirname(filename_out), exist_ok=True)
        logger.info(f"Saving tokenized file to {filename_out}")
        ak.to_parquet(tokens_int, filename_out)

    print("p4s_original z", p4s_original.z)
    print("p4s_original energy", p4s_original.energy)

    return tokens_int, p4s_original, data_showers


def reconstruct_shower_file(
    filename_in: str,
    model_ckpt_path: str,
    config_path: str,
    start_token_included: bool = False,
    end_token_included: bool = False,
    shift_tokens_by_minus_one: bool = False,
    filename_out: str = None,
    print_model: bool = False,
    device: str = "cuda",
    merge_duplicates: bool = True,
):
    """Reconstruct a single file using a trained model and the tokenized file.

    Parameters
    ----------
    filename_in : str
        Path to the file to be tokenized.
    model_ckpt_path : strgemini
        Path to the model checkpoint.
    config_path : str
        Path to the config file.
    filename_out : str, optional
        Path to the output file.
    start_token_included : bool, optional
        Whether the start token is included in the tokenized sequence.
    end_token_included : bool, optional
        Whether the end token is included in the tokenized sequence.
    shift_tokens_by_minus_one : bool, optional
        Whether to shift the tokens by -1.
    print_model : bool, optional
        Whether to print the model architecture.
    device : str, optional
        Device to use for the model.
    merge_duplicates : bool, optional
        Whether to merge the duplicate voxels.

    Returns
    -------
    p4s_reco : ak.Array
        Momentum4D array of the reconstructed particles.
    x_reco_ak : ak.Array
        Array of the reconstructed particles.
    labels_onehot : np.ndarray
        One-hot encoded labels of the shower type. Only returned if return_labels is True.
    """

    # --- Model and config loading ---
    ckpt_path = Path(model_ckpt_path)
    cfg = OmegaConf.load(config_path)
    logger.info(f"Loaded config from {config_path}")
    model = VQVAELightning.load_from_checkpoint(ckpt_path)
    if print_model:
        logger.info(model)
    pp_dict = cfg.data.dataset_kwargs_common["feature_dict"]
    # logger.info("Preprocessing dictionary:")
    # for key, value in pp_dict.items():
    #     logger.info(f" | {key}: {value}")

    model = model.to(device)
    model.eval()
    # --------------------------------

    tokens = ak.from_parquet(filename_in)
    logger.info(f"tokens: {tokens}")

    if end_token_included:
        logger.info("Removing end token")
        tokens = tokens[:, :-1]
        logger.info(f"Tokens with end token removed: {tokens}")
    if start_token_included:
        logger.info("Removing start token and shifting tokens by -1")
        tokens = tokens[:, 1:]
        logger.info(f"Tokens with start token removed: {tokens}")
    if shift_tokens_by_minus_one:
        logger.info("Shifting tokens by -1")
        tokens = tokens - 1
        logger.info(f"Tokens shifted by -1: {tokens}")

    logger.info(f"Smallest token in file: {ak.min(tokens)}")
    logger.info(f"Largest token in file:  {ak.max(tokens)}")

    x_reco_ak = model.reconstruct_shower_ak_tokens(tokens, pp_dict, hide_pbar=False, batch_size=2)

    logger.info(f"x_reco_ak x: {x_reco_ak.x}")
    logger.info(f"x_reco_ak energy: {x_reco_ak.energy}")

    logger.info(f"reconstructed file: {x_reco_ak}")

    if merge_duplicates:
        x_reco_ak = merge_duplicates_numpy(
            x_reco_ak
        )  # this maps the duplicate voxels on the true grid and sums the energy
        logger.info(f"reconstructed file after merge: {x_reco_ak}")

    p4s_reco = ak.zip(
        {
            "x": x_reco_ak.x if "x" in x_reco_ak.fields else x_reco_ak.x,
            "y": x_reco_ak.y if "y" in x_reco_ak.fields else x_reco_ak.y,
            "z": x_reco_ak.z if "z" in x_reco_ak.fields else x_reco_ak.z,
            "energy": x_reco_ak.energy if "energy" in x_reco_ak.fields else x_reco_ak.energy,
        },
        with_name="Momentum4D",
    )
    logger.info(f"p4s_reco energy: {p4s_reco.energy}")
    if filename_out is not None:
        os.makedirs(os.path.dirname(filename_out), exist_ok=True)
        logger.info(f"Saving tokenized file to {filename_out}")
        ak.to_parquet(p4s_reco, filename_out)

    return p4s_reco, x_reco_ak
