import json
import warnings
from importlib.util import find_spec
from typing import Callable, List

import awkward as ak
import hydra
import numpy as np
from omegaconf import DictConfig
from pytorch_lightning import Callback
from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities import rank_zero_only

from gabbro.utils import pylogger, rich_utils

log = pylogger.get_pylogger(__name__)


def translate_bash_range(wildcard: str, verbose: bool = False):
    """Translate bash range to list of strings with the corresponding numbers.

    Parameters
    ----------
    wildcard : str
        Wildcard string with bash range (or not).
    verbose : bool, optional
        If True, print debug messages.

    Returns
    -------
    list
        List of strings with the corresponding numbers.
    """

    # raise value error if two ranges are found
    if wildcard.count("{") > 1:
        raise ValueError(
            f"Only one range is allowed in the wildcard. Provided the following wildcard: {wildcard}"
        )

    if "{" in wildcard and ".." in wildcard and "}" in wildcard:
        log.info("Bash range found in wildcard --> translating to list of remaining wildcards.")
        start = wildcard.find("{")
        end = wildcard.find("}")
        prefix = wildcard[:start]
        suffix = wildcard[end + 1 :]
        wildcard_range = wildcard[start + 1 : end]
        start_number = int(wildcard_range.split("..")[0])
        end_number = int(wildcard_range.split("..")[1])
        if verbose:
            log.info(
                f"Prefix: {prefix}, Suffix: {suffix}, Start: {start_number}, End: {end_number}"
            )
        return [f"{prefix}{i}{suffix}" for i in range(start_number, end_number + 1)]
    else:
        # print("No range found in wildcard")
        return [wildcard]


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that wraps the task function in extra utilities.

    Makes multirun more resistant to failure.

    Utilities:
    - Calling the `utils.extras()` before the task is started
    - Calling the `utils.close_loggers()` after the task is finished or failed
    - Logging the exception if occurs
    - Logging the output dir
    """

    def wrap(cfg: DictConfig):
        # execute the task
        try:
            # apply extra utilities
            extras(cfg)

            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # when using hydra plugins like Optuna, you might want to disable raising exception
            # to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # close loggers (even if exception occurs so multirun won't fail)
            close_loggers()

        return metric_dict, object_dict

    return wrap


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
    - Ignoring python warnings
    - Setting tags from command line
    - Rich config printing
    """

    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def instantiate_callbacks(callbacks_cfg: DictConfig, ckpt_path: str = None) -> List[Callback]:
    """Instantiates callbacks from config."""
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiates loggers from config."""
    logger: List[Logger] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger


@rank_zero_only
def log_hyperparameters(object_dict: dict) -> None:
    """Controls which config parts are saved by lightning loggers.

    Additionally saves:
    - Number of model parameters
    """

    hparams = {}

    cfg = object_dict["cfg"]
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")
    hparams["run_note"] = cfg.get("run_note")
    hparams["git"] = object_dict.get("git")
    hparams["slurm"] = object_dict.get("slurm")
    hparams["load_weights_from"] = object_dict.get("load_weights_from")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)


def get_metric_value(metric_dict: dict, metric_name: str) -> float:
    """Safely retrieves value of the metric logged in LightningModule."""

    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


def close_loggers() -> None:
    """Makes sure all loggers closed properly (prevents logging failure during multirun)."""

    log.info("Closing loggers...")

    if find_spec("wandb"):  # if wandb is installed
        import wandb

        if wandb.run:
            log.info("Closing wandb!")
            wandb.finish()


@rank_zero_only
def save_file(path: str, content: str) -> None:
    """Save file in rank zero mode (only on one process in multi-GPU setup)."""
    with open(path, "w+") as file:
        file.write(content)


def get0Momentum(x: float, weights: float) -> float:
    # calculate the 0-momentum
    out = (x * weights).sum(-1)
    return out / weights.sum(-1)


def get_diff_construct(original: float, reconstructed: float) -> float:
    # calculate the difference
    return reconstructed - original


def KL(data1: np.ndarray, data2: np.ndarray, bins: int) -> float:
    """Calculates the KL divergence between two probability distributions.

    Args:
        data1: The first dataset (samples).
        data2: The second dataset (samples).
        bins: The number of bins for creating histograms.

    Returns:
        The KL divergence (a non-negative float).
    """
    hist1, bin_edges1 = np.histogram(data1, bins, density=True)
    hist2, bin_edges2 = np.histogram(data2, bins, density=True)
    epsilon = 1e-8
    hist1 = np.maximum(hist1, epsilon)  # Ensure no zero values
    hist2 = np.maximum(hist2, epsilon)

    # Assuming you have your hist1 and hist2 arrays from before

    # Get bin widths
    bin_widths1 = np.diff(bin_edges1)
    bin_widths2 = np.diff(bin_edges2)

    # Calculate approximate probabilities
    hist1 = hist1 * bin_widths1
    hist2 = hist2 * bin_widths2

    # Handle cases where bins are zero in either histogram
    nonzero_mask = (hist1 != 0) & (hist2 != 0)

    # Calculate KL divergence only for non-zero bins
    kl_div = np.sum(hist1[nonzero_mask] * np.log(hist1[nonzero_mask] / hist2[nonzero_mask]))
    return kl_div


def find_max_energy_z(energy: ak.Array, z: ak.Array) -> ak.Array:
    """Finds the z-value corresponding to the maximum energy in each shower.

    Args:
        energy: Awkward array of energy values for each shower.
        z: Awkward array of z-values for each shower.

    Returns:
        Awkward array of z-values corresponding to the maximum energy in each shower.
    """
    z = ak.fill_none(z, 0)
    energy = ak.fill_none(energy, 0)
    max_energy_indices = ak.argmax(energy, axis=1)
    max_energy_indices = ak.fill_none(max_energy_indices, 0)

    shower_indices = ak.from_numpy(np.arange(len(z)))

    max_energy_z_values = z[shower_indices, max_energy_indices]

    return ak.to_numpy(max_energy_z_values)


def get_COG_ak(x: ak.Array, weights: ak.Array) -> ak.Array:
    """Calculates the 0-momentum for each individual shower (subarray) in an awkward array.

    Args:
        x: Awkward array of coordinates.
        weights: Awkward array of weights (e.g., energy).

    Returns:
        Awkward array of 0-momentum values for each shower.
    """
    # Element-wise multiplication for each shower
    weighted_x = x * weights

    # Calculate the sum of weighted x and sum of weights for each shower
    sum_weighted_x = ak.sum(weighted_x, axis=-1)
    sum_weights = ak.sum(weights, axis=-1)

    # Divide the sums to get the 0-momentum for each shower
    return sum_weighted_x / sum_weights


def find_radial_profile(x: ak.Array, y: ak.Array) -> ak.Array:
    """finds the energy-weighted distances from the incident point in the x-y-plane.

    Args:
        x: Awkward array of x-coordinates for each point in the shower.
        y: Awkward array of y-coordinates for each point in the shower.
        energy: Awkward array of energy values for each point in the shower.

    Returns:
        Awkward array of radial distances from the center for each point in the shower.
    """

    # Calculate the middle (mean) of x and y coordinates for each shower
    x_middle = 14.5
    y_middle = 14.5

    # Calculate the radial distance from the center
    radial_distance = ((x - x_middle) ** 2 + (y - y_middle) ** 2) ** 0.5
    # Fill None values with 0 to ensure each axis has the same length
    radial_distance = ak.flatten(radial_distance, axis=1)
    radial_distance = ak.to_numpy(radial_distance)

    return radial_distance


def sum_energy_per_radial_distance(x: ak.Array, y: ak.Array, energy: ak.Array) -> ak.Array:
    """Sums up the energy per radial distance bin for each shower.

    Args:
        radial_distance: Awkward array of radial distances for each point in the shower.
        energy: Awkward array of energy values for each point in the shower.
    Returns:
        Awkward array of summed energy values per radial distance bin for each shower.
    """
    x_middle = 14.5
    y_middle = 14.5

    # Calculate the radial distance from the center
    radial_distance = ((x - x_middle) ** 2 + (y - y_middle) ** 2) ** 0.5
    result = []
    radial_bins = np.arange(0, 22)
    for radial_shower, energy_shower in zip(radial_distance, energy):
        # Bin energy values according to radial distance bins
        hist, _ = np.histogram(radial_shower, bins=radial_bins, weights=energy_shower)
        result.append(hist)
    # Convert the result to an awkward array for better handling
    binned_energy = ak.Array(result)
    return binned_energy


def sum_energy_per_layer(z: ak.Array, energy: ak.Array) -> ak.Array:
    """Sums up the energy per layer (z-bin) for each shower.

    Args:
        z: Awkward array of z-coordinates for each point in the shower.
        energy: Awkward array of energy values for each point in the shower.
        z_bins: Array of z-bin edges.
    Returns:
        Awkward array of summed energy values per z-bin for each shower.
    """
    result = []
    z_bins = np.arange(0, 30)
    for z_shower, energy_shower in zip(z, energy):
        # Bin energy values according to Z-bins
        hist, _ = np.histogram(z_shower, bins=z_bins, weights=energy_shower)
        result.append(hist)
    # Convert the result to an awkward array for better handling
    binned_energy = ak.Array(result)
    return binned_energy


def write_distances_to_json(kld, wasserstein, filepath, weights, n_data, feature):
    """Writes KLD and Wasserstein distances to a JSON file, structured for plotting.

    Args:
        kld: Kullback-Leibler divergence.
        wasserstein: Wasserstein distance.
        filepath: Path to the JSON file.
        weights: "weights" or "no weights".
        n_data: Number of training data (e.g., "100", "1000", "10000").
        feature: Feature name (e.g., "energy", "energy_sum", "max_z").
    """

    # Load existing JSON data if the file exists
    try:
        with open(filepath) as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}

    # Create a nested dictionary for the feature
    if feature not in data:
        data[feature] = {}

    # Add data for the current setting
    if weights not in data[feature]:
        data[feature][weights] = {}

    if n_data not in data[feature][weights]:
        data[feature][weights][n_data] = {}

    if "kld" not in data[feature][weights][n_data]:
        data[feature][weights][n_data] = {"kld": [], "wasserstein": []}

    # Add the new entry to the data list
    data[feature][weights][n_data]["kld"].append(kld)
    data[feature][weights][n_data]["wasserstein"].append(wasserstein)

    # Write the updated data back to the JSON file
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)


# Analyze the first 10 tokens of each shower and their commonality
def analyze_first_10_tokens(token_ids: ak.Array) -> np.ndarray:
    """Analyzes the first 10 tokens of each shower and their commonality.

    Args:
        token_ids: Awkward array of token IDs for each shower.

    Returns:
        Dictionary with unique token sequences as keys and their counts as values.
    """
    first_10_tokens = ak.to_numpy(ak.pad_none(token_ids[:, 1:11], 10, clip=True))
    unique, counts = np.unique(first_10_tokens.flatten(), return_counts=True)
    counts = np.sort(counts)[::-1]
    return counts
