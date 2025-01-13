import awkward as ak
import numpy as np
import torch
import vector

vector.register_awkward()


def ak_pad(x: ak.Array, maxlen: int, axis: int = 1, fill_value=0, return_mask=False):
    """Function to pad an awkward array to a specified length. The array is padded along the
    specified axis.

    Parameters
    ----------
    x : awkward array
        Array to pad.
    maxlen : int
        Length to pad to.
    axis : int, optional
        Axis along which to pad. Default is 1.
    fill_value : float or int, optional
        Value to use for padding. Default is 0.
    return_mask : bool, optional
        If True, also return a mask array indicating which values are padded.
        Default is False.
        If the input array has fields, the mask is created from the first field.

    Returns
    -------
    awkward array
        Padded array.
    mask : awkward array
        Mask array indicating which values are padded. Only returned if return_mask is True.
    """
    padded_x = ak.fill_none(ak.pad_none(x, maxlen, axis=axis, clip=True), fill_value)
    if return_mask:
        if len(x.fields) >= 1:
            mask = ak.ones_like(x[x.fields[0]], dtype="bool")
        else:
            mask = ak.ones_like(x, dtype="bool")
        mask = ak.fill_none(ak.pad_none(mask, maxlen, axis=axis, clip=True), False)
        return padded_x, mask
    return padded_x


def np_to_ak(x: np.ndarray, names: list, mask: np.ndarray = None):
    """Function to convert a numpy array and its mask to an awkward array. The features
    corresponding to the names are assumed to correspond to the last axis of the array.

    Parameters
    ----------
    x : np.ndarray
        Array to convert.
    names : list
        List of field names (corresponding to the features in x along the last dimension).
    mask : np.ndarray, optional
        Mask array. Default is None.
    """

    if mask is None:
        mask = np.ones_like(x[..., 0], dtype="bool")

    return ak.Array(
        {
            name: ak.values_astype(
                ak.drop_none(ak.mask(ak.Array(x[..., i]), mask != 0)),
                "float32",
            )
            for i, name in enumerate(names)
        }
    )


def np_to_akward(x: np.ndarray, pp_dict: dict):
    """Function to convert a numpy array to an awkward array with specified labels.

    Parameters
    ----------
    x : np.ndarray
        Array to convert.
    pp_dict : dict
        Dictionary containing field names as keys.
    """

    return ak.Array(
        {
            name: ak.values_astype(ak.Array(x[..., i]), "float32")
            for i, name in enumerate(pp_dict.keys())
        }
    )


def ak_to_np_stack(ak_array: ak.Array, names: list = None, axis: int = -1):
    """Function to convert an awkward array to a numpy array by stacking the values of the
    specified fields. This is much faster than ak.to_numpy(ak_array) for large arrays.

    Parameters
    ----------
    ak_array : awkward array
        Array to convert.
    names : list, optional
        List of field names to convert. Default is None.
    axis : int, optional
        Axis along which to stack the values. Default is -1.
    """
    if names is None:
        raise ValueError("names must be specified")
    return ak.to_numpy(
        np.stack(
            [ak.to_numpy(ak.values_astype(ak_array[name], "float32")) for name in names],
            axis=axis,
        )
    )


def np_PtEtaPhi_to_Momentum4D(arr, mask, log_pt=False):
    """Convert numpy array with 4-momenta to ak array of Momentum4D objects.
    NOTE: the input array is assumed to be in (pT, eta, phi) format, thus mass = 0.

    Expects an array of shape (batch_size, num_particles, 3)
    where the last dimension is (pt, eta, phi)

    Returns an ak array of shape (batch_size, var, 4) of Momentum4D objects

    If log_pt is True, the corresponding variable is exponentiated
    before being passed to Momentum4D

    Parameters
    ----------
    arr : np.ndarray
        Input array of shape (batch_size, num_particles, 3)
    mask : np.ndarray
        Mask array of shape (batch_size, num_particles)
    log_pt : bool, optional
        Whether to exponentiate pt, by default False

    Returns
    -------
    ak.Array
        Array of Momentum4D objects
    """

    p4 = ak.zip(
        {
            "pt": np.clip(arr[:, :, 0], 0, None) if not log_pt else np.exp(arr[:, :, 0]),
            "eta": arr[:, :, 1],
            "phi": arr[:, :, 2],
            "mass": ak.zeros_like(arr[:, :, 0]),
        },
        with_name="Momentum4D",
    )
    # mask the array
    ak_mask = ak.Array(mask)
    return ak.drop_none(ak.mask(p4, ak_mask == 1))


def ak_select_and_preprocess(ak_array: ak.Array, pp_dict=None, inverse=False):
    """Function to select and pre-process fields from an awkward array.

    Parameters
    ----------
    ak_array : awkward array
        Array to convert.
    pp_dict : dict, optional
        Dictionary with pre-processing values for each field. Default is None.
        The dictionary should have the following format:
        {
            "field_name_1": {"multiply_by": 1, "subtract_by": 0, "func": "np.log"},
            "field_name_2": {"multiply_by": 1, "subtract_by": 0, "func": None},
            ...
        }
    inverse : bool, optional
        If True, the inverse of the pre-processing is applied. Default is False.
    """
    if pp_dict is None:
        pp_dict = {}

    # define initial mask as all True
    first_feat = list(pp_dict.keys())[0]
    selection_mask = ak.ones_like(ak_array[first_feat], dtype="bool")

    for name, params in pp_dict.items():
        if params is None:
            pp_dict[name] = {"subtract_by": 0, "multiply_by": 1, "func": None}
        else:
            if "subtract_by" not in params:
                pp_dict[name]["subtract_by"] = 0
            if "multiply_by" not in params:
                pp_dict[name]["multiply_by"] = 1
            if "func" not in params:
                pp_dict[name]["func"] = None
            if "inv_func" not in params:
                pp_dict[name]["inv_func"] = None

            if pp_dict[name]["func"] is not None:
                if pp_dict[name]["inv_func"] is None:
                    raise ValueError(
                        "If a function is specified, an inverse function must also be specified."
                    )
            else:
                if pp_dict[name]["inv_func"] is not None:
                    raise ValueError(
                        "If an inverse function is specified, a function must also be specified."
                    )
        # apply selection cuts
        if pp_dict[name].get("larger_than") is not None:
            selection_mask = selection_mask & (ak_array[name] > pp_dict[name]["larger_than"])
        if pp_dict[name].get("smaller_than") is not None:
            selection_mask = selection_mask & (ak_array[name] < pp_dict[name]["smaller_than"])

    if inverse:
        return ak.Array(
            {
                name: (
                    eval(params["inv_func"])(  # nosec
                        getattr(ak_array, name) / params["multiply_by"] + params["subtract_by"]
                    )
                    if params["inv_func"]
                    else getattr(ak_array, name) / params["multiply_by"] + params["subtract_by"]
                )
                for name, params in pp_dict.items()
            }
        )
    return ak.Array(
        {
            name: (
                (
                    eval(params["func"])(getattr(ak_array, name)[selection_mask])  # nosec
                    if params["func"]
                    else getattr(ak_array, name)[selection_mask]
                )
                - params["subtract_by"]
            )
            * params["multiply_by"]
            for name, params in pp_dict.items()
        }
    )


# define a padding function for the shower arrays
def ak_padding(x: ak.Array, maxlen: int, energy_threshold: float):
    """Pads an Awkward Array and creates a mask based on energy values.

    Args:
        x (ak.Array): The Awkward Array to pad.
        maxlen (int): The maximum length to pad to.
        energy_threshold (float): The threshold for considering energy as non-zero.

    Returns:
        tuple: A tuple containing the padded array and the mask.
    """

    # Create mask based on energy threshold
    mask = x["energy"] > energy_threshold

    # Pad both data and mask to maxlen
    padded_x = ak.pad_none(x, maxlen, axis=1, clip=True)
    padded_mask = ak.pad_none(mask, maxlen, axis=1, clip=True)

    # Fill None in data with zeros (or custom values if needed)

    return padded_x, padded_mask


# define a function to preprocess shower data for use in the model
def ak_preprocess(ak_array: ak.Array, pp_dict=None, inverse: bool = False):
    """Function to select and pre-process fields from an awkward array.

    Parameters
    ----------
    ak_array : awkward array
        Array to convert.
    pp_dict : dict, optional
        Dictionary with pre-processing values for each field. Default is None.
        The dictionary should have the following format:
        {
            "field_name_1": {"multiply_by": 1, "subtract_by": 0, "func": "np.log"},
            "field_name_2": {"multiply_by": 1, "subtract_by": 0, "func": None},
            ...
        }
    inverse : bool, optional
        If True, the inverse of the pre-processing is applied. Default is False.
    """
    if pp_dict is None:
        pp_dict = {}

    # Get the input shape
    # num_records = len(ak_array)
    # num_values_per_field = len(ak_array["x"][0])  # Assuming uniform structure
    # input_shape = (num_records, num_values_per_field)
    # define initial mask as all True
    first_feat = list(pp_dict.keys())[0]
    selection_mask = ak.ones_like(ak_array[first_feat], dtype="bool")

    for name, params in pp_dict.items():
        if params is None:
            pp_dict[name] = {"subtract_by": 0, "multiply_by": 1, "func": None}
            # pylogger.info(f"if params is None: {pp_dict[name]}")
        else:
            # pylogger.info(f"if params is not None: {pp_dict[name]}")

            if "subtract_by" not in params:
                pp_dict[name]["subtract_by"] = 0
            if "multiply_by" not in params:
                pp_dict[name]["multiply_by"] = 1
            if "func" not in params:
                pp_dict[name]["func"] = None
            if "inv_func" not in params:
                pp_dict[name]["inv_func"] = None

            if pp_dict[name]["func"] is not None:
                if pp_dict[name]["inv_func"] is None:
                    raise ValueError(
                        "If a function is specified, an inverse function must also be specified."
                    )
            else:
                if pp_dict[name]["inv_func"] is not None:
                    raise ValueError(
                        "If an inverse function is specified, a function must also be specified."
                    )
        # apply selection cuts
        if pp_dict[name].get("larger_than") is not None:
            selection_mask = selection_mask & (ak_array[name] > pp_dict[name]["larger_than"])
        if pp_dict[name].get("smaller_than") is not None:
            selection_mask = selection_mask & (ak_array[name] < pp_dict[name]["smaller_than"])

    if inverse:
        result_array = ak.Array(
            {
                name: (
                    eval(params["inv_func"])(  # nosec
                        getattr(ak_array, name) / params["multiply_by"] + params["subtract_by"]
                    )
                    if params["inv_func"]
                    else getattr(ak_array, name) / params["multiply_by"] + params["subtract_by"]
                )
                for name, params in pp_dict.items()
            }
        )
    else:
        result_array = ak.Array(
            {
                name: (
                    (
                        (
                            eval(params["func"])(getattr(ak_array, name))  # nosec
                            if params["func"]
                            else getattr(ak_array, name)
                        )
                        - params["subtract_by"]
                    )
                    * params["multiply_by"]
                )
                for name, params in pp_dict.items()
            }
        )

    # numpy_array = ak.to_numpy(result_array)
    # pylogger.info("numpy_array: ", numpy_array)
    # pylogger.info("numpy_array.shape: ", numpy_array.shape)
    # reshaped_numpy_array = numpy_array.reshape(input_shape)
    # pylogger.info("reshaped_numpy_array: ", reshaped_numpy_array)
    # return ak.from_numpy(reshaped_numpy_array)
    return result_array


# define a function to sort ak.Array by pt
def sort_by_pt(constituents: ak.Array, ascending: bool = False):
    """Sort ak.Array of jet constituents by the pt
    Args:
        constituents (ak.Array): constituents array that should be sorted by pt.
            It should have a pt attribute.
        ascending (bool, optional): If True, the first value in each sorted
            group will be smallest; if False, the order is from largest to
            smallest. Defaults to False.
    Returns:
        ak.Array: sorted constituents array
    """
    if isinstance(constituents, ak.Array):
        try:
            temppt = constituents.pt
        except AttributeError:
            raise AttributeError(
                "Trying to sort an ak.Array without a pt attribute. Please check the input."
            )
    indices = ak.argsort(temppt, axis=1, ascending=ascending)
    return constituents[indices]


def ak_smear(arr, sigma=0, seed=42):
    """Helper function to smear an array of values by a given sigma.

    Parameters
    ----------
    arr : awkward array
        The array to smear
    sigma : float, optional
        The sigma of the smearing, by default 0 (i.e. no smearing)
    seed : int, optional
        Seed for the random number generator, by default 42
    """
    # Convert it to a 1D numpy array and perform smearing
    numpy_arr = ak.to_numpy(arr.layout.content)

    if sigma != 0:
        rng = np.random.default_rng(seed)
        numpy_arr = rng.normal(numpy_arr, sigma)

    # Convert it back to awkward form
    return ak.Array(ak.contents.ListOffsetArray(arr.layout.offsets, ak.Array(numpy_arr).layout))


def ak_clip(arr, clip_min=None, clip_max=None):
    """Helper function to clip the values of an array.

    Parameters
    ----------
    arr : awkward array
        The array to clip
    clip_min : float, optional
        Minimum value to clip to, by default None
    clip_max : float, optional
        Maximum value to clip to, by default None
    """
    # Convert it to a 1D numpy array and perform clipping
    numpy_arr = ak.to_numpy(arr.layout.content)

    if clip_min is not None:
        numpy_arr = np.clip(numpy_arr, clip_min, None)

    if clip_max is not None:
        numpy_arr = np.clip(numpy_arr, None, clip_max)

    # Convert it back to awkward form
    return ak.Array(ak.contents.ListOffsetArray(arr.layout.offsets, ak.Array(numpy_arr).layout))


def count_appearances(arr, mask, count_up_to: int = 10):
    """
    Parameters
    ----------
    arr : np.ndarray
        Array of integers, shape (n_jets, n_constituents)
    mask : np.ndarray
        Mask array, shape (n_jets, n_constituents)
    count_up_to : int, optional
        The maximum number of appearances to check for, by default 10

    Returns
    -------
    np.ndarray
        Array of shape (n_jets, n_tokens) containing the counts of each token.
        I.e. if the maximum token number is 5, the array will have 5 columns
        indicating how many times each token appears in each jet.
    np.ndarray
        Array of shape (n_jets, count_up_to) containing the number of tokens
        that appear 0, 1, 2, 3, ... times in each jet.
    np.ndarray
        Array of shape (n_jets, count_up_to) containing the fraction of tokens
        that appear 0, 1, 2, 3, ... times in each jet.
    """
    # fill the masked values with one above the maximum value in the array
    arr = np.where(mask != 0, arr, np.max(arr) + 1)

    # Count the occurrences of each integer in each row
    counts = np.array([np.bincount(row) for row in arr])
    # remove the last column, which is the count of the maximum (fill) value
    counts = counts[:, :-1]

    # calculate how many tokens appear 0, 1, 2, 3, ... times
    n_token_appearances = []
    for i in range(count_up_to + 1):
        n_token_appearances.append(np.sum(np.array(counts) == i, axis=1))

    # calculate the percentages of tokens that appear 0, 1, 2, 3, ... times
    n_tokens_total = np.sum(mask, axis=1)
    frac_token_appearances = np.array(
        [n * i / n_tokens_total for i, n in enumerate(n_token_appearances)]
    )

    return counts, np.array(n_token_appearances).T, frac_token_appearances.T


def fix_padded_logits(logits, mask, factor=1e6):
    """Used to fix a tensor of logits if the sequences are padded after some token. The logits of
    the padded values are all set to 0, except for the first value, which is set to `factor`. This
    is useful when using the logits to calculate the loss.

    Parameters
    ----------
    logits : torch.Tensor
        Tensor of logits. Shape (batch_size, seq_len, n_tokens)
    mask : torch.Tensor
        Mask tensor. Shape (batch_size, seq_len)
    factor : float, optional
        Value to set the first token of the padded values to. Default is 1e6.

    Returns
    -------
    torch.Tensor
        Fixed logits.
    """
    # fix the padded logits
    logits = logits * mask.unsqueeze(dim=-1)
    # set the logits of padded values to [1e6, -1e6, -1e6, ...]
    logits = logits + torch.cat(
        [
            (~mask).unsqueeze(-1) * factor,
            torch.zeros_like(logits[:, :, 1:]),
        ],
        dim=-1,
    )
    return logits
