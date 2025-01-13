import json
import math

import awkward as ak
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
import vector
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.stats import wasserstein_distance

import gabbro.plotting.utils as plot_utils
from gabbro.metrics.utils import quantiled_kl_divergence
from gabbro.plotting.utils import plot_ratios
from gabbro.utils.utils import (
    KL,
    find_max_energy_z,
    get_COG_ak,
    sum_energy_per_layer,
    sum_energy_per_radial_distance,
    write_distances_to_json,
)

vector.register_awkward()


def binclip(x, bins, dropinf=False):
    binfirst_center = bins[0] + (bins[1] - bins[0]) / 2
    binlast_center = bins[-2] + (bins[-1] - bins[-2]) / 2
    if dropinf:
        print("Dropping inf")
        print("len(x) before:", len(x))
        x = x[~np.isinf(x)]
        print("len(x) after:", len(x))
    return np.clip(x, binfirst_center, binlast_center)


def get_bin_centers_and_bin_heights_from_hist(hist):
    """Return the bin centers and bin heights from a histogram.

    Parameters
    ----------
    hist : tuple
        The output of matplotlib hist.

    Returns
    -------
    bin_centers : array-like
        The bin centers.
    bin_heights : array-like
        The bin heights.
    """
    bin_centers = (hist[1][:-1] + hist[1][1:]) / 2
    bin_heights = hist[0]
    return bin_centers, bin_heights


def plot_hist_with_ratios(
    comp_dict: dict,
    bins: np.ndarray,
    ax_upper: plt.Axes,
    ax_ratio: plt.Axes = None,
    ref_dict: dict = None,
    ratio_range: tuple = None,
    xlabel: str = None,
    logy: bool = False,
    leg_loc: str = "best",
    underoverflow: bool = True,
    leg_title: str = None,
    leg_ncols: int = 1,
    return_hist_curve: bool = False,
):
    """Plot histograms of the reference and comparison arrays, and their ratio.

    Parameters:
    ----------
    ax_upper : plt.Axes
        Axes for the upper panel.
    ax_ratio : plt.Axes
        Axes for the ratio panel.
    ref_dict : dict
        Dict with {id: {"arr": ..., "hist_kwargs": ...}, ...} of the reference array.
    comp_dict : dict
        Dict with {id: {"arr": ..., "hist_kwargs": ...}, ...} of the comparison arrays.
    bins : np.ndarray
        Bin edges for the histograms.
    ratio_range : tuple, optional
        Range of the y-axis for the ratio plot.
    xlabel : str, optional
        Label for the x-axis.
    logy : bool, optional
        Whether to plot the y-axis in log scale.
    leg_loc : str, optional
        Location of the legend.
    underoverflow : bool, optional
        Whether to include underflow and overflow bins. Default is True.
    leg_title : str, optional
        Title of the legend.
    leg_ncols : int, optional
        Number of columns in the legend. Default is 1.
    return_hist_curve : bool, optional
        Whether to return the histogram curves in a dict. Default is False.

    Returns
    -------
    hist_curve_dict : dict
        Dict with {id: (bin_centers, bin_heights), ...} of the histogram curves.
        Only returned if `return_hist_curve` is True. Both bin_centers and bin_heights
        are array-like.
    """

    legend_handles = []
    hist_curve_dict = {}

    if ref_dict is not None:
        ref_arr = list(ref_dict.values())[0]
        ref_label = list(ref_dict.keys())[0]
        kwargs_ref = dict(histtype="stepfilled", color="k", alpha=0.25, label=ref_label)

    if leg_title is not None:
        # plot empty array with alpha 0 to create a legend entry
        ax_upper.hist([], alpha=0, label=leg_title)

    kwargs_common = dict(bins=bins, density=True)
    if ref_dict is not None:
        hist_ref = ax_upper.hist(binclip(ref_arr["arr"], bins), **kwargs_common, **kwargs_ref)

    if ax_ratio is not None:
        ax_ratio.axhline(1, color="black", linestyle="--", lw=1)

    # loop over entries in comp_dict and plot them
    for i, (arr_id, arr_dict) in enumerate(comp_dict.items()):
        kwargs_comp = dict(histtype="step") | arr_dict.get("hist_kwargs", {})
        if "linestyle" in kwargs_comp:
            if kwargs_comp["linestyle"] == "dotted":
                kwargs_comp["linestyle"] = plot_utils.get_good_linestyles("densely dotted")
        hist_comp = ax_upper.hist(binclip(arr_dict["arr"], bins), **kwargs_common, **kwargs_comp)
        if return_hist_curve:
            hist_curve_dict[arr_id] = get_bin_centers_and_bin_heights_from_hist(hist_comp)
        legend_handles.append(
            Line2D(
                [],
                [],
                color=kwargs_comp.get("color", "C1"),
                lw=kwargs_comp.get("lw", 1),
                label=kwargs_comp.get("label", arr_id),
                linestyle=kwargs_comp.get("linestyle", "-"),
            )
        )
        if ax_ratio is not None:
            # calculate and plot ratio
            ratio = hist_comp[0] / hist_ref[0]
            # duplicate the first entry to avoid a gap in the plot (due to step plot)
            ratio = np.append(np.array(ratio[0]), np.array(ratio))
            bin_edges = hist_ref[1]
            ax_ratio.step(bin_edges, ratio, where="pre", **arr_dict.get("hist_kwargs", {}))

    ax_upper.legend(
        # handles=legend_handles,
        loc=leg_loc,
        frameon=False,
        title=leg_title,
        ncol=leg_ncols,
    )
    # re-do legend, with the first handle kep and the others replaced by the new list
    old_handles, old_labels = ax_upper.get_legend_handles_labels()
    new_handles = old_handles[:1] + legend_handles if ref_dict is not None else legend_handles
    ax_upper.legend(
        handles=new_handles,
        loc=leg_loc,
        frameon=False,
        title=leg_title,
        ncol=leg_ncols,
    )
    ax_upper.set_ylabel("Normalized")

    ax_upper.set_xlim(bins[0], bins[-1])

    if ax_ratio is not None:
        ax_ratio.set_xlim(bins[0], bins[-1])
        ax_upper.set_xticks([])

    if ratio_range is not None:
        ax_ratio.set_ylim(*ratio_range)
    if xlabel is not None:
        if ax_ratio is not None:
            ax_ratio.set_xlabel(xlabel)
        else:
            ax_upper.set_xlabel(xlabel)
    if logy:
        ax_upper.set_yscale("log")
    return hist_curve_dict if return_hist_curve else None


def plot_two_shower_versions(const1, const2, label1="version1", label2="version2", title=None):
    """Plot the constituent and shower features for two shower collections.

    Parameters:
    ----------
    const1 : awkward array
        Constituents of the first shower collection.
    const2 : awkward array
        Constituents of the second shower collection.
    title : str, optional
        Title of the plot.
    """

    showers1 = ak.sum(const1, axis=1)
    showers2 = ak.sum(const2, axis=1)

    fig, axarr = plt.subplots(4, 4, figsize=(12, 8))
    histkwargs = dict(bins=100, density=True, histtype="step")

    part_feats = ["pt", "eta", "phi", "mass"]
    for i, feat in enumerate(part_feats):
        axarr[0, i].hist(ak.flatten(const1[feat]), **histkwargs, label=label1)
        axarr[0, i].hist(ak.flatten(const2[feat]), **histkwargs, label=label1)
        axarr[0, i].set_xlabel(f"Constituent {feat}")
        # plot the difference
        axarr[1, i].hist(
            ak.flatten(const2[feat]) - ak.flatten(const1[feat]),
            **histkwargs,
            label=f"{label2} - {label1}",
        )
        axarr[1, i].set_xlabel(f"Constituent {feat} resolution")

    shower_feats = ["pt", "eta", "phi", "mass"]
    for i, feat in enumerate(shower_feats):
        axarr[2, i].hist(getattr(showers1, feat), **histkwargs, label=label1)
        axarr[2, i].hist(getattr(showers2, feat), **histkwargs, label=label2)
        axarr[2, i].set_xlabel(f"shower {feat}")
        axarr[3, i].hist(
            getattr(showers2, feat) - getattr(showers1, feat),
            **histkwargs,
            label=f"{label2} - {label1}",
        )
        axarr[3, i].set_xlabel(f"shower {feat} resolution")

    axarr[0, 0].legend(frameon=False)
    axarr[1, 0].legend(frameon=False)
    axarr[2, 0].legend(frameon=False)
    axarr[3, 0].legend(frameon=False)

    if title is not None:
        fig.suptitle(title)

    fig.tight_layout()
    # plt.show()
    return fig, axarr


def plot_features(
    ak_array_dict,
    names=None,
    label_prefix=None,
    flatten=True,
    histkwargs=None,
    legend_only_on=None,
    legend_kwargs={},
    ax_rows=1,
    decorate_ax_kwargs={},
    bins_dict=None,
    colors=None,
):
    """Plot the features of the constituents or showers.

    Parameters:
    ----------
    ak_array_dict : dict of awkward array
        Dict with {"name": ak.Array, ...} of the constituents or showers to plot.
    names : list of str or dict, optional
        Names of the features to plot. Either a list of names, or a dict of {"name": "label", ...}.
    label_prefix : str, optional
        Prefix for the plot x-axis labels.
    flatten : bool, optional
        Whether to flatten the arrays before plotting. Default is True.
    histkwargs : dict, optional
        Keyword arguments passed to plt.hist.
    legend_only_on : int, optional
        Plot the legend only on the i-th subplot. Default is None.
    legend_kwargs : dict, optional
        Keyword arguments passed to ax.legend.
    ax_rows : int, optional
        Number of rows of the subplot grid. Default is 1.
    decorate_ax_kwargs : dict, optional
        Keyword arguments passed to `decorate_ax`.
    bins_dict : dict, optional
        Dict of {name: bins} for the histograms. `name` has to be the same as the keys in `names`.
    colors : list, optional
        List of colors for the histograms. Has to have the same length as the number of arrays.
        If shorter, the colors will be repeated.
    """

    default_hist_kwargs = {"density": True, "histtype": "step", "bins": 100}

    # setup colors
    if colors is not None:
        if len(colors) < len(ak_array_dict):
            print(
                "Warning: colors list is shorter than the number of arrays. "
                "Will use default colors for remaining ones."
            )
            colors = colors + [f"C{i}" for i in range(len(ak_array_dict) - len(colors))]

    if histkwargs is None:
        histkwargs = default_hist_kwargs
    else:
        histkwargs = default_hist_kwargs | histkwargs

    # create the bins dict
    if bins_dict is None:
        bins_dict = {}
    # loop over all names - if the name is not in the bins_dict, use the default bins
    for name in names:
        if name not in bins_dict:
            bins_dict[name] = histkwargs["bins"]

    # remove default bins from histkwargs
    histkwargs.pop("bins")

    if isinstance(names, list):
        names = {name: name for name in names}

    ax_cols = len(names) // ax_rows + 1

    fig, axarr = plt.subplots(ax_rows, ax_cols, figsize=(3 * ax_cols, 2 * ax_rows))
    axarr = axarr.flatten()

    legend_handles = []
    legend_labels = []

    for i_label, (label, ak_array) in enumerate(ak_array_dict.items()):
        color = colors[i_label] if colors is not None else f"C{i_label}"
        legend_labels.append(label)
        for i, (feat, feat_label) in enumerate(names.items()):
            if flatten:
                values = ak.flatten(getattr(ak_array, feat))
            else:
                values = getattr(ak_array, feat)

            if not isinstance(bins_dict[feat], int):
                values = binclip(values, bins_dict[feat])

            _, _, patches = axarr[i].hist(values, **histkwargs, bins=bins_dict[feat], color=color)
            axarr[i].set_xlabel(
                feat_label if label_prefix is None else f"{label_prefix} {feat_label}"
            )
            if i == 0:
                legend_handles.append(
                    Line2D(
                        [],
                        [],
                        color=patches[0].get_edgecolor(),
                        lw=patches[0].get_linewidth(),
                        label=label,
                        linestyle=patches[0].get_linestyle(),
                    )
                )

    legend_kwargs["handles"] = legend_handles
    legend_kwargs["labels"] = legend_labels
    legend_kwargs["frameon"] = False
    for i, _ax in enumerate(axarr):
        if legend_only_on is None:
            _ax.legend(**legend_kwargs)
        else:
            if i == legend_only_on:
                _ax.legend(**legend_kwargs)

        plot_utils.decorate_ax(_ax, **decorate_ax_kwargs)

    fig.tight_layout()
    return fig, axarr


def plot_features_pairplot(
    arr,
    names=None,
    pairplot_kwargs={},
    input_type="ak_constituents",
):
    """Plot the features of the constituents or showers using a pairplot.

    Parameters:
    ----------
    arr : awkward array or numpy array
        Constituents or showers.
    part_names : list or dict, optional
        List of names of the features to plot, or dict of {"name": "label", ...}.
    pairplot_kwargs : dict, optional
        Keyword arguments passed to sns.pairplot.
    input_type : str, optional
        Type of the input array. Can be "ak_constituents", "ak_showers", or "np_flat".
        "ak_constituents" is an awkward array of shower constituents of shape `(n_showers, <var>, n_features)`.
        "ak_showers" is an awkward array of showers of shape `(n_showers, n_features)`.
        "np_flat" is a numpy array of shape `(n_entries, n_features)`


    Returns:
    --------
    pairplot : seaborn.axisgrid.PairGrid
        Pairplot object of the features.
    """

    if isinstance(names, list):
        names = {name: name for name in names}

    sns.set_style("dark")
    # create a dataframe from the awkward array
    if input_type == "ak_constituents":
        df = pd.DataFrame(
            {feat_label: ak.flatten(getattr(arr, feat)) for feat, feat_label in names.items()}
        )
    elif input_type == "ak_showers":
        df = pd.DataFrame({feat_label: getattr(arr, feat) for feat, feat_label in names.items()})
    elif input_type == "np_flat":
        df = pd.DataFrame(
            {feat_label: arr[:, i] for i, (feat, feat_label) in enumerate(names.items())}
        )
    else:
        raise ValueError(f"Invalid input_type: {input_type}")
    pairplot = sns.pairplot(df, kind="hist", **pairplot_kwargs)
    plt.show()

    # reset the style
    plt.rcdefaults()

    return pairplot


def plot_shower_features(
    generated_features: ak = None,
    real_features: ak = None,
    colours: list = ["cornflowerblue", "darkorange"],
    labels: list = ["Real", "Generated"],
):
    """Plot the features of the constituents or showers.

    Parameters:
    ----------
    generated_features : awkward array
        Features of the generated showers.
    real_features : awkward array
        Features of the real showers.
    """

    voxel = ak.to_numpy(ak.num(real_features["x"]))
    voxel_gen = ak.to_numpy(ak.num(generated_features["x"]))

    shower_energy = ak.to_numpy(ak.sum(real_features["energy"], axis=1))
    shower_energy_gen = ak.to_numpy(ak.sum(generated_features["energy"], axis=1))

    max_z = find_max_energy_z(real_features["energy"], real_features["z"])
    max_z_gen = find_max_energy_z(generated_features["energy"], generated_features["z"])

    x_zero = ak.to_numpy(get_COG_ak(real_features["x"], real_features["energy"]))
    y_zero = ak.to_numpy(get_COG_ak(real_features["y"], real_features["energy"]))
    z_zero = ak.to_numpy(get_COG_ak(real_features["z"], real_features["energy"]))

    x_zero_gen = ak.to_numpy(get_COG_ak(generated_features["x"], generated_features["energy"]))
    y_zero_gen = ak.to_numpy(get_COG_ak(generated_features["y"], generated_features["energy"]))
    z_zero_gen = ak.to_numpy(get_COG_ak(generated_features["z"], generated_features["energy"]))

    x = ak.flatten(real_features["x"]).to_numpy()
    y = ak.flatten(real_features["y"]).to_numpy()
    z = ak.flatten(real_features["z"]).to_numpy()
    energy = ak.flatten(real_features["energy"]).to_numpy()

    x_gen = ak.flatten(generated_features["x"]).to_numpy()
    y_gen = ak.flatten(generated_features["y"]).to_numpy()
    z_gen = ak.flatten(generated_features["z"]).to_numpy()
    energy_gen = ak.flatten(generated_features["energy"]).to_numpy()

    x_bin_min = min(x) - 1.5
    x_bin_max = max(x) + 2.5
    y_bin_min = x_bin_min
    y_bin_max = x_bin_max
    z_bin_min = x_bin_min
    z_bin_max = x_bin_max

    fig = plt.figure(figsize=(18, 12), facecolor="white")
    gs = GridSpec(2, 3)
    ############################################################
    # First Histogram - Energy Plots
    ############################################################

    bins = np.logspace(np.log(0.1), np.log(max(energy)), 150, base=np.e)
    ax0 = fig.add_subplot(gs[0])
    ax0.set_title("Visible Energy")
    ax0.hist(
        [energy, energy_gen],
        bins=bins,
        histtype="step",
        lw=2,
        alpha=0.5,
        label=labels,
        color=colours,
    )
    wasserstein_dist = wasserstein_distance(energy, energy_gen)
    kl_divergence = KL(energy, energy_gen, bins)

    ax0.text(
        0.05,
        0.95,
        f"Wasserstein Distance: {wasserstein_dist:.3f}",
        transform=plt.gca().transAxes,
    )
    ax0.text(0.05, 0.90, f"KL Divergence: {kl_divergence:.3f}", transform=plt.gca().transAxes)
    ax0.set_xlabel("Visible energy (MeV)")
    ax0.set_ylabel("a.u.")
    ax0.legend(loc="upper right")
    ax0.set_xscale("log")
    ax0.set_yscale("log")

    # Energy Sum Histogram
    ax1 = fig.add_subplot(gs[3])
    ax1.set_title("Energy Sum")
    data1 = shower_energy
    data2 = shower_energy_gen
    ax1.hist(
        [data1, data2],
        bins=30,
        histtype="step",
        lw=2,
        alpha=1.0,
        label=labels,
        color=colours,
    )
    wasserstein_dist = wasserstein_distance(data1, data2)
    kl_divergence = KL(data1, data2, 30)
    ax1.text(
        0.05,
        0.95,
        f"Wasserstein Distance: {wasserstein_dist:.3f}",
        transform=plt.gca().transAxes,
    )
    ax1.text(0.05, 0.90, f"KL Divergence: {kl_divergence:.3f}", transform=plt.gca().transAxes)
    ax1.set_xlabel("Visible energy sum (MeV)")
    ax1.set_ylabel("a.u.")
    ax1.legend(loc="upper right")

    # z-start-layer

    # Create a new figure
    ax2 = fig.add_subplot(gs[4])
    ax2.set_title("z start layer")
    step = math.ceil(z_bin_max / 11)
    bins = np.arange(z_bin_min, z_bin_max)
    ax2.hist(
        [max_z, max_z_gen],
        bins=bins,
        histtype="step",
        lw=2,
        alpha=1.0,
        color=colours,
        label=labels,
    )
    wasserstein_dist = wasserstein_distance(max_z, max_z_gen)
    kl_divergence = KL(max_z, max_z_gen, bins)
    ax2.text(
        0.05,
        0.95,
        f"Wasserstein Distance: {wasserstein_dist:.3f}",
        transform=plt.gca().transAxes,
    )
    ax2.text(0.05, 0.90, f"KL Divergence: {kl_divergence:.3f}", transform=plt.gca().transAxes)
    ax2.set_xlabel("z")
    ax2.set_ylabel("a.u.")
    ax2.ticklabel_format(
        axis="y", style="sci", scilimits=(0, 0), useMathText=True
    )  # Set scientific notation for y-axis

    ax2.set_xticks(np.arange(z_bin_min, z_bin_max, step))
    ax2.legend(loc="upper right")

    # Plot for only y-scale logarithmic
    ax3 = fig.add_subplot(gs[1])
    ax3.set_title("Visible Energy")
    ax3.hist(
        [energy, energy_gen],
        bins=150,
        histtype="step",
        lw=2,
        alpha=0.5,
        label=labels,
        color=colours,
    )

    wasserstein_dist = wasserstein_distance(energy, energy_gen)
    kl_divergence = KL(energy, energy_gen, 150)
    ax3.text(
        0.05,
        0.95,
        f"Wasserstein Distance: {wasserstein_dist:.3f}",
        transform=plt.gca().transAxes,
    )
    ax3.text(0.05, 0.90, f"KL Divergence: {kl_divergence:.3f}", transform=plt.gca().transAxes)

    ax3.set_xlabel("Visible energy (MeV)")
    ax3.set_ylabel("a.u.")
    ax3.legend(loc="upper right")
    ax3.set_yscale("log")

    # Plot for only x-scale logarithmic
    ax4 = fig.add_subplot(gs[2])
    bins = np.logspace(np.log(0.1), np.log(max(energy)), 150, base=np.e)
    ax4.set_title("Visible Energy")
    ax4.hist(
        [energy, energy_gen],
        bins,
        histtype="step",
        lw=2,
        alpha=0.5,
        label=labels,
        color=colours,
    )

    ax4.set_xlabel("Visible energy (MeV)")
    ax4.set_ylabel("a.u.")
    ax4.legend(loc="upper right")
    ax4.set_xscale("log")
    wasserstein_dist = wasserstein_distance(energy, energy_gen)
    kl_divergence = KL(energy, energy_gen, bins)
    ax4.text(
        0.05,
        0.95,
        f"Wasserstein Distance: {wasserstein_dist:.3f}",
        transform=plt.gca().transAxes,
    )
    ax4.text(0.05, 0.90, f"KL Divergence: {kl_divergence:.3f}", transform=plt.gca().transAxes)

    # Number of Hits Histogram
    ax5 = fig.add_subplot(gs[5])
    ax5.set_title("Number of Hits")
    ax5.hist(
        [voxel, voxel_gen], bins=30, histtype="step", lw=2, alpha=1.0, label=labels, color=colours
    )
    ax5.set_xlabel("n_hits")
    ax5.set_ylabel("a.u.")
    ax5.legend(loc="upper right")
    wasserstein_dist = wasserstein_distance(voxel, voxel_gen)
    kl_divergence = KL(voxel, voxel_gen, 30)
    ax5.text(
        0.05,
        0.95,
        f"Wasserstein Distance: {wasserstein_dist:.3f}",
        transform=plt.gca().transAxes,
    )
    ax5.text(0.05, 0.90, f"KL Divergence: {kl_divergence:.3f}", transform=plt.gca().transAxes)

    fig.suptitle("Distributions")

    fig.tight_layout()

    ############################################################
    # Second Histogram      ---     x,y,z Distribution and 0th Moment
    ############################################################

    fig_COG = plt.figure(figsize=(18, 12), facecolor="white")
    gs2 = GridSpec(2, 3)

    ax0 = fig_COG.add_subplot(gs2[0])

    average = sum(x_zero) / len(x_zero)
    if average < 1:
        offset = 0.4
    else:
        offset = average * 0.05

    if average < 0:
        bins = np.arange(-average - offset, -average + offset, 0.005)
    else:
        bins = np.arange(average - offset, average + offset, 0.005)

    ax0.set_title("[X] distribution")
    ax0.hist(
        [x_zero, x_zero_gen],
        bins=bins,
        histtype="step",
        lw=2,
        alpha=1.0,
        color=colours,
        label=labels,
    )
    data1 = x_zero
    data2 = x_zero_gen
    wasserstein_dist = wasserstein_distance(data1, data2)
    kl_divergence = KL(data1, data2, bins)
    ax0.text(
        0.05,
        0.95,
        f"Wasserstein Distance: {wasserstein_dist:.3f}",
        transform=plt.gca().transAxes,
    )
    ax0.text(0.05, 0.90, f"KL Divergence: {kl_divergence:.3f}", transform=plt.gca().transAxes)
    ax0.set_xlabel("X")
    ax0.set_ylabel("a.u.")
    ax0.legend(loc="upper right")

    ax1 = fig_COG.add_subplot(gs2[1])
    average = sum(y_zero) / len(y_zero)
    if average < 1:
        offset = 0.4
    else:
        offset = average * 0.05

    if average < 0:
        bins = np.arange(-average - offset, -average + offset, 0.005)
    else:
        bins = np.arange(average - offset, average + offset, 0.005)
    ax1.set_title("[Y] distribution")
    ax1.hist(
        [y_zero, y_zero_gen],
        bins=bins,
        histtype="step",
        lw=2,
        alpha=1.0,
        color=colours,
        label=labels,
    )

    wasserstein_dist = wasserstein_distance(y_zero, y_zero_gen)
    kl_divergence = KL(y_zero, y_zero_gen, bins)
    ax1.text(
        0.05,
        0.95,
        f"Wasserstein Distance: {wasserstein_dist:.3f}",
        transform=plt.gca().transAxes,
    )
    ax1.text(0.05, 0.90, f"KL Divergence: {kl_divergence:.3f}", transform=plt.gca().transAxes)
    ax1.set_xlabel("Y")
    ax1.set_ylabel("a.u.")
    ax1.legend(loc="upper right")

    average = sum(z_zero) / len(z_zero)
    if average < 1:
        offset = 1.4
    else:
        offset = average * 0.45

    if average < 0:
        bins = np.arange(-average - offset, -average + offset, 0.05)
    else:
        bins = np.arange(average - offset, average + offset, 0.05)
    ax2 = fig_COG.add_subplot(gs2[2])
    ax2.set_title("[Z] distribution")
    ax2.hist(
        [z_zero, z_zero_gen],
        bins=bins,
        histtype="step",
        lw=2,
        alpha=1.0,
        color=colours,
        label=labels,
    )

    wasserstein_dist = wasserstein_distance(z_zero, z_zero_gen)
    kl_divergence = KL(z_zero, z_zero_gen, bins)
    ax2.text(
        0.05,
        0.95,
        f"Wasserstein Distance: {wasserstein_dist:.3f}",
        transform=plt.gca().transAxes,
    )
    ax2.text(0.05, 0.90, f"KL Divergence: {kl_divergence:.3f}", transform=plt.gca().transAxes)
    ax2.set_xlabel("Z")
    ax2.set_ylabel("a.u.")
    ax2.legend(loc="upper right")

    # X Distribution
    ax3 = fig_COG.add_subplot(gs2[3])
    ax3.set_title("[x] distribution")
    ax3.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
    ax3.hist(
        [x, x_gen],
        bins=np.arange(x_bin_min, x_bin_max),
        histtype="step",
        lw=2,
        alpha=1.0,
        color=colours,
        label=labels,
    )

    data1 = x
    data2 = x_gen
    wasserstein_dist = wasserstein_distance(data1, data2)
    kl_divergence = KL(data1, data2, np.arange(x_bin_min, x_bin_max))
    ax3.text(
        0.05,
        0.95,
        f"Wasserstein Distance: {wasserstein_dist:.3f}",
        transform=plt.gca().transAxes,
    )
    ax3.text(0.05, 0.90, f"KL Divergence: {kl_divergence:.3f}", transform=plt.gca().transAxes)
    ax3.set_xlabel("[x]")
    ax3.set_ylabel("Number of hits")
    ax3.set_xticks(np.arange(x_bin_min, x_bin_max, step))
    ax3.legend(loc="upper right")

    # Y Distribution
    ax4 = fig_COG.add_subplot(gs2[4])
    ax4.set_title("[y] distribution")
    ax4.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
    ax4.hist(
        [y, y_gen],
        bins=np.arange(y_bin_min, y_bin_max),
        histtype="step",
        lw=2,
        alpha=1.0,
        color=colours,
        label=labels,
    )

    data1 = y
    data2 = y_gen
    wasserstein_dist = wasserstein_distance(data1, data2)
    kl_divergence = KL(data1, data2, np.arange(y_bin_min, y_bin_max))
    ax4.text(
        0.05,
        0.95,
        f"Wasserstein Distance: {wasserstein_dist:.3f}",
        transform=plt.gca().transAxes,
    )
    ax4.text(0.05, 0.90, f"KL Divergence: {kl_divergence:.3f}", transform=plt.gca().transAxes)
    ax4.set_xlabel("[y]")
    ax4.set_ylabel("Number of hits")
    ax4.set_xticks(np.arange(y_bin_min, y_bin_max, step))
    ax4.legend(loc="upper right")

    # Z Distribution
    ax5 = fig_COG.add_subplot(gs2[5])
    ax5.set_title("[z] distribution")
    ax5.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
    ax5.hist(
        [z, z_gen],
        bins=np.arange(z_bin_min, z_bin_max),
        histtype="step",
        lw=2,
        alpha=1.0,
        color=colours,
        label=labels,
    )

    data1 = z
    data2 = z_gen
    wasserstein_dist = wasserstein_distance(data1, data2)
    kl_divergence = KL(data1, data2, np.arange(z_bin_min, z_bin_max))
    ax5.text(
        0.05,
        0.95,
        f"Wasserstein Distance: {wasserstein_dist:.3f}",
        transform=plt.gca().transAxes,
    )
    ax5.text(0.05, 0.90, f"KL Divergence: {kl_divergence:.3f}", transform=plt.gca().transAxes)
    ax5.set_xlabel("[z]")
    ax5.set_ylabel("Number of hits")
    ax5.set_xticks(np.arange(z_bin_min, z_bin_max, step))
    ax5.legend(loc="upper right")
    return fig, fig_COG


def plot_compare_gen_showers(
    feature_sets: list, labels: list = None, colors: list = None, **kwargs
):
    """Plots the features of multiple constituent or shower sets.

    Args:
        feature_sets: A list of dictionaries, each containing awkward arrays for "x", "y", "z", and "energy" features.
        labels: (Optional) A list of labels for the feature sets (defaults to 'Set 1', 'Set 2', etc.).
        colors: (Optional) A list of colors for the feature sets (defaults to a matplotlib colormap).
        kwargs: Additional keyword arguments to pass to the plotting functions.
    """

    num_sets = len(feature_sets)

    if labels is None:
        labels = [f"Set {i + 1}" for i in range(num_sets)]
    if colors is None:
        colors = plt.cm.get_cmap("tab10").colors  # Use matplotlib's colormap

    # Preprocessing & feature extraction
    extracted_features = []
    for features in feature_sets:
        # Filter voxels with energy > 0.1
        mask = features["energy"] > 0.1
        filtered_features = {
            "x": features["x"][mask],
            "y": features["y"][mask],
            "z": features["z"][mask],
            "energy": features["energy"][mask],
        }

        extracted_features.append(
            {
                "voxel": ak.to_numpy(ak.num(filtered_features["x"])),
                "energy": ak.flatten(features["energy"]).to_numpy(),  # Keep all energies here
                "shower_energy": ak.to_numpy(ak.sum(filtered_features["energy"], axis=1)),
                "max_z": find_max_energy_z(filtered_features["energy"], filtered_features["z"]),
                "x_zero": ak.to_numpy(
                    get_COG_ak(filtered_features["x"], filtered_features["energy"])
                ),
                "y_zero": ak.to_numpy(
                    get_COG_ak(filtered_features["y"], filtered_features["energy"])
                ),
                "z_zero": ak.to_numpy(
                    get_COG_ak(filtered_features["z"], filtered_features["energy"])
                ),
                "x": ak.flatten(filtered_features["x"]).to_numpy(),
                "y": ak.flatten(filtered_features["y"]).to_numpy(),
                "z": ak.flatten(filtered_features["z"]).to_numpy(),
                "distance": filtered_features["x"].to_numpy(),  # TODO maybe delete this function
                "energy_filtered": ak.flatten(filtered_features["energy"]).to_numpy(),
            }
        )

    # Plotting (two figures)
    mpl.rcParams["xtick.labelsize"] = 15
    mpl.rcParams["ytick.labelsize"] = 15
    # mpl.rcParams['font.size'] = 28
    mpl.rcParams["font.size"] = 10
    mpl.rcParams["legend.frameon"] = False
    mpl.rcParams["text.usetex"] = False
    mpl.rcParams["font.family"] = "sans-serif"

    fig = plt.figure(figsize=(18, 12), facecolor="white")
    fig_COG = plt.figure(figsize=(18, 12), facecolor="white")

    # Call the plotting functions, passing the feature sets, labels, and colors
    plot_distributions(fig, extracted_features, labels, colors, **kwargs)
    plot_cog_and_spatial(fig_COG, extracted_features, labels, colors, **kwargs)
    fig_COG.tight_layout()
    fig.tight_layout()

    return fig, fig_COG


def plot_distributions(fig, features_list, labels, colors, **kwargs):
    """Plots the distributions of energy, energy sum, number of hits, and z start layer."""
    gs = fig.add_gridspec(
        5, 3, wspace=0.3, hspace=0.1, height_ratios=[3, 0.8, 0.9, 3, 0.8]
    )  # 3 rows for the different distributions
    # print("Plotting distributions:max(features_list[z])",  max(features_list["z"]))

    # Binning setup (adjust ranges and bins as needed for your data)
    fontsize_labels = 18

    first_features = features_list[0]
    x_max = max(first_features["x"])

    if x_max < 12:  # smaller dataset
        energy_sum = 2000
        energy = 140
        z = 10.5
        n_hits = 400
    else:
        energy_sum = 2000
        energy = 70
        z = 31.5
        n_hits = 1700

    energy_bins = np.logspace(np.log10(0.01), np.log10(energy), 50)  # Logarithmic bins for energy
    energy_sum_bins = np.arange(0, energy_sum, 50)
    max_z_bins = np.arange(-1.5, z, 1)  # Linear bins for z start layer
    voxel_bins = np.arange(0, n_hits, 50)  # The number of hits
    dist_e_bins = np.arange(0, 21, 1)  # The distance

    # Energy Distribution
    ax5 = fig.add_subplot(gs[0, 0])  # vis cell energy x log
    ax0 = fig.add_subplot(gs[0, 1])  # vis cell energy x/y log
    ax4 = fig.add_subplot(gs[0, 2])  # energy over distance
    ax1 = fig.add_subplot(gs[3, 0])  # energy sum
    ax2 = fig.add_subplot(gs[3, 1])  # z start layer
    ax3 = fig.add_subplot(gs[3, 2])  # number of hits

    # looping through all input data to be plottet on the different distributions
    for features, label, color in zip(features_list, labels, colors):
        histtype = "stepfilled" if features is features_list[0] else "step"
        edgecolor = "gray" if histtype == "stepfilled" else color
        linestyle = (
            "--"
            if len(features_list) > 2
            and (
                features is features_list[2]
                or len(features_list) > 3
                and (features is features_list[3])
            )
            else "-"
        )
        alpha = 0.95
        ax0.hist(
            features["energy"],
            bins=energy_bins,
            linestyle=linestyle,
            histtype=histtype,
            edgecolor=edgecolor,
            lw=2,
            alpha=alpha,
            label=label,
            color=color,
        )
        ax1.hist(
            features["shower_energy"],
            bins=energy_sum_bins,
            histtype=histtype,
            edgecolor=edgecolor,
            linestyle=linestyle,
            lw=2,
            alpha=alpha,
            label=label,
            color=color,
        )
        ax2.hist(
            features["max_z"],
            bins=max_z_bins,
            histtype=histtype,
            edgecolor=edgecolor,
            linestyle=linestyle,
            lw=2,
            alpha=alpha,
            label=label,
            color=color,
        )
        ax3.hist(
            features["voxel"],
            bins=voxel_bins,
            histtype=histtype,
            edgecolor=edgecolor,
            linestyle=linestyle,
            lw=2,
            alpha=alpha,
            label=label,
            color=color,
        )
        ax4.hist(
            features["distance"],
            bins=dist_e_bins,
            weights=features["energy_filtered"],
            histtype=histtype,
            edgecolor=edgecolor,
            linestyle=linestyle,
            lw=2,
            alpha=alpha,
            label=label,
            color=color,
        )
        ax5.hist(
            features["energy"],
            bins=energy_bins,
            histtype=histtype,
            edgecolor=edgecolor,
            linestyle=linestyle,
            lw=2,
            alpha=alpha,
            label=label,
            color=color,
        )
    # ax0.set_xlabel("Energy (MeV)")
    ax0.set_ylabel("a.u.", fontsize=fontsize_labels)
    ax0.set_xscale("log")
    ax0.set_yscale("log")
    ax0.axvspan(0.01, 0.1, facecolor="lightgray", alpha=0.5, hatch="/")
    ax0.tick_params(axis="x", labelbottom=False)
    ymin, ymax = ax0.get_ylim()
    new_ymax = ymax + 62 * ymax
    ax0.set_ylim(ymin, new_ymax)

    # Create twin axis for ratio plot
    ax0_twin = fig.add_subplot(gs[1, 1], sharex=ax0)
    mask = [0.7, 1.3]
    plot_ratios(ax0_twin, features_list, energy_bins, "energy", labels, colors, mask=mask)
    # Add horizontal line at y=1
    ax0_twin.axhline(y=1, color="gray", linestyle="--")
    ax0_twin.axvspan(0.01, 0.1, facecolor="lightgray", alpha=0.5, hatch="/")

    # Set y-axis limits
    ax0_twin.set_ylim(mask)
    ax0_twin.set_ylabel("ratio", color="black", fontsize=fontsize_labels)
    ax0_twin.set_xlabel("visible cell energy [MeV]", fontsize=fontsize_labels)
    ax0_twin.tick_params(axis="y", labelcolor="black")

    # Energy Sum Distribution
    ax1.set_ylabel("a.u.", fontsize=fontsize_labels)
    ax1.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ax1.tick_params(axis="x", labelbottom=False)
    ymin, ymax = ax1.get_ylim()
    new_ymax = ymax + 0.35 * ymax
    ax1.set_ylim(ymin, new_ymax)
    # Create twin axis for ratio plot
    ax1_twin = fig.add_subplot(gs[4, 0], sharex=ax1)
    plot_ratios(
        ax1_twin, features_list, energy_sum_bins, "shower_energy", labels, colors, mask=mask
    )
    ax1_twin.axhline(y=1, color="gray", linestyle="--")
    # Set y-axis limits
    ax1_twin.set_ylim(mask)
    ax1_twin.set_ylabel("ratio", color="black", fontsize=fontsize_labels)
    ax1_twin.set_xlabel("energy sum [MeV]", fontsize=fontsize_labels)
    ax1_twin.tick_params(axis="y", labelcolor="black")

    # Z Start Layer Distribution
    ax2.set_ylabel("a.u.", fontsize=fontsize_labels)
    ax2.tick_params(axis="x", labelbottom=False)
    ax2.set_yscale("log")
    ymin, ymax = ax2.get_ylim()
    new_ymax = ymax + 64 * ymax
    ax2.set_ylim(ymin, new_ymax)
    # Create twin axis for ratio plot
    ax2_twin = fig.add_subplot(gs[4, 1], sharex=ax2)
    mask = [0.6, 1.4]
    plot_ratios(ax2_twin, features_list, max_z_bins, "max_z", labels, colors, mask=mask)
    ax2_twin.axhline(y=1, color="gray", linestyle="--")

    # Set y-axis limits
    ax2_twin.set_ylim(mask)
    ax2_twin.set_ylabel("ratio", color="black", fontsize=fontsize_labels)
    ax2_twin.set_xlabel("shower start layer [layer]", fontsize=fontsize_labels)
    ax2_twin.tick_params(axis="y", labelcolor="black")

    # Number of Hits (Voxel) Distribution
    ax3.set_ylabel("# showers", fontsize=fontsize_labels)
    ax3.tick_params(axis="x", labelbottom=False)
    ax3.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax3.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ymin, ymax = ax3.get_ylim()
    new_ymax = ymax + 0.44 * ymax
    ax3.set_ylim(ymin, new_ymax)

    # Create twin axis for ratio plot
    ax3_twin = fig.add_subplot(gs[4, 2], sharex=ax3)
    plot_ratios(ax3_twin, features_list, voxel_bins, "voxel", labels, colors, mask=mask)

    ax3_twin.axhline(y=1, color="gray", linestyle="--")

    # Set y-axis limits
    ax3_twin.set_ylim(mask)
    ax3_twin.set_ylabel("ratio", color="black", fontsize=fontsize_labels)
    ax3_twin.set_xlabel("number of hits", fontsize=fontsize_labels)
    ax3_twin.tick_params(axis="y", labelcolor="black")

    # Energy Distribution per Layer
    # FIXME
    ax4.set_ylabel("energy [MeV]", fontsize=fontsize_labels)
    ax4.set_yscale("log")
    ax4.tick_params(axis="x", labelbottom=False)
    ymin, ymax = ax4.get_ylim()
    new_ymax = ymax + 0.18 * ymax
    ax4.set_ylim(ymin, new_ymax)

    # Create twin axis for ratio plot
    ax4_twin = fig.add_subplot(gs[1, 2], sharex=ax4)
    mask = [0.7, 1.3]
    plot_ratios(
        ax4_twin,
        features_list,
        dist_e_bins,
        "distance",
        labels,
        colors,
        mask=mask,
        weights="energy_filtered",
    )

    ax4_twin.axhline(y=1, color="gray", linestyle="--")

    # Set y-axis limits
    ax4_twin.set_ylim(mask)
    ax4_twin.set_ylabel("ratio", color="black", fontsize=fontsize_labels)
    ax4_twin.set_xlabel("radius [pixels]", fontsize=fontsize_labels)
    ax4_twin.tick_params(axis="y", labelcolor="black")

    # Energy Distribution only x-logarithmic
    ax5.set_ylabel("a.u.", fontsize=fontsize_labels)
    ax5.set_xscale("log")
    ax5.tick_params(axis="x", labelbottom=False)
    ax5.axvspan(0.01, 0.1, facecolor="lightgray", alpha=0.5, hatch="/")
    ax5.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax5.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ymin, ymax = ax0.get_ylim()
    new_ymax = ymax + 0.34 * ymax
    ax0.set_ylim(ymin, new_ymax)
    # Create twin axis for ratio plot
    ax5_twin = fig.add_subplot(gs[1, 0], sharex=ax5)

    plot_ratios(ax5_twin, features_list, energy_bins, "energy", labels, colors, mask=mask)

    ax5_twin.axhline(y=1, color="gray", linestyle="--")

    # Set y-axis limits
    ax5_twin.set_ylim(mask)
    ax5_twin.set_ylabel("ratio", color="black", fontsize=fontsize_labels)
    ax5_twin.set_xlabel("visible cell energy [MeV]", fontsize=fontsize_labels)
    ax5_twin.tick_params(axis="y", labelcolor="black")
    ax5_twin.axvspan(0.01, 0.1, facecolor="lightgray", alpha=0.5, hatch="/")

    # Add legend to the first subplot (energy)
    legend_elements = [
        Line2D([0], [0], color=color, lw=2, label=label) for color, label in zip(colors, labels)
    ]
    # Create the figure
    ax5.legend(handles=legend_elements, loc="upper right")
    ax2.legend(handles=legend_elements, loc="upper right")
    ax3.legend(handles=legend_elements, loc="upper right")
    ax0.legend(handles=legend_elements, loc="upper right")
    ax4.legend(handles=legend_elements, loc="upper right")
    ax1.legend(handles=legend_elements, loc="upper right")

    # Add divergence metrics to the plots
    if len(features_list) == 2:
        for ax in [ax0, ax4, ax5]:
            add_divergence_metrics(
                ax,
                features_list[0]["energy"],
                features_list[1]["energy"],
                energy_bins,
                "energy",
                fontsize=fontsize_labels - 2,
                **kwargs,
            )
        add_divergence_metrics(
            ax1,
            features_list[0]["shower_energy"],
            features_list[1]["shower_energy"],
            energy_sum_bins,
            "energy_sum",
            fontsize=fontsize_labels - 2,
            **kwargs,
        )
        add_divergence_metrics(
            ax2,
            features_list[0]["max_z"],
            features_list[1]["max_z"],
            max_z_bins,
            "max_z",
            fontsize=fontsize_labels - 2,
            **kwargs,
        )
        add_divergence_metrics(
            ax3,
            features_list[0]["voxel"],
            features_list[1]["voxel"],
            voxel_bins,
            "n_hits",
            fontsize=fontsize_labels - 2,
            **kwargs,
        )


def plot_cog_and_spatial(fig_COG, features_list, labels, colors, **kwargs):
    """Plots the COG distributions and spatial distributions of x, y, and z."""
    gs2 = fig_COG.add_gridspec(
        5, 3, wspace=0.3, hspace=0.1, height_ratios=[3, 0.8, 0.9, 3, 0.8]
    )  # 3 rows for the different distributions
    fontsize_labels = 18
    legend_elements = [
        Line2D([0], [0], color=color, lw=2, label=label) for color, label in zip(colors, labels)
    ]

    # COG Distribution Plots
    for i in range(3):
        ax = fig_COG.add_subplot(gs2[0, i])
        ax_twin = fig_COG.add_subplot(gs2[1, i], sharex=ax)
        string = "x_zero" if i == 0 else "y_zero" if i == 1 else "z_zero"
        for features, label, color in zip(features_list, labels, colors):
            histtype = "stepfilled" if features is features_list[0] else "step"
            edgecolor = "gray" if histtype == "stepfilled" else color
            linestyle = (
                "--"
                if len(features_list) > 2
                and (
                    features is features_list[2]
                    or len(features_list) > 3
                    and (features is features_list[3])
                )
                else "-"
            )

            data = features[string]
            average = np.mean(data)

            if average < 7:  # smaller rebinned dataset
                average = 4.5
                # for z 1.4, for x and y 0.4
                offset = 1.4 if i == 2 else 0.4
                # for z 0.2, for x and y 0.05
                steps = 0.2 if i == 2 else 0.05

            else:  # Full resolution dataset
                average = 14.5
                # for z 1.4, for x and y 0.4
                offset = 8 if i == 2 else 0.4
                # for z 0.25, for x and y 0.05
                steps = 0.5 if i == 2 else 0.025

            bins = (
                np.arange(average - offset, average + offset, steps)
                if average >= 0
                else np.arange(-average - offset, -average + offset, steps)
            )
            ax.hist(
                data,
                bins=bins,
                histtype=histtype,
                lw=2,
                alpha=0.8,
                linestyle=linestyle,
                label=label,
                edgecolor=edgecolor,
                color=color,
            )
        mask = [0.5, 1.5]
        plot_ratios(ax_twin, features_list, bins, string, labels, colors, mask=mask)
        ax_twin.set_xlabel(
            f"center of gravity {chr(ord('X')+i)} [voxel]", fontsize=fontsize_labels
        )  # Extract the dimension (X, Y, or Z) from the title
        ax.set_ylabel("# showers", fontsize=fontsize_labels)
        ax.tick_params(axis="x", labelbottom=False)
        ax.legend(handles=legend_elements, loc="upper right")
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        ymin, ymax = ax.get_ylim()
        new_ymax = ymax + 0.28 * ymax
        ax.set_ylim(ymin, new_ymax)

        ax_twin.axhline(y=1, color="gray", linestyle="--")

        # Set y-axis limits
        ax_twin.set_ylim(mask)
        ax_twin.set_ylabel("ratio", color="black", fontsize=fontsize_labels)
        ax_twin.set_xlabel(
            f"center of gravity {chr(ord('X')+i)} [{'layer' if i == 2 else 'cell'}]",
            fontsize=fontsize_labels,
        )

        ax_twin.tick_params(axis="y", labelcolor="black")
        if len(features_list) == 2:
            add_divergence_metrics(
                ax,
                features_list[0]["x_zero" if i == 0 else "y_zero" if i == 1 else "z_zero"],
                features_list[1]["x_zero" if i == 0 else "y_zero" if i == 1 else "z_zero"],
                bins,
                "X" if i == 0 else "Y" if i == 1 else "Z",
                fontsize=fontsize_labels - 2,
                **kwargs,
            )

    # Spatial Distribution Plots
    for i in range(3):
        ax = fig_COG.add_subplot(gs2[3, i])
        ax_twin = fig_COG.add_subplot(gs2[4, i], sharex=ax)
        string = "x" if i == 0 else "y" if i == 1 else "z"
        for features, label, color in zip(features_list, labels, colors):
            histtype = "stepfilled" if features is features_list[0] else "step"
            edgecolor = "gray" if histtype == "stepfilled" else color
            linestyle = (
                "--"
                if len(features_list) > 2
                and (
                    features is features_list[2]
                    or len(features_list) > 3
                    and (features is features_list[3])
                )
                else "-"
            )
            bins = np.arange(-0.5, 31.5, 1)
            data = features[string]
            ax.hist(
                data,
                bins=bins,
                histtype=histtype,
                lw=2,
                alpha=0.8,
                linestyle=linestyle,
                label=label,
                color=color,
                edgecolor=edgecolor,
            )
        mask = [0.7, 1.3]
        plot_ratios(ax_twin, features_list, bins, string, labels, colors, mask=mask)
        ax_twin.set_xlabel(
            f"spatial distribution {chr(ord('x')+i)} [{'layer' if i == 2 else 'cell'}]",
            fontsize=fontsize_labels,
        )
        ax.set_ylabel("a.u.", fontsize=fontsize_labels)
        ax.legend(handles=legend_elements, loc="upper right")
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        ax.tick_params(axis="x", labelbottom=False)
        ymin, ymax = ax.get_ylim()
        new_ymax = ymax + 0.28 * ymax
        ax.set_ylim(ymin, new_ymax)
        ax_twin.axhline(y=1, color="gray", linestyle="--")

        # Set y-axis limits
        ax_twin.set_ylim(mask)
        ax_twin.set_ylabel("ratio", color="black", fontsize=fontsize_labels)
        ax_twin.tick_params(axis="y", labelcolor="black")
        if len(features_list) == 2:
            add_divergence_metrics(
                ax,
                features_list[0]["x" if i == 0 else "y" if i == 1 else "z"],
                features_list[1]["x" if i == 0 else "y" if i == 1 else "z"],
                bins,
                "x" if i == 0 else "y" if i == 1 else "z",
                fontsize=fontsize_labels - 2,
                **kwargs,
            )


def save_metrics_to_file(file_path, metrics):
    """Save metrics to a JSON file.

    Parameters:
    ----------
    file_path : str
        Path to the JSON file.
    metrics : dict
        Dictionary containing the metrics to save.
    """
    try:
        with open(file_path) as file:
            data = json.load(file)
    except FileNotFoundError:
        data = []

    data.append(metrics)

    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)


def add_divergence_metrics(ax, data1, data2, bins, feature, fontsize, **kwargs):
    """Calculates and adds Wasserstein distance and KL divergence to the plot."""
    wasserstein_dist = wasserstein_distance(data1, data2)
    bins = int(len(bins))
    kl_divergence = quantiled_kl_divergence(data1, data2, bins, False)
    filepath = kwargs.get("filepath", None)
    weights = kwargs.get("weights", None)
    n_data = kwargs.get("n_data", None)
    transfer_learning = kwargs.get("transfer_learning", False)

    if transfer_learning:
        write_distances_to_json(
            kl_divergence, wasserstein_dist, filepath, weights, n_data, feature
        )

    ax.text(
        1.0,
        1.05,
        f"W-distance: {wasserstein_dist:.2e}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=fontsize,
    )
    ax.text(
        1.0,
        1.10,
        f"KLD: {kl_divergence:.2e}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=fontsize,
    )


def plot_paper_plots(feature_sets: list, labels: list = None, colors: list = None, **kwargs):
    """Plots the features of multiple constituent or shower sets.

    Args:
        feature_sets: A list of dictionaries, each containing awkward arrays for "x", "y", "z", and "energy" features.
        labels: (Optional) A list of labels for the feature sets (defaults to 'Set 1', 'Set 2', etc.).
        colors: (Optional) A list of colors for the feature sets (defaults to a matplotlib colormap).
        kwargs: Additional keyword arguments to pass to the plotting functions.
    """

    num_sets = len(feature_sets)

    if labels is None:
        labels = [f"Set {i + 1}" for i in range(num_sets)]
    if colors is None:
        colors = plt.cm.get_cmap("tab10").colors  # Use matplotlib's colormap

    # Preprocessing & feature extraction
    features_list = []
    for features in feature_sets:
        # Filter voxels with energy > 0.1
        mask = features["energy"] > 0.1
        filtered_features = {
            "x": features["x"][mask],
            "y": features["y"][mask],
            "z": features["z"][mask],
            "energy": features["energy"][mask],
        }

        features_list.append(
            {
                "voxel": ak.to_numpy(ak.num(filtered_features["x"])),
                "energy": ak.flatten(features["energy"]).to_numpy(),  # Keep all energies here
                "shower_energy": ak.to_numpy(ak.sum(filtered_features["energy"], axis=1)),
                # "max_z": find_max_energy_z(filtered_features["energy"], filtered_features["z"]),
                "x_zero": ak.to_numpy(
                    get_COG_ak(filtered_features["x"], filtered_features["energy"])
                ),
                "y_zero": ak.to_numpy(
                    get_COG_ak(filtered_features["y"], filtered_features["energy"])
                ),
                "z_zero": ak.to_numpy(
                    get_COG_ak(filtered_features["z"], filtered_features["energy"])
                ),
                "x": ak.flatten(filtered_features["x"]).to_numpy(),
                "y": ak.flatten(filtered_features["y"]).to_numpy(),
                "z": ak.flatten(filtered_features["z"]).to_numpy(),
                "distance": np.mean(
                    sum_energy_per_radial_distance(
                        filtered_features["x"], filtered_features["y"], filtered_features["energy"]
                    ),
                    axis=0,
                ),
                "energy_filtered": ak.flatten(filtered_features["energy"]).to_numpy(),
                "energy_per_layer": np.mean(
                    sum_energy_per_layer(filtered_features["z"], filtered_features["energy"]),
                    axis=0,
                ),
                "pixel": np.arange(0, 21) + 0.5,
                "hits": np.arange(0, 29) + 0.5,
            }
        )

    # Plotting (two figures)
    mpl.rcParams["xtick.labelsize"] = 15
    mpl.rcParams["ytick.labelsize"] = 15
    # mpl.rcParams['font.size'] = 28
    mpl.rcParams["font.size"] = 10
    mpl.rcParams["legend.frameon"] = False
    mpl.rcParams["text.usetex"] = False
    mpl.rcParams["font.family"] = "sans-serif"

    fig = plt.figure(figsize=(18, 12), facecolor="white")

    """Plots the distributions of energy, energy sum, number of hits, and z start layer."""
    gs = fig.add_gridspec(
        5, 3, wspace=0.3, hspace=0.1, height_ratios=[3, 0.8, 0.9, 3, 0.8]
    )  # 3 rows for the different distributions
    # print("Plotting distributions:max(features_list[z])",  max(features_list["z"]))

    # Binning setup (adjust ranges and bins as needed for your data)
    fontsize_labels = 18

    energy_sum = 2000
    energy = 70
    n_hits = 1700

    energy_bins = np.logspace(np.log10(0.01), np.log10(energy), 50)  # Logarithmic bins for energy
    energy_sum_bins = np.arange(0, energy_sum, 75)
    voxel_bins = np.arange(0, n_hits, 50)  # The number of hits
    dist_e_bins = np.arange(0, 21, 1)  # The distance
    bins_cog = np.arange(8, 22, 0.5)
    bins_z = np.arange(0, 31.5, 1)

    # Energy Distribution
    ax0 = fig.add_subplot(gs[0, 0])  # vis cell energy x/y log
    ax1 = fig.add_subplot(gs[0, 1])  # energy sum
    ax2 = fig.add_subplot(gs[0, 2])  # number of hits
    ax3 = fig.add_subplot(gs[3, 0])  # center of gravity Z
    ax4 = fig.add_subplot(gs[3, 1])  # spatial distribution Z
    ax5 = fig.add_subplot(gs[3, 2])  # energy over distance

    # looping through all input data to be plottet on the different distributions
    for features, label, color in zip(features_list, labels, colors):
        histtype = "stepfilled" if features is features_list[0] else "step"
        edgecolor = "gray" if histtype == "stepfilled" else color
        linestyle = (
            "--"
            if len(features_list) > 2
            and (
                features is features_list[2]
                or len(features_list) > 3
                and (features is features_list[3])
            )
            else "-"
        )
        alpha = 0.95
        ax0.hist(
            features["energy"],
            bins=energy_bins,
            linestyle=linestyle,
            histtype=histtype,
            edgecolor=edgecolor,
            lw=2,
            alpha=alpha,
            label=label,
            color=color,
        )
        ax1.hist(
            features["shower_energy"],
            bins=energy_sum_bins,
            histtype=histtype,
            edgecolor=edgecolor,
            linestyle=linestyle,
            lw=2,
            alpha=alpha,
            label=label,
            density=True,
            color=color,
        )
        ax2.hist(
            features["voxel"],
            bins=voxel_bins,
            histtype=histtype,
            edgecolor=edgecolor,
            linestyle=linestyle,
            lw=2,
            alpha=alpha,
            label=label,
            density=True,
            color=color,
        )
        ax3.hist(
            features["z_zero"],
            bins=bins_cog,
            histtype=histtype,
            lw=2,
            alpha=alpha,
            linestyle=linestyle,
            label=label,
            edgecolor=edgecolor,
            density=True,
            color=color,
        )
        ax4.hist(
            features["hits"],
            bins=bins_z,
            histtype=histtype,
            lw=2,
            alpha=alpha,
            label=label,
            color=color,
            linestyle=linestyle,
            weights=features["energy_per_layer"],
        )
        ax5.hist(
            features["pixel"],
            bins=dist_e_bins,
            weights=features["distance"],
            histtype=histtype,
            edgecolor=edgecolor,
            linestyle=linestyle,
            lw=2,
            alpha=alpha,
            label=label,
            color=color,
        )
    # ax0.set_xlabel("Energy (MeV)")
    ax0.set_ylabel("a.u.", fontsize=fontsize_labels)
    ax0.set_xscale("log")
    ax0.set_yscale("log")
    ax0.set_xlim(left=0.01)
    ax0.axvspan(0.01, 0.1, ymin=0, ymax=0.73, facecolor="lightgray", alpha=0.2, hatch="/")
    ax0.tick_params(axis="x", labelbottom=False)
    ymin, ymax = ax0.get_ylim()
    new_ymax = ymax + 1620 * ymax
    ax0.set_ylim(ymin, new_ymax)
    # Create twin axis for ratio plot

    mask = [0.7, 1.3]
    ax0_twin = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax0_twin.set_xlim(left=0.01)
    plot_ratios(ax0_twin, features_list, energy_bins, "energy", labels, colors, mask=mask)
    # Add horizontal line at y=1
    ax0_twin.axhline(y=1, color="gray", linestyle="--")
    ax0_twin.axvspan(0.01, 0.1, facecolor="lightgray", alpha=0.5, hatch="/")
    # Set y-axis limits
    ax0_twin.set_ylim(mask)
    ax0_twin.set_ylabel("ratio", color="black", fontsize=fontsize_labels)
    ax0_twin.set_xlabel("visible cell energy [MeV]", fontsize=fontsize_labels)
    ax0_twin.tick_params(axis="y", labelcolor="black")

    # Energy Sum Distribution
    ax1.set_ylabel("normalized", fontsize=fontsize_labels)
    ax1.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ax1.tick_params(axis="x", labelbottom=False)
    ymin, ymax = ax1.get_ylim()
    new_ymax = ymax + 0.45 * ymax
    ax1.set_ylim(ymin, new_ymax)
    # Create twin axis for ratio plot
    ax1_twin = fig.add_subplot(gs[1, 1], sharex=ax1)
    plot_ratios(
        ax1_twin, features_list, energy_sum_bins, "shower_energy", labels, colors, mask=mask
    )
    ax1_twin.axhline(y=1, color="gray", linestyle="--")
    # Set y-axis limits
    ax1_twin.set_ylim(mask)
    ax1_twin.set_ylabel("ratio", color="black", fontsize=fontsize_labels)
    ax1_twin.set_xlabel("energy sum [MeV]", fontsize=fontsize_labels)
    ax1_twin.tick_params(axis="y", labelcolor="black")

    # Number of Hits (Voxel) Distribution
    mask = [0.6, 1.4]
    ax2.set_ylabel("normalized", fontsize=fontsize_labels)
    ax2.tick_params(axis="x", labelbottom=False)
    ax2.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ymin, ymax = ax2.get_ylim()
    new_ymax = ymax + 0.44 * ymax
    ax2.set_ylim(ymin, new_ymax)

    # Create twin axis for ratio plot
    ax2_twin = fig.add_subplot(gs[1, 2], sharex=ax2)
    plot_ratios(ax2_twin, features_list, voxel_bins, "voxel", labels, colors, mask)

    ax2_twin.axhline(y=1, color="gray", linestyle="--")

    # Set y-axis limits
    ax2_twin.set_ylim(mask)
    ax2_twin.set_ylabel("ratio", color="black", fontsize=fontsize_labels)
    ax2_twin.set_xlabel("number of hits", fontsize=fontsize_labels)
    ax2_twin.tick_params(axis="y", labelcolor="black")

    # Center of Gravity Z Distribution
    ax3.set_ylabel("normalized", fontsize=fontsize_labels)
    ax3.tick_params(axis="x", labelbottom=False)
    ax3.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax3.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ymin, ymax = ax3.get_ylim()
    new_ymax = ymax + 0.48 * ymax
    ax3.set_ylim(ymin, new_ymax)

    # Create twin axis for ratio plot
    ax3_twin = fig.add_subplot(gs[4, 0], sharex=ax3)
    mask = (0.4, 1.6)
    plot_ratios(ax3_twin, features_list, bins_cog, "z_zero", labels, colors, mask=mask)

    ax3_twin.axhline(y=1, color="gray", linestyle="--")

    # Set y-axis limits

    ax3_twin.set_ylim(mask)
    ax3_twin.set_ylabel("ratio", color="black", fontsize=fontsize_labels)
    ax3_twin.set_xlabel("center of gravity Z [layer]", fontsize=fontsize_labels)
    ax3_twin.tick_params(axis="y", labelcolor="black")

    # Z Distribution
    ax4.set_ylabel("energy [MeV]", fontsize=fontsize_labels)
    ax4.tick_params(axis="x", labelbottom=False)
    ax4.set_yscale("log")
    ax4.set_xlim(0, 30)
    ymin, ymax = ax4.get_ylim()
    new_ymax = ymax + 40 * ymax
    ax4.set_ylim(ymin, new_ymax)

    # Create twin axis for ratio plot
    ax4_twin = fig.add_subplot(gs[4, 1], sharex=ax4)
    mask = [0.7, 1.3]
    plot_ratios(
        ax4_twin, features_list, bins_z, "hits", labels, colors, mask, weights="energy_per_layer"
    )

    ax4_twin.axhline(y=1, color="gray", linestyle="--")

    # Set y-axis limits

    ax4_twin.set_ylim(mask)
    ax4_twin.set_xlim(0, 30)
    ax4_twin.set_ylabel("ratio", color="black", fontsize=fontsize_labels)
    ax4_twin.set_xlabel("layer", fontsize=fontsize_labels)
    ax4_twin.tick_params(axis="y", labelcolor="black")

    # Energy Distribution per Layer
    ax5.set_ylabel("energy [MeV]", fontsize=fontsize_labels)
    ax5.set_yscale("log")
    ax5.set_xlim(0, 21)
    ax5.tick_params(axis="x", labelbottom=False, labelsize=fontsize_labels)
    ymin, ymax = ax5.get_ylim()
    new_ymax = ymax + 40 * ymax
    ax5.set_ylim(ymin, new_ymax)

    # Create twin axis for ratio plot
    ax5_twin = fig.add_subplot(gs[4, 2], sharex=ax5)
    mask = [0.7, 1.3]
    plot_ratios(
        ax5_twin, features_list, dist_e_bins, "pixel", labels, colors, mask, weights="distance"
    )

    ax5_twin.axhline(y=1, color="gray", linestyle="--")

    # Set y-axis limits
    ax5_twin.set_ylim(mask)
    ax5_twin.set_xlim(0, 21)
    ax5_twin.set_ylabel("ratio", color="black", fontsize=fontsize_labels)
    ax5_twin.set_xlabel("radius [pixels]", fontsize=fontsize_labels)
    ax5_twin.tick_params(axis="y", labelcolor="black")

    # Add legend to the first subplot (energy)
    legend_elements = [
        Line2D(
            [0],
            [0],
            color=color,
            lw=2,
            label=label,
            linestyle="--"
            if len(features_list) > 2
            and (
                features is features_list[2]
                or len(features_list) > 3
                and (features is features_list[3])
            )
            else "-",
        )
        for color, label, features in zip(colors, labels, features_list)
    ]
    # Create the figure
    ax5.legend(handles=legend_elements, loc="upper right", fontsize=fontsize_labels - 5, ncol=2)
    ax2.legend(handles=legend_elements, loc="upper right", fontsize=fontsize_labels - 5, ncol=2)
    ax3.legend(handles=legend_elements, loc="upper right", fontsize=fontsize_labels - 5, ncol=2)
    ax0.legend(handles=legend_elements, loc="upper right", fontsize=fontsize_labels - 5, ncol=2)
    ax4.legend(handles=legend_elements, loc="upper right", fontsize=fontsize_labels - 5, ncol=2)
    ax1.legend(handles=legend_elements, loc="upper right", fontsize=fontsize_labels - 5, ncol=2)

    return fig
