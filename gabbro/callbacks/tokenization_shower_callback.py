"""Callback for evaluating the tokenization of particles."""

import math
import os

import awkward as ak
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch.distributed as dist
import vector
from matplotlib.gridspec import GridSpec
from scipy.stats import wasserstein_distance

from gabbro.utils.arrays import ak_preprocess, np_to_akward

# from gabbro.plotting.plotting_functions import plot_p4s
from gabbro.utils.pylogger import get_pylogger
from gabbro.utils.utils import KL, get0Momentum, get_diff_construct

default_labels = {"x": "$x$", "y": "$y$", "z": "$z$", "energy": "$E$"}

pylogger = get_pylogger("TokenizationEvalCallback")
vector.register_awkward()


class TokenizationEvalCallback(L.Callback):
    def __init__(
        self,
        image_path: str = None,
        image_filetype: str = "png",
        no_trainer_info_in_filename: bool = False,
        save_result_arrays: bool = None,
    ):
        """Callback for evaluating the tokenization of particles.

        Parameters
        ----------
        image_path : str
            Path to save the images to. If None, the images are saved to the
            default_root_dir of the trainer.
        image_filetype : str
            Filetype to save the images as. Default is "png".
        no_trainer_info_in_filename : bool
            If True, the filename of the images will not contain the epoch and
            global step information. Default is False.
        save_result_arrays : bool
            If True, the results are saved as parquet file. Default is None.
        """
        super().__init__()
        self.comet_logger = None
        self.image_path = image_path
        self.image_filetype = image_filetype
        self.no_trainer_info_in_filename = no_trainer_info_in_filename
        self.save_results_arrays = save_result_arrays

    def on_validation_epoch_end(self, trainer, pl_module):
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        pl_module.concat_validation_loop_predictions()
        self.plot(trainer, pl_module, stage="val")

    def on_test_epoch_end(self, trainer, pl_module):
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        pl_module.concat_test_loop_predictions()
        self.plot(trainer, pl_module, stage="test")

    def plot(self, trainer, pl_module, stage="val"):
        if stage == "val" and not hasattr(pl_module, "val_x_original_concat"):
            pylogger.info("No validation predictions found. Skipping plotting.")
            return

        pylogger.info(
            f"Running TokenizationEvalCallback epoch: {trainer.current_epoch} step:"
            f" {trainer.global_step}"
        )
        # get loggers
        for logger in trainer.loggers:
            if isinstance(logger, L.pytorch.loggers.CometLogger):
                self.comet_logger = logger.experiment
            elif isinstance(logger, L.pytorch.loggers.WandbLogger):
                self.wandb_logger = logger.experiment

        plot_dir = (
            self.image_path if self.image_path is not None else trainer.default_root_dir + "/plots"
        )
        os.makedirs(plot_dir, exist_ok=True)
        if self.no_trainer_info_in_filename:
            plot_filename = f"{plot_dir}/evaluation_overview.{self.image_filetype}"
        else:
            plot_filename = f"{plot_dir}/epoch{trainer.current_epoch}_gstep{trainer.global_step}_overview.{self.image_filetype}"

        if stage == "val":
            x_recos = pl_module.val_x_reco_concat
            x_originals = pl_module.val_x_original_concat
            masks = pl_module.val_mask_concat
            pylogger.info(f"x_recos.shape: {x_recos.shape}")
            pylogger.info(f"x_originals.shape: {x_originals.shape}")
            pylogger.info(f"masks.shape: {masks.shape}")
            # labels = pl_module.val_labels_concat
            code_idx = pl_module.val_code_idx_concat
        elif stage == "test":
            # return and print that there are no test predictions if there are none
            if not hasattr(pl_module, "test_x_original_concat"):
                pylogger.info("No test predictions found. Skipping plotting.")
                return
            x_recos = pl_module.test_x_reco_concat
            x_originals = pl_module.test_x_original_concat
            masks = pl_module.test_mask_concat
            # labels = pl_module.test_labels_concat
            code_idx = pl_module.test_code_idx_concat
        else:
            raise ValueError(f"stage {stage} not recognized")

        if stage == "test":
            pylogger.info(f"x_original_concat.shape: {x_originals.shape}")
            pylogger.info(f"x_reco_concat.shape: {x_recos.shape}")
            pylogger.info(f"masks_concat.shape: {masks.shape}")
            # pylogger.info(f"labels_concat.shape: {labels.shape}")

        pp_dict = trainer.datamodule.hparams.dataset_kwargs_common.feature_dict

        x_reco_ak_pp = np_to_akward(x_recos, pp_dict)
        x_original_ak_pp = np_to_akward(x_originals, pp_dict)

        pylogger.info(f"x rocusntructed before preprocess {x_reco_ak_pp}")
        pylogger.info(f"x_original before preprocess: {x_original_ak_pp}")
        x_reco_ak = ak_preprocess(x_reco_ak_pp, pp_dict=pp_dict, inverse=True)
        x_original_ak = ak_preprocess(x_original_ak_pp, pp_dict=pp_dict, inverse=True)
        pylogger.info(f"x_reconstructed after preprocess {x_reco_ak}")
        pylogger.info(f"x_original after preprocess: {x_original_ak}")
        x_complete = ak.to_numpy(x_original_ak["x"])
        y_complete = ak.to_numpy(x_original_ak["y"])
        z_complete = ak.to_numpy(x_original_ak["z"])
        energy_xyz_complete = ak.to_numpy(x_original_ak["energy"])
        energy_xyz_complete = energy_xyz_complete * masks
        pylogger.info(f"energy: {energy_xyz_complete}")

        x_complete_reco = ak.to_numpy(x_reco_ak["x"])
        y_complete_reco = ak.to_numpy(x_reco_ak["y"])
        z_complete_reco = ak.to_numpy(x_reco_ak["z"])
        energy_xyz_complete_reco = ak.to_numpy(x_reco_ak["energy"])
        energy_xyz_complete_reco = energy_xyz_complete_reco * masks

        x = x_complete[masks].flatten()
        y = y_complete[masks].flatten()
        z = z_complete[masks].flatten()
        energy_xyz = energy_xyz_complete[masks].flatten()

        x_reco = x_complete_reco[masks].flatten()
        y_reco = y_complete_reco[masks].flatten()
        z_reco = z_complete_reco[masks].flatten()
        energy_xyz_reco = energy_xyz_complete_reco[masks].flatten()

        pylogger.info(f"energy: {energy_xyz}")
        pylogger.info(f"energy_reco: {energy_xyz_reco}")
        x_bin_min = min(x) - 0.5
        x_bin_max = max(x) + 1.5
        y_bin_min = min(y) - 0.5
        y_bin_max = max(y) + 1.5
        z_bin_min = min(z) - 0.5
        z_bin_max = max(z) + 1.5

        fig = plt.figure(figsize=(18, 12), facecolor="white")
        fig.suptitle("Projected Showers", fontsize=30)

        gs = GridSpec(2, 3)
        ############################################################
        # First Histogram - Energy Plots
        ############################################################
        bins = np.logspace(np.log(0.001), np.log(max(energy_xyz)), 150, base=np.e)
        ax0 = fig.add_subplot(gs[0])
        ax0.axvline(0.1, linestyle="--", color="black", label="MIP")
        ax0.set_title("Visible Energy")
        ax0.hist(
            energy_xyz,
            bins=bins,
            histtype="stepfilled",
            lw=2,
            alpha=0.5,
            label="original",
            color="silver",
        )
        ax0.hist(
            energy_xyz_reco,
            bins=bins,
            histtype="step",
            lw=2,
            alpha=1.0,
            label="reconstructed",
            color="red",
        )
        wasserstein_dist = wasserstein_distance(energy_xyz, energy_xyz_reco)
        kl_divergence = KL(energy_xyz, energy_xyz_reco, bins)

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

        # Plot for y non-logarithmic
        ax1 = fig.add_subplot(gs[1])
        ax1.set_title("Visible Energy")
        ax1.axvline(0.1, linestyle="--", color="black", label="MIP")
        bins = np.logspace(np.log(0.001), np.log(max(energy_xyz)), 150, base=np.e)
        ax1.hist(
            energy_xyz,
            bins=bins,
            histtype="stepfilled",
            lw=2,
            alpha=0.5,
            label="original",
            color="silver",
        )
        ax1.hist(
            energy_xyz_reco,
            bins=bins,
            histtype="step",
            lw=2,
            alpha=1.0,
            label="reconstructed",
            color="red",
        )
        ax1.set_xlabel("Visible energy (MeV)")
        ax1.set_ylabel("a.u.")
        ax1.set_xscale("log")
        ax1.legend(loc="upper right")
        wasserstein_dist = wasserstein_distance(energy_xyz, energy_xyz_reco)
        kl_divergence = KL(energy_xyz, energy_xyz_reco, 150)
        ax1.text(
            0.05,
            0.95,
            f"Wasserstein Distance: {wasserstein_dist:.3f}",
            transform=plt.gca().transAxes,
        )
        ax1.text(0.05, 0.90, f"KL Divergence: {kl_divergence:.3f}", transform=plt.gca().transAxes)

        # Energy Sum Histogram
        ax2 = fig.add_subplot(gs[2])
        ax2.set_title("Energy Sum")

        data1 = np.sum(energy_xyz_complete, axis=-1)
        data2 = np.sum(energy_xyz_complete_reco, axis=-1)

        ax2.hist(
            data1,
            bins=50,
            histtype="stepfilled",
            lw=2,
            alpha=1.0,
            label="original",
            color="silver",
        )
        ax2.hist(
            data2,
            bins=50,
            histtype="step",
            lw=2,
            alpha=1.0,
            label="reconstructed",
            color="red",
        )
        wasserstein_dist = wasserstein_distance(data1, data2)
        kl_divergence = KL(data1, data2, 50)
        ax2.text(
            0.05,
            0.95,
            f"Wasserstein Distance: {wasserstein_dist:.3f}",
            transform=plt.gca().transAxes,
        )
        ax2.text(0.05, 0.90, f"KL Divergence: {kl_divergence:.3f}", transform=plt.gca().transAxes)
        ax2.set_xlabel("Visible energy sum (MeV)")
        ax2.set_ylabel("a.u.")
        ax2.legend(loc="upper right")

        """
        # Number of Hits Histogram
        ax2 = fig.add_subplot(gs[2])
        ax2.set_title("Number of Hits")
        ax2.hist((energy_xyz_complete != 0).reshape(-1, size_of_event).sum(axis=1), bins=50, histtype="stepfilled", lw=2, alpha=1.0, label="original",color="silver")
        ax2.hist(
            (energy_xyz_complete_reco != 0).reshape(-1, size_of_event).sum(axis=1),
            bins=50,
            histtype="step",
            lw=2,
            alpha=1.0,
            label="reconstructed",
            color="red",
        )
        ax2.set_xlabel("n_hits")
        ax2.set_ylabel("a.u.")
        ax2.legend(loc = "upper right")
        """

        # Plot for only y-scale to 0.1
        ax3 = fig.add_subplot(gs[3])
        bins = np.logspace(np.log(0.1), np.log(max(energy_xyz)), 150, base=np.e)
        ax3.set_title("Visible Energy")
        ax3.hist(
            energy_xyz,
            bins=bins,
            histtype="stepfilled",
            lw=2,
            alpha=0.5,
            label="original",
            color="silver",
        )
        ax3.hist(
            energy_xyz_reco,
            bins=bins,
            histtype="step",
            lw=2,
            alpha=1.0,
            label="reconstructed",
            color="red",
        )

        wasserstein_dist = wasserstein_distance(energy_xyz, energy_xyz_reco)
        kl_divergence = KL(energy_xyz, energy_xyz_reco, 150)
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
        ax3.set_xscale("log")

        # Plot for only x-scale logarithmic
        ax4 = fig.add_subplot(gs[4])
        bins = np.logspace(np.log(0.1), np.log(max(energy_xyz)), 150, base=np.e)
        ax4.set_title("Visible Energy")

        ax4.hist(
            energy_xyz,
            bins,
            histtype="stepfilled",
            lw=2,
            alpha=0.5,
            label="original",
            color="silver",
        )
        ax4.hist(
            energy_xyz_reco,
            bins,
            histtype="step",
            lw=2,
            alpha=1.0,
            label="reconstructed",
            color="red",
        )
        ax4.set_xlabel("Visible energy (MeV)")
        ax4.set_ylabel("a.u.")
        ax4.legend(loc="upper right")
        ax4.set_xscale("log")
        wasserstein_dist = wasserstein_distance(energy_xyz, energy_xyz_reco)
        kl_divergence = KL(energy_xyz, energy_xyz_reco, bins)
        ax4.text(
            0.05,
            0.95,
            f"Wasserstein Distance: {wasserstein_dist:.3f}",
            transform=plt.gca().transAxes,
        )
        ax4.text(0.05, 0.90, f"KL Divergence: {kl_divergence:.3f}", transform=plt.gca().transAxes)

        # z-start-layer

        max_energy_indices = np.argmax(energy_xyz_complete, axis=1)
        max_energy_z_values = z_complete[np.arange(len(z_complete)), max_energy_indices]

        max_energy_indices_reco = np.argmax(energy_xyz_complete_reco, axis=1)
        max_energy_z_values_reco = z_complete_reco[
            np.arange(len(z_complete_reco)), max_energy_indices_reco
        ]
        ax5 = fig.add_subplot(gs[5])
        ax5.set_title("z start layer")
        step = math.ceil(z_bin_max / 11)
        bins = np.arange(z_bin_min, z_bin_max)
        ax5.hist(
            max_energy_z_values,
            bins=bins,
            histtype="stepfilled",
            lw=2,
            alpha=1.0,
            color="silver",
            label="original",
        )
        ax5.hist(
            max_energy_z_values_reco,
            bins=bins,
            histtype="step",
            lw=2,
            alpha=1.0,
            color="red",
            label="reconstructed",
        )
        wasserstein_dist = wasserstein_distance(max_energy_z_values, max_energy_z_values_reco)
        kl_divergence = KL(max_energy_z_values, max_energy_z_values_reco, bins)
        ax5.text(
            0.05,
            0.95,
            f"Wasserstein Distance: {wasserstein_dist:.3f}",
            transform=plt.gca().transAxes,
        )
        ax5.text(0.05, 0.90, f"KL Divergence: {kl_divergence:.3f}", transform=plt.gca().transAxes)

        ax5.set_xlabel("z")
        ax5.set_ylabel("a.u.")
        ax5.ticklabel_format(
            axis="y", style="sci", scilimits=(0, 0), useMathText=True
        )  # Set scientific notation for y-axis

        ax5.set_xticks(np.arange(z_bin_min, z_bin_max, step))
        ax5.legend(loc="upper right")

        fig.suptitle("Distributions")

        fig.tight_layout()

        rep = "_overview"
        filename = plot_filename.replace(rep, "_visible_energy")
        pylogger.info(f"Saving plot to {filename}")
        fig.savefig(filename)

        ############################################################
        # Second Histogram      ---     x,y,z Distribution and 0th Moment
        ############################################################

        fig_0Moment = plt.figure(figsize=(18, 12), facecolor="white")
        fig_0Moment.suptitle("0th Moment", fontsize=30)
        gs2 = GridSpec(2, 3)

        ax0 = fig_0Moment.add_subplot(gs2[0])
        x0 = get0Momentum(x_complete, energy_xyz_complete)
        x0_reco = get0Momentum(x_complete_reco, energy_xyz_complete_reco)
        average = sum(x0) / len(x0)
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
            x0,
            bins=bins,
            histtype="stepfilled",
            lw=2,
            alpha=1.0,
            color="silver",
            label="original",
        )
        ax0.hist(
            x0_reco,
            bins=bins,
            histtype="step",
            lw=2,
            alpha=1.0,
            color="red",
            label="reconstructed",
        )
        data1 = x0
        data2 = x0_reco
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

        ax1 = fig_0Moment.add_subplot(gs2[1])
        y0 = get0Momentum(y_complete, energy_xyz_complete)
        y0_reco = get0Momentum(y_complete_reco, energy_xyz_complete_reco)
        average = sum(y0) / len(y0)
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
            y0,
            bins=bins,
            histtype="stepfilled",
            lw=2,
            alpha=1.0,
            color="silver",
            label="original",
        )
        ax1.hist(
            y0_reco,
            bins=bins,
            histtype="step",
            lw=2,
            alpha=1.0,
            color="red",
            label="reconstructed",
        )
        wasserstein_dist = wasserstein_distance(y0, y0_reco)
        kl_divergence = KL(y0, y0_reco, bins)
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

        z0 = get0Momentum(z_complete, energy_xyz_complete)
        z0_reco = get0Momentum(z_complete_reco, energy_xyz_complete_reco)
        average = sum(z0) / len(z0)
        if average < 1:
            offset = 1.4
        else:
            offset = average * 0.45

        if average < 0:
            bins = np.arange(-average - offset, -average + offset, 0.05)
        else:
            bins = np.arange(average - offset, average + offset, 0.05)
        ax2 = fig_0Moment.add_subplot(gs2[2])
        ax2.set_title("[Z] distribution")
        ax2.hist(
            z0,
            bins=bins,
            histtype="stepfilled",
            lw=2,
            alpha=1.0,
            color="silver",
            label="original",
        )
        ax2.hist(
            z0_reco,
            bins=bins,
            histtype="step",
            lw=2,
            alpha=1.0,
            color="red",
            label="reconstructed",
        )
        wasserstein_dist = wasserstein_distance(z0, z0_reco)
        kl_divergence = KL(z0, z0_reco, bins)
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
        ax3 = fig_0Moment.add_subplot(gs2[3])
        ax3.set_title("[x] distribution")
        ax3.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
        ax3.hist(
            x,
            bins=np.arange(x_bin_min, x_bin_max),
            histtype="stepfilled",
            lw=2,
            alpha=1.0,
            color="silver",
            label="original",
        )
        ax3.hist(
            x_reco,
            bins=np.arange(x_bin_min, x_bin_max),
            histtype="step",
            lw=2,
            alpha=1.0,
            label="reconstructed",
            color="red",
        )
        data1 = x
        data2 = x_reco
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
        ax4 = fig_0Moment.add_subplot(gs2[4])
        ax4.set_title("[y] distribution")
        ax4.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
        ax4.hist(
            y,
            bins=np.arange(y_bin_min, y_bin_max),
            histtype="stepfilled",
            lw=2,
            alpha=1.0,
            color="silver",
            label="original",
        )
        ax4.hist(
            y_reco,
            bins=np.arange(y_bin_min, y_bin_max),
            histtype="step",
            lw=2,
            alpha=1.0,
            label="reconstructed",
            color="red",
        )
        data1 = y
        data2 = y_reco
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
        ax5 = fig_0Moment.add_subplot(gs2[5])
        ax5.set_title("[z] distribution")
        ax5.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
        ax5.hist(
            z,
            bins=np.arange(z_bin_min, z_bin_max),
            histtype="stepfilled",
            lw=2,
            alpha=1.0,
            color="silver",
            label="original",
        )
        ax5.hist(
            z_reco,
            bins=np.arange(z_bin_min, z_bin_max),
            histtype="step",
            lw=2,
            alpha=1.0,
            label="reconstructed",
            color="red",
        )
        data1 = z
        data2 = z_reco
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

        ############################################################

        fig.suptitle("Distributions")

        fig.tight_layout()

        rep = "_overview"
        filename = plot_filename.replace(rep, "_test")
        pylogger.info(f"Saving plot to {filename}")
        fig.savefig(filename)

        fig_0Moment.suptitle("Distributions")

        fig_0Moment.tight_layout()
        rep = "_overview"
        filename2 = plot_filename.replace(rep, "_0Moment_test")
        pylogger.info(f"Saving plot to {filename2}")
        fig_0Moment.savefig(filename2)

        ############################################################
        # Third Histogram           ----    Error
        ############################################################
        fig_shift = plt.figure(figsize=(24, 12), facecolor="white")
        fig_shift.suptitle("shift", fontsize=30)
        gs3 = GridSpec(2, 4)

        x_diff = get_diff_construct(x, x_reco)
        y_diff = get_diff_construct(y, y_reco)
        z_diff = get_diff_construct(z, z_reco)
        e_diff = get_diff_construct(energy_xyz, energy_xyz_reco)

        bins = np.arange(-0.05, 0.05, 0.001)
        bins_e = np.arange(-1, 1, 0.02)

        ax0 = fig_shift.add_subplot(gs3[0])
        ax0.set_title("$x_{reco} - x_{true}$ Distribution")
        ax0.hist(
            x_diff,
            bins=bins,
            histtype="stepfilled",
            lw=2,
            alpha=1.0,
            color="silver",
            label="error",
        )
        ax0.axvline(0, linestyle="--", color="black", label="center")
        ax0.set_xlabel("$x_{diff}$")
        ax0.set_ylabel("a.u.")
        ax0.legend(loc="upper right")

        ax1 = fig_shift.add_subplot(gs3[1])
        ax1.set_title("$y_{reco} - y_{true}$ Distribution")
        ax1.hist(
            y_diff,
            bins=bins,
            histtype="stepfilled",
            lw=2,
            alpha=1.0,
            color="silver",
            label="error",
        )
        ax1.axvline(0, linestyle="--", color="black", label="center")
        ax1.set_xlabel("$y_{diff}$")
        ax1.set_ylabel("a.u.")
        ax1.legend(loc="upper right")

        ax2 = fig_shift.add_subplot(gs3[2])
        ax2.set_title("$z_{reco} - z_{true}$ Distribution")
        ax2.hist(
            z_diff,
            bins=bins,
            histtype="stepfilled",
            lw=2,
            alpha=1.0,
            color="silver",
            label="error",
        )
        ax2.axvline(0, linestyle="--", color="black", label="center")
        ax2.set_xlabel("$z_{diff}$")
        ax2.set_ylabel("a.u.")
        ax2.legend(loc="upper right")

        ax3 = fig_shift.add_subplot(gs3[3])
        ax3.set_title("$energy_{reco} - energy_{true}$ Distribution")
        ax3.hist(
            e_diff,
            bins=bins_e,
            histtype="stepfilled",
            lw=2,
            alpha=1.0,
            color="silver",
            label="error",
        )
        ax3.axvline(0, linestyle="--", color="black", label="center")
        ax3.set_xlabel("$energy_{diff}$")
        ax3.set_ylabel("a.u.")
        ax3.legend(loc="upper right")

        ############################################################
        # Create the second row of the 3th plot with the larger errors
        ############################################################

        bins = np.arange(-10, 10, 0.25)
        bins_e = np.arange(-10, 10, 0.2)

        ax4 = fig_shift.add_subplot(gs3[4])
        ax4.set_title("$x_{reco} - x_{true}$ Distribution")
        ax4.hist(
            x_diff,
            bins=bins,
            histtype="stepfilled",
            lw=2,
            alpha=1.0,
            color="silver",
            label="error",
        )
        ax4.axvline(0, linestyle="--", color="black", label="center")
        ax4.set_xlabel("$x_{diff}$")
        ax4.set_ylabel("a.u.")
        ax4.set_yscale("log")
        ax4.legend(loc="upper right")

        ax5 = fig_shift.add_subplot(gs3[5])
        ax5.set_title("$y_{reco} - y_{true}$ Distribution")
        ax5.hist(
            y_diff,
            bins=bins,
            histtype="stepfilled",
            lw=2,
            alpha=1.0,
            color="silver",
            label="error",
        )
        ax5.axvline(0, linestyle="--", color="black", label="center")
        ax5.set_xlabel("$y_{diff}$")
        ax5.set_ylabel("a.u.")
        ax5.set_yscale("log")
        ax5.legend(loc="upper right")

        ax6 = fig_shift.add_subplot(gs3[6])
        ax6.set_title("$z_{reco} - z_{true}$ Distribution")
        ax6.hist(
            z_diff,
            bins=bins,
            histtype="stepfilled",
            lw=2,
            alpha=1.0,
            color="silver",
            label="error",
        )
        ax6.axvline(0, linestyle="--", color="black", label="center")
        ax6.set_xlabel("$z_{diff}$")
        ax6.set_ylabel("a.u.")
        ax6.legend(loc="upper right")
        ax6.set_yscale("log")

        ax7 = fig_shift.add_subplot(gs3[7])
        ax7.set_title("$energy_{reco} - energy_{true}$ Distribution")
        ax7.hist(
            e_diff,
            bins=bins_e,
            histtype="stepfilled",
            lw=2,
            alpha=1.0,
            color="silver",
            label="error",
        )
        ax7.set_yscale("log")
        ax7.axvline(0, linestyle="--", color="black", label="center")
        ax7.set_xlabel("$energy_{diff}$")
        ax7.set_ylabel("a.u.")
        ax7.legend(loc="upper right")

        fig_shift.suptitle("Distributions of the error")

        fig_shift.tight_layout()
        rep = "_overview"
        filename3 = plot_filename.replace(rep, "_shift_test")
        pylogger.info(f"Saving plot to {filename3}")
        fig_shift.savefig(filename3)

        # log the plots
        if self.comet_logger is not None:
            for fname in [filename, filename2, filename3]:
                self.comet_logger.log_image(
                    fname, name=fname.split("/")[-1], step=trainer.global_step
                )

        # calculate per-feature mean abs error
        shape = x_recos.shape
        x_recos_reshaped = x_recos.reshape(-1, shape[-1])
        x_originals_reshaped = x_originals.reshape(-1, shape[-1])
        particle_feature_mean_absolute_error = np.mean(
            np.abs(x_recos_reshaped - x_originals_reshaped), axis=1
        )
        particle_feature_mean_error = np.mean(x_recos_reshaped - x_originals_reshaped, axis=1)
        # calculate codebook utilization
        n_codes = pl_module.model.vq_kwargs["num_codes"]
        codebook_utilization = len(np.unique(code_idx)) / n_codes

        # log the mean squared error
        if self.comet_logger is not None:
            self.comet_logger.log_metric(
                f"{stage}_codebook_utilization", codebook_utilization, step=trainer.global_step
            )
            for i, feature in enumerate(pp_dict.keys()):
                self.comet_logger.log_metric(
                    f"{stage}_mean_abserr_{feature}",
                    particle_feature_mean_absolute_error[i],
                    step=trainer.global_step,
                )
                self.comet_logger.log_metric(
                    f"{stage}_mean_err_{feature}",
                    particle_feature_mean_error[i],
                    step=trainer.global_step,
                )
