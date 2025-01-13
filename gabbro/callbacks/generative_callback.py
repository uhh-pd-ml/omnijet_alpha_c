"""Callback for evaluating the generative token model."""

import os
from pathlib import Path

import awkward as ak
import lightning as L
import numpy as np
import torch.distributed as dist
import vector

import gabbro.plotting.utils as plot_utils
from gabbro.models.backbone import BackboneNextTokenPredictionLightning
from gabbro.plotting.feature_plotting import plot_paper_plots
from gabbro.utils.arrays import np_to_ak
from gabbro.utils.pylogger import get_pylogger
from gabbro.utils.utils import analyze_first_10_tokens

pylogger = get_pylogger("GenEvalCallback")
vector.register_awkward()


class GenEvalCallback(L.Callback):
    def __init__(
        self,
        image_path: str = None,
        image_filetype: str = "png",
        no_trainer_info_in_filename: bool = False,
        save_result_arrays: bool = None,
        n_val_gen_jets: int = 10,
        starting_at_epoch: int = 0,
        every_n_epochs: int = 1,
        batch_size_for_generation: int = 2,
        plot_best_checkpoint: bool = False,
        data_dir: str = "data",
        seed_shuffle_val_data: int = None,
        n_data: int = 1000,
        weights: str = "some_weights",
        file_path: str = "some_file_path",
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
        n_val_gen_jets : int
            Number of validation jets to generate. Default is 10.
        starting_at_epoch : int
            Start evaluating the model at this epoch. Default is 0.
        every_n_epochs : int
            Evaluate the model every n epochs. Default is 1.
        batch_size_for_generation : int
            Batch size for generating the jets. Default is 512.
        plot_best_checkpoint : bool
            If True, the best checkpoint is used for generating the showers.
        n_data : int
            This is just the number of showers used for training
        weights : str
            This is the path to the weights used for the backbone
        """
        super().__init__()
        self.comet_logger = None
        self.image_path = image_path
        self.n_val_gen_jets = n_val_gen_jets
        self.image_filetype = image_filetype
        self.no_trainer_info_in_filename = no_trainer_info_in_filename
        self.save_results_arrays = save_result_arrays
        self.every_n_epochs = every_n_epochs
        self.starting_at_epoch = starting_at_epoch
        self.batch_size_for_generation = batch_size_for_generation
        self.best_checkpoint = plot_best_checkpoint
        self.data_dir = data_dir
        self.seed_shuffle_val_data = seed_shuffle_val_data
        self.n_data = str(n_data)
        if weights is None:
            self.weights = "no_weights"
        elif weights == "some_weights":
            self.weights = "no_weights"
        elif weights == "":
            self.weights = "no_weights"
        else:
            self.weights = weights
        pylogger.info(f"the label used for the saving of the data: {self.weights}")
        pylogger.info(f"the actual input for the weights: {weights}")
        self.filepath = file_path

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch < self.starting_at_epoch:
            pylogger.info(
                "Skipping generation. Starting evaluating with this callback"
                f" at epoch {self.starting_at_epoch}."
            )
            return None
        if trainer.current_epoch % self.every_n_epochs != 0:
            pylogger.info(
                f"Skipping generation. Only evaluating every {self.every_n_epochs} epochs."
            )
            return None
        if len(pl_module.val_token_ids_list) == 0:
            pylogger.warning(
                "No validation data available. Skipping generation in validation end."
            )
            return None
        self.plot_real_vs_gen_jets(trainer, pl_module)

    def on_test_epoch_end(self, trainer, pl_module):
        pass

    def on_train_end(self, trainer, pl_module):
        """Called at the end of fit (training + optional testing)."""
        if len(pl_module.val_token_ids_list) == 0:
            pylogger.warning("No validation data available. Skipping generation in train end.")
            return None

        self.plot_real_vs_gen_jets(trainer, pl_module)

    def plot_real_vs_gen_jets(self, trainer, pl_module):
        plot_utils.set_mpl_style()

        # get loggers
        for logger in trainer.loggers:
            if isinstance(logger, L.pytorch.loggers.CometLogger):
                self.comet_logger = logger.experiment
            elif isinstance(logger, L.pytorch.loggers.WandbLogger):
                self.wandb_logger = logger.experiment
        # convert the numpy arrays and masks of the real jets to ak arrays of token
        # ids
        pylogger.info(f"Starting generation of {self.n_val_gen_jets} jets...")
        np_real_token_ids = np.concatenate(pl_module.val_token_ids_list)
        np_real_token_masks = np.concatenate(pl_module.val_token_masks_list)

        pylogger.info(f"np_real_token_ids.shape: {np_real_token_ids.shape}")
        pylogger.info(f"np_real_token_masks.shape: {np_real_token_masks.shape}")
        real_token_ids = np_to_ak(
            x=np_real_token_ids,
            names=["part_token_id"],
            mask=np_real_token_masks,
        )

        if self.best_checkpoint:
            # This Part is new and updates to the best / not the current Checkpoint
            best_checkpoint_path = trainer.checkpoint_callback.best_model_path
            pylogger.info(f"Loading best checkpoint from {best_checkpoint_path}")
            pl_module = BackboneNextTokenPredictionLightning.load_from_checkpoint(
                best_checkpoint_path
            )  # Call on the class
            pl_module.eval()
            pl_module.load_backbone_weights(ckpt_path=best_checkpoint_path)
            ########

        self.real_token_ids = ak.values_astype(real_token_ids["part_token_id"], "int64")
        self.gen_token_ids = pl_module.generate_n_jets_batched(
            self.n_val_gen_jets, batch_size=self.batch_size_for_generation
        )
        data_dir = self.data_dir
        data_dir = Path(data_dir)
        data_dir = data_dir / "reconstructed_test.parquet"

        pylogger.info(f"real_token_ids: {self.real_token_ids}")
        pylogger.info(f"gen_token_ids: {self.gen_token_ids}")

        pylogger.info(f"Length of generated shower: {len(self.gen_token_ids)}")
        pylogger.info(f"Length of real shower: {len(self.real_token_ids)}")

        plot_dir = (
            self.image_path
            if self.image_path is not None
            else trainer.default_root_dir + "/plots/"
        )
        os.makedirs(plot_dir, exist_ok=True)
        filename_real = f"{plot_dir}/epoch{trainer.current_epoch}_gstep{trainer.global_step}_real_shower.parquet"
        # Get the rank of the GPU
        rank = dist.get_rank() if dist.is_initialized() else 0
        filename_gen = filename_real.replace("real_shower", f"gen_shower_rank{rank}")

        # log min max values of the token ids and of the number of constituents
        multiplicity_real = ak.num(self.real_token_ids)
        multiplicity_gen = ak.num(self.gen_token_ids)
        pylogger.info(
            f"Real shower: min multiplicity: {ak.min(multiplicity_real)}, "
            f"max multiplicity: {ak.max(multiplicity_real)}"
        )
        pylogger.info(
            f"Gen shower: min multiplicity: {ak.min(multiplicity_gen)}, "
            f"max multiplicity: {ak.max(multiplicity_gen)}"
        )
        pylogger.info(
            f"Real shower: min token id: {ak.min(self.real_token_ids)}, "
            f"max token id: {ak.max(self.real_token_ids)}"
        )
        pylogger.info(
            f"Gen shower: min token id: {ak.min(self.gen_token_ids)}, "
            f"max token id: {ak.max(self.gen_token_ids)}"
        )

        # check if there are nan values in the token ids
        if np.sum(np.isnan(ak.flatten(self.real_token_ids))) > 0:
            pylogger.warning("Real token ids contain NaN values.")
        if np.sum(np.isnan(ak.flatten(self.gen_token_ids))) > 0:
            pylogger.warning("Generated token ids contain NaN values.")

        # ak.to_parquet(self.real_token_ids, filename_real)
        ak.to_parquet(self.gen_token_ids, filename_gen)
        pylogger.info(f"Real shower saved to {filename_real}")
        pylogger.info(f"Generated shower saved to {filename_gen}")

        def reconstruct_ak_array(
            ak_array_filepath, start_token_included, end_token_included, shift_tokens_by_minus_one
        ):
            token_dir = Path(pl_module.token_dir)
            config_file = token_dir / "config.yaml"
            ckpt_file = token_dir / "model.ckpt"
            input_file = ak_array_filepath
            output_file = ak_array_filepath.replace(".parquet", "_reco.parquet")

            REPO_DIR = Path(__file__).resolve().parent.parent.parent
            PYTHON_COMMAND = [
                "python",
                f"{REPO_DIR}/scripts/reconstruct_shower_tokens.py",
                f"--tokens_file={input_file}",
                f"--output_file={output_file}",
                f"--ckpt_file={ckpt_file}",
                f"--config_file={config_file}",
                f"--start_token_included={start_token_included}",
                f"--end_token_included={end_token_included}",
                f"--shift_tokens_by_minus_one={shift_tokens_by_minus_one}",
            ]
            os.system(" ".join(PYTHON_COMMAND))  # nosec

            return output_file

        # self.real_reco_file = reconstruct_ak_array(filename_real, 1, 1, 1)  #Dont use this, because we dont need to reconstruct the data again and rather want to compare to the original data
        self.gen_reco_file = reconstruct_ak_array(filename_gen, 1, 0, 1)
        # TODO: make this adjustable
        p4s_real = ak.from_parquet(
            "/beegfs/desy/user/rosehenn/gabbro/notebooks/array_test.parquet"
        )
        p4s_real_token = ak.from_parquet(
            "/beegfs/desy/user/rosehenn/gabbro/compare/2024-09-21_16-54-39_max-wng062_CerousLocknut/tokenized_test_e_sorted.parquet"
        )

        # Barrier to ensure all ranks have finished processing before proceeding
        if dist.is_initialized():
            dist.barrier()

        if rank == 0:
            base_filename = filename_gen.replace("_rank0.parquet", "")
            world_size = dist.get_world_size() if dist.is_initialized() else 1

            gen_data_list = []
            gen_data_list_token = []
            for i in range(world_size):
                filename = base_filename + f"_rank{i}_reco.parquet"
                filename_token = base_filename + f"_rank{i}.parquet"
                if os.path.exists(filename):
                    gen_data = ak.from_parquet(filename)
                    gen_data_list.append(gen_data)
                else:
                    print(f"Warning: File {filename_token} does not exist.")
                if os.path.exists(filename_token):
                    gen_data_token = ak.from_parquet(filename_token)
                    gen_data_list_token.append(gen_data_token)

            # Combine all the data
            if gen_data_list:
                p4s_gen = ak.concatenate(gen_data_list)
            else:
                p4s_gen = ak.Array([])  # Handle the case where no files are found
            if gen_data_list_token:
                p4s_gen_token = ak.concatenate(gen_data_list_token)
            else:
                p4s_gen_token = ak.Array([])
            # Now p4s_gen contains the combined data from all ranks

            min_length = min(len(p4s_real), len(p4s_gen))
            min_length_token = min(len(p4s_real_token), len(p4s_gen_token))

            # Truncate to the shorter length
            p4s_real = p4s_real[:min_length]
            p4s_gen = p4s_gen[:min_length]

            p4s_real_token = p4s_real_token[:min_length_token]
            p4s_gen_token = p4s_gen_token[:min_length_token]

            real_token_counts = analyze_first_10_tokens(p4s_real_token)
            gen_token_counts = analyze_first_10_tokens(p4s_gen_token)

            pylogger.info(f"Real token counts: {real_token_counts}")
            pylogger.info(f"Generated token counts: {gen_token_counts}")

            # Analysis of the first 10 tokens
            mean_real_token_count_10 = np.mean(real_token_counts[:10])
            mean_gen_token_count_10 = np.mean(gen_token_counts[:10])

            mean_gen_token_count = np.mean(gen_token_counts)
            mean_real_token_count = np.mean(real_token_counts)

            diversity_real_tokens = len(real_token_counts)
            diversity_gen_tokens = len(gen_token_counts)

            if self.comet_logger is not None:
                self.comet_logger.log_metrics(
                    {
                        "mean_real_token_count_10": mean_real_token_count_10,
                        "mean_gen_token_count_10": mean_gen_token_count_10,
                        "mean_gen_token_count": mean_gen_token_count,
                        "mean_real_token_count": mean_real_token_count,
                        "diversity_real_tokens": diversity_real_tokens,
                        "diversity_gen_tokens": diversity_gen_tokens,
                    },
                    step=trainer.global_step,
                )

            # Plot the real vs. generated showers
            pylogger.info(f"Real shower: {p4s_real}")
            pylogger.info(f"Generated shower: {p4s_gen}")
            pylogger.info(
                f"Plotting {len(p4s_real)} real showers and {len(p4s_gen)} generated showers..."
            )
            if self.best_checkpoint:
                plot_kwargs = {
                    "filepath": self.filepath,
                    "weights": self.weights,
                    "n_data": self.n_data,
                    "transfer_learning": True,
                }
            else:
                plot_kwargs = {}

            fig = plot_paper_plots(
                feature_sets=[p4s_real, p4s_gen],
                colors=["lightgrey", "cornflowerblue"],
                labels=["Geant4", "Generated"],
                **plot_kwargs,
            )

            image_filename = f"{plot_dir}/epoch{trainer.current_epoch}_gstep{trainer.global_step}_real_vs_gen_shower.{self.image_filetype}"
            # image_filename_COG = f"{plot_dir}/epoch{trainer.current_epoch}_gstep{trainer.global_step}_real_vs_gen_shower_COG.{self.image_filetype}"
            fig.savefig(image_filename)
            # fig_COG.savefig(image_filename_COG)

            if self.comet_logger is not None:
                for fname in [image_filename]:
                    self.comet_logger.log_image(
                        fname, name=fname.split("/")[-1], step=trainer.global_step
                    )
        if dist.is_initialized():
            dist.barrier()
