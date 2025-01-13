"""Backbone model with different heads."""

import time
from typing import Any, Dict, Tuple

import awkward as ak
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import vector
from tqdm import tqdm

from gabbro.metrics.utils import calc_accuracy
from gabbro.models.gpt_model import BackboneModel
from gabbro.utils.arrays import fix_padded_logits
from gabbro.utils.pylogger import get_pylogger

vector.register_awkward()

logger = get_pylogger(__name__)

# -------------------------------------------------------------------------
# ------------ BACKBONE + Generative (next-token-prediction) head ---------
# -------------------------------------------------------------------------


class NextTokenPredictionHead(nn.Module):
    """Head for predicting the next token in a sequence."""

    def __init__(self, embedding_dim, vocab_size):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        return self.fc1(x)


class BackboneNextTokenPredictionLightning(L.LightningModule):
    """PyTorch Lightning module for training the backbone model."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        model_kwargs={},
        token_dir=None,
        verbose=False,
        exclude_padded_values_from_loss: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        # initialize the backbone
        self.module = BackboneModel(**model_kwargs)

        # initialize the model head
        self.head = NextTokenPredictionHead(
            embedding_dim=model_kwargs["embedding_dim"],
            vocab_size=model_kwargs["vocab_size"],
        )

        # initialize the loss function
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(
                [1.0] * (model_kwargs["vocab_size"] - 1) + [model_kwargs["stop_token_weight"]]
            ).to(self.device)
        )

        self.token_dir = token_dir
        self.verbose = verbose

        self.train_loss_history = []
        self.val_loss_list = []

        self.validation_cnt = 0
        self.validation_output = {}

        self.backbone_weights_path = model_kwargs.get("backbone_weights_path", "None")

        self.pylogger = get_pylogger(__name__)
        self.pylogger.info(f"Backbone weights path: {self.backbone_weights_path}")

        if self.backbone_weights_path is not None:
            if self.backbone_weights_path != "None":
                self.load_backbone_weights(self.backbone_weights_path)

    def load_backbone_weights(self, ckpt_path):
        self.pylogger.info(f"Loading backbone weights from {ckpt_path}")
        ckpt = torch.load(ckpt_path)
        # print(ckpt)
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

        # print("state_dict",state_dict)
        loaded_state_dict = {k: v for k, v in state_dict.items() if "tril" not in k}
        self.load_state_dict(loaded_state_dict, strict=False)
        print("Backbone weights loaded")

    def forward(self, x, mask=None):
        if self.module.return_embeddings:
            backbone_out = self.module(x, mask)
            logits = self.head(backbone_out)
        else:
            logits = self.module(x, mask)
        if self.verbose:
            self.pylogger.info("Logits shape: ", logits.shape)
        return logits

    def model_step(self, batch, return_logits=False):
        """Perform a single model step on a batch of data.

        Parameters
        ----------
        batch : dict
            A batch of data as a dictionary containing the input and target tensors,
            as well as the mask.
        return_logits : bool, optional
            Whether to return the logits or not. (default is False)
        """

        # all token-ids up to the last one are the input, the ones from the second
        # to the (including) last one are the target
        # this model step uses the convention that the first particle feature
        # is the token, with the tokens up to the last one
        # the second particle feature is the target token (i.e. the next token)

        X = batch["part_features"]
        X = X.squeeze().long()
        input = X[:, :, 0]
        targets = X[:, :, 1]
        mask = batch["part_mask"]

        # Add print statements after data extraction

        # compute the logits (i.e. the predictions for the next token)
        logits = self.forward(input, mask)

        if self.hparams.exclude_padded_values_from_loss:
            logits = fix_padded_logits(logits, mask, factor=1e6)

        # reshape the logits and targets to work with the loss function
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.contiguous().view(B * T)

        loss = self.criterion(logits, targets)

        if return_logits:
            return loss, X, logits, mask, targets

        return loss

    @torch.no_grad()
    def generate_batch(self, batch_size):
        """Generate a batch of shower constituents autoregressively, stopping generation for each
        sequence individually when the stop token is encountered."""
        device = next(self.module.parameters()).device
        idx = torch.zeros(batch_size, 1).long().to(device)
        stop_token_id = self.module.vocab_size - 2
        completed_sequences = torch.zeros(batch_size, dtype=torch.bool).to(device)

        for i in range(self.module.max_sequence_len):
            if torch.all(completed_sequences):
                break  # Stop if all sequences have generated the stop token

            logits = self(idx)
            self.pylogger.info(
                "Logit shape input for generation: ", logits.shape
            ) if self.verbose else None
            logits = logits[:, -1, :]

            logits = logits / self.module.temperature
            probs = F.softmax(logits[:, 1:], dim=-1)

            stop_token_probs = probs[:, stop_token_id]

            stop_token_probs[stop_token_probs < self.module.stop_token_threshold] = 0
            probs[:, stop_token_id] = stop_token_probs

            probs_sum = probs.sum(dim=-1, keepdim=True)
            probs = probs / probs_sum

            idx_next = torch.multinomial(probs, num_samples=1) + 1
            idx = torch.cat((idx, idx_next), dim=1)
            self.pylogger.info(
                "appended idx_next to original idx, shape: ", idx.shape
            ) if self.verbose else None

            completed_sequences = completed_sequences | (idx_next.squeeze(-1) == stop_token_id + 1)

        # No need to truncate, sequences stopped naturally
        gen_batch_np = idx.detach().cpu().numpy()
        gen_batch_ak = ak.from_numpy(gen_batch_np)
        gen_batch_until_stop = []

        # loop over the showers in the batch, and only keep the tokens until the stop token
        for shower in gen_batch_ak:
            stop_token_position = np.where(shower == self.module.vocab_size - 1)
            if len(stop_token_position[0]) > 0:
                stop_token_position = stop_token_position[0][0]
            else:
                stop_token_position = shower.shape[0]
            gen_batch_until_stop.append(shower[:stop_token_position])

        return ak.Array(gen_batch_until_stop)

    def generate_n_showers_batched(self, n_showers, batch_size, saveas=None):
        """Generate showers in batches.

        Parameters
        ----------
        n_showers : int
            Number of showers to generate.
        batch_size : int
            Batch size to use during generation (use as large as possible with memory.)
        saveas : str, optional
            Path to save the generated showers to (in parquet format). (default is None)

        Returns
        -------
        ak.Array
            The generated showers (i.e. their token ids, in the shape (n_showers, <var>).
        """
        n_batches = n_showers // batch_size
        generated_showers = []

        self.pylogger.info(
            f"Generating {n_showers} showers in {n_batches} batches of size {batch_size}"
        )

        for i in tqdm(range(n_batches)):
            gen_batch_ak = self.generate_batch(batch_size)
            generated_showers.append(gen_batch_ak)

        # concatenate the generated batches
        generated_showers = ak.concatenate(generated_showers)[:n_showers]

        if saveas is not None:
            self.pylogger.info(f"Saving generated showers to {saveas}")
            ak.to_parquet(generated_showers, saveas)

        return generated_showers

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set."""
        loss = self.model_step(batch)

        self.train_loss_history.append(float(loss))
        self.log(
            "train_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )

        return loss

    def on_train_start(self) -> None:
        self.pylogger.info("`on_train_start` called.")
        self.pylogger.info("Setting up the logger with the correct rank.")
        self.pylogger = get_pylogger(__name__, rank=self.trainer.global_rank)
        self.pylogger.info("Logger set up.")

        self.preprocessing_dict = (
            self.trainer.datamodule.hparams.dataset_kwargs_common.feature_dict
        )

    def on_train_epoch_start(self):
        self.pylogger.info("`on_train_epoch_start` called.")
        self.pylogger.info(f"Epoch {self.trainer.current_epoch} starting.")
        self.epoch_train_start_time = time.time()  # start timing the epoch

    def on_train_epoch_end(self):
        self.pylogger.info("`on_train_epoch_end` called.")
        self.epoch_train_end_time = time.time()
        self.epoch_train_duration_minutes = (
            self.epoch_train_end_time - self.epoch_train_start_time
        ) / 60
        self.log(
            "epoch_train_duration_minutes",
            self.epoch_train_duration_minutes,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        if len(self.train_loss_history) > 0:
            self.pylogger.info(
                f"Epoch {self.trainer.current_epoch} finished in"
                f" {self.epoch_train_duration_minutes:.1f} minutes. "
                f"Current step: {self.global_step}. Current loss: {self.train_loss_history[-1]}."
                f" rank: {self.global_rank}"
            )
        if dist.is_initialized():
            dist.barrier()
            self.pylogger.info("Barrier at epoch end.")

    def on_train_end(self) -> None:
        self.pylogger.info("`on_train_end` called.")

    def on_validation_epoch_start(self) -> None:
        self.pylogger.info("`on_validation_epoch_start` called.")
        self.val_token_ids_list = []
        self.val_token_masks_list = []

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, X, logits, mask, targets = self.model_step(batch, return_logits=True)

        self.val_token_ids_list.append(batch["part_features"].float().detach().cpu().numpy())
        self.val_token_masks_list.append(batch["part_mask"].float().detach().cpu().numpy())
        self.log(
            "val_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )
        # self.log("batch_idx", batch["part_features"], on_step=True, on_epoch=True, prog_bar=True)
        # self.pylogger.info(f"first_batch {batch['part_features'][0]}")
        # self.pylogger.info("val_token_ids_list", self.val_token_ids_list, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def on_test_epoch_start(self) -> None:
        self.pylogger.info("`on_test_epoch_start` called.")

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, X, logits, mask, targets = self.model_step(batch, return_logits=True)
        self.log("test_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        self.pylogger.info("`on_validation_epoch_end` called.")

    def on_test_epoch_end(self):
        self.pylogger.info("`on_test_epoch_end` called.")

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configures optimizers and learning-rate schedulers to be used for training."""
        self.pylogger.info("`configure_optimizers` called.")
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        return {"optimizer": optimizer}


# -------------------------------------------------------------------------
# ------------------ BACKBONE + Classification head -----------------------
# -------------------------------------------------------------------------


class NormformerCrossBlock(nn.Module):
    def __init__(self, input_dim, mlp_dim, num_heads, dropout_rate=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        # define the MultiheadAttention layer with layer normalization
        self.norm1 = nn.LayerNorm(input_dim)
        self.attn = nn.MultiheadAttention(
            input_dim, num_heads, batch_first=True, dropout=self.dropout_rate
        )
        self.norm2 = nn.LayerNorm(input_dim)

        # define the MLP with layer normalization
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),  # Add layer normalization
            nn.Linear(input_dim, mlp_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(mlp_dim, input_dim),
        )

        # initialize weights of mlp[-1] and layer norm after attn block to 0
        # such that the residual connection is the identity when the block is
        # initialized
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        nn.init.zeros_(self.norm1.weight)

    def forward(self, x, class_token, mask=None, return_attn_weights=False):
        # x: (B, S, F)
        # mask: (B, S)
        x = x * mask.unsqueeze(-1)

        # calculate cross-attention
        x_norm = self.norm1(x)
        attn_output, attn_weights = self.attn(
            query=class_token, key=x_norm, value=x_norm, key_padding_mask=mask != 1
        )
        return attn_output


class ClassifierNormformer(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_heads=4,
        dropout_rate=0.0,
        num_class_blocks=3,
        model_kwargs={"n_out_nodes": 2, "fc_params": [(100, 0.1), (100, 0.1)]},
        **kwargs,
    ):
        super().__init__()

        self.model_kwargs = model_kwargs
        self.n_out_nodes = model_kwargs["n_out_nodes"]
        self.dropout_rate = dropout_rate
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_class_blocks = num_class_blocks
        self.class_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        self.class_attention_blocks = nn.ModuleList(
            [
                NormformerCrossBlock(
                    input_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    dropout_rate=self.dropout_rate,
                    mlp_dim=self.hidden_dim,
                )
                for _ in range(self.num_class_blocks)
            ]
        )
        self.final_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.model_kwargs["n_out_nodes"]),
        )

        self.loss_history = []
        self.lr_history = []

    def forward(self, x, mask):
        # expand class token and add to mask
        class_token = self.class_token.expand(x.size(0), -1, -1)
        mask_with_token = torch.cat([torch.ones(x.size(0), 1).to(x.device), mask], dim=1)

        # pass through class attention blocks, always use the updated class token
        for block in self.class_attention_blocks:
            x_class_token_and_x_encoded = torch.cat([class_token, x], dim=1)
            # class_token = block(x_class_token_and_x_encoded, mask=mask_with_token)[:, :1, :]
            class_token = block(x_class_token_and_x_encoded, class_token, mask=mask_with_token)

        # pass through final mlp
        class_token = self.final_mlp(class_token).squeeze(1)
        return class_token


class ClassificationHead(torch.nn.Module):
    """Classification head for the backbone model."""

    def __init__(self, model_kwargs={"n_out_nodes": 2}):
        super().__init__()
        self.backbone_weights_path = None

        if "n_out_nodes" not in model_kwargs:
            model_kwargs["n_out_nodes"] = 2
        if "return_embeddings" not in model_kwargs:
            model_kwargs["return_embeddings"] = True

        self.n_out_nodes = model_kwargs["n_out_nodes"]
        model_kwargs.pop("n_out_nodes")

        self.classification_head_linear_embed = nn.Linear(
            model_kwargs["embedding_dim"],
            model_kwargs["embedding_dim"],
        )
        self.classification_head_linear_class = nn.Linear(
            model_kwargs["embedding_dim"],
            self.n_out_nodes,
        )

    def forward(self, x, mask):
        embeddings = F.relu(self.classification_head_linear_embed(x))
        embeddings_sum = torch.sum(embeddings * mask.unsqueeze(-1), dim=1)
        logits = self.classification_head_linear_class(embeddings_sum)
        return logits


class BackboneClassificationLightning(L.LightningModule):
    """Backbone with classification head."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        class_head_type: str = "summation",
        model_kwargs: dict = {},
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        # initialize the backbone
        self.module = BackboneModel(**model_kwargs)

        # initialize the model head
        if class_head_type == "summation":
            self.head = ClassificationHead(
                model_kwargs={
                    "n_out_nodes": model_kwargs["n_out_nodes"],
                    "embedding_dim": model_kwargs["embedding_dim"],
                }
            )
        elif class_head_type == "class_attention":
            self.head = ClassifierNormformer(
                input_dim=model_kwargs["embedding_dim"],
                hidden_dim=model_kwargs["embedding_dim"],
                model_kwargs={"n_out_nodes": model_kwargs["n_out_nodes"]},
                num_heads=2,
                num_class_blocks=3,
                dropout_rate=0.0,
            )
        else:
            raise ValueError(f"Invalid class_head_type: {class_head_type}")

        self.criterion = torch.nn.CrossEntropyLoss()

        self.train_loss_history = []
        self.val_loss_history = []

        self.backbone_weights_path = model_kwargs.get("backbone_weights_path", "None")
        logger.info(f"Backbone weights path: {self.backbone_weights_path}")

        if self.backbone_weights_path is not None:
            if self.backbone_weights_path != "None":
                self.load_backbone_weights(self.backbone_weights_path)

    def load_backbone_weights(self, ckpt_path=None):
        logger.info(f"Loading backbone weights from {ckpt_path}")
        if ckpt_path is None:
            ckpt_path = (
                self.hparams.model_kwargs.backbone_weights_path
            )  # Or wherever your default path is stored
            logger.info(f"Loading backbone weights from {ckpt_path}")
        ckpt = torch.load(ckpt_path)
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        self.load_state_dict(state_dict, strict=False)

    def forward(self, X, mask):
        embeddings = self.module(X, mask)
        logits = self.head(embeddings, mask)
        return logits

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        logger.info("`on_train_start` called.")

    def on_train_epoch_start(self) -> None:
        logger.info("`on_train_epoch_start` called.")
        self.train_preds_list = []
        self.train_labels_list = []
        logger.info(f"Epoch {self.trainer.current_epoch} started.")

    def model_step(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        X = batch["part_features"].to("cuda")
        mask = batch["part_mask"].to("cuda")
        shower_labels = batch["shower_type_labels"]
        if len(X.size()) == 2:
            X = X.unsqueeze(-1)
        X = X.squeeze().long()
        # one-hot encode the labels
        logits = self.forward(X, mask)
        labels = F.one_hot(shower_labels.squeeze(), num_classes=self.head.n_out_nodes).float()
        loss = self.criterion(logits.to("cuda"), labels.to("cuda"))
        return loss, logits, labels

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, logits, targets = self.model_step(batch)

        preds = torch.softmax(logits, dim=1)
        self.train_preds_list.append(preds.float().detach().cpu().numpy())
        self.train_labels_list.append(targets.float().detach().cpu().numpy())
        self.train_loss_history.append(loss.float().detach().cpu().numpy())

        acc = calc_accuracy(
            preds=preds.float().detach().cpu().numpy(),
            labels=targets.float().detach().cpu().numpy(),
        )

        self.log(
            "train_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def on_train_epoch_end(self):
        logger.info("`on_train_epoch_end` called.")
        self.train_preds = np.concatenate(self.train_preds_list)
        self.train_labels = np.concatenate(self.train_labels_list)
        logger.info(f"Epoch {self.trainer.current_epoch} finished.")
        dist.barrier()
        plt.plot(self.train_loss_history)

    def on_validation_epoch_start(self) -> None:
        logger.info("`on_validation_epoch_start` called.")
        self.val_preds_list = []
        self.val_labels_list = []

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, logits, targets = self.model_step(batch)
        preds = torch.softmax(logits, dim=1)
        self.val_preds_list.append(preds.float().detach().cpu().numpy())
        self.val_labels_list.append(targets.float().detach().cpu().numpy())
        # update and log metrics
        acc = calc_accuracy(
            preds=preds.float().detach().cpu().numpy(),
            labels=targets.float().detach().cpu().numpy(),
        )
        self.log(
            "val_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        logger.info("`on_validation_epoch_end` called.")
        self.val_preds = np.concatenate(self.val_preds_list)
        self.val_labels = np.concatenate(self.val_labels_list)

    def on_test_start(self):
        logger.info("`on_test_start` called.")
        self.test_loop_preds_list = []
        self.test_loop_labels_list = []

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set."""
        loss, logits, targets = self.model_step(batch)
        preds = torch.softmax(logits, dim=1)
        self.test_loop_preds_list.append(preds.float().detach().cpu().numpy())
        self.test_loop_labels_list.append(targets.float().detach().cpu().numpy())

        acc = calc_accuracy(
            preds=preds.float().detach().cpu().numpy(),
            labels=targets.float().detach().cpu().numpy(),
        )
        self.log(
            "test_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )
        self.log("test_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_test_epoch_end(self):
        logger.info("`on_test_epoch_end` called.")
        self.test_preds = np.concatenate(self.test_loop_preds_list)
        self.test_labels = np.concatenate(self.test_loop_labels_list)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configures optimizers and learning-rate schedulers to be used for training."""
        logger.info("`configure_optimizers` called.")
        if self.hparams.model_kwargs.keep_backbone_fixed:
            logger.info("--- Keeping backbone fixed. ---")
            optimizer = self.hparams.optimizer(
                [
                    {"params": self.module.parameters(), "lr": 0.0},
                    {"params": self.head.parameters()},
                ]
            )
        else:
            optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


class BackboneMPMLightning(L.LightningModule):
    """Backbone model with NextTokenPredictionHead used for predicting a masked particle."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        model_kwargs={},
        token_dir=None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        # --------------- load pretrained model --------------- #
        # if kwargs.get("load_pretrained", False):
        self.module = BackboneModel(**model_kwargs)

        self.head = NextTokenPredictionHead(
            embedding_dim=model_kwargs["embedding_dim"],
            vocab_size=model_kwargs["vocab_size"],
        )
        self.backbone_weights_path = model_kwargs.get("backbone_weights_path", None)
        self.token_dir = token_dir

        self.train_loss_history = []
        self.val_loss_list = []

        self.validation_cnt = 0
        self.validation_output = {}

        self.criterion = nn.CrossEntropyLoss()

        if self.backbone_weights_path is not None:
            if self.backbone_weights_path != "None":
                self.load_backbone_weights(self.backbone_weights_path)

    def load_backbone_weights(self, ckpt_path=None):  # Add ckpt_path parameter
        if ckpt_path is None:
            ckpt_path = (
                self.hparams.model_kwargs.backbone_weights_path
            )  # Or wherever your default path is stored
            logger.info(f"Loading backbone weights from {ckpt_path}")
        logger.info(f"Loading backbone weights from {ckpt_path}")
        ckpt = torch.load(ckpt_path)
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        self.load_state_dict(state_dict, strict=False)
        print("Backbone weights loaded")

    def forward(self, x, mask):
        embedding = self.module(x, mask)
        logits = self.head(embedding)
        return logits

    def multi_masking(self, showerlenghts, mask_percent, mm=True):
        showerlenghts = showerlenghts.cpu().numpy()
        if mm:
            to_mask = showerlenghts // (100 / mask_percent)
        else:
            to_mask = np.ones_like(showerlenghts)

        batch_mask = []

        for shower, mask_amount in enumerate(to_mask):
            mask_values = np.random.choice(
                showerlenghts[shower], size=int(mask_amount), replace=False
            )
            mask_part = np.arange(128) > 130
            for index in mask_values:
                mask_part[index] = True

            batch_mask.append(mask_part)
        mask = torch.from_numpy(np.asarray(batch_mask))
        return mask

    def model_step(self, batch, return_logits=False, return_output=False):
        """Perform a single model step on a batch of data."""
        # preparing the data
        X = batch["part_features"]
        X = X.squeeze().long()
        mask = batch["part_mask"]
        showerlen = mask.sum(axis=1)
        lenmask = showerlen > 1
        mask_to_fill = self.multi_masking(showerlen[lenmask], 10, mm=False).to("cuda")
        X_len_masked = X[lenmask]

        X_masked = X_len_masked.masked_fill(mask_to_fill, 8192)
        targets = X[lenmask][:, :]

        X = X_masked[:, :]
        mask = mask[lenmask][:, :]
        # forward pass
        logits = self.forward(X, mask)

        # calculating accuracy and output metrics
        B, T, C = logits.shape
        argmax = torch.argmax(logits, axis=2)

        masked_particle_mask = mask_to_fill[:, :] & mask
        total_masked_particles = masked_particle_mask.sum().item()
        correct_masked_particles = (
            (argmax[masked_particle_mask] == targets[masked_particle_mask]).sum().item()
        )
        accuracy_masked_particles = correct_masked_particles / total_masked_particles

        masked_logits = logits[masked_particle_mask].to("cpu")
        masked_targets = targets[masked_particle_mask].to("cpu")

        loss = self.criterion(masked_logits, masked_targets)

        if return_logits:
            return loss, X, masked_logits, mask, masked_targets

        return loss, accuracy_masked_particles

    @torch.no_grad()
    def predict_tokens(self, batch):
        """Mask and predict tokens on a batch of data."""
        self.to("cuda")
        X = batch["part_features"].to("cuda")  # .to("cpu")
        X_orig = X
        X = X.squeeze().long()
        mask = batch["part_mask"].to("cuda")  # .to("cpu")
        showerlen = mask.sum(axis=1)
        lenmask = showerlen > 1
        mask_to_fill = self.multi_masking(showerlen[lenmask], 10, mm=False).to("cuda")
        X_len_masked = X[lenmask]
        X_masked = X_len_masked.masked_fill(mask_to_fill, 8192)
        # all token-ids up to the last one are the input, the ones from the second
        # to the (including) last one are the target
        targets = X[lenmask][:, :]
        X = X_masked[:, :]
        mask = mask[lenmask][:, :]

        logits = self.forward(X, mask)
        B, T, C = logits.shape
        argmax = torch.argmax(logits, axis=2)
        masked_particle_mask = mask_to_fill[:, :] & mask
        total_masked_particles = masked_particle_mask.sum().item()
        correct_masked_particles = (
            (argmax[masked_particle_mask] == targets[masked_particle_mask]).sum().item()
        )
        accuracy_masked_particles = correct_masked_particles / total_masked_particles

        masked_logits = logits[masked_particle_mask].to("cpu")
        masked_targets = targets[masked_particle_mask].to("cpu")

        self.log(
            "masked particle accuracy",
            accuracy_masked_particles,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        print("masked particle accuracy:", accuracy_masked_particles)
        return (
            X_orig.cpu(),
            X_len_masked.cpu(),
            mask_to_fill.cpu(),
            masked_targets,
            masked_logits,
        )

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set."""

        loss, accuracy_masked_particles = self.model_step(batch)
        self.log(
            "masked particle accuracy",
            accuracy_masked_particles,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.train_loss_history.append(float(loss))
        self.log("train_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def on_train_start(self) -> None:
        self.preprocessing_dict = (
            self.trainer.datamodule.hparams.dataset_kwargs_common.feature_dict
        )

    def on_train_epoch_start(self):
        logger.info(f"Epoch {self.trainer.current_epoch} starting.")
        self.epoch_train_start_time = time.time()  # start timing the epoch

    def on_train_epoch_end(self):
        self.epoch_train_end_time = time.time()
        self.epoch_train_duration_minutes = (
            self.epoch_train_end_time - self.epoch_train_start_time
        ) / 60
        self.log(
            "epoch_train_duration_minutes",
            self.epoch_train_duration_minutes,
            on_epoch=True,
            prog_bar=False,
        )
        if len(self.train_loss_history) > 0:
            logger.info(
                f"Epoch {self.trainer.current_epoch} finished in"
                f" {self.epoch_train_duration_minutes:.1f} minutes. "
                f"Current step: {self.global_step}. Current loss FULL: {self.train_loss_history}."
                f"Current step: {self.global_step}. Current loss: {self.train_loss_history[-1]}."
            )

    def on_train_end(self):
        pass

    def on_validation_epoch_start(self) -> None:
        self.val_token_preds_list = []
        self.val_token_target_list = []

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, X, logits, mask, targets = self.model_step(batch, return_logits=True)
        preds = torch.softmax(logits, dim=1)
        self.val_token_preds_list.append(preds.detach().cpu().numpy())
        self.val_token_target_list.append(targets.detach().cpu().numpy())
        acc = (np.argmax(preds.detach().cpu().numpy()) == targets.detach().cpu().numpy()).mean()
        self.log("val_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        self.val_preds = np.concatenate(self.val_token_preds_list)
        self.val_labels = np.concatenate(self.val_token_target_list)

    def on_test_epoch_start(self) -> None:
        pass

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, x_original, x_reco, mask, labels, code_idx = self.model_step(batch, return_x=True)
        self.log(
            "test_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configures optimizers and learning-rate schedulers to be used for training.

        Normally you'd need one, but in the case of GANs or similar you might need multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.parameters())

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


class RegressionHead(nn.Module):
    """Head for predicting the shower features using regression.."""

    def __init__(self, embedding_dim, num_shower_features=4):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, num_shower_features)

    def forward(self, x):
        x_dim_red = x.mean(axis=1)
        return self.fc1(x_dim_red)


class BackboneRegressionLightning(L.LightningModule):
    """Backbone model with NextTokenPredictionHead used for predicting a masked particle."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        model_kwargs={},
        token_dir=None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        # --------------- load pretrained model --------------- #
        # if kwargs.get("load_pretrained", False):
        self.module = BackboneModel(**model_kwargs)

        self.head = RegressionHead(
            embedding_dim=model_kwargs["embedding_dim"],
            num_shower_features=model_kwargs["num_shower_features"],  # needs to be implemented
        )
        self.backbone_weights_path = model_kwargs.get("backbone_weights_path", None)
        self.token_dir = token_dir

        self.train_loss_history = []
        self.val_loss_list = []

        self.validation_cnt = 0
        self.validation_output = {}

        self.criterion = nn.MSELoss()

        if self.backbone_weights_path is not None:
            if self.backbone_weights_path != "None":
                self.load_backbone_weights(self.backbone_weights_path)

    def load_backbone_weights(self, ckpt_path):
        logger.info(f"Loading backbone weights from {ckpt_path}")
        ckpt = torch.load(ckpt_path)
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        self.load_state_dict(state_dict, strict=False)
        print("Backbone weights loaded")

    def forward(self, x, mask):
        embedding = self.module(x, mask)
        pred_shower_features = self.head(embedding)
        return pred_shower_features

    def model_step(self, batch, return_logits=False, return_output=False):
        """Perform a single model step on a batch of data."""
        # preparing the data
        X = batch["part_features"].to("cuda")
        targets = X.mean(axis=1).squeeze()  # needs to be changed to the shower feature arrays

        X = X.squeeze().long()
        mask = batch["part_mask"].to("cuda")

        # forward pass
        pred_shower_features = self.forward(X, mask)

        targets = batch[
            "shower_features"
        ]  # needs to be deleted after testing!!! This is just to make the test run.

        print("shape of output: ", pred_shower_features.shape)
        # calculating loss and output metrics
        loss = self.criterion(pred_shower_features, targets)
        if return_logits:
            return loss, X, mask, pred_shower_features

        return loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set."""

        loss = self.model_step(batch)

        self.train_loss_history.append(float(loss))
        self.log("train_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def on_train_start(self) -> None:
        self.preprocessing_dict = (
            self.trainer.datamodule.hparams.dataset_kwargs_common.feature_dict
        )

    def on_train_epoch_start(self):
        logger.info(f"Epoch {self.trainer.current_epoch} starting.")
        self.epoch_train_start_time = time.time()  # start timing the epoch

    def on_train_epoch_end(self):
        self.epoch_train_end_time = time.time()
        self.epoch_train_duration_minutes = (
            self.epoch_train_end_time - self.epoch_train_start_time
        ) / 60
        self.log(
            "epoch_train_duration_minutes",
            self.epoch_train_duration_minutes,
            on_epoch=True,
            prog_bar=False,
        )
        logger.info(
            f"Epoch {self.trainer.current_epoch} finished in"
            f" {self.epoch_train_duration_minutes:.1f} minutes."
        )

    def on_train_end(self):
        pass

    def on_validation_epoch_start(self) -> None:
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, X, mask, pred_shower_features = self.model_step(batch, return_logits=True)
        self.log("val_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        pass

    def on_test_epoch_start(self) -> None:
        pass

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, X, mask, pred_shower_features = self.model_step(batch, return_logits=True)
        self.log("test_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configures optimizers and learning-rate schedulers to be used for training.

        Normally you'd need one, but in the case of GANs or similar you might need multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.parameters())

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
