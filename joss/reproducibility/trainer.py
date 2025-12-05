# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# - Anna Lappe <anna.elisa.lappe@cern.ch> - CERN
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

import os
from typing import Any, Dict, Literal, Tuple

import torch
import torch.nn as nn
from src.model import Decoder, Decoder_2d_deep, GeneratorResNet, UNet
from src.utils import init_weights
from torch.utils.data import Dataset, TensorDataset

from itwinai.distributed import suppress_workers_print
from itwinai.loggers import Logger
from itwinai.torch.config import TrainingConfiguration
from itwinai.torch.trainer import TorchTrainer


class VirgoTrainingConfiguration(TrainingConfiguration):
    """Virgo TrainingConfiguration (extends base TrainingConfiguration).

    Additional fields:
    - save_best: whether to save best model on validation dataset.
    - loss: 'l1' or 'l2' for this specific use case.
    - generator: which generator architecture to use.
    """

    #: Whether to save best model on validation dataset. Defaults to True.
    save_best: bool = True
    #: Loss function. Defaults to "l1".
    loss: Literal["l1", "l2"] = "l1"
    #: Generator to train. Defaults to "unet".
    generator: Literal["simple", "deep", "resnet", "unet"] = "unet"


class NoiseGeneratorTrainer(TorchTrainer):
    """Trainer for Virgo noise generator models.

    Reuses the generic TorchTrainer:
    - uses its Ray / checkpoint / logging / profiling logic
    - customizes model, loss, dataloaders and train/validation steps
    """

    def __init__(
        self,
        num_epochs: int = 2,
        config: Dict | VirgoTrainingConfiguration | None = None,
        strategy: Literal["ddp", "deepspeed", "horovod"] | None = "ddp",
        checkpoint_path: str = "checkpoints/epoch_{}.pth",  # kept for backwards compat
        logger: Logger | None = None,
        random_seed: int | None = None,
        name: str | None = None,
        validation_every: int | None = None,
        checkpoint_every: int | None = None,
        **kwargs,
    ) -> None:
        # Normalize config to our custom VirgoTrainingConfiguration
        if config is None:
            config = {}
        if isinstance(config, dict):
            config = VirgoTrainingConfiguration(**config)
        elif not isinstance(config, VirgoTrainingConfiguration):
            # If someone passes a plain TrainingConfiguration, upgrade it
            config = VirgoTrainingConfiguration(**config.model_dump())

        # Map old `checkpoint_path` to the new `checkpoints_location` directory
        # Old default was "checkpoints/epoch_{}.pth" -> we keep only the directory.
        if ("{" in checkpoint_path) or checkpoint_path.endswith(".pth"):
            ckpt_root = os.path.dirname(checkpoint_path) or "checkpoints"
        else:
            ckpt_root = checkpoint_path

        # If user passes validation_every, use it as checkpoint_every unless overridden
        if checkpoint_every is None and validation_every is not None:
            checkpoint_every = validation_every

        super().__init__(
            config=config,
            epochs=num_epochs,
            model=None,  # created in create_model_loss_optimizer
            strategy=strategy or "ddp",
            test_every=None,  # no periodic test by default
            random_seed=random_seed,
            logger=logger,
            checkpoints_location=ckpt_root,
            checkpoint_every=checkpoint_every,
            name=name,
            **kwargs,
        )

    def create_model_loss_optimizer(self) -> None:
        """Instantiate model + loss + optimizer, then distribute them.

        We override the base implementation because:
        - we pick generator architecture from VirgoTrainingConfiguration,
        - we use L1/L2 losses,
        - we still reuse strategy.distributed() from the generic trainer.
        """
        generator = self.config.generator.lower()
        scaling = 0.02

        if generator == "simple":
            self.model = Decoder(3, norm=False)
        elif generator == "deep":
            self.model = Decoder_2d_deep(3)
        elif generator == "resnet":
            self.model = GeneratorResNet(3, 12, 1)
            scaling = 0.01
        elif generator == "unet":
            self.model = UNet(input_channels=3, output_channels=1, norm=False)
        else:
            raise ValueError(f"Unrecognized generator type! Got {generator}")

        # Weight initialization
        init_weights(self.model, "normal", scaling=scaling)

        # Loss
        loss_name = self.config.loss.lower()
        if loss_name == "l1":
            self.loss = nn.L1Loss()
        elif loss_name == "l2":
            self.loss = nn.MSELoss()
        else:
            raise ValueError(f"Unrecognized loss type for Virgo trainer! Got {loss_name}")

        # Optimizer: you can keep using the same one here, or reuse config.optimizer.
        # To keep behaviour of the old trainer, we use Adam with optim_lr from config.
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.optim_lr,
            weight_decay=self.config.optim_weight_decay,
        )

        # IMPORTANT: distribute model/optimizer/scheduler via strategy
        distribute_kwargs = self.get_default_distributed_kwargs()
        self.model, self.optimizer, self.lr_scheduler = self.strategy.distributed(
            self.model, self.optimizer, self.lr_scheduler, **distribute_kwargs
        )

    def create_dataloaders(
        self,
        train_dataset: Dataset,
        validation_dataset: Dataset | None = None,
        test_dataset: Dataset | None = None,
    ) -> None:
        """Override to use custom_collate when dataset is not a TensorDataset."""
        if isinstance(train_dataset, TensorDataset):
            # Use generic implementation (pairs (x, y) etc.)
            return super().create_dataloaders(
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                test_dataset=test_dataset,
            )

        # Custom dataset -> use custom_collate to concat elements
        self.train_dataloader = self.strategy.create_dataloader(
            dataset=train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers_dataloader,
            pin_memory=self.config.pin_gpu_memory,
            generator=self.torch_rng,
            shuffle=self.config.shuffle_train,
            collate_fn=self.custom_collate,
        )
        if validation_dataset is not None:
            self.validation_dataloader = self.strategy.create_dataloader(
                dataset=validation_dataset,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers_dataloader,
                pin_memory=self.config.pin_gpu_memory,
                generator=self.torch_rng,
                shuffle=self.config.shuffle_validation,
                collate_fn=self.custom_collate,
            )
        if test_dataset is not None:
            self.test_dataloader = self.strategy.create_dataloader(
                dataset=test_dataset,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers_dataloader,
                pin_memory=self.config.pin_gpu_memory,
                generator=self.torch_rng,
                shuffle=self.config.shuffle_test,
                collate_fn=self.custom_collate,
            )

    def custom_collate(self, batch):
        """Custom collate: filter None and concatenate along the batch dimension.

        Each item in the dataset is expected to be a tensor of shape:
            (num_channels, H, W)
        and we want a batch of shape:
            (batch_size, num_channels, H, W)
        """
        batch = [x for x in batch if x is not None]
        return torch.cat(batch)

    # -------------------------------------------------------------------------
    # Execute hook: keep suppress_workers_print but reuse TorchTrainer.execute
    # -------------------------------------------------------------------------
    @suppress_workers_print
    def execute(
        self,
        train_dataset: Dataset,
        validation_dataset: Dataset | None = None,
        test_dataset: Dataset | None = None,
    ) -> Tuple[Dataset, Dataset, Dataset, Any]:
        return super().execute(train_dataset, validation_dataset, test_dataset)

    def _split_batch(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split batch into (input, target) as used in the original trainer.

        Input layout (as in the old code):
            batch[:, 0]   -> target (main channel)
            batch[:, 1:]  -> input  (aux channels)
        """
        if isinstance(batch, list):
            batch = batch[0]

        # batch shape: (B, C, H, W)
        target = batch[:, 0].unsqueeze(1).to(self.device).float()
        inp = batch[:, 1:].to(self.device).float()
        return inp, target

    def train_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Single optimization step using Virgo layout batch."""
        inp, target = self._split_batch(batch)

        self.optimizer.zero_grad()
        generated = self.model(inp)
        loss: torch.Tensor = self.loss(generated, target)
        loss.backward()
        self.optimizer.step()

        # Log per-batch loss
        self.log(
            item=loss.item(),
            identifier="train_loss",
            kind="metric",
            step=self.train_glob_step,
            batch_idx=batch_idx,
        )

        # Optional extra metrics via torchmetrics
        metrics: Dict[str, Any] = self.compute_metrics(
            true=target,
            pred=generated,
            logger_step=self.train_glob_step,
            batch_idx=batch_idx,
            stage="train",
        )
        return loss, metrics

    def validation_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Single validation step using Virgo layout batch."""
        inp, target = self._split_batch(batch)

        with torch.no_grad():
            generated = self.model(inp)
            loss: torch.Tensor = self.loss(generated, target)

        # Log per-batch validation loss
        self.log(
            item=loss.item(),
            identifier="validation_loss",
            kind="metric",
            step=self.validation_glob_step,
            batch_idx=batch_idx,
        )

        metrics: Dict[str, Any] = self.compute_metrics(
            true=target,
            pred=generated,
            logger_step=self.validation_glob_step,
            batch_idx=batch_idx,
            stage="validation",
        )
        return loss, metrics

