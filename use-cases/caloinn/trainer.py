import os
from typing import Any, Dict, Literal, Optional, Tuple, Union

import torch
import torch.optim as optim
import math

import itertools

from model import CINN

from itwinai.loggers import Logger
from itwinai.torch.config import TrainingConfiguration
from itwinai.torch.trainer import TorchTrainer


class CaloChallengeTrainer(TorchTrainer):
    def __init__(
        self,
        num_epochs: int = 2,
        config: Union[Dict[str, Any], TrainingConfiguration] | None = None,
        strategy: Literal["ddp", "deepspeed", "horovod"] | None = "ddp",
        checkpoint_path: str = "checkpoints/epoch_{}.pth",
        logger: Logger | None = None,
        random_seed: int | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the CaloChallengeTrainer with the provided configuration.

        Args:
            num_epochs (int, optional): Number of epochs to train. Defaults to 2.
            config (Union[Dict, TrainingConfiguration], optional): Training configuration dictionary
                or instance. Defaults to None.
            strategy (Optional[Literal["ddp", "deepspeed", "horovod"]], optional): Distributed training
                strategy. Defaults to "ddp".
            checkpoint_path (str, optional): File path format to save checkpoints. Defaults to
                "checkpoints/epoch_{}.pth".
            logger (Optional[Logger], optional): Logger instance for logging training progress.
                Defaults to None.
            random_seed (int | None, optional): Random seed for reproducibility. Defaults to None.
            name (Optional[str], optional): Optional name for the training session. Defaults to None.
            **kwargs: Additional arguments passed to the base class.

        Returns:
            None
        """
        super().__init__(
            epochs=num_epochs,
            config=config,
            strategy=strategy,
            logger=logger,
            random_seed=random_seed,
            name=name,
            **kwargs,
        )
        self.save_parameters(**self.locals2params(locals()))

        if isinstance(config, dict):
            config = TrainingConfiguration(**config)

        self.config = config
        self.num_epochs = num_epochs
        self.checkpoints_location = checkpoint_path
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    def log_prob_loss(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Computes the negative log-probability loss for the given input batch.

        Args:
            x (torch.Tensor): Input tensor.
            c (torch.Tensor): Conditioning tensor.

        Returns:
            torch.Tensor: Computed negative log-probability loss.
        """
        z, log_jac_det = self.model.forward(x, c, rev=False)
        log_prob = (
            -0.5 * torch.sum(z**2, 1)
            + log_jac_det
            - z.shape[1] / 2 * math.log(2 * math.pi)
        )
        return -torch.mean(log_prob)

    def log_prob_kl_loss(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Computes the negative log-probability loss with KL divergence for Bayesian models.

        Args:
            x (torch.Tensor): Input tensor.
            c (torch.Tensor): Conditioning tensor.

        Returns:
            torch.Tensor: Computed total loss (log-probability + KL divergence).
        """
        z, log_jac_det = self.model.forward(x, c, rev=False)
        log_prob = (
            -0.5 * torch.sum(z**2, 1)
            + log_jac_det
            - z.shape[1] / 2 * math.log(2 * math.pi)
        )
        kl_loss = sum(layer.KL() for layer in self.model.bayesian_layers)
        return -torch.mean(log_prob) + kl_loss / z.shape[0]

    def create_model_loss_optimizer(self) -> None:
        """Creates the model, loss function, optimizer, and learning rate scheduler.

        Initializes the CINN model, assigns the loss function depending on model type,
        sets up the AdamW optimizer and a OneCycleLR scheduler, and applies distributed
        strategies if needed.

        Returns:
            None
        """
        self.model = CINN(
            next(itertools.islice(self.train_dataloader, 0, None))[0].shape[1],
            self.config,
        )

        if self.model.bayesian:
            self.loss = self.log_prob_kl_loss
        else:
            self.loss = self.log_prob_loss

        # TODO: include other option for optimizer and scheduler from config
        # TODO: change eps, seems the double naming happened
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.optim_lr,
            weight_decay=self.config.optim_weight_decay,
            eps=self.config.eps,
        )

        self.lr_scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            self.config.max_lr or self.config.optim_lr * 10,
            epochs=self.config.cycle_epochs or self.num_epochs,
            steps_per_epoch=len(self.train_dataloader),
            final_div_factor=1e0,
        )

        # IMPORTANT: model, optimizer, and scheduler need to be distributed
        distribute_kwargs = self.get_default_distributed_kwargs()

        # Distributed model, optimizer, and scheduler
        (self.model, self.optimizer, self.lr_scheduler) = self.strategy.distributed(
            self.model, self.optimizer, self.lr_scheduler, **distribute_kwargs
        )

    def train_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Perform a single optimization step using a batch sampled from the
        training dataset.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Batch sampled by a dataloader,
                containing input and conditioning tensors.
            batch_idx (int): Batch index within the dataloader.

        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]: Batch loss and a dictionary of computed metrics.
        """
        x, c = batch
        x, c = x.to(self.device), c.to(self.device)

        self.optimizer.zero_grad()
        loss = self.loss(x, c)
        loss.backward()
        self.optimizer.step()
        if self.lr_scheduler.last_epoch < self.lr_scheduler.total_steps - 2:
            self.lr_scheduler.step()

        self.log(
            self.lr_scheduler.optimizer.param_groups[0]["lr"],
            identifier="lr",
            kind="metric",
            step=self.train_glob_step,
            batch_idx=batch_idx,
        )

        # Log metrics
        self.log(
            item=loss.item(),
            identifier="train_loss",
            kind="metric",
            step=self.train_glob_step,
            batch_idx=batch_idx,
        )
        metrics: Dict[str, Any] = self.compute_metrics(
            true=None,
            pred=None,
            logger_step=self.train_glob_step,
            batch_idx=batch_idx,
            stage="train",
        )
        return loss, metrics

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Perform a single optimization step using a batch sampled from the
        validation dataset.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Batch sampled by a dataloader,
                containing input and conditioning tensors.
            batch_idx (int): Batch index within the dataloader.

        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]: Batch loss and a dictionary of computed metrics.
        """
        x, c = batch
        x, c = x.to(self.device), c.to(self.device)
        with torch.no_grad():
            loss: torch.Tensor = self.loss(x, c)
        self.log(
            item=loss.item(),
            identifier="validation_loss",
            kind="metric",
            step=self.validation_glob_step,
            batch_idx=batch_idx,
        )
        metrics: Dict[str, Any] = self.compute_metrics(
            true=None,
            pred=None,
            logger_step=self.validation_glob_step,
            batch_idx=batch_idx,
            stage="validation",
        )
        return loss, metrics