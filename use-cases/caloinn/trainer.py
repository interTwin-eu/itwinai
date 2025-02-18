import sys
import os
import time
from typing import Any, Dict, Literal, Optional, Tuple, Union

import torch
import torch.distributions as dist
import torch.optim as optim
import torch.nn as nn
import numpy as np
import math

import itertools

from model import CINN

from itwinai.loggers import EpochTimeTracker, Logger
from itwinai.torch.config import TrainingConfiguration
from itwinai.torch.trainer import TorchTrainer
from itwinai.torch.inference import TorchPredictor
#from ..src.itwinai.type import Batch, LrScheduler, Metric
from itwinai.serialization import ModelLoader


class CaloChallengeTrainer(TorchTrainer):
    def __init__(
        self,
        num_epochs: int = 2,
        config: Union[Dict, TrainingConfiguration] | None = None,
        strategy: Optional[Literal["ddp", "deepspeed", "horovod"]] = "ddp",
        checkpoint_path: str = "checkpoints/epoch_{}.pth",
        logger: Optional[Logger] = None,
        random_seed: Optional[int] = None,
        name: str | None = None,
        validation_every: int = 50,
        **kwargs,
    ) -> None:
        super().__init__(
            epochs=num_epochs,
            config=config,
            strategy=strategy,
            logger=logger,
            random_seed=random_seed,
            name=name,
            validation_every=validation_every,
            **kwargs,
        )
        self.save_parameters(**self.locals2params(locals()))

        if isinstance(config, dict):
            config = TrainingConfiguration(**config)

        self.config = config
        self.num_epochs = num_epochs
        self.checkpoints_location = checkpoint_path
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    def log_prob_loss(self, x, c):
        z, log_jac_det = self.model.forward(x, c, rev=False)
        log_prob = - 0.5*torch.sum(z**2, 1) + log_jac_det - z.shape[1]/2 * math.log(2*math.pi)
        return -torch.mean(log_prob)

    def log_prob_kl_loss(self, x, c):
        z, log_jac_det = self.model.forward(x, c, rev=False)
        log_prob = - 0.5*torch.sum(z**2, 1) + log_jac_det - z.shape[1]/2 * math.log(2*math.pi)
        kl_loss = sum(layer.KL() for layer in self.model.bayesian_layers)
        return -torch.mean(log_prob) + kl_loss / z.shape[0] 
        
    def create_model_loss_optimizer(self) -> None:

        self.model = CINN(next(itertools.islice(self.train_dataloader, 0, None))[0].shape[1], self.config)

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
                eps=self.config.eps
            )
        
        self.lr_scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                self.config.max_lr or self.config.optim_lr*10,
                epochs = self.num_epochs or self.config.cycle_epochs,
                steps_per_epoch=len(self.train_dataloader),
            )

        # IMPORTANT: model, optimizer, and scheduler need to be distributed
        distribute_kwargs = self.get_default_distributed_kwargs()

        # Distributed model, optimizer, and scheduler
        (self.model, self.optimizer, self.lr_scheduler) = self.strategy.distributed(
            self.model, self.optimizer, self.lr_scheduler, **distribute_kwargs
        )

    def train_step(self, batch, batch_idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Perform a single optimization step using a batch sampled from the
        training dataset.

        Args:
            batch (Batch): batch sampled by a dataloader.
            batch_idx (int): batch index in the dataloader.

        Returns:
            Tuple[Loss, Dict[str, Any]]: batch loss and dictionary of metric
            values with the same structure of ``self.metrics``.
        """
        print("IN TRAINING STEP")
        x, c = batch
        x, c = x.to(self.device), c.to(self.device)

        self.optimizer.zero_grad()
        loss = self.loss(x, c)
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        
        self.log(
            self.lr_scheduler.optimizer.param_groups[0]['lr'], 
            identifier="lr", 
            kind="metric", 
            step=self.train_glob_step,
            batch_idx=batch_idx
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
            true=None,#x,
            pred=None,#c,
            logger_step=self.train_glob_step,
            batch_idx=batch_idx,
            stage="train",
        )
        return loss, metrics 
    
    def validation_step(
        self, batch, batch_idx: int
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Perform a single optimization step using a batch sampled from the
        validation dataset.

        Args:
            batch (Batch): batch sampled by a dataloader.
            batch_idx (int): batch index in the dataloader.

        Returns:
            Tuple[Loss, Dict[str, Any]]: batch loss and dictionary of metric
            values with the same structure of ``self.metrics``.
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
            true=None,#y,
            pred=None,#pred_y,
            logger_step=self.validation_glob_step,
            batch_idx=batch_idx,
            stage="validation",
        )
        return loss, metrics
    

class CaloChallengePredictor(TorchPredictor):
    def __init__(
        self,
        model: Union[nn.Module, ModelLoader],
        config: Union[Dict, TrainingConfiguration] | None = None,
        test_dataloader_class: str = "torch.utils.data.DataLoader",
        test_dataloader_kwargs: Optional[Dict] = None,
        name: str = None
    ) -> None:
        super().__init__(
            model=model, 
            test_dataloader_class=test_dataloader_class,
            test_dataloader_kwargs=test_dataloader_kwargs,
            name=name
            )
        self.save_parameters(**self.locals2params(locals()))
        self.num_samples_to_generate = self.config.num_samples_to_generate
        self.config = config

    def execute(self,
        test_dataset,
        model,
        config
    ) -> None:
        if model is not None:
            # Overrides existing "internal" model
            self.model = model

        test_dataloader = self.test_dataloader_class(
            test_dataset, **self.test_dataloader_kwargs
        )

        self.model.generate()


        