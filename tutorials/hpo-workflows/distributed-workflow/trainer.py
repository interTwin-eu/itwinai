# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Anna Lappe
#
# Credit:
# - Anna Lappe <anna.elisa.lappe@cern.ch> - CERN
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

from typing import Dict, Literal

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torchvision.models import resnet18

from itwinai.loggers import Logger
from itwinai.torch.distributed import RayDeepSpeedStrategy
from itwinai.torch.trainer import RayTorchTrainer


class MyRayTorchTrainer(RayTorchTrainer):
    def __init__(
        self,
        config: Dict,
        strategy: Literal["ddp", "deepspeed"] = "ddp",
        name: str | None = None,
        logger: Logger | None = None,
    ) -> None:
        super().__init__(config=config, strategy=strategy, name=name, logger=logger)

    def create_model_loss_optimizer(self):
        model = resnet18(num_classes=10)
        model.conv1 = torch.nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False
        )
        # First, define strategy-wise optional configurations
        if isinstance(self.strategy, RayDeepSpeedStrategy):
            distribute_kwargs = dict(
                config_params=dict(
                    train_micro_batch_size_per_gpu=self.training_config["batch_size"]
                )
            )
        else:
            distribute_kwargs = {}
        optimizer = Adam(model.parameters(), lr=self.training_config["learning_rate"])
        self.model, self.optimizer, _ = self.strategy.distributed(
            model, optimizer, **distribute_kwargs
        )
        self.loss = CrossEntropyLoss()

    def train(self, config, data):
        self.training_config = config

        # Because of the way the ray cluster is set up,
        # the initialisation of the strategy and logger, as well as the creation of the
        # model, loss, optimizer and dataloader are done from within the train() function
        self.strategy.init()
        self.initialize_logger(
            hyperparams=self.training_config, rank=self.strategy.global_rank()
        )
        self.create_model_loss_optimizer()
        self.create_dataloaders(
            train_dataset=data[0], validation_dataset=data[1], test_dataset=data[2]
        )

        for epoch in range(self.training_config["epochs"]):
            if self.strategy.global_world_size() > 1:
                self.set_epoch(epoch)

            train_losses = []
            val_losses = []

            for images, labels in self.train_dataloader:
                if isinstance(self.strategy, RayDeepSpeedStrategy):
                    device = self.strategy.device()
                    images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                train_loss = self.loss(outputs, labels)
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
                train_losses.append(train_loss.detach().cpu().numpy())

            for images, labels in self.validation_dataloader:
                if isinstance(self.strategy, RayDeepSpeedStrategy):
                    device = self.strategy.device()
                    images, labels = images.to(device), labels.to(device)
                with torch.no_grad():
                    outputs = self.model(images)
                    val_loss = self.loss(outputs, labels)
                val_losses.append(val_loss.detach().cpu().numpy())

            self.log(np.mean(train_losses), "train_loss", kind="metric", step=epoch)
            self.log(np.mean(val_losses), "val_loss", kind="metric", step=epoch)
            checkpoint = {
                "epoch": epoch,
                "loss": train_loss,
                "val_loss": val_loss,
            }
            metrics = {"loss": val_loss.item()}
            self.checkpoint_and_report(
                epoch, tuning_metrics=metrics, checkpointing_data=checkpoint
            )
