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
from ray import train
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torchvision.models import resnet18

from itwinai.loggers import Logger
from itwinai.torch.config import TrainingConfiguration
from itwinai.torch.distributed import DeepSpeedStrategy
from itwinai.torch.trainer import TorchTrainer


class MyTrainer(TorchTrainer):
    def __init__(
        self,
        epochs: int,
        config: Dict | TrainingConfiguration,
        strategy: Literal["ddp", "deepspeed", "horovod"] = "ddp",
        name: str | None = None,
        logger: Logger | None = None,
    ) -> None:
        self.config = config
        super().__init__(
            config=config, epochs=epochs, strategy=strategy, name=name, logger=logger
        )

    def create_model_loss_optimizer(self):
        model = resnet18(num_classes=10)
        model.conv1 = torch.nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        # First, define strategy-wise optional configurations
        if isinstance(self.strategy, DeepSpeedStrategy):
            distribute_kwargs = dict(
                config_params=dict(train_micro_batch_size_per_gpu=self.config.batch_size)
            )
        else:
            distribute_kwargs = {}
        optimizer = Adam(model.parameters(), lr=self.config.optim_lr)
        self.model, self.optimizer, _ = self.strategy.distributed(
            model, optimizer, **distribute_kwargs
        )
        self.loss = CrossEntropyLoss()

    def train(self):
        for epoch in range(self.epochs):
            if self.strategy.global_world_size() > 1:
                self.set_epoch(epoch)

            train_losses = []
            val_losses = []

            for images, labels in self.train_dataloader:
                device = self.strategy.device()
                images, labels = images.to(device), labels.to(device)

                outputs = self.model(images)
                train_loss = self.loss(outputs, labels)
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
                train_losses.append(train_loss.detach().cpu().numpy())

            for images, labels in self.validation_dataloader:
                device = self.strategy.device()
                images, labels = images.to(device), labels.to(device)

                with torch.no_grad():
                    outputs = self.model(images)
                    val_loss = self.loss(outputs, labels)
                val_losses.append(val_loss.detach().cpu().numpy())

            epoch_val_loss = np.mean(val_losses)
            epoch_train_loss = np.mean(train_losses)

            self.log(epoch_train_loss, "train_loss", kind="metric", step=epoch)
            self.log(epoch_val_loss, "val_loss", kind="metric", step=epoch)

            # Report training metrics of last epoch to Ray
            train.report({"loss": epoch_val_loss})
