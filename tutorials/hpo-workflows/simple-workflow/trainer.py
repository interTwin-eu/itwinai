import os
from typing import Dict, Literal

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torchvision.models import resnet18
from ray import train

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
        optimizer = Adam(model.parameters(), lr=self.config.learning_rate)
        self.model, self.optimizer, _ = self.strategy.distributed(
            model, optimizer, **distribute_kwargs
        )
        self.loss = CrossEntropyLoss()

    def train(self):
        for epoch in range(self.config.epochs):
            if self.strategy.global_world_size() > 1:
                self.set_epoch(epoch)

            train_losses = []
            val_losses = []

            for images, labels in enumerate(self.train_dataloader):
                device = self.strategy.device()
                images, labels = images.to(device), labels.to(device)

                outputs = self.model(images)
                train_loss = self.loss(outputs, labels)
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
                train_losses.append(train_loss.detach().cpu().numpy())

            for images, labels in enumerate(self.validation_dataloader):
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
            checkpoint_filename = self.checkpoints_location.format(epoch)
            torch.save(checkpoint, checkpoint_filename)
            self.log(
                checkpoint_filename,
                os.path.basename(checkpoint_filename),
                kind="artifact",
            )
            # Report training metrics of last epoch to Ray
            train.report({"loss": val_loss})
