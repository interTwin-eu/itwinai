# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Anna Lappe
#
# Credit:
# - Anna Lappe <anna.elisa.lappe@cern.ch> - CERN
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------


import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim.adam import Adam
from torchvision.models import resnet18

from itwinai.torch.distributed import RayDeepSpeedStrategy
from itwinai.torch.trainer import TorchTrainer


class FashionMNISTTrainer(TorchTrainer):
    def create_model_loss_optimizer(self):
        model = resnet18(num_classes=10)
        model.conv1 = torch.nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False
        )
        # First, define strategy-wise optional configurations
        if isinstance(self.strategy, RayDeepSpeedStrategy):
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

    def train(self) -> None:
        device = self.strategy.device()

        for self.current_epoch in range(self.epochs):
            self.set_epoch()

            train_losses = []
            val_losses = []

            # Training epoch
            self.model.train()
            for images, labels in self.train_dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                train_loss = self.loss(outputs, labels)
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
                train_losses.append(train_loss.detach().cpu().numpy())

            # Validation epoch
            self.model.eval()
            for images, labels in self.validation_dataloader:
                images, labels = images.to(device), labels.to(device)
                with torch.no_grad():
                    outputs = self.model(images)
                    val_loss = self.loss(outputs, labels)
                val_losses.append(val_loss.detach().cpu().numpy())

            # Log metrics with itwinai loggers
            self.log(
                np.mean(train_losses), "train_loss", kind="metric", step=self.current_epoch
            )
            self.log(np.mean(val_losses), "val_loss", kind="metric", step=self.current_epoch)

            # Report metrics and checkpoint to Ray head
            checkpoint = {
                "epoch": self.current_epoch,
                "loss": train_loss,
                "val_loss": val_loss,
            }
            metrics = {"loss": val_loss.item()}
            self.ray_report(metrics=metrics, checkpoint_data=checkpoint)
