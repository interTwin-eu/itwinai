import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torchvision.models import resnet18
from itwinai.loggers import Logger
from itwinai.torch.trainer_ray import RayTorchTrainer
from itwinai.torch.distributed import RayDeepSpeedStrategy
from typing import Dict, Optional, Literal


class MyRayTrainer(RayTorchTrainer):
    def __init__(
        self,
        config: Dict,
        strategy: Optional[Literal["ddp", "deepspeed", "horovod"]] = 'ddp',
        name: Optional[str] = None,
        logger: Optional[Logger] = None
    ) -> None:
        super().__init__(
            config=config,
            strategy=strategy,
            name=name,
            logger=logger
        )

    def create_model_loss_optimizer(self):
        # Model, Loss, Optimizer
        model = resnet18(num_classes=10)
        model.conv1 = torch.nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )

        # First, define strategy-wise optional configurations
        if isinstance(self.strategy, RayDeepSpeedStrategy):
            # Batch size definition is not optional for DeepSpeedStrategy!
            distribute_kwargs = dict(
                config_params=dict(
                    train_micro_batch_size_per_gpu=self.training_config["batch_size"]
                )
            )
        else:
            distribute_kwargs = {}

        optimizer = Adam(model.parameters(), lr=self.training_config["learning_rate"])
        self.model, self.optimizer, _ = self.strategy.distributed(
            model,
            optimizer,
            **distribute_kwargs
        )
        self.loss = CrossEntropyLoss()

    def train(self, config, data):

        self.training_config = config

        if self.logger is not None:
            print("Logger is not None!")
            print(self.strategy.global_rank())
            self.logger.create_logger_context(rank=self.strategy.global_rank())
            print(f"Worker rank set: {self.logger.worker_rank}")

        self.create_model_loss_optimizer()

        self.create_dataloaders(
            train_dataset=data[0],
            validation_dataset=data[1],
            test_dataset=data[2],
            batch_size=self.training_config["batch_size"]
        )

        # Training
        for epoch in range(self.training_config["epochs"]):
            if self.strategy.global_world_size() > 1:
                self.set_epoch(epoch)

            for images, labels in self.train_dataloader:
                outputs = self.model(images)
                loss = self.loss(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # [3] Report metrics and checkpoint.
            metrics = {"loss": loss.item(), "epoch": epoch}

            # Nothing checkpointed
            self.checkpoint_and_report(
                epoch,
                tuning_metrics=metrics
            )
