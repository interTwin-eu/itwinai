import os
from typing import Any, Dict, Literal, Optional, Tuple
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
from torch import device, cuda, save
from src.pulsar_analysis.neural_network_models import (
    UNet,
    CustomLossUNet,
    OneDconvEncoder,
    Simple1DCnnClassifier,
    CustomLossClassifier,
)
from itwinai.torch.trainer import TorchTrainer, RayTorchTrainer
from itwinai.torch.config import TrainingConfiguration
from itwinai.loggers import EpochTimeTracker, Logger

from src.pulsar_analysis.train_neural_network_model import TrainImageToMaskNetworkModel,ImageToMaskDataset,InMaskToMaskDataset
from src.pulsar_analysis.neural_network_models import UNet, CustomLossUNet, UNetFilter, FilterCNN, CustomLossSemanticSeg, CNN1D, WeightedBCELoss

from itwinai.torch.distributed import DeepSpeedStrategy, RayDDPStrategy, RayDeepSpeedStrategy

class PulsarTrainer(TorchTrainer):
    """Trainer class for radio-astronomy use-case. 
       Inherited from itwinai TorchTrainer class.
    """
    def __init__(
        self,
        config: Dict | TrainingConfiguration | None = None,
        strategy: Literal["ddp", "deepspeed", "horovod"] | None = None,
        logger: Logger | None = None,
        num_epochs: int = 3,
        model: nn.Module = None,
        loss: nn.Module = None,
        store_trained_model_at: str = ".models/model.pt",
        name: Optional[str] = None,
    ) -> None:

        # these parameters are initialized with the TorchTrainer class:
        super().__init__(
            config=config,
            strategy=strategy,
            logger=logger,
            epochs=num_epochs,
            model=model,
            name=name
        )

        # set the custom loss function
        self.loss = loss 
        self.store_trained_model_at = store_trained_model_at
        os.makedirs(os.path.dirname(store_trained_model_at), exist_ok=True)

    def create_model_loss_optimizer(self) -> None:

        """Redefined method that doens't implement
        the loss function as it's defined in the constructor"""

        if self.model is None:
            raise ValueError(
                "self.model is None! Either pass it to the constructor or "
                "override create_model_loss_optimizer method."
            )

        self._optimizer_from_config()

        distribute_kwargs = self.get_default_distributed_kwargs()

        # Distributed model, optimizer, and scheduler
        (self.model, self.optimizer, self.lr_scheduler) = self.strategy.distributed(
            self.model, self.optimizer, self.lr_scheduler, **distribute_kwargs
        )

    # def create_dataloaders(
    #     self,
    #     train_dataset: Dataset,
    #     validation_dataset: Optional[Dataset] = None,
    #     test_dataset: Optional[Dataset] = None,
    # ) -> None:
    #     self.train_dataloader = DataLoader(
    #         dataset=train_dataset, batch_size=self.config.batch_size, shuffle=self.config.shuffle_train, pin_memory=True)
    #     self.validation_dataloader = DataLoader(
    #         dataset=validation_dataset, batch_size=self.config.batch_size, shuffle=self.config.shuffle_train, pin_memory=True)
    #     self.test_dataloader = DataLoader(
    #         dataset=test_dataset, batch_size=self.config.batch_size, shuffle=self.config.shuffle_train, pin_memory=True)
        
    def write_model(self) -> None:
        """Write the model to disk."""
        save(self.model.state_dict(), self.store_trained_model_at)
