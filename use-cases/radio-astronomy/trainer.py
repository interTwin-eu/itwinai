import os
from typing import Any, Dict, Literal, Optional, Tuple
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
from torch import device, cuda
from src.pulsar_analysis.neural_network_models import (
    UNet,
    CustomLossUNet,
    OneDconvEncoder,
    Simple1DCnnClassifier,
    CustomLossClassifier,
)
from itwinai.torch.trainer import TorchTrainer
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
        strategy: Literal["ddp", "deepspeed", "horovod"] | None = "ddp",
        logger: Logger | None = None,
        num_epochs: int = 3,
        model: nn.Module = None,
        store_trained_model_at: str = ".models/model.pt",
    ) -> None:
        # manually set up the config:


        # these parameters are initialized with the TorchTrainer class:
        super().__init__(
            config=config,
            strategy=strategy,
            logger=logger,
            epochs=num_epochs,
            model=model
        )

        os.makedirs(os.path.dirname(store_trained_model_at), exist_ok=True)


    def create_dataloaders(
        self,
        train_dataset: Dataset,
        validation_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
    ) -> None:
        self.train_dataloader = DataLoader(
            dataset=train_dataset, batch_size=self.config.batch_size, shuffle=self.config.shuffle_train, pin_memory=True
        )