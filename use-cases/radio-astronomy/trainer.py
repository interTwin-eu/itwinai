# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Alex Krochak
#
# Credit:
# - Alex Krochak <o.krochak@fz-juelich.de> - FZJ
# --------------------------------------------------------------------------------------

import os
from typing import Any, Dict, Literal, Optional, Tuple
from torch.utils.data import Dataset

import torch.nn as nn
from torch import save
from itwinai.torch.trainer import TorchTrainer
from itwinai.torch.config import TrainingConfiguration
from itwinai.loggers import Logger
from itwinai.torch.monitoring.monitoring import measure_gpu_utilization
from itwinai.torch.profiling.profiler import profile_torch_trainer

class PulsarTrainer(TorchTrainer):
    """Trainer class for radio-astronomy use-case.
    Inherited from itwinai TorchTrainer class."""

    def __init__(
        self,
        config: Dict | TrainingConfiguration | None = None,
        strategy: Literal["ddp", "deepspeed", "horovod"] | None = None,
        logger: Logger | None = None,
        epochs: int = 3,
        model: nn.Module = None,
        loss: nn.Module = None,
        store_trained_model_at: str = ".models/model.pt",
        name: Optional[str] = None,
        measure_gpu_data: bool = False,
        enable_torch_profiling: bool = False,
        measure_epoch_time: bool = False,
    ) -> None:

        # these parameters are initialized with the TorchTrainer class:
        super().__init__(
            config=config,
            strategy=strategy,
            logger=logger,
            epochs=epochs,
            model=model,
            name=name,
            measure_gpu_data=measure_gpu_data,
            enable_torch_profiling=enable_torch_profiling,
            measure_epoch_time=measure_epoch_time,
        )
        # set the custom loss function
        self.loss = loss
        self.store_trained_model_at = store_trained_model_at
        os.makedirs(os.path.dirname(store_trained_model_at), exist_ok=True)

    # NOTE: this would be a nice way to re-use the original __init__
    # and insert the custom loss function, but AFAIK this doesn't
    # work nicely while running from config.yaml

    # def set_attributes(self, loss, store_trained_model_at) -> None:
    #     self.loss = loss
    #     self.store_trained_model_at = store_trained_model_at
    #     os.makedirs(os.path.dirname(store_trained_model_at), exist_ok=True)

    def create_model_loss_optimizer(self) -> None:

        ### This code is almost a complete copy of this method from ###
        ### src/itwinai/torch/trainer.py, with exception of removed ###
        ### loss function definition as it is already set in the    ###
        ### constructor                                             ###

        # Model, optimizer, and lr scheduler may have already been loaded from a checkpoint

        if self.model is None:
            raise ValueError(
                "self.model is None! Either pass it to the constructor, load a checkpoint, or "
                "override create_model_loss_optimizer method."
            )
        if self.model_state_dict:
            # Load model from checkpoint
            self.model.load_state_dict(self.model_state_dict, strict=False)

        # Parse optimizer from training configuration
        # Optimizer can be changed with a custom one here!
        self._set_optimizer_from_config()

        # Parse LR scheduler from training configuration
        # LR scheduler can be changed with a custom one here!
        self._set_lr_scheduler_from_config()

        if self.optimizer_state_dict:
            # Load optimizer state from checkpoint
            # IMPORTANT: this must be after the learning rate scheduler was already initialized
            # by passing to it the optimizer. Otherwise the optimizer state just loaded will
            # be modified by the lr scheduler.
            self.optimizer.load_state_dict(self.optimizer_state_dict)

        if self.lr_scheduler_state_dict and self.lr_scheduler:
            # Load LR scheduler state from checkpoint
            self.lr_scheduler.load_state_dict(self.lr_scheduler_state_dict)

        # IMPORTANT: model, optimizer, and scheduler need to be distributed from here on

        distribute_kwargs = self.get_default_distributed_kwargs()

        # Distributed model, optimizer, and scheduler
        (self.model, self.optimizer, self.lr_scheduler) = self.strategy.distributed(
            self.model, self.optimizer, self.lr_scheduler, **distribute_kwargs
        )

    def write_model(self) -> None:
        """Write the model to disk."""
        save(self.model.state_dict(), self.store_trained_model_at)

    # Hacky way to get around the fact that the execute method
    # does not return self.model anymore
    @profile_torch_trainer
    @measure_gpu_utilization
    def execute(
        self,
        train_dataset: Dataset,
        validation_dataset: Dataset | None = None,
        test_dataset: Dataset | None = None,
    ) -> Tuple[Dataset, Dataset, Dataset, Any]:
        objs = super().execute(
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            test_dataset=test_dataset,
        )
        # return the datasets and the model
        return *(objs[:-1]), self.model
