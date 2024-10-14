# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

import abc
import os
from typing import Any, Dict, List, Literal, Optional, Union

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..components import Predictor, monitor_exec
from ..loggers import Logger
from ..serialization import ModelLoader
from .config import TrainingConfiguration
from .distributed import (
    DeepSpeedStrategy,
    HorovodStrategy,
    NonDistributedStrategy,
    TorchDDPStrategy,
    TorchDistributedStrategy,
    distributed_resources_available,
)
from .type import Batch


class TorchModelLoader(ModelLoader):
    """Loads a torch model from somewhere.

    Args:
        model_uri (str): Can be a path on local filesystem
            or an mlflow 'locator' in the form:
            'mlflow+MLFLOW_TRACKING_URI+RUN_ID+ARTIFACT_PATH'
    """

    def __call__(self) -> nn.Module:
        """Loads model from model URI.

        Raises:
            ValueError: if the model URI is not recognized
                or the model is not found.

        Returns:
            nn.Module: torch neural network.
        """
        if os.path.exists(self.model_uri):
            # Model is on local filesystem.
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = torch.load(self.model_uri, map_location=device)
            return model

        if self.model_uri.startswith('mlflow+'):
            # Model is on an MLFLow server
            # Form is 'mlflow+MLFLOW_TRACKING_URI+RUN_ID+ARTIFACT_PATH'
            import mlflow
            from mlflow import MlflowException

            _, tracking_uri, run_id, artifact_path = self.model_uri.split("+")
            mlflow.set_tracking_uri(tracking_uri)

            # Check that run exists
            try:
                mlflow.get_run(run_id)
            except MlflowException:
                raise ValueError(f"Run ID '{run_id}' was not found!")

            # Download model weights
            ckpt_path = mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path=artifact_path,
                dst_path="tmp/",
                tracking_uri=mlflow.get_tracking_uri(),
            )
            model = torch.load(ckpt_path)
            return model.eval()

        raise ValueError(
            "Unrecognized model URI: model may not be there! "
            f"Received model URI: {self.model_uri}"
        )


class TorchPredictor(Predictor):
    """Applies a pre-trained torch model to unseen data."""

    _strategy: TorchDistributedStrategy = None
    #: PyTorch ``DataLoader`` for inference dataset.
    inference_dataloader: DataLoader = None
    #: Pre-trained PyTorch model used to make predictions.
    model: nn.Module = None
    #: PyTorch random number generator (PRNG).
    torch_rng: torch.Generator = None
    #: itwinai ``itwinai.Logger``
    logger: Logger = None

    def __init__(
        self,
        config: Union[Dict, TrainingConfiguration],
        model: Union[nn.Module, ModelLoader],
        strategy: Literal["ddp", "deepspeed", "horovod"] = 'ddp',
        logger: Optional[Logger] = None,
        checkpoints_location: str = "checkpoints",
        name: str = None
    ) -> None:
        super().__init__(model=model, name=name)
        self.save_parameters(**self.locals2params(locals()))
        if isinstance(config, dict):
            self.config = TrainingConfiguration(**config)
        else:
            self.config = config
        self.model = self.model.eval()
        self.strategy = strategy
        self.logger = logger
        self.checkpoints_location = checkpoints_location

    @property
    def strategy(self) -> TorchDistributedStrategy:
        """Strategy currently in use."""
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: Union[str, TorchDistributedStrategy]) -> None:
        if isinstance(strategy, TorchDistributedStrategy):
            self._strategy = strategy
        else:
            self._strategy = self._detect_strategy(strategy)

    @property
    def device(self) -> str:
        """Current device from distributed strategy."""
        return self.strategy.device()

    def _detect_strategy(self, strategy: str) -> TorchDistributedStrategy:
        if not distributed_resources_available():
            print("WARNING: falling back to non-distributed strategy.")
            dist_str = NonDistributedStrategy()
        elif strategy == 'ddp':
            dist_str = TorchDDPStrategy(backend='nccl')
        elif strategy == 'horovod':
            dist_str = HorovodStrategy()
        elif strategy == 'deepspeed':
            dist_str = DeepSpeedStrategy(backend='nccl')
        else:
            raise NotImplementedError(
                f"Strategy '{strategy}' is not recognized/implemented.")
        return dist_str

    def _init_distributed_strategy(self) -> None:
        if not self.strategy.is_initialized:
            self.strategy.init()

    def distribute_model(self) -> None:
        """
        Distribute the torch model with the chosen strategy.
        """
        if self.model is None:
            raise ValueError(
                "self.model is None! Mandatory constructor argument "
            )
        distribute_kwargs = {}
        # Distributed model, optimizer, and scheduler
        self.model, _, _ = self.strategy.distributed(
            self.model, None, None, **distribute_kwargs
        )

    def create_dataloaders(
        self,
        inference_dataset: Dataset
    ) -> None:
        """
        Create inference dataloader.

        Args:
            inference_dataset (Dataset): inference dataset object.
        """

        self.inference_dataloader = self.strategy.create_dataloader(
            dataset=inference_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers_dataloader,
            pin_memory=self.config.pin_gpu_memory,
            generator=self.torch_rng,
            shuffle=self.config.shuffle_test
        )

    @monitor_exec
    def execute(
        self,
        inference_dataset: Dataset,
        model: nn.Module = None,
    ) -> Dict[str, Any]:
        """Applies a torch model to a dataset for inference.

        Args:
            inference_dataset (Dataset[str, Any]): each item in this dataset is a
                couple (item_unique_id, item)
            model (nn.Module, optional): torch model. Overrides the existing
                model, if given. Defaults to None.

        Returns:
            Dict[str, Any]: maps each item ID to the corresponding predicted
                value(s).
        """
        self._init_distributed_strategy()
        if model is not None:
            # Overrides existing "internal" model
            self.model = model

        self.create_dataloaders(
            inference_dataset=inference_dataset
        )

        self.distribute_model()

        if self.logger:
            self.logger.create_logger_context(rank=self.strategy.global_rank())
            hparams = self.config.model_dump()
            hparams['distributed_strategy'] = self.strategy.__class__.__name__
            self.logger.save_hyperparameters(hparams)

        all_predictions = dict()
        for ids, (samples_ids, samples) in enumerate(self.inference_dataloader):
            with torch.no_grad():
                pred = self.model(samples.to(self.device))
            pred = self.transform_predictions(pred)
            for idx, pre in zip(samples_ids, pred):
                # For each item in the batch
                if pre.numel() == 1:
                    pre = pre.item()
                else:
                    pre = pre.to_dense().tolist()
                all_predictions[idx] = pre

        if self.logger:
            self.logger.destroy_logger_context()

        self.strategy.clean_up()

        return all_predictions

    def log(
        self,
        item: Union[Any, List[Any]],
        identifier: Union[str, List[str]],
        kind: str = 'metric',
        step: Optional[int] = None,
        batch_idx: Optional[int] = None,
        **kwargs
    ) -> None:
        """Log ``item`` with ``identifier`` name of ``kind`` type at ``step``
        time step.

        Args:
            item (Union[Any, List[Any]]): element to be logged (e.g., metric).
            identifier (Union[str, List[str]]): unique identifier for the
                element to log(e.g., name of a metric).
            kind (str, optional): type of the item to be logged. Must be one
                among the list of self.supported_types. Defaults to 'metric'.
            step (Optional[int], optional): logging step. Defaults to None.
            batch_idx (Optional[int], optional): DataLoader batch counter
                (i.e., batch idx), if available. Defaults to None.
        """
        if self.logger:
            self.logger.log(
                item=item,
                identifier=identifier,
                kind=kind,
                step=step,
                batch_idx=batch_idx,
                **kwargs
            )

    @abc.abstractmethod
    def transform_predictions(self, batch: Batch) -> Batch:
        """
        Post-process the predictions of the torch model (e.g., apply
        threshold in case of multi-label classifier).
        """


class MulticlassTorchPredictor(TorchPredictor):
    """Applies a pre-trained torch model to unseen data for multiclass classification."""

    def transform_predictions(self, batch: Batch) -> Batch:
        batch = batch.argmax(-1)
        return batch


class MultilabelTorchPredictor(TorchPredictor):
    """Applies a pre-trained torch model to unseen data for
    multilabel classification, applying a threshold on the
    output of the neural network.
    """

    #: Threshold to transform probabilities into class predictions.
    #: Defaults to 0.5.
    threshold: float = 0.5

    def __init__(
        self,
        model: Union[nn.Module, ModelLoader],
        test_dataloader_class: str = "torch.utils.data.DataLoader",
        test_dataloader_kwargs: Optional[Dict] = None,
        threshold: float = 0.5,
        name: str = None,
    ) -> None:
        super().__init__(model, test_dataloader_class, test_dataloader_kwargs, name)
        self.threshold = threshold

    def transform_predictions(self, batch: Batch) -> Batch:
        return (batch > self.threshold).float()


class RegressionTorchPredictor(TorchPredictor):
    """Applies a pre-trained torch model to unseen data for
    regression, leaving untouched the output of the neural
    network.
    """

    def transform_predictions(self, batch: Batch) -> Batch:
        return batch
