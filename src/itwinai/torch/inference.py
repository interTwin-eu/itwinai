# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# - Rakesh Sarma <r.sarma@fz-juelich.de> - Juelich
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# --------------------------------------------------------------------------------------

from pathlib import Path
from typing import Any, Dict, List, Literal

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..components import Predictor, monitor_exec
from ..loggers import Logger
from ..serialization import ModelLoader
from .config import TrainingConfiguration
from .distributed import TorchDistributedStrategy
from .trainer import TorchTrainer


class TorchModelLoader(ModelLoader):
    """Loads a torch model from somewhere.

    Args:
        model_uri (str): Can be a path on local filesystem
            or an mlflow 'locator' in the form:
            'mlflow+MLFLOW_TRACKING_URI+RUN_ID+ARTIFACT_PATH'
    """

    def __init__(self, model_uri: str, model_class: nn.Module | None = None):
        self.model_uri = model_uri
        self.model_class = model_class

    def __call__(self) -> nn.Module:
        """Loads model from model URI.

        Raises:
            ValueError: if the model URI is not recognized
                or the model is not found.

        Returns:
            nn.Module: torch neural network.
        """
        if Path(self.model_uri).exists():
            # Model is on local filesystem.
            checkpoint = torch.load(self.model_uri, weights_only=False)

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                if self.model_class is None:
                    raise ValueError(
                        "model_class required to instantiate model when checkpoint is dict."
                    )
                model = self.model_class()
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            else:
                model = checkpoint

            return model.eval()

        if self.model_uri.startswith("mlflow+"):
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
            checkpoint = torch.load(ckpt_path)

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                if self.model_class is None:
                    raise ValueError("model_class is required to instantiate the model.")
                model = self.model_class()
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            else:
                model = checkpoint

            return model.eval()

        raise ValueError(
            "Unrecognized model URI: model may not be there! "
            f"Received model URI: {self.model_uri}"
        )


class TorchPredictor(TorchTrainer, Predictor):
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
        config: Dict | TrainingConfiguration,
        model: nn.Module | ModelLoader,
        strategy: Literal["ddp", "deepspeed", "horovod"] = "ddp",
        logger: Logger | None = None,
        checkpoints_location: str = "checkpoints",
        name: str | None = None,
    ) -> None:
        if isinstance(config, dict):
            self.config = TrainingConfiguration(**config)
        else:
            self.config = config

        epochs = getattr(self.config, "epochs", 1)

        super().__init__(model=model, config=self.config, epochs=epochs, name=name)
        self.save_parameters(**self.locals2params(locals()))
        self.strategy = strategy
        self.logger = logger
        self.checkpoints_location = checkpoints_location

    def distribute_model(self) -> None:
        """Distribute the torch model with the chosen strategy."""
        if self.model is None:
            raise ValueError(
                "self.model is None! Please ensure it is set before calling this method."
            )
        distribute_kwargs = {}
        # Distributed model
        self.model, _, _ = self.strategy.distributed(
            model=self.model, optimizer=None, lr_scheduler=None, **distribute_kwargs
        )

    def create_dataloaders(self, inference_dataset: Dataset) -> DataLoader:
        """Create inference dataloader.

        Args:
            inference_dataset (Dataset): inference dataset object.

        Returns:
            DataLoader: Instance of DataLoader for the given inference dataset.
        """

        self.inference_dataloader = self.strategy.create_dataloader(
            dataset=inference_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers_dataloader,
            pin_memory=self.config.pin_gpu_memory,
            generator=self.torch_rng,
            shuffle=self.config.shuffle_test,
        )

        return self.inference_dataloader

    def predict(self) -> Dict[str, Any]:
        """Predicts or runs inference on a trained ML model.

        Returns:
            Dict[str, Any]: maps each item ID to the corresponding predicted values.
        """
        all_predictions = {}
        for samples_ids, samples in self.inference_dataloader:
            with torch.no_grad():
                pred_batch = self.model(samples.to(self.device))
            pred_batch = self.transform_predictions(pred_batch)
            for idx, pred in zip(samples_ids, pred_batch, strict=True):
                # For each item in the batch
                if pred.numel() == 1:
                    pred = pred.item()
                else:
                    pred = pred.to_dense().tolist()
                all_predictions[idx] = pred
        return all_predictions

    @monitor_exec
    def execute(
        self,
        inference_dataset: Dataset,
        model: nn.Module | None = None,
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
        elif isinstance(self.model, TorchModelLoader):
            self.model = self.model()

        self.create_dataloaders(inference_dataset=inference_dataset)

        self.distribute_model()

        if self.logger:
            self.logger.create_logger_context(rank=self.strategy.global_rank())
            hparams = self.config.model_dump()
            hparams["distributed_strategy"] = self.strategy.__class__.__name__
            self.logger.save_hyperparameters(hparams)

        all_predictions = self.predict()

        if self.logger:
            self.logger.destroy_logger_context()

        self.strategy.clean_up()

        return all_predictions

    def log(
        self,
        item: Any | List[Any],
        identifier: str | List[str],
        kind: str = "metric",
        step: int | None = None,
        batch_idx: int | None = None,
        **kwargs,
    ) -> None:
        """Log ``item`` with ``identifier`` name of ``kind`` type at ``step``
        time step.

        Args:
            item (Union[Any, List[Any]]): element to be logged (e.g., metric).
            identifier (Union[str, List[str]]): unique identifier for the
                element to log(e.g., name of a metric).
            kind (str): Type of the item to be logged. Must be one
                among the list of self.supported_types. Defaults to 'metric'.
            step (int | None): logging step. Defaults to None.
            batch_idx (int | None): DataLoader batch counter
                (i.e., batch idx), if available. Defaults to None.
        """
        if self.logger:
            self.logger.log(
                item=item,
                identifier=identifier,
                kind=kind,
                step=step,
                batch_idx=batch_idx,
                **kwargs,
            )

    def transform_predictions(self, batch: torch.Tensor) -> torch.Tensor:
        """Post-process the predictions of the torch model (e.g., apply
        threshold in case of multi-label classifier).
        """


class MulticlassTorchPredictor(TorchPredictor):
    """Applies a pre-trained torch model to unseen data for multiclass classification."""

    def transform_predictions(self, batch: torch.Tensor) -> torch.Tensor:
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
        model: nn.Module | ModelLoader,
        test_dataloader_class: str = "torch.utils.data.DataLoader",
        test_dataloader_kwargs: Dict | None = None,
        threshold: float = 0.5,
        name: str = None,
    ) -> None:
        super().__init__(model, test_dataloader_class, test_dataloader_kwargs, name)
        self.threshold = threshold

    def transform_predictions(self, batch: torch.Tensor) -> torch.Tensor:
        return (batch > self.threshold).float()


class RegressionTorchPredictor(TorchPredictor):
    """Applies a pre-trained torch model to unseen data for
    regression, leaving untouched the output of the neural
    network.
    """

    def transform_predictions(self, batch: torch.Tensor) -> torch.Tensor:
        return batch
