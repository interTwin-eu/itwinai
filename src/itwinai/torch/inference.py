from typing import Optional, Dict, Any, Union
import os
import abc

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..utils import dynamically_import_class, clear_key
from ..components import Predictor, monitor_exec
from .type import Batch
from ..serialization import ModelLoader


class TorchModelLoader(ModelLoader):
    """Loads a torch model from somewhere.

    Args:
        model_uri (str): Can be a path on local filesystem
            or an mlflow 'locator' in the form:
            'mlflow+MLFLOW_TRACKING_URI+RUN_ID+ARTIFACT_PATH'
    """

    def __call__(self) -> nn.Module:
        """"Loads model from model URI.

        Raises:
            ValueError: if the model URI is not recognized
                or the model is not found.

        Returns:
            nn.Module: torch neural network.
        """
        if os.path.exists(self.model_uri):
            # Model is on local filesystem.
            model = torch.load(self.model_uri)
            return model.eval()

        if self.model_uri.startswith('mlflow+'):
            # Model is on an MLFLow server
            # Form is 'mlflow+MLFLOW_TRACKING_URI+RUN_ID+ARTIFACT_PATH'
            import mlflow
            from mlflow import MlflowException
            _, tracking_uri, run_id, artifact_path = self.model_uri.split('+')
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
                dst_path='tmp/',
                tracking_uri=mlflow.get_tracking_uri()
            )
            model = torch.load(ckpt_path)
            return model.eval()

        raise ValueError(
            'Unrecognized model URI: model may not be there! '
            f'Received model URI: {self.model_uri}'
        )


class TorchPredictor(Predictor):
    """Applies a pre-trained torch model to unseen data."""

    #: Pre-trained PyTorch model used to make predictions.
    model: nn.Module = None
    #: ``Dataset`` on which to make predictions (ML inference).
    test_dataset: Dataset
    #: ``DataLoader`` for test dataset.
    test_dataloader: DataLoader = None

    def __init__(
        self,
        model: Union[nn.Module, ModelLoader],
        test_dataloader_class: str = 'torch.utils.data.DataLoader',
        test_dataloader_kwargs: Optional[Dict] = None,
        name: str = None
    ) -> None:
        super().__init__(model=model, name=name)
        self.save_parameters(**self.locals2params(locals()))
        self.model = self.model.eval()

        # Train and validation dataloaders
        self.test_dataloader_class = dynamically_import_class(
            test_dataloader_class
        )
        test_dataloader_kwargs = (
            test_dataloader_kwargs
            if test_dataloader_kwargs is not None else {}
        )
        self.test_dataloader_kwargs = clear_key(
            test_dataloader_kwargs, 'train_dataloader_kwargs', 'dataset'
        )

    @monitor_exec
    def execute(
        self,
        test_dataset: Dataset,
        model: nn.Module = None,
    ) -> Dict[str, Any]:
        """Applies a torch model to a dataset for inference.

        Args:
            test_dataset (Dataset[str, Any]): each item in this dataset is a
                couple (item_unique_id, item)
            model (nn.Module, optional): torch model. Overrides the existing
                model, if given. Defaults to None.

        Returns:
            Dict[str, Any]: maps each item ID to the corresponding predicted
                value(s).
        """
        if model is not None:
            # Overrides existing "internal" model
            self.model = model

        test_dataloader = self.test_dataloader_class(
            test_dataset, **self.test_dataloader_kwargs
        )

        all_predictions = dict()
        for samples_ids, samples in test_dataloader:
            with torch.no_grad():
                pred = self.model(samples)
            pred = self.transform_predictions(pred)
            for idx, pre in zip(samples_ids, pred):
                # For each item in the batch
                if pre.numel() == 1:
                    pre = pre.item()
                else:
                    pre = pre.to_dense().tolist()
                all_predictions[idx] = pre
        return all_predictions

    @abc.abstractmethod
    def transform_predictions(self, batch: Batch) -> Batch:
        """
        Post-process the predictions of the torch model (e.g., apply
        threshold in case of multi-label classifier).
        """


class MulticlassTorchPredictor(TorchPredictor):
    """
    Applies a pre-trained torch model to unseen data for
    multiclass classification.
    """

    def transform_predictions(self, batch: Batch) -> Batch:
        batch = batch.argmax(-1)
        return batch


class MultilabelTorchPredictor(TorchPredictor):
    """
    Applies a pre-trained torch model to unseen data for
    multilabel classification, applying a threshold on the
    output of the neural network.
    """

    #: Threshold to transform probabilities into class predictions.
    #: Defaults to 0.5.
    threshold: float = 0.5

    def __init__(
        self,
        model: Union[nn.Module, ModelLoader],
        test_dataloader_class: str = 'torch.utils.data.DataLoader',
        test_dataloader_kwargs: Optional[Dict] = None,
        threshold: float = 0.5,
        name: str = None
    ) -> None:
        super().__init__(
            model, test_dataloader_class, test_dataloader_kwargs, name
        )
        self.threshold = threshold

    def transform_predictions(self, batch: Batch) -> Batch:
        return (batch > self.threshold).float()


class RegressionTorchPredictor(TorchPredictor):
    """
    Applies a pre-trained torch model to unseen data for
    regression, leaving untouched the output of the neural
    network.
    """

    def transform_predictions(self, batch: Batch) -> Batch:
        return batch
