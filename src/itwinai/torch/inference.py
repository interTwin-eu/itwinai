from typing import Optional, Tuple, Dict, Any, List, Union
import os

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..utils import dynamically_import_class
from .utils import clear_key
from ..components import Predictor
from .types import TorchDistributedStrategy as StrategyT
from .types import Metric
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
            'Unrecognized model URI: model may not be there!'
        )


class TorchPredictor(Predictor):
    """Applies a pre-trained torch model to unseen data.

    Args:
        model (nn.Module): neural network instance.
        test_dataloader_class (str, optional): test dataloader class path.
            Defaults to 'torch.utils.data.DataLoader'.
        test_dataloader_kwargs (Optional[Dict], optional): constructor
            arguments of the test dataloader, except for the dataset
            instance. Defaults to None.
        strategy (Optional[TorchDistributedStrategy], optional): distributed
            strategy. Defaults to StrategyT.NONE.value.
        backend (TorchDistributedBackend, optional): computing backend.
            Defaults to BackendT.NCCL.value.
        shuffle_dataset (bool, optional): whether shuffle dataset before
            sampling batches from dataloader. Defaults to False.
        use_cuda (bool, optional): whether to use GPU. Defaults to True.
        benchrun (bool, optional): sets up a debug run. Defaults to False.
        testrun (bool, optional): deterministic training seeding everything.
            Defaults to False.
        seed (Optional[int], optional): random seed. Defaults to None.
        logger (Optional[List[Logger]], optional): logger. Defaults to None.
        checkpoint_every (int, optional): how often (epochs) to checkpoint the
            best model. Defaults to 10.
        cluster (Optional[ClusterEnvironment], optional): cluster environment
            object describing the context in which the trainer is executed.
            Defaults to None.
        train_metrics (Optional[Dict[str, Metric]], optional):
            list of metrics computed in the training step on the predictions.
            It's a dictionary with the form
            ``{'metric_unique_name': CallableMetric}``. Defaults to None.
        validation_metrics (Optional[Dict[str, Metric]], optional): same
            as ``training_metrics``. If not given, it mirrors the training
            metrics. Defaults to None.

    Raises:
        RuntimeError: When trying to use DDP without CUDA support.
        NotImplementedError: when trying to use a strategy different from the
            ones provided by TorchDistributedStrategy.
    """

    model: nn.Module = None
    test_dataset: Dataset
    test_dataloader: DataLoader = None
    _strategy: StrategyT = StrategyT.NONE.value
    epoch_idx: int = 0
    train_glob_step: int = 0
    validation_glob_step: int = 0
    train_metrics: Dict[str, Metric]
    validation_metrics: Dict[str, Metric]

    def __init__(
        self,
        model: Union[nn.Module, ModelLoader],
        test_dataloader_class: str = 'torch.utils.data.DataLoader',
        test_dataloader_kwargs: Optional[Dict] = None,
        # strategy: str = StrategyT.NONE.value,
        # seed: Optional[int] = None,
        # logger: Optional[List[Logger]] = None,
        # cluster: Optional[ClusterEnvironment] = None,
        # test_metrics: Optional[Dict[str, Metric]] = None,
    ) -> None:
        super().__init__()
        self.model = model() if isinstance(model, ModelLoader) else model
        # self.seed = seed
        # self.strategy = strategy
        # self.cluster = cluster

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

        # # Loggers
        # self.logger = logger if logger is not None else ConsoleLogger()

        # # Metrics
        # self.train_metrics = (
        #     {} if train_metrics is None else train_metrics
        # )
        # self.validation_metrics = (
        #     self.train_metrics if validation_metrics is None
        #     else validation_metrics
        # )

    def execute(
        self,
        test_dataset: Dataset,
        model: nn.Module = None,
        config: Optional[Dict] = None
    ) -> Tuple[Optional[Tuple], Optional[Dict]]:
        self.test_dataset = test_dataset
        self.test_dataloader = self.test_dataloader_class(
            test_dataset, **self.test_dataloader_kwargs
        )
        # Update model passed for "interactive" use
        if model is not None:
            self.model = model
        result = self.predict()
        return ((result,), config)

    def predict(self) -> List[Any]:
        """Returns a list of predictions."""
        # TODO: complete

        return []
