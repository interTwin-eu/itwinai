from typing import Optional, Tuple, Dict, Any, List

from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..utils import dynamically_import_class
from .utils import clear_key
from ..components import Predictor
from .types import TorchDistributedStrategy as StrategyT
from .types import Metric


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
        model: nn.Module,
        test_dataloader_class: str = 'torch.utils.data.DataLoader',
        test_dataloader_kwargs: Optional[Dict] = None,
        # strategy: str = StrategyT.NONE.value,
        # seed: Optional[int] = None,
        # logger: Optional[List[Logger]] = None,
        # cluster: Optional[ClusterEnvironment] = None,
        # test_metrics: Optional[Dict[str, Metric]] = None,
    ) -> None:
        super().__init__()
        self.model = model
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

        return []
