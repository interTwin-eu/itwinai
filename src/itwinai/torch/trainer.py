"""Provides training logic for PyTorch models via Trainer classes."""

from typing import (
    Optional, Dict, Union, Tuple, Type, List, Any, Literal
)
import time
import os
import sys

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
from torch.optim.optimizer import Optimizer

import horovod.torch as hvd

from ..components import Trainer, monitor_exec
from .utils import par_allgather_obj, clear_key
from .types import (
    Batch, Loss, LrScheduler, Metric
)
from .types import TorchDistributedStrategy as StrategyT
from ..loggers import LogMixin, Logger, ConsoleLogger
from ..utils import dynamically_import_class
from ..cluster import ClusterEnvironment
from .reproducibility import seed_worker, set_seed
from .distributed import (
    TorchDistributedStrategy,
    TorchDDPStrategy,
    HorovodStrategy,
    DeepSpeedStrategy,
    NonDistributedStrategy,
    distributed_resources_available
)


class TorchTrainer(Trainer):
    """Trainer class for torch training algorithms.

    Args:
        config (Dict): training configuration containing hyperparameters.
        epochs (int): number of training epochs.
        model (Optional[nn.Module], optional): model to train.
        Defaults to None.
        strategy (Literal[&quot;ddp&quot;, &quot;deepspeed&quot;,
        &quot;horovod&quot;], optional): distributed strategy.
        Defaults to 'ddp'.
        validation_every (Optional[int], optional): run a validation epoch
        every ``validation_every`` epochs. Disabled if None. Defaults to 1.
        test_every (Optional[int], optional): run a test epoch
        every ``test_every`` epochs. Disabled if None. Defaults to None.
        random_seed (Optional[int], optional): set random seed for
        reproducibility. If None, the seed is not set. Defaults to None.
        logger (Optional[Logger], optional): logger for ML tracking.
        Defaults to None.
        name (Optional[str], optional): trainer custom name. Defaults to None.
    """

    _strategy: TorchDistributedStrategy = None

    train_dataloader: DataLoader = None
    validation_dataloader: DataLoader = None
    test_dataloader: DataLoader = None

    model: nn.Module = None
    loss: Loss = None
    optimizer: Optimizer = None
    lr_scheduler: LrScheduler = None

    torch_rng: torch.Generator = None
    logger: Logger = None

    def __init__(
        self,
        config: Dict,
        epochs: int,
        model: Optional[nn.Module] = None,
        strategy: Literal["ddp", "deepspeed", "horovod"] = 'ddp',
        validation_every: Optional[int] = 1,
        test_every: Optional[int] = None,
        random_seed: Optional[int] = None,
        logger: Optional[Logger] = None,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name)
        self.save_parameters(**self.locals2params(locals()))

        # config is mean to store all hyperparameters, which can very from use
        # case to use case
        # and include learning_rate, batch_size....
        self.config = config
        self.epochs = epochs
        self.model = model
        self.strategy = strategy
        self.validation_every = validation_every
        self.test_every = test_every
        self.random_seed = random_seed
        self.logger = logger

    @property
    def strategy(self) -> TorchDistributedStrategy:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: Union[str, TorchDistributedStrategy]) -> None:
        if isinstance(strategy, TorchDistributedStrategy):
            self._strategy = strategy
        else:
            self._strategy = self._detect_strategy(strategy)

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

    def create_model_loss_optimizer(self) -> None:
        """
        Instantiate a torch model, loss, optimizer, and LR scheduler using the
        configuration provided in the Trainer constructor.
        Generally a user-define method.
        """
        ###################################
        # Dear user, this is a method you #
        # may be interested to override!  #
        ###################################

        if self.model is None:
            # Model was not passed to the constructor.
            # Create a model here
            raise ValueError(
                "self.model is None! Either pass it to the constructor or "
                "override this method."
            )

        # A simple NLLLoss
        self.loss = nn.functional.nll_loss

        # TODO: improve robustness of getting from config
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config.lr,
            momentum=self.config.momentum
        )
        # Create self.lr_scheduler if needed

        # IMPORTANT: model, optimizer, and scheduler need to be distributed

        # First, define strategy-wise optional configurations
        # TODO: improve robustness of getting from config
        if isinstance(self.strategy, DeepSpeedStrategy):
            # Batch size definition is not optional for DeepSpeedStrategy!
            distribute_kwargs = dict(
                config_params=dict(
                    train_micro_batch_size_per_gpu=self.config.batch_size
                )
            )
        elif isinstance(self.strategy, HorovodStrategy):
            distribute_kwargs = dict(
                compression=(
                    hvd.Compression.fp16 if self.config.fp16_allreduce
                    else hvd.Compression.none
                ),
                op=hvd.Adasum if self.config.use_adasum else hvd.Average,
                gradient_predivide_factor=self.config.gradient_predivide_factor
            )
        else:
            distribute_kwargs = {}

        # Distributed model, optimizer, and scheduler
        (
            self.model,
            self.optimizer,
            self.lr_scheduler
        ) = self.strategy.distributed(
            self.model, self.optimizer, self.lr_scheduler, **distribute_kwargs
        )

    def create_dataloaders(
        self,
        train_dataset: Dataset,
        validation_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None
    ) -> None:
        """
        Create train, validation and test dataloaders using the
        configuration provided in the Trainer constructor.
        Generally a user-define method.

        Args:
            train_dataset (Dataset): training dataset object.
            validation_dataset (Optional[Dataset]): validation dataset object.
            Default None.
            test_dataset (Optional[Dataset]): test dataset object.
            Default None.
        """

        ###################################
        # Dear user, this is a method you #
        # may be interested to override!  #
        ###################################

        # TODO: improve robustness of getting from config
        self.train_dataloader = self.strategy.create_dataloader(
            dataset=train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            generator=self.torch_rng
        )
        if validation_dataset is not None:
            self.validation_dataloader = self.strategy.create_dataloader(
                dataset=train_dataset,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                generator=self.torch_rng
            )
        if test_dataset is not None:
            self.test_dataloader = self.strategy.create_dataloader(
                dataset=train_dataset,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                generator=self.torch_rng
            )

    @monitor_exec
    def execute(
        self,
        train_dataset: Dataset,
        validation_dataset: Dataset,
        test_dataset: Dataset
    ) -> Tuple[Dataset, Dataset, Dataset, Any]:
        """Trains a machine learning model.

        Args:
            train_dataset (Dataset): training dataset.
            validation_dataset (Dataset): validation dataset.
            test_dataset (Dataset): test dataset.

        Returns:
            Tuple[Dataset, Dataset, Dataset, Any]: training dataset,
            validation dataset, test dataset, trained model.
        """
        self.torch_rng = set_seed(self.random_seed)
        self._init_distributed_strategy()
        self.create_dataloaders(
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            test_dataset=test_dataset
        )
        self.create_model_loss_optimizer()
        self.train()
        return train_dataset, validation_dataset, test_dataset, self.model

    def _set_epoch_dataloaders(self, epoch: int):
        """
        Sets epoch in the distributed sampler of a dataloader when using it.
        """
        if self.strategy.is_distributed:
            self.train_dataloader.sampler.set_epoch(epoch)
            if self.validation_dataloader is not None:
                self.validation_dataloader.sampler.set_epoch(epoch)
            if self.test_dataloader is not None:
                self.test_dataloader.sampler.set_epoch(epoch)

    def train(self):
        """Main training logic (training loop)."""
        # start_time = time.perf_counter()
        for epoch in range(self.epochs):
            self._set_epoch_dataloaders(epoch)
            self.train_epoch(epoch)
            if self.validation_every and self.validation_every % epoch == 0:
                self.validation_epoch(epoch)
            if self.test_every and self.test_every % epoch == 0:
                self.test_epoch(epoch)

    def train_epoch(self, epoch: int):
        pass

    def validation_epoch(self, epoch: int):
        pass

    def test_epoch(self, epoch: int):
        pass

    def save_state(self):
        return super().save_state()

    def load_state(self):
        return super().load_state()


def preproc_dataloader(dataloader: DataLoader, gwsize, grank):
    """Makes a Dataloader distributed."""
    sampler = DistributedSampler(
        dataloader.dataset,
        num_replicas=gwsize,
        rank=grank,
        shuffle=True
    )
    # Recreate dataloader, with updated sampler
    return DataLoader(
        dataloader.dataset,
        batch_size=dataloader.batch_size,
        sampler=sampler,
        num_workers=dataloader.num_workers,
        collate_fn=dataloader.collate_fn,
        pin_memory=dataloader.pin_memory,
        drop_last=dataloader.drop_last,
        timeout=dataloader.timeout,
        worker_init_fn=seed_worker,  # dataloader.worker_init_fn,
        multiprocessing_context=dataloader.multiprocessing_context,
        generator=dataloader.generator,
        prefetch_factor=dataloader.prefetch_factor,
        persistent_workers=dataloader.persistent_workers,
        pin_memory_device=dataloader.pin_memory_device
    )


def distributed(func):
    """The decorated function must have a standard signature.
    Its first arguments must be:
    model, train_dataloader, validation_dataloader, device (in this order).

    Additional args or kwargs are allowed consistently with the signature
    of the decorated function.
    """
    def dist_train(
            model, train_dataloader, validation_dataloader=None, device='cpu',
            *args, **kwargs
    ):
        if torch.cuda.is_available():
            dist.init_process_group(backend='nccl')

        if torch.cuda.is_available():
            lwsize = torch.cuda.device_count()  # local world size - per node
            gwsize = dist.get_world_size()     # global world size - per run
            grank = dist.get_rank()            # global rank - assign per run
            lrank = dist.get_rank() % lwsize   # local rank - assign per node
        else:
            gwsize = 1
            grank = 0
            lrank = 0

        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu', lrank)
        if torch.cuda.is_available():
            torch.cuda.set_device(lrank)

        model = model.to(device)
        model = DDP(model, device_ids=[device], output_device=device)

        train_dataloader = preproc_dataloader(train_dataloader, gwsize, grank)
        if validation_dataloader is not None:
            validation_dataloader = preproc_dataloader(
                validation_dataloader, gwsize, grank)

        try:
            func(model, train_dataloader, validation_dataloader, device,
                 *args, **kwargs)
        finally:
            if torch.cuda.is_available():
                dist.barrier()
                dist.destroy_process_group()
    return dist_train


class TorchTrainerMG(Trainer, LogMixin):
    """
    Torch trainer for optionally distributed data-parallel (DDP) workload.
    Multi-GPU distribution.

    Args:
        model (nn.Module): neural network instance.
        loss (Loss): torch loss function instance.
        optimizer_class (str): path to optimizer class
            (e.g., 'torch.optim.SGD')
        optimizer_kwargs (Optional[Dict], optional): optimizer constructor
            arguments (except from parameters). Defaults to None.
        lr_scheduler_class (Optional[str], optional): path to learning
            rate scheduler class. Defaults to None.
        lr_scheduler_kwargs (Optional[Dict], optional): constructor arguments
            of the learning rate scheduler, except for the optimizer.
            Defaults to None.
        train_dataloader_class (str, optional): train dataloader class path.
            Defaults to 'torch.utils.data.DataLoader'.
        train_dataloader_kwargs (Optional[Dict], optional): constructor
            arguments of the train dataloader, except for the dataset
            instance. Defaults to None.
        validation_dataloader_class (str, optional): validation dataloader
            class path. Defaults to 'torch.utils.data.DataLoader'.
        validation_dataloader_kwargs (Optional[Dict], optional): constructor
            arguments of the validation dataloader, except for the dataset
            instance. If None, it replicates `train_dataloader_kwargs`.
            Defaults to None.
        epochs (int, optional): number of training epochs. Defaults to 1.
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
    loss: Loss = None
    optimizer: Optimizer = None
    lr_scheduler = None
    _strategy: StrategyT = StrategyT.NONE.value
    train_dataset: Dataset
    validation_dataset: Dataset
    train_dataloader: DataLoader = None
    validation_dataloader: DataLoader = None
    epoch_idx: int = 0
    train_glob_step: int = 0
    validation_glob_step: int = 0
    train_metrics: Dict[str, Metric]
    validation_metrics: Dict[str, Metric]

    def __init__(
        self,
        model: nn.Module,
        loss: Loss,
        optimizer_class: str,
        optimizer_kwargs: Optional[Dict] = None,
        lr_scheduler_class: Optional[str] = None,
        lr_scheduler_kwargs: Optional[Dict] = None,
        train_dataloader_class: str = 'torch.utils.data.DataLoader',
        train_dataloader_kwargs: Optional[Dict] = None,
        validation_dataloader_class: str = 'torch.utils.data.DataLoader',
        validation_dataloader_kwargs: Optional[Dict] = None,
        epochs: int = 1,
        strategy: str = StrategyT.NONE.value,
        benchrun: bool = False,
        testrun: bool = False,
        seed: Optional[int] = None,
        logger: Optional[List[Logger]] = None,
        checkpoint_every: int = 10,
        cluster: Optional[ClusterEnvironment] = None,
        train_metrics: Optional[Dict[str, Metric]] = None,
        validation_metrics: Optional[Dict[str, Metric]] = None
    ) -> None:
        """Sets up the distributed backend and loggers.
        Makes the model a DDP model.
        """
        super().__init__()
        self.save_parameters(**self.locals2params(locals()))
        self.model = model
        self.loss = loss
        self.epochs = epochs
        self.testrun = testrun
        self.seed = seed
        self.strategy = strategy
        self.benchrun = benchrun
        self.cluster = cluster
        # Checkpoint every n epochs
        self.checkpoint_every = checkpoint_every

        # Train and validation dataloaders
        self.train_dataloader_class = dynamically_import_class(
            train_dataloader_class
        )
        self.validation_dataloader_class = dynamically_import_class(
            validation_dataloader_class
        )
        train_dataloader_kwargs = (
            train_dataloader_kwargs
            if train_dataloader_kwargs is not None else {}
        )
        self.train_dataloader_kwargs = clear_key(
            train_dataloader_kwargs, 'train_dataloader_kwargs', 'dataset'
        )
        # If validation_dataloader_kwargs is not given,
        # copy train_dataloader_kwargs
        validation_dataloader_kwargs = (
            validation_dataloader_kwargs if validation_dataloader_kwargs
            is not None else train_dataloader_kwargs
        )
        self.validation_dataloader_kwargs = clear_key(
            validation_dataloader_kwargs, 'validation_dataloader_kwargs',
            'dataset'
        )

        # Optimizer and scheduler
        optim_class = dynamically_import_class(optimizer_class)
        optimizer_kwargs = (
            optimizer_kwargs if optimizer_kwargs is not None else {}
        )
        optimizer_kwargs = clear_key(
            optimizer_kwargs, 'optimizer_kwargs', 'parameters'
        )
        self.optimizer: Optimizer = optim_class(
            self.model.parameters(), **optimizer_kwargs
        )
        if lr_scheduler_class is not None:
            scheduler_class = dynamically_import_class(lr_scheduler_class)
            lr_scheduler_kwargs = (
                lr_scheduler_kwargs if lr_scheduler_kwargs is not None else {}
            )
            lr_scheduler_kwargs = clear_key(
                lr_scheduler_kwargs, 'lr_scheduler_kwargs', 'optimizer'
            )
            self.lr_scheduler: LrScheduler = scheduler_class(
                self.optimizer, **lr_scheduler_kwargs
            )

        # Loggers
        self.logger = logger if logger is not None else ConsoleLogger()

        # Metrics
        self.train_metrics = (
            {} if train_metrics is None else train_metrics
        )
        self.validation_metrics = (
            self.train_metrics if validation_metrics is None
            else validation_metrics
        )

    @property
    def strategy(self) -> Optional[str]:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy_name) -> None:
        if strategy_name not in StrategyT:
            raise ValueError(
                "Unrecognized 'strategy' field. Allowed values "
                f"are: {StrategyT.list()}. Received '{strategy_name}'")
        self._strategy = strategy_name

    @property
    def global_step(self) -> int:
        return self.train_glob_step + self.validation_glob_step

    def set_seed(self, seed: Optional[int] = None):
        """Deterministic operations for reproducibility.
        Sets the random seed.

        Args:
            seed (Optional[int], optional): if not None, overrides
                `self.seed`. Defaults to None.
        """
        seed = seed if seed is not None else self.seed
        np.random.seed(seed)
        self.torch_rng = torch.Generator()
        if seed is not None:
            torch.manual_seed(seed)
            self.torch_rng.manual_seed(seed)
            if self.cluster.is_cuda_available():
                torch.cuda.manual_seed(seed)

    @monitor_exec
    def execute(
        self,
        train_dataset: Dataset,
        validation_dataset: Dataset,
        model: nn.Module = None,
        optimizer: Optimizer = None,
        lr_scheduler: LrScheduler = None,
    ) -> Any:
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset

        # Update parameters passed for "interactive" use
        if model is not None:
            self.model = model
        if optimizer is not None:
            self.optimizer = optimizer
        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler

        # Start training
        if self.cluster.distributed:
            # Make training distributed
            result = mp.spawn(self._train, nprocs=self.cluster.ngpus_per_node)
        else:
            result = self._train(0)

        # Return value compliant with Executable.execute format
        return result

    def _train(
        self,
        worker_id: int
    ):
        # Each worker has a different deterministic seed
        # Here, 'worker' = replica of the training function
        worker_seed = (
            self.seed + worker_id if self.seed is not None else self.seed
        )
        self.set_seed(worker_seed)

        # Instantiate dataloaders
        self.train_dataloader = self._instantiate_dataloader(
            dataloader_class=self.train_dataloader_class,
            dataset=self.train_dataset,
            init_kwargs=self.train_dataloader_kwargs
        )
        if self.validation_dataset is not None:
            self.validation_dataloader = self._instantiate_dataloader(
                dataloader_class=self.validation_dataloader_class,
                dataset=self.validation_dataset,
                init_kwargs=self.validation_dataloader_kwargs
            )

        # Launch actual training:

        # Single worker case
        if not self.cluster.distributed:
            with self.cluster.init_dist_gpu(worker_id) as device:
                self.device: torch.device = device
                self.model = self.model.to(self.device)
                self.setup_logger()
                self._setup_metrics()
                try:
                    train_result = self.train()
                except Exception as exc:
                    print(exc)
                    raise exc
                finally:
                    print("INFO: Training ended")
                    self.destroy_logger()
                    train_result = None
                return train_result

        # Init / connect to distributed backend
        with self.cluster.init_dist_gpu(worker_id) as device:
            self.device: torch.device = device
            self._distribute_model()
            self.setup_logger()
            self._setup_metrics()
            try:
                train_result = self.train()
            except Exception as exc:
                print(exc)
                raise exc
            finally:
                print("INFO: Training ended")
                self.destroy_logger()
                train_result = None
        return train_result

    def _instantiate_dataloader(
        self,
        dataloader_class: Type,
        dataset: Dataset,
        init_kwargs: Dict
    ) -> DataLoader:
        """Make dataloader distributed if using distributed training strategy.

        Args:
            dataloader_class (Type): some torch DataLoader type.
            dataset (Dataset): torch dataset instance.
            init_kwargs (Dict): constructor args.
        """
        init_kwargs['generator'] = init_kwargs.get(
            'generator', self.torch_rng
        )
        init_kwargs['worker_init_fn'] = init_kwargs.get(
            'worker_init_fn', seed_worker
        )

        if self.strategy == StrategyT.DDP.value and self.cluster.distributed:
            sampler = DistributedSampler(
                dataset=dataset,
                num_replicas=self.cluster.global_world_size,
                rank=self.cluster.global_rank,
                shuffle=init_kwargs.get(
                    'shuffle', False
                )
            )
            # Overwrite existing sampler, if given.
            # TODO: improve using wrapper:
            # https://discuss.pytorch.org/t/how-to-use-my-own-sampler-when-i-already-use-distributedsampler/62143?page=2
            init_kwargs['sampler'] = sampler
            if init_kwargs.get('shuffle') is not None:
                # sampler option is mutually exclusive with shuffle
                del init_kwargs['shuffle']

        return dataloader_class(dataset, **init_kwargs)

    def _setup_metrics(self):
        for m_name, metric in self.train_metrics.items():
            self.train_metrics[m_name] = metric.to(self.device)
        for m_name, metric in self.validation_metrics.items():
            self.validation_metrics[m_name] = metric.to(self.device)

    def _distribute_model(self):
        if self.cluster.distributed:
            # Distribute model
            self.model = self.model.to(self.device)
            if self.strategy == StrategyT.NONE.value:
                print(
                    "WARNING: A GPU cluster is available but no distributed "
                    "strategy was given... Falling back to single worker...")
                if not self.cluster.is_main_worker():
                    # Use only GPU:0 for single worker
                    sys.exit(0)
            elif self.strategy == StrategyT.DDP.value:
                self.model = DDP(
                    self.model,
                    device_ids=[self.device.index],
                    output_device=self.device
                )
            else:
                raise NotImplementedError("Only DDP strategy is implemented.")
        else:
            raise RuntimeError(
                "Trying to distribute a model when a "
                "distributed cluster is not available."
            )

    def setup_logger(self):
        if self.cluster.is_main_worker():
            # Only setup loggers on main worker
            if isinstance(self.logger, list):
                for logger in self.logger:
                    logger.create_logger_context()
            elif isinstance(self.logger, Logger):
                self.logger.create_logger_context()
            else:
                raise TypeError(
                    "Unrecognized self.logger. Allowed types are 'list' and "
                    f"'Logger'. Received {type(self.logger)}"
                )
        else:
            self.logger = []

    def destroy_logger(self):
        if self.cluster.is_main_worker():
            if isinstance(self.logger, list):
                for logger in self.logger:
                    logger.destroy_logger_context()
            elif isinstance(self.logger, Logger):
                self.logger.destroy_logger_context()
            else:
                raise TypeError(
                    "Unrecognized self.logger. Allowed types are 'list' and "
                    f"'Logger'. Received {type(self.logger)}"
                )

    def log(
        self,
        item: Union[Any, List[Any]],
        identifier: Union[str, List[str]],
        kind: str = 'metric',
        step: Optional[int] = None,
        batch_idx: Optional[int] = None,
        every_worker: bool = False,
        **kwargs
    ) -> None:
        if self.cluster.is_main_worker() or every_worker:
            # Only log on main worker if not specified otherwise
            if isinstance(self.logger, list):
                for logger in self.logger:
                    logger.log(
                        item=item,
                        identifier=identifier,
                        kind=kind,
                        step=step,
                        batch_idx=batch_idx,
                        **kwargs
                    )
            elif isinstance(self.logger, Logger):
                self.logger.log(
                    item=item,
                    identifier=identifier,
                    kind=kind,
                    step=step,
                    batch_idx=batch_idx,
                    **kwargs
                )
            else:
                raise TypeError(
                    "Unrecognized self.logger. Allowed types are 'list' and "
                    f"'Logger'. Received {type(self.logger)}"
                )

    def compute_metrics(
        self,
        metrics: Dict[str, Metric],
        true: Batch,
        pred: Batch,
        logger_step: int,
        batch_idx: Optional[int],
        stage: str = 'train'
    ) -> Dict[str, Any]:
        """Compute and log metrics.

        Args:
            metrics (Dict[str, Metric]): metrics dict. Can be
                ``self.train_metrics`` or ``self.validation_metrics``.
            true (Batch): true values.
            pred (Batch): predicted values.
            logger_step (int): global step to pass to the logger.
            stage (str): 'train', 'validation'...

        Returns:
            Dict[str, Any]: metric values.
        """
        m_values = {}
        for m_name, metric in metrics.items():
            # metric = metric.to(self.device)
            m_val = metric(pred, true).detach().cpu().numpy()
            self.log(
                item=m_val,
                identifier=f'{m_name}_{stage}',
                kind='metric',
                step=logger_step,
                batch_idx=batch_idx
            )
            m_values[m_name] = m_val
        return m_values

    def training_step(
        self,
        batch: Batch,
        batch_idx: int
    ) -> Tuple[Loss, Dict[str, Any]]:
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        pred_y = self.model(x)
        loss: Loss = self.loss(pred_y, y)
        self.log(
            item=loss.item(),
            identifier='training_loss',
            kind='metric',
            step=self.train_glob_step,
            batch_idx=batch_idx
        )
        metrics: Dict[str, Any] = self.compute_metrics(
            metrics=self.train_metrics,
            true=y,
            pred=pred_y,
            logger_step=self.train_glob_step,
            batch_idx=batch_idx,
            stage='training'
        )
        return loss, metrics

    def validation_step(
        self,
        batch: Batch,
        batch_idx: int
    ) -> Tuple[Loss, Dict[str, Any]]:
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        pred_y = self.model(x)
        loss: Loss = self.loss(pred_y, y)
        self.log(
            item=loss.item(),
            identifier='validation_loss',
            kind='metric',
            step=self.validation_glob_step,
            batch_idx=batch_idx
        )
        metrics: Dict[str, Any] = self.compute_metrics(
            metrics=self.validation_metrics,
            true=y,
            pred=pred_y,
            logger_step=self.validation_glob_step,
            batch_idx=batch_idx,
            stage='validation'
        )
        return loss, metrics

    def training_epoch(self) -> Loss:
        self.model.train()
        train_losses = []
        for batch_idx, train_batch in enumerate(self.train_dataloader):
            loss, metrics = self.training_step(
                batch=train_batch,
                batch_idx=batch_idx
            )
            # TODO: merge and log batch metrics and loss into epoch metrics
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_losses.append(loss)
            # Important: update counter
            self.train_glob_step += 1

        # Aggregate and log losses
        avg_loss = torch.mean(torch.stack(train_losses)).detach().cpu()
        self.log(
            item=avg_loss.item(),
            identifier='training_loss_epoch',
            kind='metric',
            step=self.train_glob_step,
        )
        return avg_loss

    def validation_epoch(self) -> Loss:
        if self.validation_dataloader is not None:
            self.model.eval()
            validation_losses = []
            for batch_idx, val_batch \
                    in enumerate(self.validation_dataloader):
                # TODO: merge and log batch metrics and loss into epoch metrics
                loss, metrics = self.validation_step(
                    batch=val_batch,
                    batch_idx=batch_idx
                )
                validation_losses.append(loss)
                # Important: update counter
                self.validation_glob_step += 1

            # Aggregate and log losses
            avg_loss = torch.mean(
                torch.stack(validation_losses)
            ).detach().cpu()
            self.log(
                item=avg_loss.item(),
                identifier='validation_loss_epoch',
                kind='metric',
                step=self.validation_glob_step,
            )
            return avg_loss

    def train(self):

        if self.optimizer is None:
            raise ValueError("Undefined optimizer!")

        if self.loss is None:
            raise ValueError("Undefined loss function!")

        st = time.time()

        # Resume state
        self.start_epoch = 1
        self.best_loss = np.Inf
        self.load_state()

        # start training/testing loop
        if self.cluster.is_main_worker():
            print(f'TIMER: broadcast: {time.time()-st}s')
            print('DEBUG: start training')
            print('-'*56)

        ##############################
        # Start training: run epochs #
        ##############################

        et = time.time()
        for self.epoch_idx in range(self.start_epoch, self.epochs + 1):
            lt = time.time()

            #######################################################
            # Perform one training epoch and one validation epoch #
            #######################################################

            if self.benchrun and self.epoch_idx == self.epochs:
                # TODO: move profiler into cluster environment
                # profiling (done on last epoch - slower!)
                with torch.autograd.profiler.profile(
                    use_cuda=self.cluster.is_cuda_available(),
                    profile_memory=True
                ) as prof:
                    train_loss = self.training_epoch()
            else:
                train_loss = self.training_epoch()
            val_loss = self.validation_epoch()

            #####################################
            # Save checkpoint if model improved #
            #####################################

            ref_loss = val_loss if val_loss is not None else train_loss
            is_best = ref_loss < self.best_loss
            if (self.epoch_idx % self.checkpoint_every == 0
                    and not self.benchrun):
                self.save_state(
                    loss_val=ref_loss,
                    is_best=is_best
                )
                self.best_loss = min(ref_loss, self.best_loss)

            ###########################
            # End of epoch operations #
            ###########################

            # save first epoch timer
            if self.epoch_idx == self.start_epoch:
                first_ep_t = time.time()-lt

            # Final epoch
            if self.epoch_idx + 1 == self.epochs:
                self.train_dataloader.last_epoch = True
                self.validation_dataloader.last_epoch = True

            if self.cluster.is_main_worker():
                print(f'TIMER: epoch time: {time.time()-lt}s')
                if self.benchrun and self.epoch_idx == self.epochs:
                    print('-'*56)
                    print('benchmark of last epoch:')
                    what1 = (
                        'cuda' if self.cluster.is_cuda_available() else 'cpu'
                    )
                    print(
                        prof.key_averages().table(
                            sort_by='self_'+str(what1)+'_time_total'
                        )
                    )

        ##########################
        # Training has completed #
        ##########################

        # save final state
        if not self.benchrun:
            self.save_state(
                loss_val=ref_loss,
                is_best=is_best
            )
        if self.cluster.is_cuda_available() and self.cluster.distributed:
            dist.barrier()

        ########################
        # Print training stats #
        ########################

        if self.cluster.is_main_worker():
            print('-'*56)
            print('training results:')
            print(f'TIMER: first epoch time: {first_ep_t}s')
            print(f'TIMER: last epoch time: {time.time()-lt}s')
            print(
                f'TIMER: average epoch time: {(time.time()-et)/self.epochs}s')
            print(f'TIMER: total epoch time: {time.time()-et}s')
            if self.epoch_idx > 1:
                print(
                    f'TIMER: total epoch-1 time: {time.time()-et-first_ep_t}s'
                )
                print(
                    'TIMER: average epoch-1 time: '
                    f'{(time.time()-et-first_ep_t)/(self.epochs-1)}s')
            if self.benchrun:
                print(
                    f'TIMER: total epoch-2 time: {lt-first_ep_t}s')
                print('TIMER: average epoch-2 time: '
                      f'{(lt-first_ep_t)/(self.epochs-2)}s')
            mem = int(torch.cuda.memory_reserved(
                self.cluster.local_rank)/1024/1024)
            print(
                f'memory req: {mem} MB'
                if self.cluster.is_cuda_available()
                and self.cluster.distributed else 'memory req: - MB'
            )
            if self.cluster.is_cuda_available():
                print(
                    f'memory summary:\n {torch.cuda.memory_summary(0)}')

        if self.cluster.is_main_worker():
            print(f'TIMER: final time: {time.time()-st} s')

    def save_state(self, loss_val: Any, is_best: bool):
        """Save training state."""
        res_name = 'checkpoint.pth.tar'
        rt = time.time()

        if (self.cluster.is_cuda_available() and self.cluster.distributed):
            # find if is_best happened in any worker
            is_best_m = par_allgather_obj(
                is_best, self.cluster.global_world_size
            )
            if any(is_best_m):
                # TODO: is this strategy really good? Checkpointing when
                # at least one worker improves the loss on their local
                # data split is prone to overfitting, especially when
                # the dataset in unbalanced!

                # find which rank is_best happened - select first rank
                # if multiple
                best_rank = np.where(np.array(is_best_m))[0][0]
                if self.cluster.global_rank == best_rank:
                    self._save_sate(
                        epoch=self.epoch_idx+1,
                        loss_val=loss_val,
                        save_path=res_name
                    )
                    print(
                        f'DEBUG: state in {self.cluster.global_rank} is '
                        f'saved on epoch:{self.epoch_idx} '
                        f'in {time.time()-rt} s')
        else:
            self._save_sate(
                epoch=self.epoch_idx+1,
                loss_val=loss_val,
                save_path=res_name
            )
            print(
                f'DEBUG: state in {self.cluster.global_rank} '
                f'is saved on epoch:{self.epoch_idx} in {time.time()-rt} s')

    def _save_sate(
        self,
        epoch: int,
        loss_val: Any,
        save_path: str
    ):
        """Save state on disk."""
        sched = (
            self.lr_scheduler.state_dict()
            if self.lr_scheduler is not None else None
        )
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'best_loss': loss_val,
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': sched
        }
        self.log(
            item=state,
            identifier=save_path,
            kind='torch',
            epoch_step=self.epoch_idx,
            batch_step=0
        )

    def load_state(self):
        """Load training state."""
        res_name = 'checkpoint.pth.tar'
        if os.path.isfile(res_name) and not self.benchrun:
            try:
                if (self.cluster.is_cuda_available()
                        and self.cluster.distributed):
                    dist.barrier()
                    # Map model to be loaded to specified single gpu.
                    # loc = (
                #     {'cuda:%d' % 0: 'cuda:%d' % self.cluster.local_rank}
                #     if self.cluster.is_cuda_available()
                #     else {'cpu:%d' % 0: 'cpu:%d' % self.cluster.local_rank}
                    # )
                    # checkpoint = torch.load(res_name, map_location=loc)
                    checkpoint = torch.load(
                        res_name, map_location=self.device
                    )
                else:
                    checkpoint = torch.load(res_name, map_location='cpu')
                self.start_epoch = checkpoint['epoch']
                self.best_loss = checkpoint['best_loss']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                if self.lr_scheduler is not None:
                    self.lr_scheduler.load_state_dict(
                        checkpoint['lr_scheduler']
                    )
                if self.cluster.is_cuda_available():
                    if self.cluster.is_main_worker():
                        print(
                            f'WARNING: restarting from {self.start_epoch} '
                            'epoch')
                else:
                    print(
                        f'WARNING: restarting from {self.start_epoch} epoch')
            except Exception:
                if self.cluster.is_cuda_available():
                    if self.cluster.is_main_worker():
                        print(
                            'restart file cannot be loaded, restarting!')
                else:
                    print(
                        'WARNING: restart file cannot be loaded, restarting!')

        if self.start_epoch >= self.epochs + 1:
            if self.cluster.is_cuda_available() and self.cluster.distributed:
                if self.cluster.is_main_worker():
                    print(
                        'WARNING: given epochs are less than the '
                        'one in the restart file!')
                    print('WARNING: SYS.EXIT is issued')
                sys.exit()
            else:
                print(
                    'WARNING: given epochs are less than the '
                    'one in the restart file!')
                print('WARNING: SYS.EXIT is issued')
                sys.exit()
