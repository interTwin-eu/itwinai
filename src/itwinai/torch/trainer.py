# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# - Anna Lappe <anna.elisa.lappe@cern.ch> - CERN
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# - Henry Mutegeki <henry.mutegeki@cern.ch> - CERN
# --------------------------------------------------------------------------------------


"""Provides training logic for PyTorch models via Trainer classes."""

import logging
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import ray.train
import ray.train.horovod
import ray.train.torch
import ray.tune
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from ..components import Trainer, monitor_exec
from ..distributed import ray_cluster_is_running
from ..loggers import Logger, LogMixin
from ..utils import load_yaml
from .config import TrainingConfiguration
from .distributed import (
    DeepSpeedStrategy,
    HorovodStrategy,
    NonDistributedStrategy,
    RayDDPStrategy,
    RayDeepSpeedStrategy,
    RayHorovodStrategy,
    RayTorchDistributedStrategy,
    TorchDDPStrategy,
    TorchDistributedStrategy,
    distributed_resources_available,
)
from .ray import run_config, scaling_config, search_space, tune_config
from .reproducibility import seed_worker, set_seed
from .type import Batch, Metric

py_logger = logging.getLogger(__name__)


class TorchTrainer(Trainer, LogMixin):
    """Trainer class for torch training algorithms.

    Args:
        config (Dict | TrainingConfiguration): training configuration
            containing hyperparameters.
        epochs (int): number of training epochs.
        model (Optional[Union[nn.Module, str]], optional): pytorch model to
            train or a string identifier. Defaults to None.
        strategy (Literal['ddp', 'deepspeed', 'horovod'], optional):
            distributed strategy. Defaults to 'ddp'.
        test_every (Optional[int], optional): run a test epoch
            every ``test_every`` epochs. Disabled if None. Defaults to None.
        random_seed (int | None, optional): set random seed for
            reproducibility. If None, the seed is not set. Defaults to None.
        logger (Logger | None, optional): logger for ML tracking.
            Defaults to None.
        metrics (Optional[Dict[str, Callable]], optional): map of torchmetrics
            metrics. Defaults to None.
        checkpoints_location (str): path to checkpoints directory.
            Defaults to "checkpoints".
        checkpoint_every (int | None): save a checkpoint every
            ``checkpoint_every`` epochs. Disabled if None. Defaults to None.
        disable_tqdm (bool): whether to disable tqdm progress bar(s).
        name (str | None, optional): trainer custom name. Defaults to None.
        profiling_wait_epochs (int): how many epochs to wait before starting
            the profiler.
        profiling_warmup_epochs (int): length of the profiler warmup phase in terms of
            number of epochs.
        ray_scaling_config (Dict[str, Any], optional): scaling config for Ray Trainer.
            Defaults to None,
        ray_tune_config (Dict[str, Any], optional): tune config for Ray Tuner.
            Defaults to None.
        ray_run_config (Dict[str, Any], optional): run config for Ray Trainer.
            Defaults to None.
        ray_search_space (Dict[str, Any], optional): search space for Ray Tuner.
            Defaults to None.
        from_checkpoint (str | Path, optional): path to checkpoint directory. Defaults to None.
    """

    # TODO:
    #   - extract BaseTorchTrainer and extend it creating a set of trainer
    #     templates (e.g.. GAN, Classifier, Transformer) allowing scientists
    #     to reuse ML algos.

    _strategy: TorchDistributedStrategy | None = None

    #: PyTorch ``DataLoader`` for training dataset.
    train_dataloader: DataLoader | None = None
    #: PyTorch ``DataLoader`` for validation dataset.
    validation_dataloader: DataLoader | None = None
    #: PyTorch ``DataLoader`` for test dataset.
    test_dataloader: DataLoader | None = None
    #: PyTorch model to train.
    model: nn.Module | None = None
    #: Loss criterion.
    loss: Callable | None = None
    #: Optimizer.
    optimizer: Optimizer | None = None
    #: Learning rate scheduler.
    lr_scheduler: LRScheduler = None
    #: PyTorch random number generator (PRNG).
    torch_rng: torch.Generator | None = None
    #: itwinai ``itwinai.Logger``
    logger: Logger | None = None
    #: Total number training batches used so far, across all epochs.
    train_glob_step: int = 0
    #: Total number validation batches used so far, across all epochs.
    validation_glob_step: int = 0
    #: Total number test batches used so far, across all epochs.
    test_glob_step: int = 0
    #: Dictionary of ``torchmetrics`` metrics, indexed by user-defined names.
    metrics: Dict[str, Callable]
    #: PyTorch Profiler for communication vs. computation comparison
    profiler: Any | None

    def __init__(
        self,
        config: Dict | TrainingConfiguration,
        epochs: int,
        model: Optional[Union[nn.Module, str]] = None,
        strategy: Optional[Literal["ddp", "deepspeed", "horovod"]] = "ddp",
        test_every: Optional[int] = None,
        random_seed: Optional[int] = None,
        logger: Optional[Logger] = None,
        metrics: Optional[Dict[str, Metric]] = None,
        checkpoints_location: str | Path = "checkpoints",
        checkpoint_every: Optional[int] = None,
        disable_tqdm: bool = False,
        name: str | None = None,
        profiling_wait_epochs: int = 1,
        profiling_warmup_epochs: int = 2,
        ray_scaling_config: Dict[str, Any] | None = None,
        ray_tune_config: Dict[str, Any] | None = None,
        ray_run_config: Dict[str, Any] | None = None,
        ray_search_space: Dict[str, Any] | None = None,
        from_checkpoint: str | Path | None = None,
    ) -> None:
        super().__init__(name)
        self.save_parameters(**self.locals2params(locals()))

        # config is mean to store all hyperparameters, which can very from use
        # case to use case and include learning_rate, batch_size....
        if isinstance(config, dict):
            config = TrainingConfiguration(**config)

        self.config = config
        self.epochs = epochs
        self.model = model
        self.strategy = strategy
        self.test_every = test_every
        self.random_seed = random_seed
        self.logger = logger
        self.metrics = metrics if metrics is not None else {}
        self.checkpoints_location = checkpoints_location
        os.makedirs(self.checkpoints_location, exist_ok=True)
        self.checkpoint_every = checkpoint_every
        self.disable_tqdm = disable_tqdm
        self.profiler = None
        self.profiling_wait_epochs = profiling_wait_epochs
        self.profiling_warmup_epochs = profiling_warmup_epochs
        self.ray_scaling_config = ray_scaling_config
        self.ray_tune_config = ray_tune_config
        self.ray_run_config = ray_run_config
        self.ray_search_space = ray_search_space
        self.from_checkpoint = from_checkpoint

        if self.from_checkpoint:
            self.from_checkpoint = Path(from_checkpoint)
            if not self.from_checkpoint.exists():
                raise RuntimeError(
                    "from_checkpoint was passed, but the checkpoint is not found"
                )

        py_logger.debug(f"ray_scaling_config: {ray_scaling_config}")
        py_logger.debug(f"ray_tune_config: {ray_tune_config}")
        py_logger.debug(f"ray_run_config: {ray_run_config}")
        py_logger.debug(f"ray_search_space: {ray_search_space}")

        # Initial training state -- can be resumed from a checkpoint
        self.model_state_dict = None
        self.optimizer_state_dict = None
        self.lr_scheduler_state_dict = None
        self.torch_rng_state = None
        self.best_validation_loss = float("inf")
        self.epoch = 0

    @property
    def strategy(self) -> TorchDistributedStrategy:
        """Strategy currently in use."""
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: str | TorchDistributedStrategy) -> None:
        if isinstance(strategy, TorchDistributedStrategy):
            self._strategy = strategy
        else:
            self._strategy = self._detect_strategy(strategy)

    @property
    def device(self) -> str:
        """Current device from distributed strategy."""
        return self.strategy.device()

    def _detect_strategy(self, strategy: str) -> TorchDistributedStrategy:
        """If a Ray cluster is detected"""

        py_logger.debug(f"Strategy was set to {strategy}")

        enough_resources = distributed_resources_available() or ray_cluster_is_running()
        py_logger.debug(
            f"Enough resources? {enough_resources} "
            f"(distributed_resources_available: {distributed_resources_available()}) "
            f"(ray_cluster_is_running: {ray_cluster_is_running()})"
        )

        # NOTE: setting strategy to None prevents the trainer to run distribtued ML, regardless
        # of the availability of the resources.
        if strategy is None or not enough_resources:
            py_logger.warning("falling back to non-distributed strategy.")
            strategy_obj = NonDistributedStrategy()
        elif strategy == "ddp":
            if ray_cluster_is_running():
                # NOTE: the torch backend is passed in the Ray's torch config
                strategy_obj = RayDDPStrategy()
                py_logger.info(
                    f"Ray cluster was detected, thus the Ray equvalent for {strategy} is used"
                )
            else:
                strategy_obj = TorchDDPStrategy(backend=self.config.dist_backend)
        elif strategy == "horovod":
            if ray_cluster_is_running():
                strategy_obj = RayHorovodStrategy()
                py_logger.info(
                    f"Ray cluster was detected, thus the Ray equvalent for {strategy} is used"
                )
            else:
                strategy_obj = HorovodStrategy()
        elif strategy == "deepspeed":
            if ray_cluster_is_running():
                strategy_obj = RayDeepSpeedStrategy(backend=self.config.dist_backend)
                py_logger.info(
                    f"Ray cluster was detected, thus the Ray equvalent for {strategy} is used"
                )
            else:
                strategy_obj = DeepSpeedStrategy(backend=self.config.dist_backend)
        else:
            raise NotImplementedError(f"Strategy '{strategy}' is not recognized/implemented.")
        return strategy_obj

    def _init_distributed_strategy(self) -> None:
        if not self.strategy.is_initialized:
            self.strategy.init()

    def _optimizer_from_config(self) -> None:
        match self.config.optimizer:
            case "adadelta":
                self.optimizer = optim.Adadelta(
                    self.model.parameters(),
                    lr=self.config.optim_lr,
                    weight_decay=self.config.optim_weight_decay,
                )
            case "adam":
                self.optimizer = optim.Adam(
                    self.model.parameters(),
                    lr=self.config.optim_lr,
                    weight_decay=self.config.optim_weight_decay,
                )
            case "rmsprop":
                self.optimizer = optim.RMSprop(
                    self.model.parameters(),
                    lr=self.config.optim_lr,
                    weight_decay=self.config.optim_weight_decay,
                    momentum=self.config.optim_momentum,
                )
            case "sgd":
                self.optimizer = optim.SGD(
                    self.model.parameters(),
                    lr=self.config.optim_lr,
                    weight_decay=self.config.optim_weight_decay,
                    momentum=self.config.optim_momentum,
                )
            case _:
                raise ValueError(
                    "Unrecognized self.config.optimizer! Check the docs for "
                    "supported values and consider overriding "
                    "create_model_loss_optimizer method for more flexibility."
                )

    def _lr_scheduler_from_config(self) -> None:
        """Parse Lr scheduler from training config"""
        if not self.config.lr_scheduler:
            return
        if not self.optimizer:
            raise ValueError("Trying to instantiate a LR scheduler but the optimizer is None!")

        match self.config.lr_scheduler:
            case "constant":
                self.lr_scheduler = lr_scheduler.ConstantLR(self.optimizer)
            case "polynomial":
                self.lr_scheduler = lr_scheduler.PolynomialLR(self.optimizer)
            case "exponential":
                self.lr_scheduler = lr_scheduler.ExponentialLR(
                    self.optimizer, gamma=self.config.lr_scheduler_gamma
                )
            case "linear":
                self.lr_scheduler = lr_scheduler.LinearLR(self.optimizer)
            case "multistep":
                self.lr_scheduler = lr_scheduler.MultiStepLR(
                    self.optimizer, milestones=self.config.lr_scheduler_step_size
                )
            case "step":
                self.lr_scheduler = lr_scheduler.StepLR(
                    self.optimizer, step_size=self.config.lr_scheduler_step_size
                )
            case _:
                raise ValueError(
                    "Unrecognized self.config.lr_scheduler! Check the docs for "
                    "supported values and consider overriding "
                    "create_model_loss_optimizer method for more flexibility."
                )

    def _loss_from_config(self) -> None:
        if self.config.loss == "nllloss":
            self.loss = nn.functional.nll_loss
        elif self.config.loss == "cross_entropy":
            self.loss = nn.functional.cross_entropy
        elif self.config.loss == "mse":
            self.loss = nn.functional.mse_loss
        else:
            raise ValueError(
                "Unrecognized self.config.loss! Check the docs for "
                "supported values and consider overriding "
                "create_model_loss_optimizer method for more flexibility."
            )

    def get_default_distributed_kwargs(self) -> Dict:
        """Gives the default kwargs for the trainer's strategy's distributed() method."""

        if isinstance(self.strategy, DeepSpeedStrategy):
            # Batch size definition is not optional for DeepSpeedStrategy!
            distribute_kwargs = dict(
                config_params=dict(train_micro_batch_size_per_gpu=self.config.batch_size)
            )
        elif isinstance(self.strategy, HorovodStrategy):
            import horovod.torch as hvd

            distribute_kwargs = dict(
                compression=(
                    hvd.Compression.fp16
                    if self.config.fp16_allreduce
                    else hvd.Compression.none
                ),
                op=hvd.Adasum if self.config.use_adasum else hvd.Average,
                gradient_predivide_factor=self.config.gradient_predivide_factor,
            )
        else:
            distribute_kwargs = {}

        return distribute_kwargs

    def create_model_loss_optimizer(self) -> None:
        """
        Instantiate a torch model, loss, optimizer, and LR scheduler using the
        configuration provided in the Trainer constructor.
        Generally a user-defined method.
        """
        ###################################
        # Dear user, this is a method you #
        # may be interested to override!  #
        ###################################

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
        self._optimizer_from_config()

        # Parse LR scheduler from training configuration
        # LR scheduler can be changed with a custom one here!
        self._lr_scheduler_from_config()

        if self.optimizer_state_dict:
            # Load optimizer state from checkpoint
            # IMPORTANT: this must be after the learning rate scheduler was already initialized
            # by passing to it the optimizer. Otherwise the optimizer state just loaded will
            # be modified by the lr scheduler.
            self.optimizer.load_state_dict(self.optimizer_state_dict)

        if self.lr_scheduler_state_dict and self.lr_scheduler:
            # Load LR scheduler state from checkpoint
            self.lr_scheduler.load_state_dict(self.lr_scheduler_state_dict)

        # Parse loss from training configuration
        # Loss can be changed with a custom one here!
        self._loss_from_config()

        # # Save non-distributed copies of model, optim and lr scheduler
        # self.original_model = self.model
        # self.original_optimizer = self.optimizer
        # self.original_lr_scheduler = self.lr_scheduler

        # IMPORTANT: model, optimizer, and scheduler need to be distributed
        distribute_kwargs = self.get_default_distributed_kwargs()

        # Distributed model, optimizer, and scheduler
        (self.model, self.optimizer, self.lr_scheduler) = self.strategy.distributed(
            self.model, self.optimizer, self.lr_scheduler, **distribute_kwargs
        )

    def save_checkpoint(
        self,
        name: str,
        best_validation_loss: Optional[torch.Tensor] = None,
        checkpoints_root: str | Path | None = None,
        force: bool = False,
    ) -> str | None:
        """Save training checkpoint.

        Args:
            name (str): name of the checkpoint directory.
            best_validation_loss (Optional[torch.Tensor]): best validation loss throughout
                training so far (if available).
            checkpoints_root (str | None): path for root checkpoints dir. If None, uses
                ``self.checkpoints_location`` as base.
            force (bool): force checkpointign now.

        Returns:
            path to the checkpoint file or ``None`` when the checkpoint is not created.
        """
        if not (
            force
            or self.strategy.is_main_worker
            and self.checkpoint_every
            and (self.epoch + 1) % self.checkpoint_every == 0
        ):
            # Do nothing and return
            return

        ckpt_dir = Path(checkpoints_root or self.checkpoints_location) / name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save state (epoch, loss, optimizer, scheduler)
        state = {
            "epoch": self.epoch,
            # This could store the best validation loss
            "best_validation_loss": (
                best_validation_loss.item() if best_validation_loss is not None else None
            ),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": (
                self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None
            ),
            "torch_rng_state": self.torch_rng.get_state(),
            "random_seed": self.random_seed,
        }
        state_path = ckpt_dir / "state.pt"
        torch.save(state, state_path)

        # Save PyTorch model separately
        model_path = ckpt_dir / "model.pt"
        # TODO: check that the state dict is stripped from any distributed info
        torch.save(self.model.state_dict(), model_path)

        # Save Pydantic config as YAML
        config_path = ckpt_dir / "config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(self.config.model_dump(), f)

        # Log each file with an appropriate identifier
        self.log(str(state_path), f"{name}_state", kind="artifact")
        self.log(str(model_path), f"{name}_model", kind="artifact")
        self.log(str(config_path), f"{name}_config", kind="artifact")
        return str(ckpt_dir)

    def _load_checkpoint(self) -> None:
        """Reload training state from checkpoint."""

        if self.from_checkpoint:
            self.from_checkpoint = Path(self.from_checkpoint)
            state = torch.load(self.from_checkpoint / "state.pt")

            # Override initial training state
            self.model_state_dict = torch.load(self.from_checkpoint / "model.pt")
            self.optimizer_state_dict = state["optimizer_state_dict"]
            self.lr_scheduler_state_dict = state["lr_scheduler_state_dict"]
            self.torch_rng_state = state["torch_rng_state"]
            # Direct overrides (don't require further attention)
            self.random_seed = state["random_seed"]
            self.epoch = state["epoch"]
            if state["best_validation_loss"]:
                self.best_validation_loss = state["best_validation_loss"]

    def create_dataloaders(
        self,
        train_dataset: Dataset,
        validation_dataset: Dataset | None = None,
        test_dataset: Dataset | None = None,
    ) -> None:
        """
        Create train, validation and test dataloaders using the
        configuration provided in the Trainer constructor.
        Generally a user-defined method.

        Args:
            train_dataset (Dataset): training dataset object.
            validation_dataset (Dataset | None): validation dataset object.
                Default None.
            test_dataset (Dataset | None): test dataset object.
                Default None.
        """

        ###################################
        # Dear user, this is a method you #
        # may be interested to override!  #
        ###################################
        self.train_dataloader = self.strategy.create_dataloader(
            dataset=train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers_dataloader,
            pin_memory=self.config.pin_gpu_memory,
            generator=self.torch_rng,
            shuffle=self.config.shuffle_train,
        )
        if validation_dataset is not None:
            self.validation_dataloader = self.strategy.create_dataloader(
                dataset=validation_dataset,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers_dataloader,
                pin_memory=self.config.pin_gpu_memory,
                generator=self.torch_rng,
                shuffle=self.config.shuffle_validation,
            )
        if test_dataset is not None:
            self.test_dataloader = self.strategy.create_dataloader(
                dataset=test_dataset,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers_dataloader,
                pin_memory=self.config.pin_gpu_memory,
                generator=self.torch_rng,
                shuffle=self.config.shuffle_test,
            )

    def _setup_metrics(self):
        """Move metrics to current device."""
        for m_name, metric in self.metrics.items():
            self.metrics[m_name] = metric.to(self.device)

    @monitor_exec
    def execute(
        self,
        train_dataset: Dataset,
        validation_dataset: Dataset | None = None,
        test_dataset: Dataset | None = None,
    ) -> Tuple[Dataset, Dataset, Dataset, Any]:
        """Prepares distributed environment and data structures
        for the actual training.

        Args:
            train_dataset (Dataset): training dataset.
            validation_dataset (Dataset | None, optional): validation
                dataset. Defaults to None.
            test_dataset (Dataset | None, optional): test dataset.
                Defaults to None.

        Returns:
            Tuple[Dataset, Dataset, Dataset, Any]: training dataset,
            validation dataset, test dataset, trained model.
        """
        if isinstance(self.strategy, RayTorchDistributedStrategy):
            # Run with Ray
            # TODO: check that args can be always passed like this (e.g., large datasets or
            # weird objs)
            train_with_data = ray.tune.with_parameters(
                self._run_worker,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                test_dataset=test_dataset,
            )

            if isinstance(self.strategy, RayHorovodStrategy):
                # Using Horovod with Ray
                trainer = ray.train.horovod.HorovodTrainer(
                    train_with_data,
                    # TODO: this may have to be set to some relevant config
                    train_loop_config=None,
                    # TODO: this may need to be set
                    horovod_config=None,
                    scaling_config=scaling_config(self.ray_scaling_config),
                    run_config=run_config(
                        self.ray_run_config, default_checkpoints_root=self.checkpoints_location
                    ),
                )
            else:
                # Using DDP or DeepSpeed with Ray
                py_logger.debug(f"scaling config: {scaling_config(self.ray_scaling_config)}")
                trainer = ray.train.torch.TorchTrainer(
                    train_with_data,
                    # TODO: this may have to be set to some relevant config
                    train_loop_config=None,
                    scaling_config=scaling_config(self.ray_scaling_config),
                    run_config=run_config(
                        self.ray_run_config, default_checkpoints_root=self.checkpoints_location
                    ),
                )
            param_space = {"train_loop_config": search_space(self.ray_search_space)}
            tuner = ray.tune.Tuner(
                trainer,
                param_space=param_space,
                tune_config=tune_config(self.ray_tune_config),
            )

            result_grid = tuner.fit()

            # TODO: review returned objs
            return train_dataset, validation_dataset, test_dataset, result_grid
        else:
            if self.ray_scaling_config:
                py_logger.warning(
                    "Ray scaling config was passed, but it's ignored as Ray is not used"
                )
            if self.ray_run_config:
                py_logger.warning(
                    "Ray run config was passed, but it's ignored as Ray is not used"
                )
            if self.ray_tune_config:
                py_logger.warning(
                    "Ray tune config was passed, but it's ignored as Ray is not used"
                )
            if self.ray_search_space:
                py_logger.warning(
                    "Ray search space was passed, but it's ignored as Ray is not used"
                )
            # Run without Ray
            self._run_worker(
                config={},
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                test_dataset=test_dataset,
            )
            # TODO: review returned objs
            return train_dataset, validation_dataset, test_dataset, None

    def _run_worker(
        self,
        config: Dict,
        train_dataset: Dataset,
        validation_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
    ) -> Tuple[Dataset, Dataset, Dataset, Any]:
        self._load_checkpoint()

        self._override_config(config)

        self._set_seed()
        self._init_distributed_strategy()
        self._setup_metrics()

        self.create_dataloaders(
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            test_dataset=test_dataset,
        )
        self.create_model_loss_optimizer()

        if self.logger:
            self.logger.create_logger_context(rank=self.strategy.global_rank())
            hparams = self.config.model_dump()
            hparams["distributed_strategy"] = self.strategy.__class__.__name__
            self.logger.save_hyperparameters(hparams)

        self.train()

        if self.logger:
            self.logger.destroy_logger_context()
        self.strategy.clean_up()
        return

    def _set_seed(self):
        self.torch_rng = set_seed(self.random_seed)

        if self.torch_rng_state is not None:
            # Resume state from checkpoint
            self.torch_rng.set_state(self.torch_rng_state)

    def _override_config(self, config: Dict) -> None:
        """Overrid self.config with a sample from the search space from the Ray tuner."""
        self.config = self.config.model_copy(update=config)

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

    def set_epoch(self) -> None:
        """Set current epoch at the beginning of training."""
        if self.profiler is not None and self.epoch > 0:
            # We don't want to start stepping until after the first epoch
            self.profiler.step()
        if self.lr_scheduler:
            self.lr_scheduler.step()
        self._set_epoch_dataloaders(self.epoch)

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
            item (Any | List[Any]): element to be logged (e.g., metric).
            identifier (str | List[str]): unique identifier for the
                element to log(e.g., name of a metric).
            kind (str, optional): type of the item to be logged. Must be one
                among the list of self.supported_types. Defaults to 'metric'.
            step (int | None, optional): logging step. Defaults to None.
            batch_idx (int | None, optional): DataLoader batch counter
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

    def ray_report(
        self,
        metrics: Dict[str, float],
        checkpoint_file: str | Path | None = None,
        checkpoint_dir: str | Path | None = None,
        checkpoint_data: Any | None = None,
    ):
        """Report a dictionary of metrics and optionally a checkpoint to Ray, only when using
        Ray distributed strategies. The checkpoint could be in the form of a Python object
        (passed as ``checkpoint_data``), the path to a single file (passed as
        ``checkpoint_file``), or the path to an existing checkpoint directory (passed as
        ``checkpoint_dir``).

        Args:
            metrics (Dict[str, float]): metrics to be reported.
            checkpoint_file (str | Path | None, optional): path to the checkpoint file.
                Defaults to None.
            checkpoint_dir (str | Path | None, optional): path to the checkpoint directory.
                Defaults to None.
            checkpoint_data (Any | None, optional):object to serialize as a checkpoint.
                Defaults to None.
        """
        if isinstance(self.strategy, RayTorchDistributedStrategy):
            checkpoint = None
            if checkpoint_file:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    import shutil

                    shutil.copy(checkpoint_file, tmp_dir)
                    checkpoint = ray.train.Checkpoint.from_directory(tmp_dir)
            elif checkpoint_data:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    torch.save(checkpoint_data, tmp_dir + "ckpt.pt")
                    checkpoint = ray.train.Checkpoint.from_directory(tmp_dir)
            elif checkpoint_dir:
                checkpoint = ray.train.Checkpoint.from_directory(checkpoint_dir)
            ray.train.report(metrics, checkpoint=checkpoint)

    def compute_metrics(
        self,
        true: Batch,
        pred: Batch,
        logger_step: int,
        batch_idx: int | None,
        stage: str = "train",
    ) -> Dict[str, Any]:
        """Compute and log metrics.

        Args:
            metrics (Dict[str, Callable]): metrics dict. Can be
                ``self.train_metrics`` or ``self.validation_metrics``.
            true (Batch): true values.
            pred (Batch): predicted values.
            logger_step (int): global step to pass to the logger.
            stage (str): 'train', 'validation'...

        Returns:
            Dict[str, Any]: metric values.
        """
        m_values = {}
        for m_name, metric in self.metrics.items():
            # metric = metric.to(self.device)
            m_val = metric(pred, true).detach().cpu().numpy()
            self.log(
                item=m_val,
                identifier=f"{stage}_{m_name}",
                kind="metric",
                step=logger_step,
                batch_idx=batch_idx,
            )
            m_values[m_name] = m_val
        return m_values

    def train(self):
        """Trains a machine learning model.
        Main training loop/logic.

        Args:
            train_dataset (Dataset): training dataset.
            validation_dataset (Dataset): validation dataset.
            test_dataset (Dataset): test dataset.

        Returns:
            Tuple[Dataset, Dataset, Dataset, Any]: training dataset,
            validation dataset, test dataset, trained model.
        """

        progress_bar = tqdm(
            range(self.epoch, self.epochs),
            desc="Epochs",
            disable=self.disable_tqdm or not self.strategy.is_main_worker,
        )

        for self.epoch in progress_bar:
            progress_bar.set_description(f"Epoch {self.epoch + 1}/{self.epochs}")

            epoch_n = self.epoch + 1
            self.set_epoch()
            self.train_epoch()
            val_loss = self.validation_epoch()

            # Periodic checkpointing
            periodic_ckpt_path = self.save_checkpoint(name=f"epoch_{self.epoch}")

            # Checkpointing current best model
            best_ckpt_path = None
            worker_val_losses = self.strategy.gather(val_loss, dst_rank=0)
            if self.strategy.is_main_worker:
                avg_loss = torch.mean(torch.stack(worker_val_losses)).detach().cpu()
                if avg_loss < self.best_validation_loss and self.checkpoint_every is not None:
                    ckpt_name = "best_model"
                    best_ckpt_path = self.save_checkpoint(
                        name=ckpt_name,
                        best_validation_loss=avg_loss,
                        force=True,
                    )
                    self.best_validation_loss = avg_loss

            # Report validation metrics to Ray (useful for tuning!)
            self.ray_report(
                {"loss": val_loss.item()},
                checkpoint_dir=best_ckpt_path or periodic_ckpt_path,
            )

            if self.test_every and epoch_n % self.test_every == 0:
                self.test_epoch()

    def train_epoch(self) -> torch.Tensor:
        """Perform a complete sweep over the training dataset, completing an
        epoch of training.

        Args:
            epoch (int): current epoch number, from 0 to ``self.epochs - 1``.

        Returns:
            Loss: average training loss for the current epoch.
        """
        self.model.train()
        train_loss_sum = 0.0
        train_metrics_sum = defaultdict(float)
        batch_counter = 0

        progress_bar = tqdm(
            enumerate(self.train_dataloader),
            total=len(self.train_dataloader) // self.strategy.global_world_size(),
            desc="Train batches",
            disable=self.disable_tqdm or not self.strategy.is_main_worker,
            leave=False,  # Set this to true to see how many batches were used
        )

        for batch_idx, train_batch in progress_bar:
            loss, metrics = self.train_step(batch=train_batch, batch_idx=batch_idx)
            train_loss_sum += loss
            batch_counter += 1
            for name, val in metrics.items():
                train_metrics_sum[name] += val

            # Important: update counter
            self.train_glob_step += 1

        # Aggregate and log losses
        avg_loss = train_loss_sum / batch_counter
        self.log(
            item=avg_loss.item(),
            identifier="train_loss_epoch",
            kind="metric",
            step=self.train_glob_step,
        )
        # Aggregate and log metrics
        for m_name, m_val in train_metrics_sum.items():
            self.log(
                item=m_val / batch_counter,
                identifier="train_" + m_name + "_epoch",
                kind="metric",
                step=self.train_glob_step,
            )

        return avg_loss

    def train_step(self, batch: Batch, batch_idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Perform a single optimization step using a batch sampled from the
        training dataset.

        Args:
            batch (Batch): batch sampled by a dataloader.
            batch_idx (int): batch index in the dataloader.

        Returns:
            Tuple[Loss, Dict[str, Any]]: batch loss and dictionary of metric
            values with the same structure of ``self.metrics``.
        """
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        self.optimizer.zero_grad()
        pred_y = self.model(x)
        loss = self.loss(pred_y, y)
        loss.backward()
        self.optimizer.step()

        # Log metrics
        self.log(
            item=loss.item(),
            identifier="train_loss",
            kind="metric",
            step=self.train_glob_step,
            batch_idx=batch_idx,
        )
        metrics: Dict[str, Any] = self.compute_metrics(
            true=y,
            pred=pred_y,
            logger_step=self.train_glob_step,
            batch_idx=batch_idx,
            stage="train",
        )
        return loss, metrics

    def validation_epoch(self) -> torch.Tensor:
        """Perform a complete sweep over the validation dataset, completing an
        epoch of validation.

        Returns:
            Loss | None: average validation loss for the current epoch if
                self.validation_dataloader is not None
        """
        if self.validation_dataloader is None:
            return

        progress_bar = tqdm(
            enumerate(self.validation_dataloader),
            total=len(self.validation_dataloader) // self.strategy.global_world_size(),
            desc="Validation batches",
            disable=self.disable_tqdm or not self.strategy.is_main_worker,
            leave=False,  # Set this to true to see how many batches were used
        )

        self.model.eval()
        validation_loss_sum = 0.0
        validation_metrics_sum = defaultdict(float)
        batch_counter = 0
        for batch_idx, val_batch in progress_bar:
            loss, metrics = self.validation_step(batch=val_batch, batch_idx=batch_idx)
            validation_loss_sum += loss
            batch_counter += 1
            for name, val in metrics.items():
                validation_metrics_sum[name] += val

            # Important: update counter
            self.validation_glob_step += 1

        # Aggregate and log losses
        avg_loss = validation_loss_sum / batch_counter
        self.log(
            item=avg_loss.item(),
            identifier="validation_loss_epoch",
            kind="metric",
            step=self.validation_glob_step,
        )
        # Aggregate and log metrics
        for m_name, m_val in validation_metrics_sum.items():
            self.log(
                item=m_val / batch_counter,
                identifier="validation_" + m_name + "_epoch",
                kind="metric",
                step=self.validation_glob_step,
            )

        return avg_loss

    def validation_step(
        self, batch: Batch, batch_idx: int
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Perform a single optimization step using a batch sampled from the
        validation dataset.

        Args:
            batch (Batch): batch sampled by a dataloader.
            batch_idx (int): batch index in the dataloader.

        Returns:
            Tuple[Loss, Dict[str, Any]]: batch loss and dictionary of metric
            values with the same structure of ``self.metrics``.
        """
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        with torch.no_grad():
            pred_y = self.model(x)
            loss: torch.Tensor = self.loss(pred_y, y)
        self.log(
            item=loss.item(),
            identifier="validation_loss",
            kind="metric",
            step=self.validation_glob_step,
            batch_idx=batch_idx,
        )
        metrics: Dict[str, Any] = self.compute_metrics(
            true=y,
            pred=pred_y,
            logger_step=self.validation_glob_step,
            batch_idx=batch_idx,
            stage="validation",
        )
        return loss, metrics

    def test_epoch(self) -> torch.Tensor:
        """Perform a complete sweep over the test dataset, completing an
        epoch of test.

        Returns:
            Loss: average test loss for the current epoch.
        """
        raise NotImplementedError()

    def test_step(self, batch: Batch, batch_idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Perform a single predictions step using a batch sampled from the
        test dataset.

        Args:
            batch (Batch): batch sampled by a dataloader.
            batch_idx (int): batch index in the dataloader.

        Returns:
            Tuple[Loss, Dict[str, Any]]: batch loss and dictionary of metric
            values with the same structure of ``self.metrics``.
        """
        raise NotImplementedError()


class GANTrainer(TorchTrainer):
    """Trainer class for GAN models using pytorch.

    Args:
        config (Dict | TrainingConfiguration): training configuration
            containing hyperparameters.
        epochs (int): number of training epochs.
        discriminator (nn.Module): pytorch discriminator model to train GAN.
        generator (nn.Module): pytorch generator model to train GAN.
        strategy (Literal['ddp', 'deepspeed', 'horovod'], optional):
            distributed strategy. Defaults to 'ddp'.
        test_every (Optional[int], optional): run a test epoch
            every ``test_every`` epochs. Disabled if None. Defaults to None.
        random_seed (int | None, optional): set random seed for
            reproducibility. If None, the seed is not set. Defaults to None.
        logger (Logger | None, optional): logger for ML tracking.
            Defaults to None.
        metrics (Optional[Dict[str, Callable]], optional): map of torch metrics
            metrics. Defaults to None.
        checkpoints_location (str): path to checkpoints directory.
            Defaults to "checkpoints".
        checkpoint_every (int | None): save a checkpoint every
            ``checkpoint_every`` epochs. Disabled if None. Defaults to None.
        name (Optional[str], optional): trainer custom name. Defaults to None.
        profiling_wait_epochs (int): how many epochs to wait before starting
            the profiler.
        profiling_warmup_epochs (int): length of the profiler warmup phase in terms of
            number of epochs.
        ray_scaling_config (Dict[str, Any], optional): scaling config for Ray Trainer.
            Defaults to None,
        ray_tune_config (Dict[str, Any], optional): tune config for Ray Tuner.
            Defaults to None.
        ray_run_config (Dict[str, Any], optional): run config for Ray Trainer.
            Defaults to None.
        ray_search_space (Dict[str, Any], optional): search space for Ray Tuner.
            Defaults to None.
        from_checkpoint (str | Path, optional): path to checkpoint directory. Defaults to None.
    """

    def __init__(
        self,
        config: Dict | TrainingConfiguration,
        epochs: int,
        discriminator: nn.Module,
        generator: nn.Module,
        strategy: Literal["ddp", "deepspeed"] = "ddp",
        test_every: Optional[int] = None,
        random_seed: Optional[int] = None,
        logger: Optional[Logger] = None,
        metrics: Optional[Dict[str, Metric]] = None,
        checkpoints_location: str = "checkpoints",
        checkpoint_every: Optional[int] = None,
        name: Optional[str] = None,
        profiling_wait_epochs: int = 1,
        profiling_warmup_epochs: int = 2,
        ray_scaling_config: Dict[str, Any] | None = None,
        ray_tune_config: Dict[str, Any] | None = None,
        ray_run_config: Dict[str, Any] | None = None,
        ray_search_space: Dict[str, Any] | None = None,
        from_checkpoint: str | Path | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            config=config,
            epochs=epochs,
            model=None,
            strategy=strategy,
            test_every=test_every,
            random_seed=random_seed,
            logger=logger,
            metrics=metrics,
            checkpoints_location=checkpoints_location,
            checkpoint_every=checkpoint_every,
            name=name,
            profiling_wait_epochs=profiling_wait_epochs,
            profiling_warmup_epochs=profiling_warmup_epochs,
            ray_scaling_config=ray_scaling_config,
            ray_tune_config=ray_tune_config,
            ray_run_config=ray_run_config,
            ray_search_space=ray_search_space,
            from_checkpoint=from_checkpoint,
            **kwargs,
        )
        self.save_parameters(**self.locals2params(locals()))

        # Initial training state -- can be resumed from a checkpoint
        self.discriminator = discriminator
        self.generator = generator
        self.optimizerD_state_dict = None
        self.optimizerG = None

    def create_model_loss_optimizer(self) -> None:
        # Model, optimizer, and lr scheduler may have already been loaded from a checkpoint
        # TODO: this is incomplete
        if not self.optimizerD:
            self.optimizerD = optim.Adam(
                self.discriminator.parameters(), lr=self.config.lr, betas=(0.5, 0.999)
            )
        if not self.optimizerG:
            self.optimizerG = optim.Adam(
                self.generator.parameters(), lr=self.config.lr, betas=(0.5, 0.999)
            )
        self.criterion = nn.BCELoss()

        # https://stackoverflow.com/a/67437077
        self.discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.discriminator)
        self.generator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.generator)

        # First, define strategy-wise optional configurations
        if isinstance(self.strategy, DeepSpeedStrategy):
            # Batch size definition is not optional for DeepSpeedStrategy!
            distribute_kwargs = dict(
                config_params=dict(train_micro_batch_size_per_gpu=self.config.batch_size)
            )
        else:
            distribute_kwargs = {}

        # Distribute discriminator and its optimizer
        self.discriminator, self.optimizerD, _ = self.strategy.distributed(
            self.discriminator, self.optimizerD, **distribute_kwargs
        )
        self.generator, self.optimizerG, _ = self.strategy.distributed(
            self.generator, self.optimizerG, **distribute_kwargs
        )

    def train_epoch(self, epoch: int):
        self.discriminator.train()
        self.generator.train()
        gen_train_losses = []
        disc_train_losses = []
        disc_train_accuracy = []
        for batch_idx, (real_images, _) in enumerate(self.train_dataloader):
            lossG, lossD, accuracy_disc = self.train_step(real_images, batch_idx)
            gen_train_losses.append(lossG)
            disc_train_losses.append(lossD)
            disc_train_accuracy.append(accuracy_disc)

            self.train_glob_step += 1
        # Aggregate and log losses and accuracy
        avg_disc_accuracy = torch.mean(torch.stack(disc_train_accuracy))
        self.log(
            item=avg_disc_accuracy.item(),
            identifier="disc_train_accuracy_per_epoch",
            kind="metric",
            step=epoch,
        )
        avg_gen_loss = torch.mean(torch.stack(gen_train_losses))
        self.log(
            item=avg_gen_loss.item(),
            identifier="gen_train_loss_per_epoch",
            kind="metric",
            step=epoch,
        )

        avg_disc_loss = torch.mean(torch.stack(disc_train_losses))
        self.log(
            item=avg_disc_loss.item(),
            identifier="disc_train_loss_per_epoch",
            kind="metric",
            step=epoch,
        )

        self.save_fake_generator_images(epoch)

    def validation_epoch(self, epoch: int):
        gen_validation_losses = []
        gen_validation_accuracy = []
        disc_validation_losses = []
        disc_validation_accuracy = []
        self.discriminator.eval()
        self.generator.eval()
        for batch_idx, (real_images, _) in enumerate(self.validation_dataloader):
            loss_gen, accuracy_gen, loss_disc, accuracy_disc = self.validation_step(
                real_images, batch_idx
            )
            gen_validation_losses.append(loss_gen)
            gen_validation_accuracy.append(accuracy_gen)
            disc_validation_losses.append(loss_disc)
            disc_validation_accuracy.append(accuracy_disc)
            self.validation_glob_step += 1

        # Aggregate and log metrics
        disc_validation_loss = torch.mean(torch.stack(disc_validation_losses))
        self.log(
            item=disc_validation_loss.item(),
            identifier="disc_valid_loss_per_epoch",
            kind="metric",
            step=epoch,
        )
        disc_validation_accuracy = torch.mean(torch.stack(disc_validation_accuracy))
        self.log(
            item=disc_validation_accuracy.item(),
            identifier="disc_valid_accuracy_epoch",
            kind="metric",
            step=epoch,
        )
        gen_validation_loss = torch.mean(torch.stack(gen_validation_losses))
        self.log(
            item=gen_validation_loss.item(),
            identifier="gen_valid_loss_per_epoch",
            kind="metric",
            step=epoch,
        )
        gen_validation_accuracy = torch.mean(torch.stack(gen_validation_accuracy))
        self.log(
            item=gen_validation_accuracy.item(),
            identifier="gen_valid_accuracy_epoch",
            kind="metric",
            step=epoch,
        )

        return gen_validation_loss

    def train_step(self, real_images, batch_idx):
        real_images = real_images.to(self.device)
        batch_size = real_images.size(0)
        real_labels = torch.ones((batch_size,), dtype=torch.float, device=self.device)
        fake_labels = torch.zeros((batch_size,), dtype=torch.float, device=self.device)

        # Train Discriminator with real images
        output_real = self.discriminator(real_images)
        lossD_real = self.criterion(output_real, real_labels)
        # Generate fake images and train Discriminator
        noise = torch.randn(batch_size, self.config.z_dim, 1, 1, device=self.device)

        fake_images = self.generator(noise)
        output_fake = self.discriminator(fake_images.detach())
        lossD_fake = self.criterion(output_fake, fake_labels)

        lossD = (lossD_real + lossD_fake) / 2

        self.optimizerD.zero_grad()
        lossD.backward()
        self.optimizerD.step()

        accuracy = ((output_real > 0.5).float() == real_labels).float().mean() + (
            (output_fake < 0.5).float() == fake_labels
        ).float().mean()
        accuracy_disc = accuracy.mean()

        # Train Generator
        output_fake = self.discriminator(fake_images)
        lossG = self.criterion(output_fake, real_labels)
        self.optimizerG.zero_grad()
        lossG.backward()
        self.optimizerG.step()
        self.log(
            item=accuracy_disc,
            identifier="disc_train_accuracy_per_batch",
            kind="metric",
            step=self.train_glob_step,
            batch_idx=batch_idx,
        )
        self.log(
            item=lossG,
            identifier="gen_train_loss_per_batch",
            kind="metric",
            step=self.train_glob_step,
            batch_idx=batch_idx,
        )
        self.log(
            item=lossD,
            identifier="disc_train_loss_per_batch",
            kind="metric",
            step=self.train_glob_step,
            batch_idx=batch_idx,
        )

        return lossG, lossD, accuracy_disc

    def validation_step(self, real_images, batch_idx):
        real_images = real_images.to(self.device)
        batch_size = real_images.size(0)
        real_labels = torch.ones((batch_size,), dtype=torch.float, device=self.device)
        fake_labels = torch.zeros((batch_size,), dtype=torch.float, device=self.device)

        # Validate with real images
        output_real = self.discriminator(real_images)
        loss_real = self.criterion(output_real, real_labels)

        # Generate and validate fake images
        noise = torch.randn(batch_size, self.config.z_dim, 1, 1, device=self.device)

        with torch.no_grad():
            fake_images = self.generator(noise)
            output_fake = self.discriminator(fake_images.detach())
        loss_fake = self.criterion(output_fake, fake_labels)

        # Generator's attempt to fool the discriminator
        loss_gen = self.criterion(output_fake, real_labels)
        accuracy_gen = ((output_fake > 0.5).float() == real_labels).float().mean()

        # Calculate total discriminator loss and accuracy
        d_total_loss = (loss_real + loss_fake) / 2
        accuracy = ((output_real > 0.5).float() == real_labels).float().mean() + (
            (output_fake < 0.5).float() == fake_labels
        ).float().mean()
        d_accuracy = accuracy.item() / 2

        self.log(
            item=loss_gen.item(),
            identifier="gen_valid_loss_per_batch",
            kind="metric",
            step=self.validation_glob_step,
            batch_idx=batch_idx,
        )
        self.log(
            item=accuracy_gen.item(),
            identifier="gen_valid_accuracy_per_batch",
            kind="metric",
            step=self.validation_glob_step,
            batch_idx=batch_idx,
        )

        self.log(
            item=d_total_loss.item(),
            identifier="disc_valid_loss_per_batch",
            kind="metric",
            step=self.validation_glob_step,
            batch_idx=batch_idx,
        )
        self.log(
            item=d_accuracy,
            identifier="disc_valid_accuracy_per_batch",
            kind="metric",
            step=self.validation_glob_step,
            batch_idx=batch_idx,
        )
        return loss_gen, accuracy_gen

    def save_checkpoint(self, name, epoch, loss=None):
        """Save training checkpoint with both optimizers."""
        if not os.path.exists(self.checkpoints_location):
            os.makedirs(self.checkpoints_location)

        checkpoint_path = os.path.join(self.checkpoints_location, f"{name}")
        checkpoint = {
            "epoch": epoch,
            "loss": loss.item() if loss is not None else None,
            "discriminator_state_dict": self.discriminator.state_dict(),
            "generator_state_dict": self.generator.state_dict(),
            "optimizerD_state_dict": self.optimizerD.state_dict(),
            "optimizerG_state_dict": self.optimizerG.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load models and optimizers from checkpoint."""
        checkpoint = torch.load(checkpoint_path)

        self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.optimizerD.load_state_dict(checkpoint["optimizerD_state_dict"])
        self.optimizerG.load_state_dict(checkpoint["optimizerG_state_dict"])

        if "lr_scheduler" in checkpoint:
            if checkpoint["lr_scheduler"] is not None:
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        print(f"Checkpoint loaded from {checkpoint_path}")

    def save_fake_generator_images(self, epoch):
        """
        plot and save fake images from generator

         Args:
            epoch (int): epoch number, from 0 to ``epochs-1``.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        self.generator.eval()
        noise = torch.randn(64, self.config.z_dim, 1, 1, device=self.device)
        fake_images = self.generator(noise)
        fake_images_grid = torchvision.utils.make_grid(fake_images, normalize=True)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_axis_off()
        ax.set_title(f"Fake images for epoch {epoch}")
        ax.imshow(np.transpose(fake_images_grid.cpu().numpy(), (1, 2, 0)))
        self.log(
            item=fig,
            identifier=f"fake_images_epoch_{epoch}.png",
            kind="figure",
            step=epoch,
        )


class TorchLightningTrainer(Trainer):
    """Generic trainer for torch Lightning workflows.

    Args:
        config (Dict | str): `Lightning configuration`_
            which can be the path to a file or a Python dictionary.
        mlflow_saved_model (str, optional): name of the model created in
            MLFlow. Defaults to 'my_model'.

    .. _Lightning configuration:
        https://pytorch-lightning.readthedocs.io/en/1.6.5/common/lightning_cli.html
    """

    def __init__(self, config: Dict | str, mlflow_saved_model: str = "my_model"):
        self.save_parameters(**self.locals2params(locals()))
        super().__init__()
        if isinstance(config, str) and os.path.isfile(config):
            # Load from YAML
            config = load_yaml(config)
        self.conf = config
        self.mlflow_saved_model = mlflow_saved_model

    @monitor_exec
    def execute(self) -> Any:
        import lightning as L
        from lightning.pytorch.cli import LightningCLI

        from .mlflow import init_lightning_mlflow, teardown_lightning_mlflow

        init_lightning_mlflow(
            self.conf, tmp_dir="/tmp", registered_model_name=self.mlflow_saved_model
        )
        old_argv = sys.argv
        sys.argv = ["some_script_placeholder.py"]
        cli = LightningCLI(
            args=self.conf,
            model_class=L.LightningModule,
            datamodule_class=L.LightningDataModule,
            run=False,
            save_config_kwargs={
                "overwrite": True,
                "config_filename": "pl-training.yml",
            },
            subclass_mode_model=True,
            subclass_mode_data=True,
        )
        sys.argv = old_argv
        cli.trainer.fit(cli.model, datamodule=cli.datamodule)
        teardown_lightning_mlflow()


def _distributed_dataloader(dataloader: DataLoader, gwsize, grank):
    """Makes a Dataloader distributed."""
    sampler = DistributedSampler(
        dataloader.dataset, num_replicas=gwsize, rank=grank, shuffle=True
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
        pin_memory_device=dataloader.pin_memory_device,
    )


def distributed(func):
    """The decorated function must have a standard signature.
    Its first arguments must be:
    model, train_dataloader, validation_dataloader, device (in this order).

    Additional args or kwargs are allowed consistently with the signature
    of the decorated function.
    """

    def dist_train(
        model, train_dataloader, validation_dataloader=None, device="cpu", *args, **kwargs
    ):
        if torch.cuda.is_available():
            dist.init_process_group(backend="nccl")

        if torch.cuda.is_available():
            lwsize = torch.cuda.device_count()  # local world size - per node
            gwsize = dist.get_world_size()  # global world size - per run
            grank = dist.get_rank()  # global rank - assign per run
            lrank = dist.get_rank() % lwsize  # local rank - assign per node
        else:
            gwsize = 1
            grank = 0
            lrank = 0

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu", lrank)
        if torch.cuda.is_available():
            torch.cuda.set_device(lrank)

        model = model.to(device)
        model = DDP(model, device_ids=[device], output_device=device)

        train_dataloader = _distributed_dataloader(train_dataloader, gwsize, grank)
        if validation_dataloader is not None:
            validation_dataloader = _distributed_dataloader(
                validation_dataloader, gwsize, grank
            )

        try:
            func(model, train_dataloader, validation_dataloader, device, *args, **kwargs)
        finally:
            if torch.cuda.is_available():
                dist.barrier()
                dist.destroy_process_group()

    return dist_train
