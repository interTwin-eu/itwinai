# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# - Anna Lappe <anna.elisa.lappe@cern.ch> - CERN
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# - Linus Eickhoff <linus.maximilian.eickhoff@cern.ch> - CERN
# --------------------------------------------------------------------------------------


"""Provides training logic for PyTorch models via Trainer classes."""

import logging
import sys
import tempfile
from collections import defaultdict
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, List, Literal, Tuple

import mlflow
import mlflow.tracking
import ray.train
import ray.tune
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import yaml
from ray.train import Checkpoint, DataConfig, ScalingConfig
from ray.train.torch import TorchConfig
from ray.train.torch import TorchTrainer as RayTorchTrainer
from ray.tune import RunConfig, TuneConfig
from torch.optim import SGD, Adadelta, Adam, AdamW, RMSprop
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Metric
from tqdm import tqdm

from itwinai.torch.monitoring.monitoring import measure_gpu_utilization
from itwinai.torch.profiling.profiler import profile_torch_trainer

from ..components import Trainer, monitor_exec
from ..distributed import ray_cluster_is_running
from ..loggers import Logger, LogMixin, get_mlflow_logger
from ..utils import generate_random_name, load_yaml, time_and_log, to_uri
from .config import TrainingConfiguration
from .distributed import (
    DeepSpeedStrategy,
    HorovodStrategy,
    NonDistributedStrategy,
    RayDDPStrategy,
    RayDeepSpeedStrategy,
    RayTorchDistributedStrategy,
    TorchDDPStrategy,
    TorchDistributedStrategy,
    distributed_resources_available,
)
from .reproducibility import set_seed
from .tuning import search_space

py_logger = logging.getLogger(__name__)


def _get_tuning_metric_name(tune_config: TuneConfig | None) -> str:
    """Extracts the metric name from TuneConfig or scheduler in a generic way."""
    DEFAULT_NAME = "loss"

    if not tune_config:
        return DEFAULT_NAME

    # Try to get from TuneConfig
    if tune_config.metric:
        return tune_config.metric

    # Try to get from the scheduler (if defined)
    scheduler = tune_config.scheduler
    if scheduler and hasattr(scheduler, "metric") and scheduler.metric:
        return scheduler.metric

    return DEFAULT_NAME


class TorchTrainer(Trainer, LogMixin):
    """Trainer class for torch training algorithms.

    Args:
        config (Dict | TrainingConfiguration): training configuration containing
            hyperparameters.
        epochs (int): number of training epochs.
        model (nn.Module | None, optional): pytorch model to train or a string identifier.
            Defaults to None.
        strategy (Literal['ddp', 'deepspeed', 'horovod'], optional): distributed strategy.
            Defaults to 'ddp'.
        test_every (int | None, optional): run a test epoch every ``test_every`` epochs.
            Disabled if None. Defaults to None.
        random_seed (int | None, optional): set random seed for reproducibility. If None, the
            seed is not set. Defaults to None.
        logger (Logger | None, optional): logger for ML tracking. Defaults to None.
        metrics (Dict[str, Callable] | None, optional): map of torchmetrics metrics. Defaults
            to None.
        checkpoints_location (str): path to checkpoints directory. Defaults to "checkpoints".
        checkpoint_every (int | None): save a checkpoint every ``checkpoint_every`` epochs.
            Disabled if None. Defaults to None.
        disable_tqdm (bool): whether to disable tqdm progress bar(s).
        name (str | None, optional): trainer custom name. Defaults to None.
        profiling_wait_epochs (int): how many epochs to wait before starting the profiler.
        profiling_warmup_epochs (int): length of the profiler warmup phase in terms of number
            of epochs.
        measure_gpu_data (bool): enable the collection of data on average GPU utilization and
            total energy consumption throughout training. Defaults to False.
        enable_torch_profiling (bool): enable the profiling of computation. It uses the torch
            profiler and it may slow down training. Defaults to False.
        measure_epoch_time (bool): enable the measurement of epoch duration (in seconds).
            Defaults to False.
        ray_scaling_config (ScalingConfig, optional): scaling config for Ray Trainer. Defaults
            to None.
        ray_tune_config (TuneConfig, optional): tune config for Ray Tuner. Defaults to None.
        ray_run_config (ray.tune.RunConfig, optional): run config for Ray Tuner. Distributed
            training with Ray but without HPO will still be wrapped into a Ray Tuner, to keep
            everything homogeneous. Defaults to None.
        ray_search_space (Dict[str, Any], optional): search space for Ray Tuner. Defaults to
            None.
        ray_torch_config (TorchConfig, optional): torch configuration for Ray's TorchTrainer.
            Defaults to None.
        ray_data_config (DataConfig, optional): dataset configuration for Ray. Defaults to
            None.
        from_checkpoint (str | Path, optional): path to checkpoint directory. Defaults to None.
        initial_best_validation_metric (str): initial value for the best validation metric.
            Usually the validation metric is a loss to be minimized and this value exceeds the
            highest possible loss value, so that it will be overwritten when the first
            validation loss is computed. Example values are "inf" and "-inf", depending on
            wether the best validation metric should be minimized or maximized.
            Defaults to "inf".
        run_name (str, optional): name used to identify a specific run when collecting
            metrics on the trainer (e.g. GPU utilization). Defaults to None.
        time_ray (bool): whether to time and log the execution of Ray functions. Defaults to
            False.
    """

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
    lr_scheduler: LRScheduler | None = None
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
    #: PyTorch Profiler for computation ratio profiling.
    profiler: Any | None
    #: Toggle for GPU utilization monitoring
    measure_gpu_data: bool = False
    #: Toggle for computation fraction profiling
    enable_torch_profiling: bool = False
    #: Store PyTorch Profiling traces
    store_torch_profiling_traces: bool = False
    #: Toggle for epoch time tracking
    measure_epoch_time: bool = False
    #: Run ID
    run_name: str
    #: Toggle for Ray time logging
    time_ray: bool = False
    # Tune run id for nested runs in mlflow
    mlflow_tune_run_id: str | None = None
    # train run id
    mlflow_train_run_id: str | None = None
    # worker run_id
    mlflow_worker_run_id: str | None = None

    def __init__(
        self,
        config: Dict | TrainingConfiguration,
        epochs: int,
        model: nn.Module | None = None,
        strategy: Literal["ddp", "deepspeed", "horovod"] = "ddp",
        test_every: int | None = None,
        random_seed: int | None = None,
        logger: Logger | None = None,
        metrics: Dict[str, Metric] | None = None,
        checkpoints_location: str | Path = "checkpoints",
        checkpoint_every: int | None = None,
        disable_tqdm: bool = False,
        name: str | None = None,
        profiling_wait_epochs: int = 0,
        profiling_warmup_epochs: int = 0,
        measure_gpu_data: bool = False,
        enable_torch_profiling: bool = False,
        store_torch_profiling_traces: bool = False,
        measure_epoch_time: bool = False,
        ray_scaling_config: ScalingConfig | None = None,
        ray_tune_config: TuneConfig | None = None,
        ray_run_config: RunConfig | None = None,
        ray_search_space: Dict[str, Any] | None = None,
        ray_torch_config: TorchConfig | None = None,
        ray_data_config: DataConfig | None = None,
        from_checkpoint: Path | str | None = None,
        initial_best_validation_metric: str = "inf",
        run_name: str | None = None,
        time_ray: bool = False,
    ) -> None:
        super().__init__(name)
        self.save_parameters(**self.locals2params(locals()))

        # config is meant to store all hyperparameters, which can vary from use
        # case to use case and include learning_rate, batch_size....
        config = {} if config is None else config
        if isinstance(config, dict):
            config = TrainingConfiguration(**config)

        if store_torch_profiling_traces and not enable_torch_profiling:
            raise ValueError(
                "`store_torch_profiling_traces` is True, but `enable_torch_profiling` is"
                " False. Cannot store traces without enabling profiling."
            )

        self.config = config
        self.epochs = epochs
        self.model = model
        self.strategy = strategy
        self.test_every = test_every
        self.random_seed = random_seed
        self.logger = logger
        self.metrics = metrics if metrics is not None else {}
        self.checkpoints_location = checkpoints_location
        Path(self.checkpoints_location).mkdir(exist_ok=True, parents=True)
        self.checkpoint_every = checkpoint_every
        self.disable_tqdm = disable_tqdm

        self.profiler = None
        self.profiling_wait_epochs = profiling_wait_epochs
        self.profiling_warmup_epochs = profiling_warmup_epochs
        self.measure_gpu_data = measure_gpu_data
        self.enable_torch_profiling = enable_torch_profiling
        self.store_torch_profiling_traces = store_torch_profiling_traces
        self.measure_epoch_time = measure_epoch_time

        self.ray_scaling_config = ray_scaling_config
        self.ray_tune_config = ray_tune_config
        self.ray_run_config = ray_run_config
        self.ray_search_space = ray_search_space
        self.ray_torch_config = ray_torch_config
        self.ray_data_config = ray_data_config
        self.from_checkpoint = from_checkpoint
        self.time_ray = time_ray

        if self.from_checkpoint:
            self.from_checkpoint = Path(self.from_checkpoint)
            if not self.from_checkpoint.exists():
                raise RuntimeError(
                    "from_checkpoint argument was passed, but the checkpoint is not found "
                    f"at {self.from_checkpoint}"
                )

        if self.checkpoints_location:
            Path(self.checkpoints_location).mkdir(exist_ok=True, parents=True)

        py_logger.debug(f"ray_scaling_config: {ray_scaling_config}")
        py_logger.debug(f"ray_tune_config: {ray_tune_config}")
        py_logger.debug(f"ray_run_config: {ray_run_config}")
        py_logger.debug(f"ray_torch_config: {ray_torch_config}")
        py_logger.debug(f"ray_data_config: {ray_data_config}")
        py_logger.debug(f"ray_search_space: {ray_search_space}")

        # Initial training state -- can be resumed from a checkpoint
        self.model_state_dict = None
        self.optimizer_state_dict = None
        self.lr_scheduler_state_dict = None
        self.torch_rng_state = None
        # This is initialized to inf as it usually represents a loss to minimize.
        # If the validation metric is meant to be maximized, change this to -inf.
        self.best_validation_metric = float(initial_best_validation_metric)
        self.current_epoch = 0

        self.mlflow_logger = get_mlflow_logger(logger)
        if self.mlflow_logger:
            self.mlflow_client = mlflow.tracking.MlflowClient(self.mlflow_logger.tracking_uri)

        if run_name is None:
            run_name = generate_random_name()

        self.run_name = run_name

    @property
    def strategy(self) -> TorchDistributedStrategy:
        """Strategy currently in use."""
        assert self._strategy is not None, "Expected strategy to be initialized before access"
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: str | TorchDistributedStrategy) -> None:
        if isinstance(strategy, TorchDistributedStrategy):
            self._strategy = strategy
        else:
            self._strategy = self._detect_distributed_strategy(strategy)

    @property
    def device(self) -> str:
        """Current device from distributed strategy."""
        return self.strategy.device()

    def _detect_distributed_strategy(self, strategy: str) -> TorchDistributedStrategy:
        """When a Ray cluster is detected the Ray-equivalent distributed strategy is
        automatically selected, without needing the user to explicitly set it.
        """

        py_logger.debug(f"Strategy was set to {strategy}")

        dist_resources = distributed_resources_available()
        ray_cluster_running = ray_cluster_is_running()
        enough_resources = dist_resources or ray_cluster_running
        py_logger.debug(
            f"Enough resources? {enough_resources}"
            f" (distributed_resources_available: {dist_resources},"
            f" ray_cluster_is_running: {ray_cluster_running})"
        )

        # NOTE: setting strategy to None prevents the trainer to run distributed ML, regardless
        # of the availability of the resources.
        if strategy is None or not enough_resources:
            py_logger.warning("Falling back to non-distributed strategy.")
            return NonDistributedStrategy()

        if ray_cluster_is_running():
            py_logger.info(
                f"Ray cluster was detected, thus the Ray equivalent for {strategy} is used"
            )

        match strategy, ray_cluster_is_running():
            case "ddp", True:
                return RayDDPStrategy()

            case "ddp", False:
                return TorchDDPStrategy(backend=self.config.dist_backend)

            case "horovod", True:
                py_logger.warning(
                    "Horovod strategy is no longer supported with Ray V2. See "
                    "https://github.com/ray-project/ray/issues/49454#issuecomment-2899138398. "
                    "Falling back to HorovodStrategy without Ray."
                )
                return HorovodStrategy()

            case "horovod", False:
                return HorovodStrategy()

            case "deepspeed", True:
                return RayDeepSpeedStrategy(backend=self.config.dist_backend)

            case "deepspeed", False:
                return DeepSpeedStrategy(backend=self.config.dist_backend)

            case _:
                raise ValueError(f"Strategy '{strategy}' is not recognized.")

    def _init_distributed_strategy(self) -> None:
        if not self.strategy.is_initialized:
            self.strategy.init()

    def _set_optimizer_from_config(self) -> None:
        if self.model is None:
            raise ValueError(
                "self.model should be initialized before setting optimizer from configuration."
            )
        match self.config.optimizer:
            case "adadelta":
                self.optimizer = Adadelta(
                    self.model.parameters(),
                    lr=self.config.optim_lr,
                    weight_decay=self.config.optim_weight_decay,
                )
            case "adam":
                self.optimizer = Adam(
                    self.model.parameters(),
                    lr=self.config.optim_lr,
                    betas=self.config.optim_betas,
                    weight_decay=self.config.optim_weight_decay,
                )
            case "adamw":
                self.optimizer = AdamW(
                    self.model.parameters(),
                    lr=self.config.optim_lr,
                    betas=self.config.optim_betas,
                    weight_decay=self.config.optim_weight_decay,
                )
            case "rmsprop":
                self.optimizer = RMSprop(
                    self.model.parameters(),
                    lr=self.config.optim_lr,
                    weight_decay=self.config.optim_weight_decay,
                    momentum=self.config.optim_momentum,
                )
            case "sgd":
                self.optimizer = SGD(
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

    def _set_lr_scheduler_from_config(self) -> None:
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

    def _set_loss_from_config(self) -> None:
        match self.config.loss:
            case "nllloss":
                self.loss = nn.functional.nll_loss
            case "cross_entropy":
                self.loss = nn.functional.cross_entropy
            case "mse":
                self.loss = nn.functional.mse_loss
            case "bceloss":
                self.loss = nn.functional.binary_cross_entropy
            case _:
                raise ValueError(
                    "Unrecognized self.config.loss! Check the docs for "
                    "supported values and consider overriding "
                    "create_model_loss_optimizer method for more flexibility."
                )

    def get_default_distributed_kwargs(self) -> Dict:
        """Gives the default kwargs for the trainer's strategy's distributed() method."""

        if isinstance(self.strategy, DeepSpeedStrategy):
            # Batch size definition is not optional for DeepSpeedStrategy!
            distribute_kwargs = {
                "config_params": {"train_micro_batch_size_per_gpu": self.config.batch_size}
            }
        elif isinstance(self.strategy, HorovodStrategy):
            import horovod.torch as hvd

            distribute_kwargs = {
                "compression": (
                    hvd.Compression.fp16
                    if self.config.fp16_allreduce
                    else hvd.Compression.none
                ),
                "op": hvd.Adasum if self.config.use_adasum else hvd.Average,
                "gradient_predivide_factor": self.config.gradient_predivide_factor,
            }
        else:
            distribute_kwargs = {}

        return distribute_kwargs

    def create_model_loss_optimizer(self) -> None:
        """Instantiate a torch model, loss, optimizer, and LR scheduler using the
        configuration provided in the Trainer constructor.
        Generally a user-defined method.

        Raises:
            ValueError: If ``self.model`` is None.
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

        # Parse loss from training configuration
        # Loss can be changed with a custom one here!
        self._set_loss_from_config()

        # IMPORTANT: model, optimizer, and scheduler need to be distributed from here on

        distribute_kwargs = self.get_default_distributed_kwargs()

        # Distributed model, optimizer, and scheduler
        (self.model, self.optimizer, self.lr_scheduler) = self.strategy.distributed(
            self.model, self.optimizer, self.lr_scheduler, **distribute_kwargs
        )

    def save_checkpoint(
        self,
        name: str,
        best_validation_metric: torch.Tensor | None = None,
        checkpoints_root: str | Path | None = None,
        force: bool = False,
    ) -> str | None:
        """Save training checkpoint.

        Args:
            name (str): name of the checkpoint directory.
            best_validation_metric (torch.Tensor | None): best validation metric throughout
                training so far (if available). Usually this is the validation loss.
            checkpoints_root (str | None): path for root checkpoints dir. If None, uses
                ``self.checkpoints_location`` as base.
            force (bool): force checkpointing now.

        Returns:
            path to the checkpoint file or ``None`` when the checkpoint is not created.
        """
        # Determine whether a checkpoint should be created
        should_checkpoint = self.strategy.is_main_worker and (
            force
            or self.checkpoint_every
            and (self.current_epoch + 1) % self.checkpoint_every == 0
        )

        ckpt_dir = Path(checkpoints_root or self.checkpoints_location) / name
        py_logger.info(f"Saving checkpoint at {ckpt_dir.resolve()}? {should_checkpoint}")

        if not should_checkpoint:
            # Do nothing and return
            return

        ckpt_dir = Path(checkpoints_root or self.checkpoints_location) / name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save state (epoch, loss, optimizer, scheduler)
        state = {
            "epoch": self.current_epoch,
            # This could store the best validation loss
            "best_validation_metric": (
                best_validation_metric.item() if best_validation_metric is not None else None
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

        assert state_path.exists()
        assert model_path.exists()
        assert config_path.exists()

        py_logger.info(f"Saved checkpoint at {ckpt_dir.resolve()}")

        return str(ckpt_dir)

    def load_checkpoint(self) -> None:
        """Reload training state from checkpoint."""
        if not self.from_checkpoint:
            # A checkpoint path was NOT provided
            return

        # A checkpoint path was provided
        py_logger.info(f"Loading from existing checkpoint at {self.from_checkpoint}")

        if not isinstance(self.strategy, RayTorchDistributedStrategy):
            # Not using Ray, falling back to simple checkpoint reload
            py_logger.debug("Loading from existing checkpoint without using Ray")
            self._load_checkpoint(checkpoint_dir=self.from_checkpoint)
            return

        # A Ray checkpoint directory was passed to the trainer -- assuming to be inside a trial
        checkpoint = ray.train.get_checkpoint()
        if not checkpoint:
            py_logger.warning(
                "A checkpoint path was passed, but Ray could not find a valid "
                "checkpoint directory. Skipping loading from checkpoint."
            )
            return
        with checkpoint.as_directory() as checkpoint_dir:
            py_logger.debug("Loading from existing Ray checkpoint")
            self._load_checkpoint(checkpoint_dir=checkpoint_dir)

    def _load_checkpoint(self, checkpoint_dir: str | Path) -> None:
        """Load checkpoint from path."""
        checkpoint_dir = Path(checkpoint_dir)
        state = torch.load(checkpoint_dir / "state.pt")

        # Override initial training state
        self.model_state_dict = torch.load(checkpoint_dir / "model.pt")
        self.optimizer_state_dict = state["optimizer_state_dict"]
        self.lr_scheduler_state_dict = state["lr_scheduler_state_dict"]
        self.torch_rng_state = state["torch_rng_state"]
        # Direct overrides (don't require further attention)
        self.random_seed = state["random_seed"]
        self.current_epoch = state["epoch"] + 1  # Start from next epoch
        if state["best_validation_metric"]:
            self.best_validation_metric = state["best_validation_metric"]

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

    def _setup_metrics(self) -> None:
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
            # Execute with Ray
            return self._execute_with_ray(
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                test_dataset=test_dataset,
            )

        # Execute without ray
        if self.ray_scaling_config:
            py_logger.warning(
                "Ray scaling config was passed, but it's ignored as Ray is not used"
            )
        if self.ray_run_config:
            py_logger.warning("Ray run config was passed, but it's ignored as Ray is not used")
        if self.ray_tune_config:
            py_logger.warning(
                "Ray tune config was passed, but it's ignored as Ray is not used"
            )
        if self.ray_search_space:
            py_logger.warning(
                "Ray search space was passed, but it's ignored as Ray is not used"
            )
        if self.ray_torch_config:
            py_logger.warning(
                "Ray torch config was passed, but it's ignored as Ray is not used"
            )
        if self.ray_data_config:
            py_logger.warning(
                "Ray dataset config was passed, but it's ignored as Ray is not used"
            )

        self._run_worker(
            config={},
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            test_dataset=test_dataset,
        )
        return train_dataset, validation_dataset, test_dataset, None

    def _execute_with_ray(
        self,
        train_dataset: Dataset,
        validation_dataset: Dataset | None = None,
        test_dataset: Dataset | None = None,
    ) -> Tuple[Dataset, Dataset, Dataset, Any]:
        """Launch training and, optionally, hyperparameter tuning with Ray

        Args:
            train_dataset (Dataset): training dataset.
            validation_dataset (Dataset | None, optional): validation
                dataset. Defaults to None.
            test_dataset (Dataset | None, optional): test dataset.
                Defaults to None.

        Returns:
            Dataset: The training dataset
            Dataset: The validation dataset
            Dataset: The test dataset
            Any: The trained model
        """
        if self.ray_run_config and self.ray_run_config.storage_path:
            # Create Ray checkpoints dir if it does not exist yet
            ckpt_dir = Path(self.ray_run_config.storage_path)
            ckpt_dir.mkdir(parents=True, exist_ok=True)

        if (
            self.ray_scaling_config
            and getattr(self.ray_scaling_config, "num_workers", 1) > 1
            and getattr(self.ray_scaling_config.resources_per_worker, "GPU", 0) > 0.0
            and getattr(self.ray_scaling_config.resources_per_worker, "GPU", 0) < 1.0
        ):
            raise ValueError(
                "Distributed trials with fractional gpu resources are not supported."
                " Please ensure that either num_workers is set to 1 or GPUs in"
                " resources_per_worker is 0 or 1"
            )

        if (
            self.ray_tune_config
            and self.ray_tune_config.scheduler is not None
            and self.measure_gpu_data
        ):
            py_logger.info(
                "A Trial scheduler for Ray is specified"
                f" ({type(self.ray_tune_config.scheduler)}), while measuring gpu data."
                " Trials stopped by the scheduler might not close logger context in time,"
                " leaving the status of the mlflow run in 'pending'. This is just a visual"
                " caveat and can be ignored."
            )

        if self.mlflow_logger:
            # Create mlflow runs per trial (will be started by the trial's main worker)
            tune_run = self.mlflow_client.create_run(
                self.mlflow_logger.experiment_id, run_name=self.run_name
            )
            # Start and stop run to ensure it exists in MLflow before logging
            self.mlflow_tune_run_id = tune_run.info.run_id

        # Passes datasets to workers efficiently through Ray storage
        train_with_data = ray.tune.with_parameters(
            self._run_worker,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            test_dataset=test_dataset,
        )

        if self.from_checkpoint:
            # Create trainer from checkpoint
            if RayTorchTrainer.can_restore(to_uri(self.from_checkpoint)):
                trainer = RayTorchTrainer.restore(
                    path=to_uri(self.from_checkpoint),
                    train_loop_per_worker=train_with_data,
                    train_loop_config=None,
                )
            else:
                # Ray is unable to restore the checkpoint implicitly, but it's passing
                # it to the trial
                trainer = RayTorchTrainer(
                    train_loop_per_worker=train_with_data,
                    train_loop_config=None,
                    scaling_config=self.ray_scaling_config,
                    torch_config=self.ray_torch_config,
                    dataset_config=self.ray_data_config,
                    resume_from_checkpoint=Checkpoint(to_uri(self.from_checkpoint)),
                )
        else:
            # Create trainer without checkpoint
            trainer = RayTorchTrainer(
                train_loop_per_worker=train_with_data,
                train_loop_config=None,
                scaling_config=self.ray_scaling_config,
                torch_config=self.ray_torch_config,
                dataset_config=self.ray_data_config,
            )

        # Create the parameter space for hyperparameter tuning
        param_space = {"train_loop_config": search_space(self.ray_search_space)}

        # Create the tuner with the driver function
        tuner = ray.tune.Tuner(
            trainable=trainer,
            param_space=param_space,
            tune_config=self.ray_tune_config,
            run_config=self.ray_run_config,
        )

        # Run the tuner and capture results
        if self.time_ray and self.logger is not None:
            self.logger.create_logger_context(run_id=self.mlflow_tune_run_id)
            self.tune_result_grid = time_and_log(
                func=tuner.fit,
                logger=self.logger,
                identifier="ray_fit_time_s",
                step=0,
            )
        else:
            # Run the tuner and capture results
            self.tune_result_grid = tuner.fit()

        return train_dataset, validation_dataset, test_dataset, None

    def _run_worker(
        self,
        config: Dict,
        train_dataset: Dataset,
        validation_dataset: Dataset | None = None,
        test_dataset: Dataset | None = None,
    ) -> None:
        self.load_checkpoint()

        self._override_config(config)

        self._set_seed()
        self._init_distributed_strategy()
        self._setup_metrics()

        if self.logger:
            py_logger.debug(f"Using logger: {self.logger.__class__.__name__}")
            worker_run_name = f"worker_{self.strategy.global_rank()}"

            if self.strategy.is_main_worker and self.mlflow_logger:
                # Set the tracking uri and experiment for the main worker
                mlflow.set_tracking_uri(self.mlflow_logger.tracking_uri)
                mlflow.set_experiment(self.mlflow_logger.experiment_name)
                # If a tune_run_id is set, we create a nested run (Ray)
                if self.mlflow_tune_run_id:
                    train_run_name = ray.tune.get_context().get_trial_name()
                    train_run = self.mlflow_logger.mlflow.start_run(
                        experiment_id=self.mlflow_logger.experiment_id,
                        run_name=train_run_name,
                        parent_run_id=self.mlflow_tune_run_id,
                    )
                else:
                    train_run_name = self.run_name
                    train_run = self.mlflow_logger.mlflow.start_run(
                        experiment_id=self.mlflow_logger.experiment_id,
                        run_name=train_run_name,
                    )

                # store the mlflow run id as a parent for the worker runs
                self.mlflow_train_run_id = train_run.info.run_id
                # Stop train run to remove pending status in mlflow
                # (metrics are logged to workers)
                self.mlflow_logger.mlflow.end_run()
                worker_run_name += " (main)"

            # Broadcast trial_run_id from main worker to all workers
            # Ensure the broadcasted value is not None (Horovod otherwise deadlocks)
            self.mlflow_train_run_id = self.strategy.broadcast_obj(
                self.mlflow_train_run_id or "", src_rank=0
            )
            if self.mlflow_logger and not self.strategy.is_main_worker:
                # Set the tracking uri and experiment for other workers after main worker
                mlflow.set_tracking_uri(self.mlflow_logger.tracking_uri)
                mlflow.set_experiment(self.mlflow_logger.experiment_name)

            py_logger.debug(
                f"Broadcasted mlflow_trial_run_id {self.mlflow_train_run_id} to all workers"
            )
            # Create logger on worker level
            self.logger.create_logger_context(
                rank=self.strategy.global_rank(),
                parent_run_id=self.mlflow_train_run_id,
                run_name=worker_run_name,
            )
            self.log(
                item=self.strategy.name,
                identifier="strategy",
                kind="param",
            )
            self.log(
                item=self.strategy.global_rank(),
                identifier="global_rank",
                kind="param",
            )
            self.log(
                item=self.strategy.global_world_size(),
                identifier="global_world_size",
                kind="param",
            )
            if self.mlflow_logger and self.mlflow_logger.should_log():
                self.mlflow_worker_run_id = self.mlflow_logger.active_run.info.run_id

            py_logger.debug("...the logger has been initialized")

            hparams = self.config.model_dump()
            hparams["distributed_strategy"] = self.strategy.__class__.__name__
            self.logger.save_hyperparameters(hparams)

        self.create_dataloaders(
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            test_dataset=test_dataset,
        )
        self.create_model_loss_optimizer()

        self.train()

        if self.logger:
            self.logger.destroy_logger_context()

        self.strategy.clean_up()

    def _set_seed(self) -> None:
        py_logger.debug(f"Using random seed: {self.random_seed}")
        self.torch_rng = set_seed(self.random_seed)

        if self.torch_rng_state is not None:
            # Resume state from checkpoint
            py_logger.debug("Resuming torch PRNG state from checkpoint")
            self.torch_rng.set_state(self.torch_rng_state)

    def _override_config(self, config: Dict) -> None:
        """Override self.config with a sample from the search space from the Ray tuner."""
        self.config = self.config.model_copy(update=config)
        py_logger.debug("Overrode self.config with trial config (if given)")

    def _set_epoch_dataloaders(self, epoch: int) -> None:
        """Sets epoch in the distributed sampler of a dataloader when using it."""
        if not self.strategy.is_distributed:
            return

        self.train_dataloader.sampler.set_epoch(epoch)
        if self.validation_dataloader is not None:
            self.validation_dataloader.sampler.set_epoch(epoch)
        if self.test_dataloader is not None:
            self.test_dataloader.sampler.set_epoch(epoch)

    def set_epoch(self) -> None:
        """Set current epoch at the beginning of training."""
        # We don't want to start stepping until after the first epoch
        if self.profiler and self.current_epoch > 0:
            # Always step the profiler at the beginning of the epoch
            self.profiler.step()

        if self.lr_scheduler:
            self.lr_scheduler.step()
        self._set_epoch_dataloaders(self.current_epoch)

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
    ) -> None:
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
        if not isinstance(self.strategy, RayTorchDistributedStrategy):
            # Ray is not used, thus do nothing
            return

        if checkpoint_file:
            # A checkpoint is given as a file
            with tempfile.TemporaryDirectory() as tmp_dir:
                import shutil

                shutil.copy(checkpoint_file, tmp_dir)
                checkpoint = ray.train.Checkpoint.from_directory(tmp_dir)
                ray.train.report(metrics, checkpoint=checkpoint)

        elif checkpoint_data:
            # A checkpoint is given as a python object which needs to be serialized
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_dir = Path(tmp_dir)
                ckpt_file = tmp_dir / "ckpt.pt"
                torch.save(checkpoint_data, ckpt_file)
                checkpoint = ray.train.Checkpoint.from_directory(tmp_dir)
                ray.train.report(metrics, checkpoint=checkpoint)

        elif checkpoint_dir:
            # A checkpoint is given as a directory
            checkpoint = ray.train.Checkpoint.from_directory(checkpoint_dir)
            ray.train.report(metrics, checkpoint=checkpoint)

        else:
            # No checkpoint is given: only report metrics
            ray.train.report(metrics)

    def compute_metrics(
        self,
        true: torch.Tensor,
        pred: torch.Tensor,
        logger_step: int,
        batch_idx: int | None,
        stage: str = "train",
    ) -> Dict[str, Any]:
        """Compute and log metrics.

        Args:
            true (torch.Tensor): true values.
            pred (torch.Tensor): predicted values.
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

    @profile_torch_trainer
    @measure_gpu_utilization
    def train(self) -> None:
        """Trains a machine learning model.
        Main training loop/logic.

        Args:
            train_dataset (Dataset): training dataset.
            validation_dataset (Dataset): validation dataset.
            test_dataset (Dataset): test dataset.

        Returns:
            Dataset: The training dataset.
            Dataset: The validation dataset.
            Dataset: The test dataset.
            Any: The trained model
        """

        progress_bar = tqdm(
            range(self.current_epoch, self.epochs),
            desc="Epochs",
            disable=self.disable_tqdm or not self.strategy.is_main_worker,
        )

        for self.current_epoch in progress_bar:
            if self.strategy.is_main_worker:
                epoch_start_time = perf_counter()
            progress_bar.set_description(f"Epoch {self.current_epoch + 1}/{self.epochs}")

            self.set_epoch()
            self.train_epoch()
            val_metric = self.validation_epoch()

            # Periodic checkpointing
            periodic_ckpt_path = self.save_checkpoint(name=f"epoch_{self.current_epoch}")

            # Checkpointing current best model
            best_ckpt_path = None
            worker_val_metrics = self.strategy.gather(val_metric, dst_rank=0)

            if self.strategy.is_main_worker:
                avg_metric = torch.mean(torch.stack(worker_val_metrics)).detach().cpu()
                self.log(
                    item=avg_metric.item(),
                    identifier="global_validation_loss_epoch",
                    kind="metric",
                    step=self.current_epoch,
                )
                if avg_metric < self.best_validation_metric:
                    best_ckpt_path = self.save_checkpoint(
                        name="best_model",
                        best_validation_metric=avg_metric,
                        force=True,
                    )
                    self.best_validation_metric = avg_metric

            # Report validation metrics to Ray (useful for tuning!)
            metric_name = _get_tuning_metric_name(self.ray_tune_config)
            if metric_name is None:
                raise ValueError("Could not find a metric in the TuneConfig")

            if (
                self.time_ray
                and self.logger is not None
                and isinstance(self.strategy, RayTorchDistributedStrategy)
            ):
                time_and_log(
                    func=partial(
                        self.ray_report,
                        metrics={metric_name: val_metric.item()},
                        checkpoint_dir=best_ckpt_path or periodic_ckpt_path,
                    ),
                    logger=self.logger,
                    identifier="ray_report_time_s_per_epoch",
                    step=self.current_epoch,
                )
            else:
                self.ray_report(
                    metrics={metric_name: val_metric.item()},
                    checkpoint_dir=best_ckpt_path or periodic_ckpt_path,
                )

            if self.test_every and (self.current_epoch + 1) % self.test_every == 0:
                self.test_epoch()

            # Measure epoch time and log
            if self.strategy.is_main_worker:
                epoch_time = perf_counter() - epoch_start_time
                self.log(
                    item=epoch_time,
                    identifier="epoch_time_s",
                    kind="metric",
                    step=self.current_epoch,
                )

    def train_epoch(self) -> torch.Tensor:
        """Perform a complete sweep over the training dataset, completing an epoch of training.

        Args:
            epoch (int): current epoch number, from 0 to ``self.epochs - 1``.

        Returns:
            torch.Tensor: average training loss for the current epoch.
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

    def train_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Perform a single optimization step using a batch sampled from the training dataset.

        Args:
            batch (torch.Tensor): batch sampled by a dataloader.
            batch_idx (int): batch index in the dataloader.

        Returns:
            torch.Tensor: The batch loss.
            Dict[str, Any]: Dictionary of metric values (same structure as ``self.metrics``).
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

    def validation_epoch(self) -> torch.Tensor | None:
        """Perform a complete sweep over the validation dataset, completing an epoch of
        validation.

        Returns:
            torch.Tensor | None: average validation loss for the current epoch if
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
        self, batch: torch.Tensor, batch_idx: int
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Perform a single optimization step using a batch sampled from the validation
        dataset.

        Args:
            batch (torch.Tensor): batch sampled by a dataloader.
            batch_idx (int): batch index in the dataloader.

        Returns:
            torch.Tensor: Batch loss.
            Dict[str, Any]: Dictionary of metric values (same structure as ``self.metrics``).
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
        """Perform a complete sweep over the test dataset, completing an epoch of test.

        Returns:
            torch.Tensor: average test loss for the current epoch.
        """
        raise NotImplementedError()

    def test_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Perform a single predictions step using a batch sampled from the test dataset.

        Args:
            batch (torch.Tensor): batch sampled by a dataloader.
            batch_idx (int): batch index in the dataloader.

        Returns:
            torch.Tensor: The batch loss
            Dict[str, Any]: Dictionary of metric values (same structure as ``self.metrics``).
        """
        raise NotImplementedError()


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
        if isinstance(config, str) and Path(config).is_file():
            # Load from YAML
            config = load_yaml(config)
        self.conf = config
        self.mlflow_saved_model = mlflow_saved_model

    @monitor_exec
    def execute(self) -> Any:
        from lightning import LightningDataModule, LightningModule
        from lightning.pytorch.cli import LightningCLI

        from .mlflow import init_lightning_mlflow, teardown_lightning_mlflow

        init_lightning_mlflow(
            self.conf, tmp_dir="/tmp", registered_model_name=self.mlflow_saved_model
        )
        old_argv = sys.argv
        sys.argv = ["some_script_placeholder.py"]
        cli = LightningCLI(
            args=self.conf,
            model_class=LightningModule,
            datamodule_class=LightningDataModule,
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
