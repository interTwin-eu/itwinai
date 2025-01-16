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

import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from ..components import Trainer, monitor_exec
from ..loggers import Logger, LogMixin
from ..utils import load_yaml
from .config import TrainingConfiguration
from .distributed import (
    DeepSpeedStrategy,
    HorovodStrategy,
    NonDistributedStrategy,
    RayDDPStrategy,
    RayDeepSpeedStrategy,
    TorchDDPStrategy,
    TorchDistributedStrategy,
    distributed_resources_available,
)
from .reproducibility import seed_worker, set_seed
from .type import Batch, LrScheduler, Metric


class TorchTrainer(Trainer, LogMixin):
    """Trainer class for torch training algorithms.

    Args:
        config (Union[Dict, TrainingConfiguration]): training configuration
            containing hyperparameters.
        epochs (int): number of training epochs.
        model (Optional[Union[nn.Module, str]], optional): pytorch model to
            train or a string identifier. Defaults to None.
        strategy (Literal['ddp', 'deepspeed', 'horovod'], optional):
            distributed strategy. Defaults to 'ddp'.
        validation_every (Optional[int], optional): run a validation epoch
            every ``validation_every`` epochs. Disabled if None. Defaults to 1.
        test_every (Optional[int], optional): run a test epoch
            every ``test_every`` epochs. Disabled if None. Defaults to None.
        random_seed (Optional[int], optional): set random seed for
            reproducibility. If None, the seed is not set. Defaults to None.
        logger (Optional[Logger], optional): logger for ML tracking.
            Defaults to None.
        metrics (Optional[Dict[str, Metric]], optional): map of torchmetrics
            metrics. Defaults to None.
        checkpoints_location (str): path to checkpoints directory.
            Defaults to "checkpoints".
        checkpoint_every (Optional[int]): save a checkpoint every
            ``checkpoint_every`` epochs. Disabled if None. Defaults to None.
        disable_tqdm (bool): whether to disable tqdm progress bar(s).
        name (Optional[str], optional): trainer custom name. Defaults to None.
        profiling_wait_epochs (int): how many epochs to wait before starting
            the profiler.
        profiling_warmup_epochs (int): length of the profiler warmup phase in terms of
            number of epochs.
    """

    # TODO:
    #   - extract BaseTorchTrainer and extend it creating a set of trainer
    #     templates (e.g.. GAN, Classifier, Transformer) allowing scientists
    #     to reuse ML algos.

    _strategy: TorchDistributedStrategy = None

    #: PyTorch ``DataLoader`` for training dataset.
    train_dataloader: DataLoader = None
    #: PyTorch ``DataLoader`` for validation dataset.
    validation_dataloader: DataLoader = None
    #: PyTorch ``DataLoader`` for test dataset.
    test_dataloader: DataLoader = None
    #: PyTorch model to train.
    model: nn.Module = None
    #: Loss criterion.
    loss: Callable = None
    #: Optimizer.
    optimizer: Optimizer = None
    #: Learning rate scheduler.
    lr_scheduler: LrScheduler = None
    #: PyTorch random number generator (PRNG).
    torch_rng: torch.Generator = None
    #: itwinai ``itwinai.Logger``
    logger: Logger = None
    #: Total number training batches used so far, across all epochs.
    train_glob_step: int = 0
    #: Total number validation batches used so far, across all epochs.
    validation_glob_step: int = 0
    #: Total number test batches used so far, across all epochs.
    test_glob_step: int = 0
    #: Dictionary of ``torchmetrics`` metrics, indexed by user-defined names.
    metrics: Dict[str, Metric]
    #: PyTorch Profiler for communication vs. computation comparison
    profiler: Optional[Any]

    def __init__(
        self,
        config: Union[Dict, TrainingConfiguration],
        epochs: int,
        model: Optional[Union[nn.Module, str]] = None,
        strategy: Optional[Literal["ddp", "deepspeed", "horovod"]] = "ddp",
        validation_every: Optional[int] = 1,
        test_every: Optional[int] = None,
        random_seed: Optional[int] = None,
        logger: Optional[Logger] = None,
        metrics: Optional[Dict[str, Metric]] = None,
        checkpoints_location: str = "checkpoints",
        checkpoint_every: Optional[int] = None,
        disable_tqdm: bool = False,
        name: Optional[str] = None,
        profiling_wait_epochs: int = 1,
        profiling_warmup_epochs: int = 2,
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
        self.validation_every = validation_every
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
        if strategy is None or not distributed_resources_available():
            print("WARNING: falling back to non-distributed strategy.")
            strategy_obj = NonDistributedStrategy()
        elif strategy == "ddp":
            strategy_obj = TorchDDPStrategy(backend=self.config.dist_backend)
        elif strategy == "horovod":
            strategy_obj = HorovodStrategy()
        elif strategy == "deepspeed":
            strategy_obj = DeepSpeedStrategy(backend=self.config.dist_backend)
        else:
            raise NotImplementedError(f"Strategy '{strategy}' is not recognized/implemented.")
        return strategy_obj

    def _init_distributed_strategy(self) -> None:
        if not self.strategy.is_initialized:
            self.strategy.init()

    def _optimizer_from_config(self) -> None:
        if self.config.optimizer == "adadelta":
            self.optimizer = optim.Adadelta(
                self.model.parameters(),
                lr=self.config.optim_lr,
                weight_decay=self.config.optim_weight_decay,
            )
        elif self.config.optimizer == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.optim_lr,
                weight_decay=self.config.optim_weight_decay,
            )
        elif self.config.optimizer == "rmsprop":
            self.optimizer = optim.RMSprop(
                self.model.parameters(),
                lr=self.config.optim_lr,
                weight_decay=self.config.optim_weight_decay,
                momentum=self.config.optim_momentum,
            )
        elif self.config.optimizer == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.optim_lr,
                weight_decay=self.config.optim_weight_decay,
                momentum=self.config.optim_momentum,
            )
        else:
            raise ValueError(
                "Unrecognized self.config.optimizer! Check the docs for "
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

        if self.model is None:
            raise ValueError(
                "self.model is None! Either pass it to the constructor or "
                "override create_model_loss_optimizer method."
            )

        # Parse optimizer from training configuration
        # Optimizer can be changed with a custom one here!
        self._optimizer_from_config()

        # Parse loss from training configuration
        # Loss can be changed with a custom one here!
        self._loss_from_config()

        # IMPORTANT: model, optimizer, and scheduler need to be distributed
        distribute_kwargs = self.get_default_distributed_kwargs()

        # Distributed model, optimizer, and scheduler
        (self.model, self.optimizer, self.lr_scheduler) = self.strategy.distributed(
            self.model, self.optimizer, self.lr_scheduler, **distribute_kwargs
        )

    def create_dataloaders(
        self,
        train_dataset: Dataset,
        validation_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
    ) -> None:
        """
        Create train, validation and test dataloaders using the
        configuration provided in the Trainer constructor.
        Generally a user-defined method.

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
        validation_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
    ) -> Tuple[Dataset, Dataset, Dataset, Any]:
        """Prepares distributed environment and data structures
        for the actual training.

        Args:
            train_dataset (Dataset): training dataset.
            validation_dataset (Optional[Dataset], optional): validation
                dataset. Defaults to None.
            test_dataset (Optional[Dataset], optional): test dataset.
                Defaults to None.

        Returns:
            Tuple[Dataset, Dataset, Dataset, Any]: training dataset,
            validation dataset, test dataset, trained model.
        """
        self.torch_rng = set_seed(self.random_seed)
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

    def set_epoch(self, epoch: int) -> None:
        """Set current epoch at the beginning of training.

        Args:
            epoch (int): epoch number, from 0 to ``epochs-1``.
        """
        if self.profiler is not None and epoch > 0:
            # We don't want to start stepping until after the first epoch
            self.profiler.step()
        self._set_epoch_dataloaders(epoch)

    def log(
        self,
        item: Union[Any, List[Any]],
        identifier: Union[str, List[str]],
        kind: str = "metric",
        step: Optional[int] = None,
        batch_idx: Optional[int] = None,
        **kwargs,
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
                **kwargs,
            )

    def save_checkpoint(
        self, name: str, epoch: int, loss: Optional[torch.Tensor] = None
    ) -> None:
        """Save training checkpoint.

        Args:
            name (str): name of the checkpoint.
            epoch (int): current training epoch.
            loss (Optional[torch.Tensor]): current loss (if available).
        """
        state = dict(
            epoch=epoch,
            loss=loss,
            optimizer=self.optimizer.state_dict(),
            model=self.model.state_dict(),
            lr_scheduler=self.lr_scheduler,
        )
        ckpt_path = os.path.join(self.checkpoints_location, name)
        torch.save(state, ckpt_path)
        # print(f"Saved '{name}' checkpoint at {ckpt_path}")

        # Save checkpoint to logger
        self.log(ckpt_path, name, kind="artifact")

    def load_checkpoint(self, name: str) -> None:
        """Load state from a checkpoint.

        Args:
            name (str): name of the checkpoint to load, assuming it
                is under ``self.checkpoints_location`` location.
        """
        ckpt_path = os.path.join(self.checkpoints_location, name)
        state = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.lr_scheduler = state["lr_scheduler"]

    def compute_metrics(
        self,
        true: Batch,
        pred: Batch,
        logger_step: int,
        batch_idx: Optional[int],
        stage: str = "train",
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
        best_loss = float("inf")

        progress_bar = tqdm(
            range(self.epochs),
            desc="Epochs",
            disable=self.disable_tqdm or not self.strategy.is_main_worker,
        )

        for epoch in progress_bar:
            progress_bar.set_description(f"Epoch {epoch + 1}/{self.epochs}")

            epoch_n = epoch + 1
            self.set_epoch(epoch)
            self.train_epoch(epoch)
            if self.validation_every and epoch_n % self.validation_every == 0:
                val_loss = self.validation_epoch(epoch)

                # Checkpointing current best model
                worker_val_losses = self.strategy.gather(val_loss, dst_rank=0)
                if self.strategy.is_main_worker:
                    avg_loss = torch.mean(torch.stack(worker_val_losses)).detach().cpu()
                    if avg_loss < best_loss and self.checkpoint_every is not None:
                        ckpt_name = "best_model.pth"
                        self.save_checkpoint(name=ckpt_name, epoch=epoch, loss=avg_loss)
                        best_loss = avg_loss

            if self.test_every and epoch_n % self.test_every == 0:
                self.test_epoch(epoch)

            # Periodic checkpointing
            if (
                self.strategy.is_main_worker
                and self.checkpoint_every
                and epoch_n % self.checkpoint_every == 0
            ):
                ckpt_name = f"epoch_{epoch}.pth"
                self.save_checkpoint(name=ckpt_name, epoch=epoch)

    def train_epoch(self, epoch: int) -> torch.Tensor:
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

    def validation_epoch(self, epoch: int) -> torch.Tensor:
        """Perform a complete sweep over the validation dataset, completing an
        epoch of validation.

        Args:
            epoch (int): current epoch number, from 0 to ``self.epochs - 1``.

        Returns:
            Optional[Loss]: average validation loss for the current epoch if
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

    def test_epoch(self, epoch: int) -> torch.Tensor:
        """Perform a complete sweep over the test dataset, completing an
        epoch of test.

        Args:
            epoch (int): current epoch number, from 0 to ``self.epochs - 1``.

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
        config (Union[Dict, TrainingConfiguration]): training configuration
            containing hyperparameters.
        epochs (int): number of training epochs.
        discriminator (nn.Module): pytorch discriminator model to train GAN.
        generator (nn.Module): pytorch generator model to train GAN.
        strategy (Literal['ddp', 'deepspeed', 'horovod'], optional):
            distributed strategy. Defaults to 'ddp'.
        validation_every (Optional[int], optional): run a validation epoch
            every ``validation_every`` epochs. Disabled if None. Defaults to 1.
        test_every (Optional[int], optional): run a test epoch
            every ``test_every`` epochs. Disabled if None. Defaults to None.
        random_seed (Optional[int], optional): set random seed for
            reproducibility. If None, the seed is not set. Defaults to None.
        logger (Optional[Logger], optional): logger for ML tracking.
            Defaults to None.
        metrics (Optional[Dict[str, Metric]], optional): map of torch metrics
            metrics. Defaults to None.
        checkpoints_location (str): path to checkpoints directory.
            Defaults to "checkpoints".
        checkpoint_every (Optional[int]): save a checkpoint every
            ``checkpoint_every`` epochs. Disabled if None. Defaults to None.
        name (Optional[str], optional): trainer custom name. Defaults to None.
    """

    def __init__(
        self,
        config: Union[Dict, TrainingConfiguration],
        epochs: int,
        discriminator: nn.Module,
        generator: nn.Module,
        strategy: Literal["ddp", "deepspeed"] = "ddp",
        validation_every: Optional[int] = 1,
        test_every: Optional[int] = None,
        random_seed: Optional[int] = None,
        logger: Optional[Logger] = None,
        metrics: Optional[Dict[str, Metric]] = None,
        checkpoints_location: str = "checkpoints",
        checkpoint_every: Optional[int] = None,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            config=config,
            epochs=epochs,
            model=None,
            strategy=strategy,
            validation_every=validation_every,
            test_every=test_every,
            random_seed=random_seed,
            logger=logger,
            metrics=metrics,
            checkpoints_location=checkpoints_location,
            checkpoint_every=checkpoint_every,
            name=name,
            **kwargs,
        )
        self.save_parameters(**self.locals2params(locals()))
        self.discriminator = discriminator
        self.generator = generator

    def create_model_loss_optimizer(self) -> None:
        self.optimizerD = optim.Adam(
            self.discriminator.parameters(), lr=self.config.lr, betas=(0.5, 0.999)
        )
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
        config (Union[Dict, str]): `Lightning configuration`_
            which can be the path to a file or a Python dictionary.
        mlflow_saved_model (str, optional): name of the model created in
            MLFlow. Defaults to 'my_model'.

    .. _Lightning configuration:
        https://pytorch-lightning.readthedocs.io/en/1.6.5/common/lightning_cli.html
    """

    def __init__(self, config: Union[Dict, str], mlflow_saved_model: str = "my_model"):
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


class RayTorchTrainer(Trainer):
    """A trainer class for distributed training and hyperparameter optimization
    using Ray Train/ Tune and PyTorch.

    Args:
        config (Dict): A dictionary of configuration settings for the trainer.
        strategy (Optional[Literal["ddp", "deepspeed"]]):
            The distributed training strategy to use. Defaults to "ddp".
        name (Optional[str]): Optional name for the trainer instance. Defaults to None.
        logger (Optional[Logger]): Optional logger instance. Defaults to None.
    """

    def __init__(
        self,
        config: Dict,
        strategy: Literal["ddp", "deepspeed"] = "ddp",
        name: str | None = None,
        logger: Logger | None = None,
        random_seed: int = 1234,
    ) -> None:
        super().__init__(name=name)
        self.logger = logger
        self._set_strategy_and_init_ray(strategy)
        self._set_configs(config=config)
        self.torch_rng = set_seed(random_seed)

        import ray.train
        import ray.tune

        self.ray_train = ray.train
        self.ray_tune = ray.tune

    def _set_strategy_and_init_ray(self, strategy: str) -> None:
        """Set the distributed training strategy. This will initialize the ray backend.

        Args:
            strategy (str): The strategy to use for distributed training.
                Must be one of ["ddp", "deepspeed"].

        Raises:
            ValueError: If an unsupported strategy is provided.
        """
        if strategy == "ddp":
            self.strategy = RayDDPStrategy()
        elif strategy == "deepspeed":
            self.strategy = RayDeepSpeedStrategy(backend="nccl")
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

    def _set_configs(self, config: Dict) -> None:
        self.config = config
        self._set_scaling_config()
        self._set_tune_config()
        self._set_run_config()
        self._set_train_loop_config()

    @property
    def device(self) -> str:
        """Get the current device from distributed strategy.
        Returns:
            str: Device string (e.g., "cuda:0").
        """
        return self.strategy.device()

    def create_dataloaders(
        self,
        train_dataset: Dataset,
        validation_dataset: Dataset | None = None,
        test_dataset: Dataset | None = None,
        batch_size: int = 1,
        num_workers_dataloader: int = 4,
        pin_memory: bool = False,
        shuffle_train: bool | None = False,
        shuffle_test: bool | None = False,
        shuffle_validation: bool | None = False,
        sampler: Union[Sampler, Iterable, None] = None,
        collate_fn: Callable[[List], Any] | None = None,
    ) -> None:
        """Create data loaders for training, validation, and testing.

        Args:
            train_dataset (Dataset): The training dataset.
            validation_dataset (Dataset, optional): The validation dataset. Defaults to None.
            test_dataset (Dataset, optional): The test dataset. Defaults to None.
            batch_size (int, optional): Batch size for data loaders. Defaults to 1.
            shuffle_train (bool, optional): Whether to shuffle the training dataset.
                Defaults to False.
            shuffle_test (bool, optional): Whether to shuffle the test dataset.
                Defaults to False.
            shuffle_validation (bool, optional): Whether to shuffle the validation dataset.
                Defaults to False.
            sampler (Union[Sampler, Iterable, None], optional): Sampler for the datasets.
                Defaults to None.
            collate_fn (Callable[[List], Any], optional):
                Function to collate data samples into batches. Defaults to None.
        """
        self.train_dataloader = self.strategy.create_dataloader(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=num_workers_dataloader,
            pin_memory=pin_memory,
            generator=self.torch_rng,
            shuffle=shuffle_train,
            sampler=sampler,
            collate_fn=collate_fn,
        )
        if validation_dataset is not None:
            self.validation_dataloader = self.strategy.create_dataloader(
                dataset=validation_dataset,
                batch_size=batch_size,
                num_workers=num_workers_dataloader,
                pin_memory=pin_memory,
                generator=self.torch_rng,
                shuffle=shuffle_validation,
                sampler=sampler,
                collate_fn=collate_fn,
            )
        else:
            self.validation_dataloader = None
        if test_dataset is not None:
            self.test_dataloader = self.strategy.create_dataloader(
                dataset=test_dataset,
                batch_size=batch_size,
                num_workers=num_workers_dataloader,
                pin_memory=pin_memory,
                generator=self.torch_rng,
                shuffle=shuffle_test,
                sampler=sampler,
                collate_fn=collate_fn,
            )
        else:
            self.test_dataloader = None

    @monitor_exec
    def execute(
        self,
        train_dataset: Dataset,
        validation_dataset: Dataset | None = None,
        test_dataset: Dataset | None = None,
    ) -> Tuple[Dataset, Dataset, Dataset, Any]:
        """Execute the training pipeline with the given datasets.

        Args:
            train_dataset (Dataset): Training dataset.
            validation_dataset (Optional[Dataset], optional): Validation dataset.
                Defaults to None.
            test_dataset (Optional[Dataset], optional): Test dataset. Defaults to None.

        Returns:
            Tuple[Dataset, Dataset, Dataset, Any]:
                A tuple containing the datasets and the training result grid.
        """
        import ray.train.torch

        train_with_data = self.ray_tune.with_parameters(
            self.train, data=[train_dataset, validation_dataset, test_dataset]
        )
        trainer = ray.train.torch.TorchTrainer(
            train_with_data,
            train_loop_config=self.train_loop_config,
            scaling_config=self.scaling_config,
            run_config=self.run_config,
        )
        param_space = {"train_loop_config": self.train_loop_config}
        tuner = self.ray_tune.Tuner(
            trainer, param_space=param_space, tune_config=self.tune_config
        )

        result_grid = tuner.fit()

        return train_dataset, validation_dataset, test_dataset, result_grid

    def set_epoch(self, epoch: int) -> None:
        self.train_dataloader.sampler.set_epoch(epoch)
        if self.validation_dataloader is not None:
            self.validation_dataloader.sampler.set_epoch(epoch)
        if self.test_dataloader is not None:
            self.test_dataloader.sampler.set_epoch(epoch)

    def _set_tune_config(self) -> None:
        from .tuning import get_raytune_scheduler, get_raytune_search_alg

        tune_config = self.config.get("tune_config", {})

        if not tune_config:
            print(
                "WARNING: Empty Tune Config configured. Using the default configuration with "
                "a single trial."
            )

        search_alg = get_raytune_search_alg(tune_config)
        scheduler = get_raytune_scheduler(tune_config)

        metric = tune_config.get("metric", "loss")
        mode = tune_config.get("mode", "min")

        try:
            self.tune_config = self.ray_tune.TuneConfig(
                **tune_config,
                search_alg=search_alg,
                scheduler=scheduler,
                metric=metric,
                mode=mode,
            )
        except AttributeError as e:
            print(
                "Could not set Tune Config. Please ensure that you have passed the "
                "correct arguments for it. You can find more information for which "
                "arguments to set at "
                "https://docs.ray.io/en/latest/tune/api/doc/ray.tune.TuneConfig.html."
            )
            print(e)

    def _set_scaling_config(self) -> None:
        scaling_config = self.config.get("scaling_config", {})

        if not scaling_config:
            print("WARNING: No Scaling Config configured. Running trials non-distributed.")

        try:
            self.scaling_config = self.ray_train.ScalingConfig(**scaling_config)
        except AttributeError as e:
            print(
                "Could not set Scaling Config. Please ensure that you have passed the "
                "correct arguments for it. You can find more information for which "
                "arguments to set at "
                "https://docs.ray.io/en/latest/train/api/doc/ray.train.ScalingConfig.html"
            )
            print(e)

    def _set_run_config(self) -> None:
        run_config = self.config.get("run_config", {})

        if not run_config:
            print("WARNING: No RunConfig provided. Assuming local or single-node execution.")

        try:
            storage_path = Path(run_config.pop("storage_path")).resolve()

            if not storage_path:
                print(
                    "INFO: Empty storage path provided. Using default path 'ray_checkpoints'"
                )
                storage_path = Path("ray_checkpoints").resolve()

            self.run_config = self.ray_train.RunConfig(**run_config, storage_path=storage_path)
        except AttributeError as e:
            print(
                "Could not set Run Config. Please ensure that you have passed the "
                "correct arguments for it. You can find more information for which "
                "arguments to set at "
                "https://docs.ray.io/en/latest/train/api/doc/ray.train.RunConfig.html"
            )
            print(e)

    def _set_train_loop_config(self) -> None:
        self.train_loop_config = self.config.get("train_loop_config", {})

        if not self.train_loop_config:
            print(
                "WARNING: No training_loop_config detected. "
                "If you want to tune any hyperparameters, make sure to define them here."
            )
            return

        try:
            for name, param in self.train_loop_config.items():
                if not isinstance(param, dict):
                    continue

                # Convert specific keys to float if necessary
                for key in ["lower", "upper", "mean", "std"]:
                    if key in param:
                        param[key] = float(param[key])

                param_type = param.pop("type")
                param = getattr(self.ray_tune, param_type)(**param)
                self.train_loop_config[name] = param

        except AttributeError as e:
            print(
                f"{param} could not be set. Check that this parameter type is "
                "supported by Ray Tune at "
                "https://docs.ray.io/en/latest/tune/api/search_space.html"
            )
            print(e)

    # TODO: Can I also log the checkpoint?
    def checkpoint_and_report(self, epoch, tuning_metrics, checkpointing_data=None):
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            checkpoint = None

            should_checkpoint = epoch % self.config.get("checkpoint_freq", 1)

            if checkpointing_data and should_checkpoint:
                torch.save(checkpointing_data, os.path.join(temp_checkpoint_dir, str(epoch)))
                checkpoint = self.ray_train.Checkpoint.from_directory(temp_checkpoint_dir)

        self.ray_train.report(tuning_metrics, checkpoint=checkpoint)

    def initialize_logger(self, hyperparams: Optional[Dict], rank):
        if not self.logger:
            return

        self.logger.create_logger_context(rank=rank)
        print(f"Logger initialized with rank {rank}")

        if hyperparams:
            self.logger.save_hyperparameters(hyperparams)

    def close_logger(self):
        if self.logger:
            self.logger.destroy_logger_context()

    def log(
        self,
        item: Union[Any, List[Any]],
        identifier: Union[str, List[str]],
        kind: str = "metric",
        step: Optional[int] = None,
        batch_idx: Optional[int] = None,
        **kwargs,
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
                **kwargs,
            )
        else:
            print(
                "INFO: The log method was called, but no logger was configured for this "
                "Trainer."
            )
