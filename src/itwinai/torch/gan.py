# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# - Henry Mutegeki <henry.mutegeki@cern.ch> - CERN
# --------------------------------------------------------------------------------------


"""Provides training logic for PyTorch models via Trainer classes."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
import yaml
from ray.train import DataConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchConfig
from ray.tune import TuneConfig
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from ..loggers import Logger
from .config import TrainingConfiguration
from .trainer import TorchTrainer
from .type import Metric

if TYPE_CHECKING:
    from ray.train.horovod import HorovodConfig

py_logger = logging.getLogger(__name__)


class GANTrainingConfiguration(TrainingConfiguration):
    """Configuration object for training a GAN. Extends the base TrainingConfiguration."""

    #: Name of the optimizer to use for the generator. Defaults to 'adam'.
    optimizer_generator: Literal["adadelta", "adam", "adamw", "rmsprop", "sgd"] = "adam"
    #: Learning rate used by the optimizer for the generator. Defaults to 1e-3.
    optim_generator_lr: float = 1e-3
    #: Momentum used by some optimizers (e.g., SGD) for the generator. Defaults to 0.9.
    optim_generator_momentum: float = 0.9
    #: Betas of Adam optimized (if used) for the generator. Defaults to (0.5, 0.999).
    optim_generator_betas: Tuple[float, float] = (0.5, 0.999)
    #: Weight decay parameter for the optimizer for the generator. Defaults to 0.
    optim_generator_weight_decay: float = 0.0
    #: Learning rate scheduler algorithm for the generator optimizer.
    #: Defaults to None (not used).
    lr_scheduler_generator: (
        Literal["step", "multistep", "constant", "linear", "exponential", "polynomial"] | None
    ) = None
    #: Learning rate scheduler step size, if needed by the scheduler. Defaults to 10 (epochs).
    lr_scheduler_generator_step_size: int | Iterable[int] = 10
    #: Learning rate scheduler step size, if needed by the scheduler.
    #: Usually this is used by the ExponentialLR.
    # : Defaults to 0.5.
    lr_scheduler_generator_gamma: float = 0.95

    #: Name of the optimizer to use for the discriminator. Defaults to 'adam'.
    optimizer_discriminator: Literal["adadelta", "adam", "adamw", "rmsprop", "sgd"] = "adam"
    #: Learning rate used by the optimizer for the discriminator. Defaults to 1e-3.
    optim_discriminator_lr: float = 1e-3
    #: Momentum used by some optimizers (e.g., SGD) for the discriminator. Defaults to 0.9.
    optim_discriminator_momentum: float = 0.9
    #: Betas of Adam optimized (if used) for the discriminator. Defaults to (0.5, 0.999).
    optim_discriminator_betas: Tuple[float, float] = (0.5, 0.999)
    #: Weight decay parameter for the optimizer for the discriminator. Defaults to 0.
    optim_discriminator_weight_decay: float = 0.0
    #: Learning rate scheduler algorithm for the discriminator optimizer.
    #: Defaults to None (not used).
    lr_scheduler_discriminator: (
        Literal["step", "multistep", "constant", "linear", "exponential", "polynomial"] | None
    ) = None
    #: Learning rate scheduler step size, if needed by the scheduler. Defaults to 10 (epochs).
    lr_scheduler_discriminator_step_size: int | Iterable[int] = 10
    #: Learning rate scheduler step size, if needed by the scheduler.
    #: Usually this is used by the ExponentialLR.
    # : Defaults to 0.5.
    lr_scheduler_discriminator_gamma: float = 0.95

    #: Classification criterion to be used for generator and discriminator losses. Defaults to
    #: "bceloss".
    loss: str = "bceloss"

    #: Generator input size (random noise size). Defaults to 100.
    z_dim: int = 100


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
        ray_scaling_config (ScalingConfig, optional): scaling config for Ray Trainer.
            Defaults to None,
        ray_tune_config (TuneConfig, optional): tune config for Ray Tuner.
            Defaults to None.
        ray_run_config (RunConfig, optional): run config for Ray Trainer.
            Defaults to None.
        ray_search_space (Dict[str, Any], optional): search space for Ray Tuner.
            Defaults to None.
        ray_torch_config (TorchConfig, optional): torch configuration for Ray's TorchTrainer.
            Defaults to None.
        ray_data_config (DataConfig, optional): dataset configuration for Ray.
            Defaults to None.
        ray_horovod_config (HorovodConfig, optional): horovod configuration for Ray's
            HorovodTrainer. Defaults to None.
        from_checkpoint (str | Path, optional): path to checkpoint directory. Defaults to None.
    """

    #: PyTorch generator to train.
    generator: nn.Module | None = None
    #: PyTorch discriminator to train.
    discriminator: nn.Module | None = None
    #: Classification loss criterion used in the generator and discriminator losses.
    loss: Callable | None = None
    #: Optimizer for the generator.
    optimizer_generator: Optimizer | None = None
    #: Optimizer for the discriminator.
    optimizer_discriminator: Optimizer | None = None
    #: Learning rate scheduler for the optimizer of the generator.
    lr_scheduler_generator: LRScheduler = None
    #: Learning rate scheduler for the optimizer of the discriminator.
    lr_scheduler_discriminator: LRScheduler = None

    def __init__(
        self,
        config: Dict | GANTrainingConfiguration,
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
        ray_scaling_config: ScalingConfig | None = None,
        ray_tune_config: TuneConfig | None = None,
        ray_run_config: RunConfig | None = None,
        ray_search_space: Dict[str, Any] | None = None,
        ray_torch_config: TorchConfig | None = None,
        ray_data_config: DataConfig | None = None,
        ray_horovod_config: Optional["HorovodConfig"] = None,
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
            ray_horovod_config=ray_horovod_config,
            from_checkpoint=from_checkpoint,
            ray_torch_config=ray_torch_config,
            ray_data_config=ray_data_config,
            **kwargs,
        )
        self.save_parameters(**self.locals2params(locals()))

        self.discriminator = discriminator
        self.generator = generator

        if isinstance(config, dict):
            config = GANTrainingConfiguration(**config)
        self.config = config

        # Initial training state -- can be resumed from a checkpoint
        self.discriminator_state_dict = None
        self.generator_state_dict = None
        self.optimizer_discriminator_state_dict = None
        self.optimizer_generator_state_dict = None
        self.lr_scheduler_generator_state_dict = None
        self.lr_scheduler_discriminator_state_dict = None

    def _optimizer_from_config(self) -> None:
        match self.config.optimizer_generator:
            case "adadelta":
                self.optimizer_generator = optim.Adadelta(
                    self.generator.parameters(),
                    lr=self.config.optim_generator_lr,
                    weight_decay=self.config.optim_generator_weight_decay,
                )
            case "adam":
                self.optimizer_generator = optim.Adam(
                    self.generator.parameters(),
                    lr=self.config.optim_generator_lr,
                    betas=self.config.optim_generator_betas,
                    weight_decay=self.config.optim_generator_weight_decay,
                )
            case "adamw":
                self.optimizer_generator = optim.AdamW(
                    self.generator.parameters(),
                    lr=self.config.optim_generator_lr,
                    betas=self.config.optim_generator_betas,
                    weight_decay=self.config.optim_generator_weight_decay,
                )
            case "rmsprop":
                self.optimizer_generator = optim.RMSprop(
                    self.generator.parameters(),
                    lr=self.config.optim_generator_lr,
                    weight_decay=self.config.optim_generator_weight_decay,
                    momentum=self.config.optim_generator_momentum,
                )
            case "sgd":
                self.optimizer_generator = optim.SGD(
                    self.generator.parameters(),
                    lr=self.config.optim_generator_lr,
                    weight_decay=self.config.optim_generator_weight_decay,
                    momentum=self.config.optim_generator_momentum,
                )
            case _:
                raise ValueError(
                    "Unrecognized self.config.optimizer_generator! Check the docs for "
                    "supported values and consider overriding "
                    "create_model_loss_optimizer method for more flexibility."
                )

        match self.config.optimizer_discriminator:
            case "adadelta":
                self.optimizer_discriminator = optim.Adadelta(
                    self.discriminator.parameters(),
                    lr=self.config.optim_discriminator_lr,
                    weight_decay=self.config.optim_discriminator_weight_decay,
                )
            case "adam":
                self.optimizer_discriminator = optim.Adam(
                    self.discriminator.parameters(),
                    lr=self.config.optim_discriminator_lr,
                    betas=self.config.optim_discriminator_betas,
                    weight_decay=self.config.optim_discriminator_weight_decay,
                )
            case "adamw":
                self.optimizer_discriminator = optim.AdamW(
                    self.discriminator.parameters(),
                    lr=self.config.optim_discriminator_lr,
                    betas=self.config.optim_discriminator_betas,
                    weight_decay=self.config.optim_discriminator_weight_decay,
                )
            case "rmsprop":
                self.optimizer_discriminator = optim.RMSprop(
                    self.discriminator.parameters(),
                    lr=self.config.optim_discriminator_lr,
                    weight_decay=self.config.optim_discriminator_weight_decay,
                    momentum=self.config.optim_discriminator_momentum,
                )
            case "sgd":
                self.optimizer_discriminator = optim.SGD(
                    self.discriminator.parameters(),
                    lr=self.config.optim_discriminator_lr,
                    weight_decay=self.config.optim_discriminator_weight_decay,
                    momentum=self.config.optim_discriminator_momentum,
                )
            case _:
                raise ValueError(
                    "Unrecognized self.config.optimizer_discriminator! Check the docs for "
                    "supported values and consider overriding "
                    "create_model_loss_optimizer method for more flexibility."
                )

    def _lr_scheduler_from_config(self) -> None:
        """Parse Lr scheduler from training config"""
        if self.config.lr_scheduler_generator:
            if not self.optimizer_generator:
                raise ValueError(
                    "Trying to instantiate a LR scheduler but the optimizer_generator is None!"
                )

            match self.config.lr_scheduler_generator:
                case "constant":
                    self.lr_scheduler_generator = lr_scheduler.ConstantLR(
                        self.optimizer_generator
                    )
                case "polynomial":
                    self.lr_scheduler_generator = lr_scheduler.PolynomialLR(
                        self.optimizer_generator
                    )
                case "exponential":
                    self.lr_scheduler_generator = lr_scheduler.ExponentialLR(
                        self.optimizer_generator,
                        gamma=self.config.lr_scheduler_generator_gamma,
                    )
                case "linear":
                    self.lr_scheduler_generator = lr_scheduler.LinearLR(
                        self.optimizer_generator
                    )
                case "multistep":
                    self.lr_scheduler_generator = lr_scheduler.MultiStepLR(
                        self.optimizer_generator,
                        milestones=self.config.lr_scheduler_generator_step_size,
                    )
                case "step":
                    self.lr_scheduler_generator = lr_scheduler.StepLR(
                        self.optimizer_generator,
                        step_size=self.config.lr_scheduler_generator_step_size,
                    )
                case _:
                    raise ValueError(
                        "Unrecognized self.config.lr_scheduler_generator! Check the docs for "
                        "supported values and consider overriding "
                        "create_model_loss_optimizer method for more flexibility."
                    )

        if self.config.lr_scheduler_discriminator:
            if not self.optimizer_discriminator:
                raise ValueError(
                    "Trying to instantiate a LR scheduler but the optimizer_discriminator "
                    "is None!"
                )

            match self.config.lr_scheduler_discriminator:
                case "constant":
                    self.lr_scheduler_discriminator = lr_scheduler.ConstantLR(
                        self.optimizer_discriminator
                    )
                case "polynomial":
                    self.lr_scheduler_discriminator = lr_scheduler.PolynomialLR(
                        self.optimizer_discriminator
                    )
                case "exponential":
                    self.lr_scheduler_discriminator = lr_scheduler.ExponentialLR(
                        self.optimizer_discriminator,
                        gamma=self.config.lr_scheduler_discriminator_gamma,
                    )
                case "linear":
                    self.lr_scheduler_discriminator = lr_scheduler.LinearLR(
                        self.optimizer_discriminator
                    )
                case "multistep":
                    self.lr_scheduler_discriminator = lr_scheduler.MultiStepLR(
                        self.optimizer_discriminator,
                        milestones=self.config.lr_scheduler_discriminator_step_size,
                    )
                case "step":
                    self.lr_scheduler_discriminator = lr_scheduler.StepLR(
                        self.optimizer_discriminator,
                        step_size=self.config.lr_scheduler_discriminator_step_size,
                    )
                case _:
                    raise ValueError(
                        "Unrecognized self.config.lr_scheduler_discriminator! Check the "
                        "docs for supported values and consider overriding "
                        "create_model_loss_optimizer method for more flexibility."
                    )

    def create_model_loss_optimizer(self) -> None:
        """Instantiate a torch model, loss, optimizer, and LR scheduler using the
        configuration provided in the Trainer constructor.
        Generally a user-defined method.
        """
        ###################################
        # Dear user, this is a method you #
        # may be interested to override!  #
        ###################################

        # Model, optimizer, and lr scheduler may have already been loaded from a checkpoint

        if self.generator is None or self.discriminator is None:
            raise ValueError(
                "self.generator or self.discrimintaor is None! "
                "Either pass it to the constructor, load a checkpoint, or "
                "override create_model_loss_optimizer method."
            )

        if self.generator_state_dict:
            # Load generator from checkpoint
            self.generator.load_state_dict(self.generator_state_dict, strict=False)

        if self.discriminator_state_dict:
            # Load discriminator from checkpoint
            self.discriminator.load_state_dict(self.discriminator_state_dict, strict=False)

        # Parse optimizers from training configuration
        # Optimizers can be changed with a custom one here!
        self._optimizer_from_config()

        # Parse LR schedulers from training configuration
        # LR schedulers can be changed with a custom one here!
        self._lr_scheduler_from_config()

        if self.optimizer_generator_state_dict:
            # Load optimizer state from checkpoint
            # IMPORTANT: this must be after the learning rate scheduler was already initialized
            # by passing to it the optimizer. Otherwise the optimizer state just loaded will
            # be modified by the lr scheduler.
            self.optimizer_generator.load_state_dict(self.optimizer_generator_state_dict)
        if self.optimizer_discriminator_state_dict:
            # Load optimizer state from checkpoint
            # IMPORTANT: this must be after the learning rate scheduler was already initialized
            # by passing to it the optimizer. Otherwise the optimizer state just loaded will
            # be modified by the lr scheduler.
            self.optimizer_discriminator.load_state_dict(
                self.optimizer_discriminator_state_dict
            )

        if self.lr_scheduler_generator_state_dict and self.lr_scheduler_generator:
            # Load LR scheduler state from checkpoint
            self.lr_scheduler_generator.load_state_dict(self.lr_scheduler_generator_state_dict)
        if self.lr_scheduler_discriminator_state_dict and self.lr_scheduler_discriminator:
            # Load LR scheduler state from checkpoint
            self.lr_scheduler_discriminator.load_state_dict(
                self.lr_scheduler_discriminator_state_dict
            )

        # Parse loss from training configuration
        # Loss can be change with a custom one here!
        self._loss_from_config()
        self.criterion = self.loss

        # if not self.optimizer_discriminator:
        #     self.optimizer_discriminator = optim.Adam(
        #         self.discriminator.parameters(), lr=self.config.lr, betas=(0.5, 0.999)
        #     )
        # if not self.optimizer_generator:
        #     self.optimizer_generator = optim.Adam(
        #         self.generator.parameters(), lr=self.config.lr, betas=(0.5, 0.999)
        #     )
        # self.criterion = nn.BCELoss()

        # https://stackoverflow.com/a/67437077
        self.discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.discriminator)
        self.generator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.generator)

        # IMPORTANT: model, optimizer, and scheduler need to be distributed from here on

        distribute_kwargs = self.get_default_distributed_kwargs()

        # Distribute discriminator and its optimizer
        self.discriminator, self.optimizer_discriminator, _ = self.strategy.distributed(
            self.discriminator, self.optimizer_discriminator, **distribute_kwargs
        )
        self.generator, self.optimizer_generator, _ = self.strategy.distributed(
            self.generator, self.optimizer_generator, **distribute_kwargs
        )

    def save_checkpoint(
        self,
        name: str,
        best_validation_metric: Optional[torch.Tensor] = None,
        checkpoints_root: str | Path | None = None,
        force: bool = False,
    ) -> str | None:
        """Save training checkpoint.

        Args:
            name (str): name of the checkpoint directory.
            best_validation_metric (Optional[torch.Tensor]): best validation loss throughout
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
            "best_validation_metric": (
                best_validation_metric.item() if best_validation_metric is not None else None
            ),
            "optimizer_generator_state_dict": self.optimizer_generator.state_dict(),
            "optimizer_discriminator_state_dict": self.optimizer_discriminator.state_dict(),
            "lr_scheduler_generator_state_dict": (
                self.lr_scheduler_generator.state_dict()
                if self.lr_scheduler_generator is not None
                else None
            ),
            "lr_scheduler_discriminator_state_dict": (
                self.lr_scheduler_discriminator.state_dict()
                if self.lr_scheduler_discriminator is not None
                else None
            ),
            "torch_rng_state": self.torch_rng.get_state(),
            "random_seed": self.random_seed,
        }
        state_path = ckpt_dir / "state.pt"
        torch.save(state, state_path)

        # Save PyTorch models separately
        # TODO: check that the state dict is stripped from any distributed info
        generator_path = ckpt_dir / "generator.pt"
        torch.save(self.generator.state_dict(), generator_path)
        discriminator_path = ckpt_dir / "discriminator.pt"
        torch.save(self.discriminator.state_dict(), discriminator_path)

        # Save Pydantic config as YAML
        config_path = ckpt_dir / "config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(self.config.model_dump(), f)

        # Log each file with an appropriate identifier
        self.log(str(state_path), f"{name}_state", kind="artifact")
        self.log(str(generator_path), f"{name}_generator", kind="artifact")
        self.log(str(discriminator_path), f"{name}_discriminator", kind="artifact")
        self.log(str(config_path), f"{name}_config", kind="artifact")
        return str(ckpt_dir)

    def _load_checkpoint(self, checkpoint_dir: str | Path) -> None:
        """Load checkpoint from path."""
        checkpoint_dir = Path(checkpoint_dir)
        state = torch.load(checkpoint_dir / "state.pt")

        # Override initial training state
        self.generator_state_dict_state_dict = torch.load(checkpoint_dir / "generator.pt")
        self.discriminator_state_dict = torch.load(checkpoint_dir / "discriminator.pt")
        self.optimizer_generator_state_dict = state["optimizer_generator_state_dict"]
        self.optimizer_discriminator_state_dict = state["optimizer_discriminator_state_dict"]
        self.lr_scheduler_generator_state_dict = state["lr_scheduler_generator_state_dict"]
        self.lr_scheduler_discriminator_state_dict = state[
            "lr_scheduler_discriminator_state_dict"
        ]
        self.torch_rng_state = state["torch_rng_state"]
        # Direct overrides (don't require further attention)
        self.random_seed = state["random_seed"]
        self.epoch = state["epoch"] + 1  # Start from next epoch
        if state["best_validation_metric"]:
            self.best_validation_metric = state["best_validation_metric"]

    def train_epoch(self):
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
            step=self.epoch,
        )
        avg_gen_loss = torch.mean(torch.stack(gen_train_losses))
        self.log(
            item=avg_gen_loss.item(),
            identifier="gen_train_loss_per_epoch",
            kind="metric",
            step=self.epoch,
        )

        avg_disc_loss = torch.mean(torch.stack(disc_train_losses))
        self.log(
            item=avg_disc_loss.item(),
            identifier="disc_train_loss_per_epoch",
            kind="metric",
            step=self.epoch,
        )

        self.save_fake_generator_images()

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

        self.optimizer_discriminator.zero_grad()
        lossD.backward()
        self.optimizer_discriminator.step()

        accuracy = ((output_real > 0.5).float() == real_labels).float().mean() + (
            (output_fake < 0.5).float() == fake_labels
        ).float().mean()
        accuracy_disc = accuracy.mean()

        # Train Generator
        output_fake = self.discriminator(fake_images)
        lossG = self.criterion(output_fake, real_labels)
        self.optimizer_generator.zero_grad()
        lossG.backward()
        self.optimizer_generator.step()
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

    def validation_epoch(self) -> torch.Tensor:
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
            step=self.epoch,
        )
        disc_validation_accuracy = torch.mean(torch.stack(disc_validation_accuracy))
        self.log(
            item=disc_validation_accuracy.item(),
            identifier="disc_valid_accuracy_epoch",
            kind="metric",
            step=self.epoch,
        )
        gen_validation_loss = torch.mean(torch.stack(gen_validation_losses))
        self.log(
            item=gen_validation_loss.item(),
            identifier="gen_valid_loss_per_epoch",
            kind="metric",
            step=self.epoch,
        )
        gen_validation_accuracy = torch.mean(torch.stack(gen_validation_accuracy))
        self.log(
            item=gen_validation_accuracy.item(),
            identifier="gen_valid_accuracy_epoch",
            kind="metric",
            step=self.epoch,
        )
        # TODO: return IS or FID metrics instead, more suitable for GAN validation
        return gen_validation_loss

    def validation_step(
        self, real_images, batch_idx
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        d_accuracy = accuracy / 2

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
            item=d_accuracy.item(),
            identifier="disc_valid_accuracy_per_batch",
            kind="metric",
            step=self.validation_glob_step,
            batch_idx=batch_idx,
        )
        return loss_gen, accuracy_gen, d_total_loss, d_accuracy

    def save_fake_generator_images(self):
        """Plot and save fake images from generator

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
        ax.set_title(f"Fake images for epoch {self.epoch}")
        ax.imshow(np.transpose(fake_images_grid.cpu().numpy(), (1, 2, 0)))
        self.log(
            item=fig,
            identifier=f"fake_images_epoch_{self.epoch}.png",
            kind="figure",
            step=self.epoch,
        )
