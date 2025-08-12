# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Javad Komijani - ETHZ
#
# Credit:
# - Javad Komijani <jkomijani@gmail.com> - ETHZ
# - Gaurav Sinha Ray <sinha@ifca.unican.es> - CSIC
# - Rakesh Sarma <r.sarma@fz-juelich.de> - Juelich
# --------------------------------------------------------------------------------------

# Copyright (c) 2021-2024 Javad Komijani

"""This module contains high-level classes for normalizing flow techniques,
with the central `Model` class integrating essential components such as priors,
networks, and actions. It provides utilities for training and sampling,
along with support for MCMC sampling. It also integrates with the itwinai package
to perform distributed training, profiling and logging.
"""

import torch
import torch.distributed as dist
import time
import logging, os, sys
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional, Tuple
from torch.utils.data import Dataset
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np

from itwinai.constants import EPOCH_TIME_DIR

from .mcmc import MCMCSampler, BlockedMCMCSampler
from .lib.combo import estimate_logz, fmt_val_err
from normflow.prior import Prior
from normflow.nn import ModuleList_
from normflow.action.scalar_action import ScalarPhi4Action

from itwinai.torch.trainer import TorchTrainer
from itwinai.torch.distributed import DeepSpeedStrategy
from itwinai.loggers import EpochTimeTracker, Logger
from itwinai.torch.monitoring.monitoring import measure_gpu_utilization
from itwinai.torch.profiling.profiler import profile_torch_trainer
from itwinai.distributed import suppress_workers_print
from itwinai.torch.config import TrainingConfiguration


class Model:
    """The central high-level class of the package, which integrates instances of
    essential classes (`prior`, `net_`, and `action`) to provide utilities for
    training and sampling. This class interfaces with various core components
    to facilitate training, posterior inference and MCMC sampling.

    Args:
        prior (Prior): instance of a `Prior` class
            An instance of a Prior class (e.g., `NormalPrior`) representing the
            model's prior distribution.
        net_ (Module_): instance of a `Module_` class
            A model component responsible for the transformations required in the
            model. The trailing underscore indicates that the associated forward
            method computes and returns the Jacobian of the transformation, which
            is crucial in the method of normalizing flows.
        action (Action): instance of an `ScalarPhi4Action` class
            Defines the model's action, which specified the target distribution
            during training.

    Attributes:
        posterior (Posterior): An instance of the Posterior class,
            which manages posterior inference on the model parameters.
        mcmc (MCMCSampler): An instance of the MCMCSampler class, enabling
            MCMC sampling for posterior distributions.
        blocked_mcmc (BlockedMCMCSampler): An instance of the BlockedMCMCSampler class,
            providing blockwise MCMC sampling for improved sampling efficiency.
    """

    def __init__(self, *, prior: Prior, net_: ModuleList_, action: ScalarPhi4Action):
        self.net_ = net_
        self.prior = prior
        self.action = action

        self.posterior = Posterior(self)
        self.mcmc = MCMCSampler(self)
        self.blocked_mcmc = BlockedMCMCSampler(self)


class Posterior:
    """Creates samples directly from a trained probabilistic model.
    The `Posterior` class generates samples from a specified model without
    using an accept-reject step, making it suitable for tasks that require
    quick, direct sampling. All methods in this class use `torch.no_grad()`
    to prevent gradient computation.

    Args:
        model (Model): A trained model to sample from.
    """

    def __init__(self, model: Model):
        self._model = model

    @torch.no_grad()
    def sample(self, batch_size: int = 1, **kwargs) -> torch.Tensor:
        """Draws samples from the model.

        Args:
            batch_size (int, optional): Number of samples to draw. Defaults to 1.

        Returns:
            torch.Tensor: Generated samples.
        """
        return self.sample_(batch_size=batch_size, **kwargs)[0]

    @torch.no_grad()
    def sample_(
        self,
        batch_size: int = 1,
        preprocess_func: Callable | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draws samples and their log probabilities from the model.

        Args:
            batch_size (int, optional): Number of samples to draw. Defaults to 1.
            preprocess_func (Callable, optional): A function to adjust the prior
                samples if needed. It should take samples and log probabilities as
                input and return modified values.

        Returns:
            torch.Tensor: Generated Samples (`y`)
            torch.Tensor: Log probabilities of the samples (`logq`)
        """
        x, logr = self._model.prior.sample_(batch_size)

        if preprocess_func is not None:
            x, logr = preprocess_func(x, logr)

        y, logj = self._model.net_(x)
        logq = logr - logj
        return y, logq

    @torch.no_grad()
    def sample_log(
        self,
        batch_size: int = 1,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Similar to `sample_`, but also returns the log probability of the
        target distribution from `model.action`.

        Args:
            batch_size (int, optional): Number of samples to draw. Defaults to 1.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - `y`: Generated samples.
                - `logq`: Log probabilities of the samples.
                - `logp`: Log probabilities from the target distribution.
        """
        y, logq = self.sample_(batch_size=batch_size, **kwargs)
        logp = -self._model.action(y)  # logp is log(p_{non-normalized})
        return y, logq, logp

    @torch.no_grad()
    def log_prob(self, y: torch.Tensor) -> torch.Tensor:
        """Computes the log probability of the provided samples.

        Args:
            y (torch.Tensor): Samples for which to calculate the log probability.

        Returns:
            torch.Tensor: Log probabilities of the samples.
        """
        x, minus_logj = self._model.net_.reverse(y)
        logr = self._model.prior.log_prob(x)
        logq = logr + minus_logj
        return logq


class Fitter(TorchTrainer):
    """A class for training a given model."""
    def __init__(
        self,
        model: Model,
        epochs: int,
        config: Dict | TrainingConfiguration | None = None,
        strategy: Literal["ddp", "deepspeed", "horovod"] = 'ddp',
        logger: Logger | None = None,
        **kwargs
    ):
        super().__init__(
            config=config,
            epochs=epochs,
            strategy=strategy,
            logger=logger,
            **kwargs
        )
        self._model = model
        self.epochs = epochs
        # Global training configuration
        if isinstance(config, dict):
            config = TrainingConfiguration(**config)
        self.config = config
        self.checkpoint_dict = dict(
            display=self.config.ckpt_disp,
            print_stride=self.config.print_stride,
            print_batch_size=self.config.print_batch_size,
            snapshot_path=self.config.snapshot_path or 'checkpoint.pth',
            epochs_run=self.config.epochs_run
        )
        self.train_batch_size = 1
        self.train_history = dict(
            loss=[], logqp=[], logz=[], ess=[], rho=[], accept_rate=[]
        )
        self.hyperparam = dict(lr=self.config.optim_lr, weight_decay=self.config.weight_decay)

    def setup_seed(self, rank: int, world_size: int) -> None:
        """Sets up random seed for each worker in a distributed setting.
        This methods ensures that each workers receives a unique seed. The main worker
        generates the seeds and broadcasts to other workers.

        Args:
            rank (int): The rank of the current worker.
            world_size (int): The total number of workers in the distributed setting
        """
        if self.strategy.is_main_worker:
            # Generate unique seed for each worker based on its rank
            seeds_torch = [torch.randint(2**32 - 1, (1,)).item() for _ in range(world_size)]
            print(f"Generated seeds for workers: {seeds_torch}\n")
        else:
            seeds_torch = [None] * world_size

        # Broadcast seed list to all workers
        seeds_torch_list = self.strategy.allgather_obj(seeds_torch)
        seeds_torch = next(s for s in seeds_torch_list if s is not None)

        # Set seed for current worker based on its rank
        seed = seeds_torch[rank]
        torch.manual_seed(seed)

        print(f"Rank {rank} has been assigned seed {seed}\n")

    def create_dataloaders(
        self,
        train_dataset: Dataset,
        validation_dataset: Dataset | None = None,
        test_dataset: Dataset | None = None
    ) -> None:
        """Overrides create_dataloaders method from itwinai TorchTrainer and returns
        nothing since this is not needed in this use-case.
        """
        return

    def create_model_loss_optimizer(self) -> None:
        """Creates create_model_loss_optimizer method to setup model, seeds,
        loss function and optimizer.
        """
        self._model.prior.to(device=self.device)

        # First, define strategy-wise optional configurations
        if isinstance(self.strategy, DeepSpeedStrategy):
            # Batch size definition is not optional for DeepSpeedStrategy!
            distribute_kwargs = dict(
                config_params=dict(
                    train_micro_batch_size_per_gpu=self.config.batch_size
                )
            )
        else:
            distribute_kwargs = {}

        optimizer_class = eval(self.config.optimizer_class)
        if '_groups' in self._model.net_.__dict__.keys():
            parameters = self._model.net_.grouped_parameters()
        else:
            parameters = self._model.net_.parameters()
        self.optimizer = optimizer_class(parameters, **self.hyperparam)

        # Distributed model, optimizer, and scheduler
        self._model.net_, self.optimizer, _ = self.strategy.distributed(
            self._model.net_, self.optimizer, **distribute_kwargs
        )

        # Call setup_seed to set seed with current rank and world_size
        self.setup_seed(self.strategy.global_rank(), self.strategy.global_world_size())

        # decide whether to load a snapshot
        snapshot_path = self.checkpoint_dict['snapshot_path']
        if (snapshot_path is not None) and os.path.exists(snapshot_path):
            print(f"Trying to load snapshot from {snapshot_path}")
            self._load_snapshot()

        self.loss_fn = Fitter.calc_kl_mean if self.config.loss_fn == "None" else self.config.loss_fn

        if not self.config.scheduler or self.config.scheduler == "None":
            self.scheduler = None
        else:
            try:
                scheduler_class = getattr(lr_scheduler, self.config.scheduler)
                self.scheduler = scheduler_class(self.optimizer)
            except AttributeError:
                raise ValueError(f"Invalid scheduler name: {self.config.scheduler}")

    @suppress_workers_print
    def execute(
        self,
        train_dataset: Dataset | None = None,
        validation_dataset: Dataset | None = None,
        test_dataset: Dataset | None = None
    ) -> None:
        """Overrides execute method from itwinai TorchTrainer class to ignore
        train_dataset since it is not needed in this usecase.

        Args:
            train_dataset (Dataset | None): training dataset object. Defaults to None.
            validation_dataset (Dataset | None): validation dataset object.
                Defaults to None.
            test_dataset (Dataset | None): test dataset object. Defaults to None.
        """
        super().execute(train_dataset, validation_dataset, test_dataset)

    def _load_snapshot(self) -> None:
        """Method to load a snapshot from a path provided in checkpoint_dict"""
        snapshot_path = self.checkpoint_dict['snapshot_path']
        if torch.cuda.is_available():
            gpu_id = self.strategy.global_rank()
            loc = f"cuda:{gpu_id}"
            print(f"GPU: Attempting to load saved model into {loc}")
        else:
            loc = None  # cpu training
            print("CPU: Attempting to load saved model")
        snapshot = torch.load(snapshot_path, map_location=loc)
        self._model.net_.load_state_dict(snapshot["MODEL_STATE"])
        self.checkpoint_dict['epochs_run'] = snapshot['EPOCHS_RUN']
        print(f"Snapshot found: {snapshot_path}\nResuming training via Saved Snapshot at Epoch {snapshot['EPOCHS_RUN']}")

    def _save_snapshot(self, epoch: int) -> None:
        """Save snapshot of training for analysis and/or to continue training at a later date.

        Args:
            epoch (int): Number of the epoch at which the snapshot is saved.
        """
        snapshot_path = self.checkpoint_dict['snapshot_path']
        epochs_run = epoch + self.checkpoint_dict['epochs_run']
        snapshot_new_path = snapshot_path.rsplit('.',2)[0] + ".E" + str(epochs_run) + ".tar"
        snapshot = {
            "MODEL_STATE": self._model.net_.state_dict(),
            "EPOCHS_RUN": epochs_run
        }
        torch.save(snapshot, snapshot_new_path)
        print(f"Epoch {epochs_run} | Model Snapshot saved at {snapshot_new_path}")

    def set_epoch(self, epoch: int) -> None:
        """Sets current epoch at beginning of training.

        Args:
            epoch (int): current epoch number.
        """
        if self.profiler is not None:
            self.profiler.step()

    @profile_torch_trainer
    @measure_gpu_utilization
    def train(self) -> None:
        """Trains the neural network model."""

        # Track epoch time for scaling statistics
        if self.strategy.is_main_worker and self.strategy.is_distributed:
            try:
                num_nodes = int(os.environ["SLURM_NNODES"])
            except (KeyError, ValueError):
                logging.warning("SLURM_NNODES is not set or invalid; defaulting num_nodes to 1.")
                num_nodes = 1
            epoch_time_output_dir = Path(f"scalability-metrics/{EPOCH_TIME_DIR}")
            epoch_time_file_name = f"epochtime_{self.strategy.name}_{num_nodes}N.csv"
            epoch_time_output_path = epoch_time_output_dir / epoch_time_file_name

            epoch_time_logger= EpochTimeTracker(
                strategy_name=self.strategy.name,
                save_path=epoch_time_output_path,
                num_nodes=num_nodes,
                should_log=self.measure_epoch_time
            )

        if self.config.save_every == "None":
            self.config.save_every = self.epochs

        start_time = time.time()
        for epoch in range(1, self.epochs+1):
            self.set_epoch(epoch)
            loss, logqp = self.step(self.config.batch_size)
            self.checkpoint(epoch, loss, self.config.save_every)
            if self.scheduler is not None:
                self.scheduler.step()
        end_time = time.time()
        if self.strategy.is_main_worker:
            print(f"({self.strategy.device()}) Time = {end_time - start_time:.3g} sec.")

    def step(self, batch_size: int) -> None:
        """Perform a train step with a batch of inputs.

        Args:
            batch_size (int): Specifies the batch size for the training.
        """
        net_ = self._model.net_
        prior = self._model.prior
        action = self._model.action

        x, logr = prior.sample_(batch_size)
        y, logJ = net_(x)
        logq = logr - logJ
        logp = -action(y)
        loss = self.loss_fn(logq, logp)

        # clears old gradients from last steps
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss, logq - logp

    def checkpoint(self, epoch: int, loss: torch.Tensor, save_every: int) -> None:
        """Handles model checkpointing and logging during training.

        Args:
            epoch (int): Epoch at which checkpoint is created.
            loss (torch.Tensor): Loss value at the epoch.
            save_every (int): interval at which checkpoints are saved.
        """
        print_stride = self.checkpoint_dict['print_stride']
        print_batch_size = self.checkpoint_dict['print_batch_size']
        snapshot_path = self.checkpoint_dict['snapshot_path']

        # Always save loss on rank 0
        if self.strategy.is_main_worker:
            self.train_history['loss'].append(loss.item())
            # Save model as well
            if snapshot_path is not None and (epoch % save_every == 0):
                self._save_snapshot(epoch)

        if epoch != 1 or epoch % print_stride != 0:
            return

        print_batch_size = print_batch_size // self.strategy.global_world_size()
        _, logq, logp = self._model.posterior.sample_log(print_batch_size)
        logq = self.strategy.gather(logq)
        logp = self.strategy.gather(logp)

        if not self.strategy.is_main_worker:
            return

        logq = torch.cat(logq, dim=0)
        logp = torch.cat(logp, dim=0)
        loss_ = self.loss_fn(logq, logp)
        self._append_to_train_history(logq, logp)
        self.print_fit_status(epoch, loss=loss_)

    @staticmethod
    def calc_kl_mean(logq: torch.Tensor, logp: torch.Tensor) -> torch.Tensor:
        """Return Kullback-Leibler divergence estimated from logq and logp.

        Args:
            logq (torch.Tensor): Log probabilities from the approximate distribution `q`.
            logp (torch.Tensor): Log probabilities from the target distribution `p`.

        Returns:
            torch.Tensor: Estimated KL divergence.
        """
        return (logq - logp).mean()  # KL, assuming samples from q

    @staticmethod
    def calc_kl_var(logq: torch.Tensor, logp: torch.Tensor) -> torch.Tensor:
        """Compute variance of the Kullback-Leibler (KL) divergence.

        Args:
            logq (torch.Tensor): Log probabilities from the approximate distribution `q`.
            logp (torch.Tensor): Log probabilities from the target distribution `p`.

        Returns:
            torch.Tensor: Variance of the KL divergence.
        """
        return (logq - logp).var()

    @staticmethod
    def calc_corrcoef(logq: torch.Tensor, logp: torch.Tensor) -> torch.Tensor:
        """Compute the Pearson correlation coefficient between logq and logp.

        Args:
            logq (torch.Tensor): Log probabilities from the approximate distribution `q`.
            logp (torch.Tensor): Log probabilities from the target distribution `p`.

        Returns:
            torch.Tensor: Pearson correlation coefficient.
        """
        return torch.corrcoef(torch.stack([logq, logp]))[0, 1]

    @staticmethod
    def calc_direct_kl_mean(logq: torch.Tensor, logp: torch.Tensor) -> torch.Tensor:
        """Compute the direct KL divergence.

        Args:
            logq (torch.Tensor): Log probabilities from the approximate distribution `q`.
            logp (torch.Tensor): Log probabilities from the target distribution `p`.

        Returns:
            torch.Tensor: KL divergence
        """
        logpq = logp - logq
        logz = torch.logsumexp(logpq, dim=0) - np.log(logp.shape[0])
        logpq = logpq - logz  # p is now normalized
        p_by_q = torch.exp(logpq)
        return (p_by_q * logpq).mean()

    @staticmethod
    def calc_minus_logz(logq: torch.Tensor, logp: torch.Tensor) -> torch.Tensor:
        """Compute the negative log partition function.

        Args:
            logq (torch.Tensor): Log probabilities from the approximate distribution `q`.
            logp (torch.Tensor): Log probabilities from the target distribution `p`.

        Returns:
            torch.Tensor: Negative log partition function.
        """
        logz = torch.logsumexp(logp - logq, dim=0) - np.log(logp.shape[0])
        return -logz

    @staticmethod
    def calc_ess(logq: torch.Tensor, logp: torch.Tensor) -> torch.Tensor:
        """Return effective sample size (ESS).

        Args:
            logq (torch.Tensor): Log probabilities from the approximate distribution `q`.
            logp (torch.Tensor): Log probabilities from the target distribution `p`.

        Returns:
            torch.Tensor: Estimated effective sample size (ESS).
        """
        logqp = logq - logp
        log_ess = 2*torch.logsumexp(-logqp, dim=0) - torch.logsumexp(-2*logqp, dim=0)
        ess = torch.exp(log_ess) / len(logqp)  # normalized
        return ess

    @staticmethod
    def calc_minus_logess(logq: torch.Tensor, logp: torch.Tensor) -> torch.Tensor:
        """Return logarithm of inverse of effective sample size.

        Args:
            logq (torch.Tensor): Log probabilities from the approximate distribution `q`.
            logp (torch.Tensor): Log probabilities from the target distribution `p`.

        Returns:
            torch.Tensor: logarithm of inverse of effective sample size.
        """
        logqp = logq - logp
        log_ess = 2*torch.logsumexp(-logqp, dim=0) - torch.logsumexp(-2*logqp, dim=0)
        return - log_ess + np.log(len(logqp))  # normalized

    @torch.no_grad()
    def _append_to_train_history(self, logq: torch.Tensor, logp: torch.Tensor) -> None:
        """Update train history with log probability estimates.

        Args:
            logq (torch.Tensor): Log probabilities from the approximate distribution `q`.
            logp (torch.Tensor): Log probabilities from the target distribution `p`.
        """
        logqp = logq - logp
        logz = estimate_logz(logqp, method='jackknife')  # returns (mean, std)
        accept_rate = self._model.mcmc.estimate_accept_rate(logqp)
        ess = self.calc_ess(logqp, 0)
        rho = self.calc_corrcoef(logq, logp)
        logqp = (logqp.mean().item(), logqp.std().item())
        self.train_history['logqp'].append(logqp)
        self.train_history['logz'].append(logz)
        self.train_history['ess'].append(ess)
        self.train_history['rho'].append(rho)
        self.train_history['accept_rate'].append(accept_rate)

    def print_fit_status(self, epoch: int, loss: torch.Tensor | None = None) -> None:
        """Print training progress and key metrics.

        Args:
            epoch (int): Current epoch.
            loss (torch.Tensor, optional): Loss value for the current epoch.
        """
        mydict = self.train_history
        if loss is None:
            loss = mydict['loss'][-1]
        logqp_mean, logqp_std = mydict['logqp'][-1]
        logz_mean, logz_std = mydict['logz'][-1]
        accept_rate_mean, accept_rate_std = mydict['accept_rate'][-1]
        # We now incorporate the effect of estimated log(z) to mean of log(q/p)
        adjusted_logqp_mean = logqp_mean + logz_mean
        ess = mydict['ess'][-1]
        rho = mydict['rho'][-1]

        if epoch == 1:
            print(f"\n>>> Training progress ({self.device}) <<<\n")
            print("Note: log(q/p) is estimated with normalized p; " \
                  + "mean & error are obtained from samples in a batch\n")

        epoch += self.checkpoint_dict['epochs_run']
        print(f"Epoch: {epoch} | loss: {loss:.4f} | ess: {ess:.4f}")
        self.log(loss.detach().cpu().numpy(),'epoch_loss',kind='metric')


# =============================================================================
@torch.no_grad()
def reverse_flow_sanitychecker(
    model: Model,
    n_samples: int = 4,
    net_: torch.nn.Module | None = None
) -> None:
    """Performs a sanity check on the reverse method of modules.
    Args:
        model (Model): Model containing prior and transformation networks.
        n_samples (int, optional): Number of samples to test. Defaults to 4.
        net_ (torch.nn.Module, optional): The transformation network.
    """
    if net_ is None:
        net_ = model.net_

    x = model.prior.sample(n_samples)
    y, logj = net_(x)
    x_hat, minus_logj = net_.backward(y)

    mean = lambda z: z.abs().mean().item()

    print("reverse method is OK if following values vanish (up to round off):")
    print(f"{mean(x - x_hat):g} & {mean(1 + minus_logj / logj):g}")
