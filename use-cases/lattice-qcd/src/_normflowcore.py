# Copyright (c) 2021-2024 Javad Komijani

"""
This module contains high-level classes for normalizing flow techniques,
with the central `Model` class integrating essential components such as priors,
networks, and actions. It provides utilities for training and sampling,
along with support for MCMC sampling and device management.
"""

import torch
import torch.distributed as dist
import time
import os, sys
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple
from torch.utils.data import Dataset

import numpy as np

from .mcmc import MCMCSampler, BlockedMCMCSampler
from .lib.combo import estimate_logz, fmt_val_err
from normflow.prior import Prior
from normflow.nn import ModuleList_
from normflow.action.scalar_action import ScalarPhi4Action

from itwinai.torch.trainer import TorchTrainer
from itwinai.torch.distributed import DeepSpeedStrategy
from itwinai.loggers import Logger
from itwinai.torch.profiling.profiler import profile_torch_trainer
from itwinai.torch.config import TrainingConfiguration


class Model:
    """
    The central high-level class of the package, which integrates instances of
    essential classes (`prior`, `net_`, and `action`) to provide utilities for
    training and sampling. This class interfaces with various core components
    to facilitate training, posterior inference, MCMC sampling, and device
    management.

    Parameters
    ----------
    prior : instance of a `Prior` class
        An instance of a Prior class (e.g., `NormalPrior`) representing the
        model's prior distribution.

    net_ : instance of a `Module_` class
        A model component responsible for the transformations required in the
        model. The trailing underscore indicates that the associated forward
        method computes and returns the Jacobian of the transformation, which
        is crucial in the method of normalizing flows.

    action : instance of an `Action` class
        Defines the model's action, which specified the target distribution
        during training.

    Attributes
    ----------
    fit : Fitter
        An instance of the Fitter class, responsible for training the model.
        `fit` is aliased to `train` for flexibility in usage.

    posterior : Posterior
        An instance of the Posterior class, which manages posterior inference
        on the model parameters.

    mcmc : MCMCSampler
        An instance of the MCMCSampler class, enabling MCMC sampling for
        posterior distributions.

    blocked_mcmc : BlockedMCMCSampler
        An instance of the BlockedMCMCSampler class, providing blockwise
        MCMC sampling for improved sampling efficiency.

    device_handler : ModelDeviceHandler
        Manages the device (CPU/GPU) for model training and inference, ensuring
        seamless operation across hardware setups.
    """

    def __init__(self,
        *,
        prior: Prior,
        net_: ModuleList_,
        action: ScalarPhi4Action,
        config=None,
        epochs=1000,
        logger=None
    ):
        self.net_ = net_
        self.prior = prior
        self.action = action

        # Components for training, sampling, and device handling
        self.fit = Fitter(self, config=config, epochs=epochs, logger=logger)
        self.train = self.fit  # Alias for `fit`

        self.posterior = Posterior(self)
        self.mcmc = MCMCSampler(self)
        self.blocked_mcmc = BlockedMCMCSampler(self)


class Posterior:
    """
    Creates samples directly from a trained probabilistic model.

    The `Posterior` class generates samples from a specified model without
    using an accept-reject step, making it suitable for tasks that require
    quick, direct sampling. All methods in this class use `torch.no_grad()`
    to prevent gradient computation.

    Parameters
    ----------
    model : Model
        A trained model to sample from.

    Methods
    -------
    sample(batch_size=1, **kwargs)
        Returns a specified number of samples from the model.

    sample_(batch_size=1, preprocess_func=None)
        Returns samples and their log probabilities, with an optional
        preprocessing function.

    sample__(batch_size=1, **kwargs)
        Similar to `sample_`, but also returns the log probability of the
        target distribution.

    log_prob(y)
        Computes the log probability of given samples.
    """

    def __init__(self, model: Model):
        self._model = model

    @torch.no_grad()
    def sample(self, batch_size=1, **kwargs):
        """
        Draws samples from the model.

        Parameters
        ----------
        batch_size : int, optional
            Number of samples to draw, default is 1.

        Returns
        -------
        Tensor
            Generated samples.
        """
        return self.sample_(batch_size=batch_size, **kwargs)[0]

    @torch.no_grad()
    def sample_(self, batch_size=1, preprocess_func=None):
        """
        Draws samples and their log probabilities from the model.

        Parameters
        ----------
        batch_size : int, optional
            Number of samples to draw, default is 1.

        preprocess_func : function or None, optional
            A function to adjust the prior samples if needed. It should take
            samples and log probabilities as input and return modified values.

        Returns
        -------
        tuple
            - `y`: Generated samples.
            - `logq`: Log probabilities of the samples.
        """
        x, logr = self._model.prior.sample_(batch_size)

        if preprocess_func is not None:
            x, logr = preprocess_func(x, logr)

        y, logj = self._model.net_(x)
        logq = logr - logj
        return y, logq

    @torch.no_grad()
    def sample__(self, batch_size=1, **kwargs):
        """
        Similar to `sample_`, but also returns the log probability of the
        target distribution from `model.action`.

        Parameters
        ----------
        batch_size : int, optional
            Number of samples to draw, default is 1.

        Returns
        -------
        tuple
            - `y`: Generated samples.
            - `logq`: Log probabilities of the samples.
            - `logp`: Log probabilities from the target distribution.
        """
        y, logq = self.sample_(batch_size=batch_size, **kwargs)
        logp = -self._model.action(y)  # logp is log(p_{non-normalized})
        return y, logq, logp

    @torch.no_grad()
    def log_prob(self, y):
        """
        Computes the log probability of the provided samples.

        Parameters
        ----------
        y : torch.Tensor
            Samples for which to calculate the log probability.

        Returns
        -------
        Tensor
            Log probabilities of the samples.
        """
        x, minus_logj = self._model.net_.reverse(y)
        logr = self._model.prior.log_prob(x)
        logq = logr + minus_logj
        return logq


class Fitter(TorchTrainer):
    """A class for training a given model."""
    profiler: Optional[Any]
    def __init__(
        self,
        model: Model,
        epochs: int = 100,
        config: Dict | TrainingConfiguration | None = None,
        strategy: Literal["ddp", "deepspeed", "horovod"] = 'ddp',
        logger: Logger | None = None,
        profiling_wait_epochs: int = 1,
        profiling_warmup_epochs: int = 2,
    ):
        super().__init__(config=config, epochs=epochs, strategy=strategy, logger=logger)
        self._model = model
        self.epochs = epochs

        self.train_batch_size = 1
        self.train_history = dict(
                loss=[], logqp=[], logz=[], ess=[], rho=[], accept_rate=[]
                )
        self.hyperparam = dict(lr=0.001, weight_decay=0.01)
        self.checkpoint_dict = dict(
            display=False,
            print_stride=10,
            print_batch_size=1024,
            snapshot_path=None,
            epochs_run=0
            )
        # Global training configuration
        if isinstance(config, dict):
            config = TrainingConfiguration(**config)
        self.config = config
        self.profiler = None
        self.profiling_wait_epochs = profiling_wait_epochs
        self.profiling_warmup_epochs = profiling_warmup_epochs

    def setup_seed(self, rank, world_size):
        """Sets up random seed for each worker in a distributed setting."""
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

        sys.stdout.write(f"Rank {rank} has been assigned seed {seed}\n")

    def execute(
        self,
        n_epochs=100,
        save_every=None,
        batch_size=64,
        optimizer_class=torch.optim.AdamW,
        scheduler=None,
        loss_fn=None,
        hyperparam={},
        checkpoint_dict={},
        profiling_wait_epochs=1,
        profiling_warmup_epochs=1
    ):

        """Fit the model; i.e. train the model.

        Parameters
        ----------
        n_epochs : int
            Number of epochs of training.

        save_every: int
            save a model every <save_every> epochs.

        batch_size : int
            Size of samples used at each epoch.

        optimizer_class : optimization class, optional
            By default is set to torch.optim.AdamW, but can be changed.

        scheduler : scheduler class, optional
            By default no scheduler is used.

        loss_fn : None or function, optional
            The default value is None, which translates to using KL divergence.

        hyperparam : dict, optional
            Can be used to set hyperparameters like the learning rate and decay
            weights.

        checkpoint_dict : dict, optional
            Can be set to control the printing and saving of the training status.
        """

        self.hyperparam.update(hyperparam)
        self.checkpoint_dict.update(checkpoint_dict)

        snapshot_path = self.checkpoint_dict['snapshot_path']

        if save_every is None:
            save_every = n_epochs

        self.epochs = n_epochs

        self._init_distributed_strategy()

        if self.strategy.is_main_worker and self.logger:
            self.logger.create_logger_context()

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
        if (snapshot_path is not None) and os.path.exists(snapshot_path):
            print(f"Trying to load snapshot from {snapshot_path}")
            self._load_snapshot()

        self.loss_fn = Fitter.calc_kl_mean if loss_fn is None else loss_fn

        if scheduler is None:
            self.scheduler = None
        else:
            self.scheduler = scheduler(self.optimizer)

        if n_epochs > 0:
            self._train(n_epochs, batch_size, save_every)

        if self.strategy.is_main_worker and self.logger:
            self.logger.destroy_logger_context()

        self.strategy.clean_up()

    def _load_snapshot(self):
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

    def _save_snapshot(self, epoch):
        """Save snapshot of training for analysis and/or to continue training
        at a later date.
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

    @profile_torch_trainer
    def _train(self, n_epochs: int, batch_size: int, save_every: int):

        T1 = time.time()
        for epoch in range(1, n_epochs+1):

            if self.profiler is not None:
                self.profiler.step()
            loss, logqp = self.step(batch_size)
            self.checkpoint(epoch, loss, save_every)
            if self.scheduler is not None:
                self.scheduler.step()
        T2 = time.time()
        if n_epochs > 0 and self.strategy.is_main_worker:
            print(f"({self.device}) Time = {T2 - T1:.3g} sec.")

    def step(self, batch_size):
        """Perform a train step with a batch of inputs"""
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

    def checkpoint(self, epoch, loss, save_every):

        print_stride = self.checkpoint_dict['print_stride']
        print_batch_size = self.checkpoint_dict['print_batch_size']
        snapshot_path = self.checkpoint_dict['snapshot_path']

        # Always save loss on rank 0
        if self.strategy.is_main_worker:
            self.train_history['loss'].append(loss.item())
            # Save model as well
            if snapshot_path is not None and (epoch % save_every == 0):
                self._save_snapshot(epoch)

        print_batch_size = print_batch_size // self.strategy.global_world_size()

        if epoch == 1 or (epoch % print_stride == 0):

            _, logq, logp = self._model.posterior.sample__(print_batch_size)
            logq = self.strategy.gather(logq)
            logp = self.strategy.gather(logp)

            if self.strategy.is_main_worker:
                logq = torch.cat(logq, dim=0)
                logp = torch.cat(logp, dim=0)

                loss_ = self.loss_fn(logq, logp)
                self._append_to_train_history(logq, logp)
                self.print_fit_status(epoch, loss=loss_)

    @staticmethod
    def calc_kl_mean(logq, logp):
        """Return Kullback-Leibler divergence estimated from logq and logp."""
        return (logq - logp).mean()  # KL, assuming samples from q

    @staticmethod
    def calc_kl_var(logq, logp):
        return (logq - logp).var()

    @staticmethod
    def calc_corrcoef(logq, logp):
        return torch.corrcoef(torch.stack([logq, logp]))[0, 1]

    @staticmethod
    def calc_direct_kl_mean(logq, logp):
        logpq = logp - logq
        logz = torch.logsumexp(logpq, dim=0) - np.log(logp.shape[0])
        logpq = logpq - logz  # p is now normalized
        p_by_q = torch.exp(logpq)
        return (p_by_q * logpq).mean()

    @staticmethod
    def calc_minus_logz(logq, logp):
        logz = torch.logsumexp(logp - logq, dim=0) - np.log(logp.shape[0])
        return -logz

    @staticmethod
    def calc_ess(logq, logp):
        """Rerturn effective sample size (ESS)."""
        logqp = logq - logp
        log_ess = 2*torch.logsumexp(-logqp, dim=0) \
                - torch.logsumexp(-2*logqp, dim=0)
        ess = torch.exp(log_ess) / len(logqp)  # normalized
        return ess

    @staticmethod
    def calc_minus_logess(logq, logp):
        """Return logarithm of inverse of effective sample size."""
        logqp = logq - logp
        log_ess = 2*torch.logsumexp(-logqp, dim=0) \
                - torch.logsumexp(-2*logqp, dim=0)
        return - log_ess + np.log(len(logqp))  # normalized

    @torch.no_grad()
    def _append_to_train_history(self, logq, logp):
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

    def print_fit_status(self, epoch, loss=None):
        mydict = self.train_history
        if loss is None:
            loss = mydict['loss'][-1]
        else:
            pass  # the printed loss can be different from mydict['loss'][-1]
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
        str1 = f"Epoch: {epoch} | loss: {loss:.4f} | ess: {ess:.4f}"
        print(str1)
        self.log(loss.detach().cpu().numpy(),'epoch_loss',kind='metric')


# =============================================================================
@torch.no_grad()
def reverse_flow_sanitychecker(model, n_samples=4, net_=None):
    """Performs a sanity check on the reverse method of modules."""

    if net_ is None:
        net_ = model.net_

    x = model.prior.sample(n_samples)
    y, logj = net_(x)
    x_hat, minus_logj = net_.backward(y)

    mean = lambda z: z.abs().mean().item()

    print("reverse method is OK if following values vanish (up to round off):")
    print(f"{mean(x - x_hat):g} & {mean(1 + minus_logj / logj):g}")
