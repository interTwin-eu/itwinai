# Copyright (c) 2021-2022 Javad Komijani


"""This module is for introducing priors..."""


import torch
import copy
import numpy as np

from abc import abstractmethod, ABC
from typing import Dict, Tuple


class Prior(ABC):
    """A template class to initiate a prior distribution."""

    propagate_density = False

    def __init__(self, dist, seed=None, **kwargs):
        self.dist = dist
        Prior.manual_seed(seed)
        self.extra_params = kwargs

    def sample(self, batch_size=1):
        return self.dist.sample((batch_size,))

    def sample_(self, batch_size=1):
        x = self.dist.sample((batch_size,))
        return x, self.log_prob(x)

    def log_prob(self, x):
        log_prob_density = self.dist.log_prob(x)
        if self.propagate_density:
            return log_prob_density
        else:
            dim = range(1, len(log_prob_density.shape))  # 0: batch axis
            return torch.sum(log_prob_density, dim=tuple(dim))

    @staticmethod
    def manual_seed(seed):
        if isinstance(seed, int):
            torch.manual_seed(seed)

    @property
    def nvar(self):
        return np.product(self.shape)

    @abstractmethod
    def to(self, *args, **kwargs):
        """
        Moves the distibution parameters to a device, implying that the samples
        also will also be created on the same device.
        """
        pass

    @property
    @abstractmethod
    def parameters(self):
        """Returns all parameters needed to define the prior in a dict."""
        pass


class UniformPrior(Prior):
    """Creates a uniform distribution parameterized by low and high;
    uniform in [low, hight].
    """

    def __init__(self, low=None, high=None, shape=None, seed=None, **kwargs):
        """If shape is None, low & high must be of similar shape."""
        if shape is not None:
            low = torch.zeros(shape)
            high = torch.ones(shape)
        else:
            shape = low.shape
        dist = torch.distributions.uniform.Uniform(low, high)
        super().__init__(dist, seed, **kwargs)
        self.shape = shape

    def to(self, *args, **kwargs):
        """
        Moves the distibution parameters to a device, implying that the samples
        also will also be created on the same device.
        """
        self.dist.low = self.dist.low.to(*args, **kwargs)
        self.dist.high = self.dist.high.to(*args, **kwargs)

    @property
    def parameters(self):
        """Returns all parameters needed to define the prior in a dict."""
        return dict(low=self.dist.low, high=self.dist.high)


class NormalPrior(Prior):
    """Creates a normal distribution parameterized by loc and scale."""

    def __init__(
        self,
        loc: float | torch.Tensor = None,
        scale: float | torch.Tensor = None,
        shape: Tuple[int, ...] | None = None,
        seed: int | None = None,
        **kwargs: Dict
    ):
        self.loc = loc
        self.scale = scale
        self.shape = shape
        self.seed = seed

        """If shape is None, loc & scale must be of similar shape."""
        if shape is not None:
            loc = torch.zeros(shape)  # i.e. mean
            scale = torch.ones(shape)  # i.e. sigma
        else:
            shape = loc.shape
        dist = torch.distributions.normal.Normal(loc, scale)
        super().__init__(dist, seed, **kwargs)
        self.shape = shape

    def setup_blockupdater(self, block_len):
        # For simplicity we assume that loc & scale are identical everywhere.
        chopped_prior = NormalPrior(
                loc=self.dist.loc.ravel()[:block_len],
                scale=self.dist.scale.ravel()[:block_len]
                )
        self.blockupdater = BlockUpdater(chopped_prior, block_len)

    def to(self, *args, **kwargs):
        """
        Moves the distibution parameters to a device, implying that the samples
        also will also be created on the same device.
        """
        self.dist.loc = self.dist.loc.to(*args, **kwargs)
        self.dist.scale = self.dist.scale.to(*args, **kwargs)

    @property
    def parameters(self):
        """Returns all parameters needed to define the Prior in a dict."""
        return dict(loc=self.dist.loc, scale=self.dist.scale)


class PriorList:

    def __init__(self, prior_list):
        self.prior_list = prior_list

    def sample(self, batch_size=1):
        return [prior.sample(batch_size) for prior in self.prior_list]

    def sample_(self, batch_size=1):
        x = [prior.sample(batch_size) for prior in self.prior_list]
        return x, self.log_prob(x)

    def log_prob(self, x):
        return [prior.log_prob(x_) for prior, x_ in zip(self.prior_list, x)]

    @property
    def nvar(self):
        return sum([prior.nvar for prior in self.prior_list])

    def to(self, *args, **kwargs):
        """
        Moves the distibution parameters to a device, implying that the samples
        also will also be created on the same device.
        """
        for prior in self.prior_list:
            prior.to(*args, **kwargs)

    @property
    def parameters(self):
        """Returns all parameters needed to define the priors in a dict."""
        return [prior.parameters for prior in self.prior_list]


class BlockUpdater:

    def __init__(self, chopped_prior, block_len):
        self.block_len = block_len
        self.chopped_prior = chopped_prior
        self.backup_block = None

    def __call__(self, x, block_ind):
        """In-place updater"""
        batch_size = x.shape[0]
        view = x.view(batch_size, -1, self.block_len)
        self.backup_block = copy.deepcopy(view[:, block_ind])
        view[:, block_ind] = self.chopped_prior.sample(batch_size)

    def restore(self, x, block_ind, restore_ind=slice(None)):
        batch_size = x.shape[0]
        view = x.view(batch_size, -1, self.block_len)
        view[restore_ind, block_ind] = self.backup_block[restore_ind]

