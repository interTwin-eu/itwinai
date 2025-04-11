# Copyright (c) 2021-2022 Javad Komijani

"""This module introduces a neural network to handle the mean field of a field.

The classes defined here are children of Module_, and like Module_, the trailing
underscore implies that the associated forward and backward methods handle the
Jacobians of the transformation.
"""


import torch
import numpy as np

from .modules_ import DistConvertor_
from .._core import Module_, ModuleList_


class MeanFieldNet_(Module_):
    """A probability distribution convertor of the mean field part of the process."""

    def __init__(self, dc_, label='mean-field'):
        super().__init__(label)
        self.dc_ = dc_

    def forward(self, x, log0=0, rvol=None):
        # To normalize data, multiply \& divide by square root of volume
        if rvol is None:
            dim = list(range(1, x.dim()))
            rvol = np.prod(x.shape[1:])**0.5  # square root of volume
            x_mean = torch.mean(x, dim=dim).reshape(-1, *[1 for _ in dim])
            x_mean_new_scaled, log0 = self.dc_.forward(x_mean * rvol, log0)
            return x + (x_mean_new_scaled/rvol - x_mean), log0
        else:
            # assume x is already the mean of a field
            x_mean_new_scaled, log0 = self.dc_.forward(x * rvol, log0)
            return x_mean_new_scaled / rvol, log0

    def backward(self, x, log0=0, rvol=None):
        # To normalize data, multiply \& divide by square root of volume
        if rvol is None:
            dim = list(range(1, x.dim()))
            rvol = np.prod(x.shape[1:])**0.5  # square root of volume
            x_mean = torch.mean(x, dim=dim).reshape(-1, *[1 for _ in dim])
            x_mean_new_scaled, log0 = self.dc_.backward(x_mean * rvol, log0)
            return x + (x_mean_new_scaled/rvol - x_mean), log0
        else:
            # assume x is already the mean of a field
            x_mean_new_scaled, log0 = self.dc_.backward(x * rvol, log0)
            return x_mean_new_scaled / rvol, log0

    def _hack(self, x, log0=0):
        """Similar to the forward method, except that returns `x_mean` from
        begining to the end; useful for examining effects of each block.
        """
        dim = list(range(1, x.dim()))
        rvol = np.prod(x.shape[1:])**0.5  # square root of volume
        x_mean = torch.mean(x, dim=dim).reshape(-1, *[1 for _ in dim])
        stack = [(x_mean.ravel(), log0)]
        x_mean_scaled, log0 = self.dc_.forward(x_mean * rvol, log0)
        stack.append((x_mean_scaled.ravel() / rvol, log0))
        return stack

    @staticmethod
    def build(knots_len=10, **kwargs):
        dc_ = DistConvertor_(knots_len, **kwargs)
        return MeanFieldNet_(dc_)
