# Copyright (c) 2021-2022 Javad Komijani

"""This module introduces a neural network to handle the PSD of a field.

The classes defined here are children of Module_, and like Module_, the trailing
underscore implies that the associated forward and backward methods handle the
Jacobians of the transformation.
"""


import torch
import numpy as np

from .._core import Module_


class PSDBlock_(Module_):
    """Power Spectral Density Block"""

    def __init__(self, *, mfnet_, fftnet_, label='psd-block'):
        super().__init__(label=label)
        self.mfnet_ = mfnet_
        self.fftnet_ = fftnet_

    def forward(self, x, log0=0):
        dim = list(range(1, x.dim()))
        rvol = np.prod(x.shape[1:])**0.5  # square root of volume
        x_mean = torch.mean(x, dim=dim).reshape(-1, *[1 for _ in dim])
        y_mf, logJ_mf = self.mfnet_.forward(x_mean, rvol=rvol)
        y_fft, logJ_fft = self.fftnet_.forward(x - x_mean)
        return (y_mf + y_fft), (log0 + logJ_mf + logJ_fft)

    def backward(self, x, log0=0):
        dim = list(range(1, x.dim()))
        rvol = np.prod(x.shape[1:])**0.5  # square root of volume
        x_mean = torch.mean(x, dim=dim).reshape(-1, *[1 for _ in dim])
        y_mf, logJ_mf = self.mfnet_.backward(x_mean, rvol=rvol)
        y_fft, logJ_fft = self.fftnet_.backward(x - x_mean)
        return (y_mf + y_fft), (log0 + logJ_mf + logJ_fft)

    def _hack(self, x, log0=0):
        """Similar to the forward method, but returns intermediate parts too."""
        dim = list(range(1, x.dim()))
        rvol = np.prod(x.shape[1:])**0.5  # square root of volume
        x_mean = torch.mean(x, dim=dim).reshape(-1, *[1 for _ in dim])
        y_mf, logJ_mf = self.mfnet_.forward(x_mean, rvol=rvol)
        y_fft, logJ_fft = self.fftnet_.forward(x - x_mean)
        stack = [(x_mean, log0), (y_mf, logJ_mf), (y_fft, logJ_fft),
                 ((y_mf + y_fft), (log0 + logJ_mf + logJ_fft))
                ]
        return stack

    def transfer(self, **kwargs):
        return self.__class__(
                mfnet_ = self.mfnet_.transfer(**kwargs),
                fftnet_ = self.fftnet_.transfer(**kwargs)
                )
