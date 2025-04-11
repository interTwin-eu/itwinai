# Copyright (c) 2021-2023 Javad Komijani

"""
This module contains new neural networks that are subclasses of Module_ and
do not couple sites to each other.

As in Module_, the trailing underscore implies that the associated forward and
backward methods handle the Jacobians of the transformation.
"""


import torch
import copy
import numpy as np

from .modules import SplineNet
from .._core import Module_, ModuleList_


class Identity_(Module_):

    def __init__(self, label='identity_'):
        super().__init__(label=label)

    def forward(self, x, log0=0, **extra):
        return x, log0

    def backward(self, x, log0=0, **extra):
        return x, log0


class Clone_(Module_):

    def __init__(self, label='clone_'):
        super().__init__(label=label)

    def forward(self, x, log0=0, **extra):
        return x.clone(), log0

    def backward(self, x, log0=0, **extra):
        return x.clone(), log0


class ScaleNet_(Module_):
    """Scales the input by a positive weight."""
    # TODO: one can consider different weights for different channels

    softplus = torch.nn.Softplus(beta=np.log(2))

    def __init__(self, label='scale_'):
        super().__init__(label=label)
        self._weight = torch.nn.Parameter(torch.zeros(1))

    @property
    def weight(self):
        return self.softplus(self._weight)

    def forward(self, x, log0=0):
        return x * self.weight, log0 + self.log_jacobian(x.shape)

    def backward(self, x, log0=0):
        return x / self.weight, log0 - self.log_jacobian(x.shape)

    def log_jacobian(self, x_shape):
        if Module_.propagate_density:
            return torch.log(self.weight) * torch.ones(x_shape)
        else:
            logwscaled = torch.log(self.weight) * np.prod(x_shape[1:])
            return logwscaled * torch.ones(x_shape[0], device=self._weight.device)


class Tanh_(Module_):

    def forward(self, x, log0=0):
        logJ = -2 * self.sum_density(torch.log(torch.cosh(x)))
        return torch.tanh(x), log0 + logJ

    def backward(self, x, log0=0):
        return ArcTanh_().forward(x, log0)


class ArcTanh_(Module_):

    def forward(self, x, log0=0):
        y = torch.atanh(x)
        logJ = 2 * self.sum_density(torch.log(torch.cosh(y)))
        return y, log0 + logJ

    def backward(self, x, log0=0):
        return Tanh_().forward(x, log0)


class Expit_(Module_):
    """This can be also called Sigmoid_"""

    def forward(self, x, log0=0):
        y = 1 / (1 + torch.exp(-x))
        logJ = self.sum_density(-x + 2 * torch.log(y))
        return y, log0 + logJ

    def backward(self, x, log0=0):
        return Logit_().forward(x, log0)


class Logit_(Module_):
    """This is inverse of Sigmoid_"""

    def forward(self, x, log0=0):
        y = torch.log(x / (1 - x))
        logJ = - self.sum_density(torch.log(x * (1 - x)))
        return y, log0 + logJ

    def backward(self, x, log0=0):
        return Expit_().forward(x, log0)


class Pade11_(Module_):
    r"""An invertible transformation as a Pade approximant of order 1/1

    .. math::

        f(x; t) = x / (x + e^t * (1 - x))

    that maps :math:`[0, 1] \to [0, 1]` and is useful for input and output
    variables that vary between zero and one.

    This transformation is equivalent to math:`f(x; t) = \expit(\logit(x) - t)`
    and its inverse is :math:`f(.; -t)`.
    """

    softplus = torch.nn.Softplus(beta=np.log(2))

    def __init__(self, n_channels=1, channels_axis=1, label='pade11'):
        super().__init__(label=label)
        self.w1 = torch.nn.Parameter(torch.zeros(n_channels))
        self.n_channels = n_channels
        self.channels_axis = channels_axis

    def forward(self, x, log0=0):
        d1 = self.get_derivative_reshaped(x.shape)
        denom = x + (1 - x) * d1
        logJ = self.sum_density(torch.log(d1) - 2 * torch.log(denom))
        return x / denom, log0 + logJ

    def backward(self, x, log0=0):
        d1 = self.get_derivative_reshaped(x.shape)
        denom = x + (1 - x) / d1
        logJ = self.sum_density(-torch.log(d1) - 2 * torch.log(denom))
        return x / denom, log0 + logJ

    def get_derivative_reshaped(self, shape):
        if self.n_channels == 1:
            w1 = self.w1
        else:
            shape = [1 for _ in shape]
            shape[self.channels_axis] = self.n_channels
            w1 = self.w1.reshape(*shape)
        return self.softplus(w1)


class Pade22_(Module_):
    r"""An invertible transformation as a Pade approximant of order 2/2

    .. math::

        f(x; t) = (x^2 + a x (1 - x)) / (1 + b x (1 - x))

    that maps :math:`[0, 1] \to [0, 1]` and is useful for input and output
    variables that vary between zero and one.
    """

    softplus = torch.nn.Softplus(beta=np.log(2))

    def __init__(
            self, n_channels=1, channels_axis=1, symmetric=False, label='pade22'
            ):
        super().__init__(label=label)
        self.w0 = torch.nn.Parameter(torch.zeros(n_channels))
        if not symmetric:
            self.w1 = torch.nn.Parameter(torch.zeros(n_channels))
        else:
            self.w1 = self.w0
        self.n_channels = n_channels
        self.channels_axis = channels_axis
        self.symmetric = symmetric

    def forward(self, x, log0=0):
        d0, d1 = self.get_derivatives_reshaped(x.shape)
        denom = (1 + (d1 + d0 - 2) * x * (1 - x))
        g_0 = x * (x + d0 * (1 - x)) / denom
        g_1 = (d0 + 2 * (1 - d0) * x + (d1 + d0 - 2) * x**2) / denom**2
        return g_0, log0 + self.sum_density(torch.log(g_1))

    def backward(self, y, log0=0):
        d0, d1 = self.get_derivatives_reshaped(y.shape)

        def invert(y):
            # returns the positive solution of $a x^2 + b x + c = 0$
            c = y
            b = (d1 + d0 - 2) * y - d0
            a = -1 - b
            delta = torch.sqrt(b**2 - 4 * c * a)
            x = (-b - delta) / (2 * a)
            x[a == 0] = (-c / b)[a == 0]
            return x

        x = invert(y)
        denom = (1 + (d1 + d0 - 2) * x * (1 - x))
        g_1 = (d0 + 2 * (1 - d0) * x + (d1 + d0 - 2) * x**2) / denom**2
        return x, log0 - self.sum_density(torch.log(g_1))

    def get_derivatives_reshaped(self, shape):
        if self.n_channels == 1:
            w0 = self.w0
            w1 = self.w1
        else:
            shape = [1 for _ in shape]
            shape[self.channels_axis] = self.n_channels
            w0 = self.w0.reshape(*shape)
            w1 = self.w1.reshape(*shape)

        return self.softplus(w0), self.softplus(w1)


class Pade32_(Module_):
    r"""An invertible transformation as a Pade approximant of order 3/2

    .. math::

        f(x) = x (a + x^2) / (1 + a x^2)

    which is invertible for :math:`0 < a < 3`.
    Note that this transformation is not the most general invertible Pade 3/2,
    but is has the following traits: it is odd and regular at any finite real
    :math:`x`, it has three fixed points at zero and plus/minus unity,
    and it is proportional to :math:`x` as :math:`x \to \pm \infty`.

    Moreover, this can be used as a nonlinear activation (if :math:`a \neq 1`).
    """

    def __init__(self, n_channels=1, channels_axis=1, label='pade32'):
        super().__init__(label=label)
        self.w0 = - torch.nn.Parameter(torch.log(2 * torch.ones(n_channels)))
        self.n_channels = n_channels
        self.channels_axis = channels_axis

    def forward(self, x, log0=0):
        a = self.get_derivative_reshaped(x.shape)  # a is derivative at x=0
        s = x**2
        y = x * (a + s) / (1 + a * s)
        dy_by_dx = (a * s**2 + (3 - a**2) * s + a) / (1 + a * s)**2
        logJ = self.sum_density(torch.log(dy_by_dx))
        return y, log0 + logJ

    def backward(self, y, log0=0):
        """We solve a cubic relation that has only one real solution."""
        a = self.get_derivative_reshaped(x.shape)  # a is derivative at x=0
        delta0 = a**2 -  3 * a / y**2
        delta1 = - 2 * a**3 + (9 * a**2 - 27) / y**2
        delta = 2**(-1/3) * (- delta1 + torch.sqrt(delta1**2 - 4*delta0**3))**(1/3)
        x = y * (a + delta + delta0 / delta) / 3
        s = x**2
        dy_by_dx = (a * s**2 + (3 - a**2) * s + a) / (1 + a * s)**2
        logJ = self.sum_density(-torch.log(dy_by_dx))
        return x, log0 + logJ

    def get_derivative_reshaped(self, shape):
        if self.n_channels == 1:
            w0 = self.w0
        else:
            shape = [1 for _ in shape]
            shape[self.channels_axis] = self.n_channels
            w0 = self.w0.reshape(*shape)
        return 3 * torch.special.expit(w0)


class SplineNet_(SplineNet, Module_):
    """Identical to SplineNet, except for taking care of log_jacobian.

    This can be used as a probability distribution convertor for random
    variables with nonzero probability in [0, 1].
    """

    def forward(self, x, log0=0):
        spline = self.make_spline()
        if len(self.spline_shape) > 0:
            fx, g = spline(x, grad=True)  # g is gradient of the spline @ x
        else:
            fx, g = spline(x.ravel(), grad=True)  # g is gradient of spline @ x
            fx, g = fx.reshape(x.shape), g.reshape(x.shape)
        logJ = self.sum_density(torch.log(g))
        return fx, log0 + logJ

    def backward(self, x, log0=0):
        spline = self.make_spline()
        if len(self.spline_shape) > 0:
            fx, g = spline.backward(x, grad=True)  # g is gradient @ x
        else:
            fx, g = spline.backward(x.ravel(), grad=True)  # g is gradient @ x
            fx, g = fx.reshape(x.shape), g.reshape(x.shape)
        logJ = self.sum_density(torch.log(g))
        return fx, log0 + logJ


class UnityDistConvertor_(SplineNet_):
    """As a PDF convertor for random variables in range [0, 1]."""

    def __init__(self, knots_len, symmetric=False, **kwargs):

        if symmetric:
            extra = dict(xlim=(0.5, 1), ylim=(0.5, 1), extrap={'left':'anti'})
        else:
            extra = {}

        super().__init__(knots_len, **kwargs, **extra)


class PhaseDistConvertor_(SplineNet_):
    """As a PDF convertor for random variables in range [-pi, pi]."""

    def __init__(self, knots_len, symmetric=False, label='phase-dc_', **kwargs):

        pi = np.pi

        if symmetric:
            extra = dict(xlim=(0, pi), ylim=(0, pi), extrap={'left':'anti'})
        else:
            extra = dict(xlim=(-pi, pi), ylim=(-pi, pi))

        super().__init__(knots_len, label=label, **kwargs, **extra)


class DistConvertor_(ModuleList_):
    """As a PDF convertor for real random variables (from minus to plus
    infinity).

    Steps: pass through Expit_, SplineNet_, and Logit_
    """
    def __init__(self,
        knots_len: int,
        symmetric: bool = False,
        label: str = 'dc_',
        sgnbias: bool = False,
        initial_scale: bool = False,
        final_scale: bool = False,
        **kwargs
    ):
        self.knots_len = knots_len
        self.symmetric = symmetric
        self.label = label
        self.sgnbias = sgnbias
        self.initial_scale = initial_scale
        self.final_scale = final_scale

        if symmetric:
            extra = dict(xlim=(0.5, 1), ylim=(0.5, 1), extrap={'left':'anti'})
        else:
            extra = dict(xlim=(0, 1), ylim=(0, 1))

        kwargs.update(extra)

        if knots_len > 1:
            spline_ = SplineNet_(knots_len, label='spline_', **kwargs)
            nets_ = [Expit_(label='expit_'), spline_, Logit_(label='logit_')]
        else:
            nets_ = []

        if initial_scale:
            nets_ = [ScaleNet_(label='scale_')] + nets_
        elif final_scale:
            nets_ = nets_ + [ScaleNet_(label='scale_')]

        if sgnbias:  # SgnBiasNet_() **must** come first if exits
            nets_ = [SgnBiasNet_()] + nets_

        super().__init__(nets_)
        self.label = label


class SgnBiasNet_(Module_):
    """This module should be used only and only in the first layer, where the
    input does not depend on the parameters of the net. Otherwise, because it
    is not continuous, the derivatives will be messed up.
    """

    def __init__(self, size=[1], label='sgnbias_'):
        super().__init__(label=label)
        self.w = torch.nn.Parameter(torch.rand(*size)/10)

    def forward(self, x, log0=0):
        return x + torch.sgn(x) * self.w**2, log0

    def backward(self, x, log0=0):
        return x - torch.sgn(x) * self.w**2, log0
