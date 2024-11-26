# Copyright (c) 2021-2022 Javad Komijani

r"""This module introduces a new neural network called `FFTNet_`.

Theory: The Effective Action
--------------------------
[e.g. Shrednicki's book, chapter 21]

We define the effective action

.. math:

    \Gamma[\phi] S = \int d^n k (
        \tilde \phi(-k) (\kappa k^2 + m^2 - \Pi(k^2)) \tilde \phi(k) + \cdots
        ).

where :math:`\tilde \phi(k)` is the Fourier transform of :math:`\phi(x)`.
The effective action has the property that the tree-level Feynman diagram that
it generates reproduces the complete scattering amplitude of the original
theory.
"""


import torch
import copy
import numpy as np

from .modules import SplineNet
from .._core import Module_
from ...lib.indexing import outer_arange


irfft, rfft = torch.fft.irfftn, torch.fft.rfftn


# =============================================================================
class FFTNet_(Module_):
    """FFT Network...

    One can use FFTNet without specifying any channels. Then the parameters
    have the same dimensions as the lattice. The data, however, can have one
    more dimension for batches as the outermost axis.
    (Note that the data can also come without any batch axis.)
    For example, let us first define

        >>> net = normflow.FFTNet((4, 4))
        >>> prior = normflow.NormalPrior(shape=(4, 4))

    where the samples of prior look like

        >>> prior.sample(13).size()
        torch.Size([13, 4, 4])

    The netwrok can be used as either of these two ways (with \& without
    batches):

        >>> net(prior.sample(13)).size()
        torch.Size([13, 4, 4])
        >>> net(prior.sample(13)[0]).size()
        torch.Size([4, 4])

    **CHANNELS** are removed
    If channels are added, then it is assumed that the 0 axis in the data is
    always the batch axis, and there is one more axis in the data as the
    channels axis, where its length is either 1 or the number of channels.
    For example let us define:

        >>> net = normflow.FFTNet((4, 4), channels=7, channels_axis=1)

    The netwrok can be used as either of these two ways:

        >>> net(prior.sample(13).unsqueeze(1)).size()
        torch.Size([13, 7, 4, 4])
        >>> y = net(prior.sample(13).unsqueeze(1))
        >>> y.size()
        torch.Size([13, 7, 4, 4])
        >>> net(y).size()
        torch.Size([13, 7, 4, 4])

    There are exceptioncal cases that the data does not have eiher of the batch
    or channels axes, but the calculation go through, e.g.

        >>> net(prior.sample(7)).size()
        torch.Size([7, 4, 4])

    while the correct usage could have been:

        >>> net(prior.sample(7).unsqueeze(1)).size()
        torch.Size([7, 7, 4, 4])


    Parameters
    ----------
    lat_shape: list/tuple like
    ipsd_net: a NN for...
    ignore_zeromode: bool
         ignore/include contribution of zero mode to log(J)
    """
    
    def __init__(self, lat_shape, ipsd_net, ignore_zeromode=False, label='fftnet_'):
        super().__init__(label=label)
        self.lat_ndim = len(lat_shape)
        self.lat_shape = lat_shape
        self.ipsd_net = ipsd_net
        self.ignore_zeromode = ignore_zeromode

        # we use rfftn instead of fftn, which means the length of the
        # last axis of data changes from n to n//2 + 1
        # self.param_shape = [n for n in lat_shape[:-1]] + [lat_shape[-1]//2 + 1]

        # With negative indices, define the dimensions (axes) to feed to rfftn
        self.rfft_dim = list(range(-self.lat_ndim, 0, 1))
        # Negative indices allow for data (not) to have a batch axis
        self.rfft_axis = -1  # the axis reduced by rfft for redundancy

        freetheory = FreeScalar(self.lat_shape)
        lat_k2 = freetheory.calc_lattice_k2()
        self.register_buffer('norm_lat_k2', lat_k2/torch.max(lat_k2))
        self.register_buffer('max_lat_k2', torch.max(lat_k2))

    def forward(self, x, log0=0):
        """Take `rfftn`, multiply by `weights`, and take `irfftn`."""
        w = 1/self.ipsd**0.5
        dim = self.rfft_dim
        return irfft(rfft(x, dim=dim) * w, dim=dim), log0 + self.log_jacobian(w)

    def backward(self, x, log0=0):
        """Take `rfftn`, multiply by `weights`, and take `irfftn`."""
        w = 1/self.ipsd**0.5
        dim = self.rfft_dim
        return irfft(rfft(x, dim=dim) / w, dim=dim), log0 - self.log_jacobian(w)

    @property
    def ipsd(self):
        """ipsd: Inverse Power Spectral Density"""
        return self.ipsd_net(self.norm_lat_k2)

    @staticmethod
    def build(lat_shape, knots_len=10, eff_mass2=1, eff_kappa=1, a=1,
            ignore_zeromode=False, nozeromode=False, **ipsd_kwargs
            ):
        # better not to use nozeromode, obsolete option, will be removed

        freetheory = FreeScalar(lat_shape)
        max_lat_k2 = torch.max(freetheory.calc_lattice_k2())

        if knots_len < 2:
            knots_len = 2
            ipsd_kwargs.update(dict(smooth=True))
            # with the above two commands, ipsd_net would be like identity

        logm2 = torch.log(torch.tensor(eff_mass2))
        logk2 = torch.log(eff_kappa * max_lat_k2)
        mydict = dict(a=a, ndim=len(lat_shape))

        if nozeromode and not ignore_zeromode:
            logy = IPSDnozeromode.apply_scale(torch.tensor([logk2]), **mydict)
            ipsd_net = IPSDnozeromode(knots_len, logy=logy, **ipsd_kwargs)
        else:
            logy = IPSD.apply_scale(torch.tensor([logm2, logk2]), **mydict)
            ipsd_net = IPSD(knots_len, logy=logy,
                    ignore_zeromode=ignore_zeromode, **ipsd_kwargs
                    )

        return FFTNet_(lat_shape, ipsd_net, ignore_zeromode=ignore_zeromode)

    def log_jacobian(self, weights):
        """Return logarithm of the Jacobian of multiplication by `weights`.
        Note that the Jacobian of FFT is 1, so one only needs to take care of
        the `weights`.
        """
        sumlog = lambda w: torch.sum(torch.log(w), dim=self.rfft_dim)

        w = weights
        logJ = 2 * sumlog(w)  # for 2 see below
        # Because we use rfftn: the factor 2 is for (..., k) & (..., -k),
        # except at k=0 & pi/a, which we subtract below
        logJ -= (sumlog(w[..., 0:1]) + sumlog(w[..., -1:]))

        return self.create_density(logJ)
    
    @property
    def infrared_mass(self):
        """dimension-less mass (in lattice units)"""
        return self.ipsd_net.infrared_mass(self.max_lat_k2)

    def transfer(self, scale_factor=1, shape=None, **extra):
        """Map the weights of the current lattice to a new lattice.

        Parameters
        ----------
        shape : tuple of integers
            The shape of the lattice ...

        scale_factor : float
            The factor for improving the resolution; inverse of ratio of the
            lattice spacing of the new lattice compared to the current one.

        **How to set the scale_factor**:
        Set scale_factor to 10 for converting a netwrok corresponding to $a = 2$
        fm to another one corresponding to $a = 0.2$ fm.
        """
        shape = self.lat_shape if shape is None else shape
        ipsd_net = self.ipsd_net.transfer(scale_factor=scale_factor, ndim=self.lat_ndim)
        try:
            return self.__class__(shape, ipsd_net=ipsd_net,
                    ignore_zeromode=self.ignore_zeromode)
        except:
            return self.__class__(shape, ipsd_net=ipsd_net)

    def create_density(self, logJ):
        """logJ is scalar for each sample; this funtion decides whether to
        assign a density to it or not!
        """
        if Module_.propagate_density:
            n = np.prod(self.lat_shape)
            ones = torch.ones(*logJ.shape, n)
            return (logJ.unsqueeze(-1) / n * ones).reshape(*logJ.shape, *self.lat_shape)
        else:
            return logJ


# =============================================================================
class IPSD(SplineNet):
    """A class for Inverse Power Spectral Density."""

    def __init__(self, knots_len, *, logy, ignore_zeromode=False, **kwargs):
        super().__init__(knots_len, **kwargs)
        self.logy = torch.nn.Parameter(logy)
        self.ignore_zeromode = ignore_zeromode

    def forward(self, x):
        y = torch.exp(self.logy)
        sigma_k2 = y[0] + y[1] * super().forward(x)
        if self.ignore_zeromode:
            ind = tuple([0]*x.dim())
            sigma_k2[ind] = 1  # replace 0 with 1 to get correct jacobian item
        return sigma_k2

    def _backward(self, x):
        # Note that this typically is not used!
        y = torch.exp(self.logy)
        return super().backward((x - y[0]) / y[1])

    def transfer(self, scale_factor=1, ndim=1):
        ipsd = copy.deepcopy(self)
        state_dict = ipsd.state_dict()
        logy = self.apply_scale(state_dict['logy'], a=1/scale_factor, ndim=ndim)
        state_dict.update(dict(logy=logy))
        ipsd.load_state_dict(state_dict)
        return ipsd

    @staticmethod
    @torch.no_grad()
    def apply_scale(logy, *, a, ndim):
        a = torch.tensor(a)
        logm2 = logy[0] + torch.log(a) * ndim
        logk2 = logy[1] + torch.log(a) * (ndim - 2)
        return torch.tensor([logm2, logk2])

    @torch.no_grad()
    def infrared_mass(self, max_lat_k2):
        return torch.exp(0.5 * self.logy[0])


class IPSDnozeromode(SplineNet):
    """A class for Inverse Power Spectral Density; a case without zero mode.

    The coefficient of zero mode is set to 1 (DO NOT CHANGE THIS).

    ***COMMENT***
    It is better to use IPSD with ignore_zeromode option rather than this one.
    We still keep it this one here just in case.
    """

    def __init__(self, knots_len, *, logy, **kwargs):
        super().__init__(knots_len, **kwargs)
        self.logy = torch.nn.Parameter(logy)

    def forward(self, x):
        y = torch.exp(self.logy)
        sigma_k2 = y[0] * super().forward(x)
        ind = tuple([0]*x.dim())
        sigma_k2[ind] = 1  # replace 0 with 1 to avoid divergence in 1/ipsd
        return sigma_k2

    def _backward(self, x):
        # Note that this typically is not used!
        y = torch.exp(self.logy)
        x = x / y[0]
        ind = tuple([0]*x.dim())
        x[ind] = 0  # replace back 1 with the original 0
        return super().backward(x)

    def transfer(self, scale_factor=1, ndim=1):
        ipsd = copy.deepcopy(self)
        state_dict = ipsd.state_dict()
        logy = self.apply_scale(state_dict['logy'], a=1/scale_factor, ndim=ndim)
        state_dict.update(dict(logy=logy))
        ipsd.load_state_dict(state_dict)
        return ipsd

    @staticmethod
    @torch.no_grad()
    def apply_scale(logy, *, a, ndim):
        a = torch.tensor(a)
        logk2 = logy[0] + torch.log(a) * (ndim - 2)
        return torch.tensor([logk2])

    @torch.no_grad()
    def infrared_mass(self, max_lat_k2):
        z = self.forward(torch.tensor([1e-6/max_lat_k2, 2e-6/max_lat_k2]))
        factor = (z[1] - z[0]) / 1e-6  # factor is 1 for free theory
        return (z[0] / factor)**0.5


# =============================================================================
class FreeScalar:

    def __init__(self, lat_shape, kappa=None, m_sq=None):
        self.lat_shape = lat_shape
        self.kappa = kappa
        self.m_sq = m_sq

    def calc_lattice_k2(self):
        list_ = [(0, 2*np.pi*(1-1/n), n) for n in self.lat_shape]
        lat_k2 = outer_lattice_k2(tuple(list_))
        lat_k2 = lat_k2[..., :(1 + self.lat_shape[-1]//2)]  # trim for rfftn
        return lat_k2


# =============================================================================
def outer_lattice_k2(args):
    """Calculate a multi-dimensional lattice `k^2` grid (in lattice units).

    See `outer_arange` for the implementation; the only differences are:
    1. `arange_gen` is set to a function that calculates `k^2`
    2. `rule` is set to sum the inputs.

    Eaxmple:
        >>> outer_lattice_k2(tuple([(0, 1, 3) for _ in range(2)]))
        >>> tensor([[0.0000, 0.2448, 0.9194],
                    [0.2448, 0.4897, 1.1642],
                    [0.9194, 1.1642, 1.8388]])
    """
    def arange_gen(*k_tuple):
        lat_k2 = lambda k: 4 * torch.sin(k/2)**2  # lattice k^2
        return lat_k2(torch.linspace(*k_tuple))
    return outer_arange(args, rule=lambda a, b: a+b, arange_gen=arange_gen)
