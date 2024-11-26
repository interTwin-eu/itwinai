# Copyright (c) 2021-2022 Javad Komijani

"""This is a module for defining gauge actions..."""


import torch

from numpy import pi


class GaugeAction:
    r"""The action is defined as

    .. math::

        S = \int d^n x (...).
    """
    def __init__(self, *, beta, ndim, nc):
        self.beta = beta
        self.ndim = ndim
        self.nc = nc  # number of colors, i.e. dimension of the gauge matrix

    def reset_parameters(self, *, beta):
        self.beta = beta
       
    def __call__(self, cfgs):
        return self.action(cfgs)
  
    def action(self, cfgs):
        """Returns action corresponding to input configurations."""
        dim = tuple(range(1, 1+self.ndim))  # 0 axis is the batch axis
        action = 0
        for mu in range(1, self.ndim):
            for nu in range(mu):
                action += torch.sum(self.calc_plaq(cfgs, mu=mu, nu=nu), dim=dim)
        return -self.beta * action

    def action_density(self, cfgs):
        """Returns action density corresponding to input configurations."""
        dim = tuple(range(1, 1+self.ndim))  # 0 axis is the batch axis
        action_density = 0
        for mu in range(1, self.ndim):
            for nu in range(mu):
                action_density += self.calc_plaq(cfgs, mu=mu, nu=nu)
        return -self.beta * action_density

    def calc_plaq(self, cfgs, *, mu, nu, real=True):
        x_mu = cfgs[:, mu]
        x_nu = cfgs[:, nu]
        plaq = self.plaq_rule(
                x_mu,
                torch.roll(x_nu, -1, dims=1 + mu),
                torch.roll(x_mu, -1, dims=1 + nu),
                x_nu
                )
        return torch.real(plaq) if real else plaq

    @staticmethod
    def plaq_rule(a, b, c, d):
        mul = torch.matmul
        plaq = mul(mul(a, b), mul(d, c).adjoint())
        return calc_reduced_trace(plaq)

    def log_prob(self, x, action_logz=0):
        """Returns log probability up to an additive constant."""
        return -self.action(x) - action_logz

    @property
    def parameters(self):
        return dict(beta=self.beta, ndim=self.ndim)


class U1GaugeAction(GaugeAction):
    """A special case of GaugeAction with special `plaq_rule`, ...."""

    def __init__(self, **kwargs):
        super().__init__(nc=1, **kwargs)

    @staticmethod
    def plaq_rule(a, b, c, d):
        return a * b * torch.conj(d * c)

    def calc_topo_charge(self, cfgs):
        topo_charge = 0
        for mu in range(1, self.ndim):
            for nu in range(mu):
                angle_plaq = torch.angle(
                                self.calc_plaq(cfgs, mu=mu, nu=nu, real=False)
                                )
                dim = tuple(range(1, len(angle_plaq.shape)))
                topo_charge += torch.sum(angle_plaq, dim=dim) / (2 * pi)
        return topo_charge


def calc_trace(x):
    return torch.sum(torch.diagonal(x, dim1=-2, dim2=-1), dim=-1)


def calc_reduced_trace(x):  # reduced trace = 1/n trace()
    return torch.mean(torch.diagonal(x, dim1=-2, dim2=-1), dim=-1)
