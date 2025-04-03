# Copyright (c) 2021-2023 Javad Komijani

"""This is a module for defining actions..."""


import torch


class ScalarPhi4Action:
    r"""The action is defined as

    .. math::
        S = \int d^n x (
            \frac{\kappa}{2} (\partial_\mu \phi(x))^2
            + \frac{m^2}{2} \phi(x)^2
            + \lambda \phi(x)^4
            ).
    """
    def __init__(self, *, m_sq: float, lambd: float, kappa: float = 1.0, a: float = 1.0):
        self.kappa = kappa
        self.m_sq = m_sq
        self.lambd = lambd
        self.a = a

    def get_coef(self, lat_ndim):
        """Returns coefficients of different terms after absorbing appropriate
        powers of the lattice spacing in them.
        """
        a = self.a
        kappa = self.kappa * a**(lat_ndim - 2)
        m_sq = self.m_sq * a**lat_ndim
        lambd = self.lambd * a**lat_ndim
        w_0 = 0.5 * (2 * kappa)
        w_2 = 0.5 * (m_sq + 2 * kappa * lat_ndim)
        w_4 = lambd
        return w_0, w_2, w_4

    def __call__(self, cfgs):
        return self.action(cfgs)

    def action(self, cfgs):
        """Returns action corresponding to input configurations."""
        dim = tuple(range(1, cfgs.ndim))  # 0 axis -> batch axis
        w0, w2, w4 = self.get_coef(cfgs.ndim - 1)
        action = torch.sum(w2 * cfgs**2 + w4 * cfgs**4, dim=dim)
        for mu in dim:
            action -= w0 * torch.sum(cfgs * torch.roll(cfgs, 1, mu), dim=dim)

        return action

    def action_density(self, cfgs):
        """Returns action density corresponding to input configurations.

        Action density is not unique; a version is returned that is symmetric
        and also its kinetic term is always positive.
        """
        dim = tuple(range(1, cfgs.ndim))  # 0 axis -> batch axis
        w0, w2, w4 = self.get_coef(cfgs.ndim - 1)
        w2 = w2 - w0 * (cfgs.ndim - 1)  # this m_sq * a**ndim
        action_density = w2 * cfgs**2 + w4 * cfgs**4
        for mu in dim:
            action_density += (w0/4) * (cfgs - torch.roll(cfgs, -1, mu))**2
            action_density += (w0/4) * (cfgs - torch.roll(cfgs, +1, mu))**2

        return action_density

    def potential(self, x):
        return self.m_sq * x**2 + self.lambd * x**4

    def log_prob(self, x, action_logz=0):
        """Returns log probability up to an additive constant."""
        return -self.action(x) - action_logz

