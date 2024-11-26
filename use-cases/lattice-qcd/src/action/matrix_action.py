# Copyright (c) 2021-2022 Javad Komijani

"""This is a module for defining matrix models..."""


import torch


class MatrixAction:
    r"""The action is defined for $n \times n$ matrix M as

    .. math::
        S = beta / n  Tr f(M) \Gamma \\
        f(M) = Re \sum a_k M^k.
    """
    def __init__(self, *, beta, staples_matrix=None):
        # staples_matrix is the Gamma matrix above
        self.beta = beta
        self.staples_matrix = staples_matrix

    def reset_parameters(self, *, beta):
        self.beta = beta
       
    def __call__(self, cfgs):
        return self.action(cfgs)
  
    def action(self, cfgs):
        """Returns action corresponding to input configurations."""
        action = self.action_density(cfgs)
        # The following if True for more-than-one-point matrix models.
        if action.ndim > 1:
            dim = tuple(range(1, action.ndim))  # 0 axis is the batch axis
            action = torch.sum(action, dim=dim)

        return action
   
    def action_density(self, cfgs):
        """Returns action density corresponding to input configurations."""
        if self.staples_matrix is not None:
            cfgs = cfgs @ self.staples_matrix
        return -self.beta * calc_reduced_trace(cfgs).real

    def log_prob(self, x, action_logz=0):
        """Returns log probability up to an additive constant."""
        return -self.action(x) - action_logz

    @property
    def parameters(self):
        return {'beta': self.beta}


def calc_trace(x):
    return torch.sum(torch.diagonal(x, dim1=-2, dim2=-1), dim=-1)


def calc_reduced_trace(x):  # reduced trace = 1/n trace()
    return torch.mean(torch.diagonal(x, dim1=-2, dim2=-1), dim=-1)
