# Copyright (c) 2021-2022 Javad Komijani

"""This module includes utilities for masking inputs."""


import torch


class ZebraPlanarMask:
    """
    Parameters
    ----------
    mu : int
        The first direction, parallel to zebra stripes
    nu : int
        The second direction, perpendicular to zebra stripes
    parity : int (0, 1)
        The even/odd parity of the zebra stripes
    """
    def __init__(self, *, mu, nu, parity=0, shape=None):
        """
        Parameters
        ----------
        shape : tuple/list (option)
        """
        self.mu = mu
        self.nu = nu
        self.parity = parity
        self.shape = shape
        # below the first axis is the batch axis.
        p, q = parity, (parity + 1) % 2
        self.white_ind = [slice(None)]*(1+nu) + [slice(p, None, 2)]
        self.black_ind = [slice(None)]*(1+nu) + [slice(q, None, 2)]

    def split(self, x):
        """Split in the (1 + self.nu) axis according to the zebra pattern,
        where the first axis is supposed to be the batch axis.
        Note that we use the builtin `slice` that returns a view of the
        original tensor.
        """
        return x[self.white_ind], x[self.black_ind]

    def cat(self, x_white, x_black):
        shape = list(x_white.shape) 
        shape[1 + self.nu] *= 2  # the 0 axis is the batch axis
        x = torch.zeros(shape, dtype=x_white.dtype, device=x_white.device)
        x[self.white_ind] = x_white
        x[self.black_ind] = x_black
        return x

    @property
    def subshape(self):
        """Shape of the split parts."""
        if self.shape is None:
            raise Exception("shape of the underlying lattice is not defined.")
        subshape = [ell for ell in self.shape]
        subshape[self.nu] = subshape[self.nu] // 2
        return subshape
