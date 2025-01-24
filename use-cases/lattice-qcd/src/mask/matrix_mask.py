# Copyright (c) 2021-2022 Javad Komijani

"""This module includes utilities for masking inputs."""


import torch

import itertools


class MatrixMask(torch.nn.Module):
    """Each mask must have two methods: `split` and `cat` to split and
    concatenate the data according to the mask. Another method called ``purify``
    is needed to make sure the data is zero where it must be zero.
    (The ``purify`` method is used by some classes, but not all.)
    """

    def __init__(self, *, lat_shape, identity_matrix=torch.eye(2), parity=0,
            anisotropic_dir=None
            ):
        """
        Parameters
        ----------
        anisotrpic_dir : int or None
            If None, the mask is isotropic, othewise indicates anistropic
            direction
        """

        super().__init__()
        self.lat_shape = lat_shape
        self.identity_matrix = identity_matrix
        mask = self.evenodd(lat_shape, parity, anisotropic_dir=anisotropic_dir)
        self.register_buffer('mask', mask)

    @staticmethod
    def evenodd(lat_shape, parity, anisotropic_dir=None):
        shape = (*lat_shape, *[1]*len(lat_shape))
        mask = torch.empty(shape, dtype=torch.uint8)
        if anisotropic_dir is None:
            for ind in itertools.product(*tuple([range(l) for l in shape])):
                mask[ind] = (sum(ind) + parity) % 2
        else:
            mu = anisotropic_dir
            assert 0 <= mu and mu < len(lat_shape)
            for ind in itertools.product(*tuple([range(l) for l in shape])):
                mask[ind] = (sum(ind) + parity - ind[mu]) % 2
        return mask

    def split(self, x):
        mask, eye = self.mask, self.identity_matrix
        return (1 - mask) * x + mask * eye, mask * x + (1 - mask) * eye

    def cat(self, x_0, x_1):
        return x_0 + x_1 - self.identity_matrix

    def purify(self, x_chnl, channel):
        mask, eye = self.mask, self.identity_matrix
        if channel == 0:
            return (1 - mask) * x_chnl + mask * eye
        else:
            return mask * x_chnl + (1 - mask) * eye
