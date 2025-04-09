# Copyright (c) 2023 Javad Komijani


import torch


def neighbor_mean(x, dim=None):
    """Return average of all neighbors."""
    if dim is None:
        dim = range(1, x.ndim)  # zero for the batch axis

    y, len_ = 0, len(dim)
    for mu in dim:
        if x.shape[mu] == 1:  # there are no neighbors in mu direction
            len_ -= 1
        else:
            y = y + (torch.roll(x, 1, mu) + torch.roll(x, -1, mu))

    return y / (2*len_)


class NeighborMean(torch.autograd.Function):
    """Return average of all neighbors. (Alternative version.)"""
    @staticmethod
    def forward(ctx, x):
        avg = neighbor_mean(x)
        return avg

    @staticmethod
    def backward(ctx, grad_avg):
        grad_x = neighbor_mean(grad_avg)
        return grad_x


alt_neighbor_mean = NeighborMean.apply  # not neces. faster than neighbor_mean
