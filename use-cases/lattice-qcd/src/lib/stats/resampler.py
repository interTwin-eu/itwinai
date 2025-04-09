# Copyright (c) 2021-2022 Javad Komijani


import torch
import numpy as np
from functools import partial


class Resampler:
    """
    Resample the data with bootstrap or jackknife method.

    Parameters
    ----------
    method : str, optional
        The method of resampling (default is bootstrap)
    """
    def __init__(self, method='bootstrap'):
        self.method = method

    def __call__(self, samples, n_resamples=100, binsize=1, batch_size=None):
        """
        Parameters:
        -----------
        samples : tensor/ndarray
            The main samples that are going to be resampled.

        n_resamples : int, optional
            The number of resamples, which is only relevant for the bootstrap
            method (default is 100).

        binsize : int, optional
            Size of bins for binning the data before sampling (default is 1).

        batch_size : int or None, optional
            Size of each bootstrap resample (irrelevant for jackknife).
            This is mainly for test, because the batch size in bootstrap is
            supposed to be equal to the number of samples in the original
            samples (default is None, indicating that set batch_size to the
            number of samples.
        """
        l_b = samples.shape[0] // binsize  # lenght of binned samples
        binned_samples = samples[:(l_b * binsize)].reshape(l_b, binsize, -1)

        if type(samples) == torch.Tensor:
            arange = partial(torch.arange, device=samples.device)
            randint = partial(torch.randint, device=samples.device)
            randperm = partial(torch.randperm, device=samples.device)
        else:
            arange, randint, randperm = np.arange, np.random.randint, np.random.permutation

        match self.method:
            case 'jackknife':
                n_resamples = l_b
                get_indices = lambda i: arange(l_b)[arange(l_b) != i]
                resample_shape = ((l_b - 1) * binsize, *samples.shape[1:])
            case 'bootstrap':
                if batch_size is None:
                    batch_size = l_b  # useful if method is not 'jackknife'
                get_indices = lambda i: randint(l_b, size=(batch_size,))
                resample_shape = (l_b * binsize, *samples.shape[1:])
            case 'shuffling':
                get_indices = lambda i: randperm(l_b)
                resample_shape = (l_b * binsize, *samples.shape[1:])

        for i in range(n_resamples):
            yield binned_samples[get_indices(i)].reshape(*resample_shape)

    def eval(self, samples, fn=lambda x: np.mean(x), **kwargs):
        """
        Operates/evaluates the given function on the (re)samples and returns
        the mean and standard deviation of the outputs.
        """
        x = [fn(q) for q in self(samples, **kwargs)]
        return np.mean(x), np.std(x)
