# Copyright (c) 2021-2022 Javad Komijani


import torch
import numpy as np

from ..stats.resampler import Resampler


# =============================================================================
def estimate_logz(logqp, n_resamples=10, method='bootstrap'):
    """Estimate log(z) from logqp = log(q) - log(p * z) by evaluating

    Integral p * z = Integral q exp(-logqp)

    which is expected to be equal to z when correctly sampled.
    """
    def calc_logz(x):
        return torch.logsumexp(x, dim=0).item() - np.log(logqp.shape[0])
    mean = calc_logz(-logqp)
    resampler = Resampler(method)
    std = np.std([calc_logz(x) for x in resampler(-logqp, n_resamples)])
    return mean, std


def fmt_val_err(value, error, err_digits=1):
    try:
        digits = -int(np.floor(np.log10(error))) + err_digits - 1
        if digits < 0:
            digits = 0
        str_ = "{0:.{2}f}({1:.0f})".format(value, error * 10**digits, digits)
    except:
        str_ = "{0}+-{1}".format(value, error)
    return str_
