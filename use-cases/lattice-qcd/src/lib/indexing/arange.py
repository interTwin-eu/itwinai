# Copyright (c) 2021-2022 Javad Komijani

"""A module for expanding `torch.arange` to higher dimension."""


import torch


def arange_like(x, dim=-1):
    """Return a tensor with shape of `x`, filled with `(0, 1, ..., n)` in the
    `dim` direction, where `n = x.shape[dim]`, and repeated in all other
    directions.
    """
    if dim == -1 or dim == x.ndim - 1:
        # a special case, which can be written in a compact form as
        return torch.arange(x.shape[-1]).repeat((*x.shape[:-1], 1))

    arange = torch.arange(x.shape[dim])
    subshape = x.shape[1+dim:]
    arange = arange.view(-1, *[1]*len(subshape)).repeat((1, *subshape))
    if dim % x.ndim > 0:
        arange = arange.repeat((*x.shape[:dim], *[1]*arange.ndim))
    return arange


def outer_arange(tuple_of_tuples,
        rule=lambda a, b: a * b,
        arange_gen=torch.arange
        ):
    """Return a multi-dimensional arange using outer product of one dimensional
    tensors obtained using `torch.arange`.

    Parameters
    ----------
    tuple_of_tuples : tuple of tuples
        Each nested tuple define the `arange()` in the corresponding axis.
    rule : a lambda function, optional
        By providing a new function, one can define new operations.
        For example, setting `rule = lambda a, b: a + b` returns
        the outer sum of the inputs rather than the outer prodcut.
    arange_gen : a lambda function, optional
        By providing a new function, e.g. `torch.linspace`,
        one can generalize the function for other applications.

    Returns
    -------
    z : tensor_like
        A tensor constructed by product of the elementray tensors.

    Example
    -------

        >>> fftflow.outer_arange(((1,3,), (1,2.5,0.5),  (1,5)))
        >>> tensor([[[ 1.0000,  2.0000,  3.0000,  4.0000],
                     [ 1.5000,  3.0000,  4.5000,  6.0000],
                     [ 2.0000,  4.0000,  6.0000,  8.0000]],

                    [[ 2.0000,  4.0000,  6.0000,  8.0000],
                     [ 3.0000,  6.0000,  9.0000, 12.0000],
                     [ 4.0000,  8.0000, 12.0000, 16.0000]]])

    """
    if not isinstance(tuple_of_tuples, tuple):
        raise Exception(f"Oops: {tuple_of_tuples} is not a tuple of tuples!")

    for i, tuple_ in enumerate(tuple_of_tuples):
        if i == 0:
            outer_arange_ = arange_gen(*tuple_)
        else:
            outer_arange_ = outer(outer_arange_, arange_gen(*tuple_), rule)

    return outer_arange_


def outer_linspace(args, **kwargs):
    """Return a multi-dimensional linspace; see `outer_arange` for more info."""
    return nd_arange(args, arange=torch.linspace, **kwargs)


def outer_sum(x, y):
    """Return the outer sum of `x` and `y`. See `outer()` for more info."""
    return outer(x, y, rule=lambda a, b: a + b)


def outer(x, y, rule=lambda a, b: a * b):
    """Return the outer product of `x` and `y`.

    Parameters
    ----------
    x : tensor_like
        The first input tensor.
    y : tensor_like
        The second input tensor.
    rule : a lambda function, optional
        By providing a new function, one can define new operations.
        For example, setting `rule = lambda a, b: a + b` returns
        the outer sum of the inputs rather than the outer prodcut.

    Returns
    -------
    z : tensor_like
        The outer product of `x` and `y`.
    """
    shape = (*x.shape, *tuple([1]*len(y.shape)))
    repeat = (*tuple([1]*len(x.shape)), *y.shape)
    return rule(x.reshape(*shape).repeat(*repeat), y[None, ...])
    # return rule(np.tile(x.reshape(*shape), repeat), y[None, ...]) for ndarray
