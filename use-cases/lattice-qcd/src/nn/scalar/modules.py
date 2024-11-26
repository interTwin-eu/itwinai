# Copyright (c) 2021-2023 Javad Komijani

"""This module contains new neural networks..."""


import torch
import copy
import numpy as np

from ...lib.spline import RQSpline
from ...lib.linalg import neighbor_mean
from .convNd import Conv4d


class AvgNeighborPool(torch.nn.Module):
    """Return average of all neighbors"""

    def forward(self, x):
        return neighbor_mean(x, dim=range(1, x.ndim))


class Abs(torch.nn.Module):
    """Added for adding to the list of activations"""

    def forward(self, x):
        return torch.abs(x)


class Expit(torch.nn.Module):
    """This can be also called Sigmoid and is basically torch.nn.Sigmoid"""

    def forward(self, x):
        return torch.special.expit


class Logit(torch.nn.Module):
    """This is inverse of Sigmoid"""

    def forward(self, x):
        return torch.special.logit


ACTIVATIONS = torch.nn.ModuleDict(
                    [['tanh', torch.nn.Tanh()],
                     ['relu', torch.nn.ReLU()],
                     ['leaky_relu', torch.nn.LeakyReLU()],
                     ['softplus', torch.nn.Softplus()],
                     ['avg_neighbor_pool', AvgNeighborPool()],
                     ['abs', Abs()],
                     ['expit', Expit()],
                     ['logit', Logit()],
                     ['none', torch.nn.Identity()]
                    ]
              )


class PlusBias(torch.nn.Module):

    def __init__(self, out_features):
        super().__init__()
        self.out_features = out_features
        self.bias = torch.nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        return x + self.bias


class ConvAct(torch.nn.Sequential):
    """
    As an extension to torch.nn.Conv2d, this network is a sequence of
    convolutional layers with possible hidden layers and activations and other
    dimensions.

    Instantiating this class with the default optional variables is equivalent
    to instantiating torch.nn.Conv2d with following optional varaibles:
    padding = 'same' and padding_mode = 'circular'.

    As an option, one can provide a list/tuple for `hidden_sizes`. Then, one
    must also provide another list/tuple for activations using the option
    `acts`; the lenght of `acts` must be equal to the lenght of `hidden_sizes`
    plus 1 (for the output layer).
    There is also another option for pre-activation of the input: `pre_act`.

    The axes of the input and output tensors are treated as
    :math:`tensor(:, ch, ...)`, where `:` stands for the batch axis,
    `ch` for the channels axis, and `...` for the features axes.

    .. math::

        out(:, ch_o, ...) = bias(ch_o) +
                        \sum_{ch_i} weight(ch_o, ...) \star input(:, ch_i, ...)

    where :math:`\star` is n-dimensional cross-correlation operator acting on
    the features axes. The supported features dinensions are 1, 2, 3, and 4.

    Parameters
    ----------
    in_channels (int):
        Number of channels in the input data
    out_channels (int):
        Number of channels produced by the convolution
    kernel_size (int or tuple):
        Size of the convolving kernel
    conv_dim (int, optional):
        Dimension of the convolving kernel (default is 2)
    hidden_sizes (list/tuple of int, optional):
        Sizes of hidden layers (default is [])
    acts (list/tuple of str or None, optional):
        Activations after each layer (default is None)
    pre_act (str or None, optional):
        A possible activation layer before the rest (default is None)
    """

    Conv = {1: torch.nn.Conv1d,
            2: torch.nn.Conv2d,
            3: torch.nn.Conv3d,
            4: Conv4d
            }

    def __init__(self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            conv_dim: int = 2,
            hidden_sizes = [],
            acts = [None],
            pre_act = None,
            **extra_kwargs  # all other kwargs to pass to torch.nn.Conv?d
            ):

        Conv = self.Conv[conv_dim]
        sizes = [in_channels, *hidden_sizes, out_channels]
        assert len(acts) == len(hidden_sizes) + 1

        conv_kwargs = dict(padding='same', padding_mode='circular')
        conv_kwargs.update(extra_kwargs)

        nets = [] if pre_act is None else [ACTIVATIONS[pre_act]]

        for i, act in enumerate(acts):
            nets.append(Conv(sizes[i], sizes[i+1], kernel_size, **conv_kwargs))
            if act is not None:
                nets.append(ACTIVATIONS[act])

        super().__init__(*nets)

        # save all inputs so that the can be used later for transfer learning
        conv_kwargs.update(
                dict(in_channels=in_channels, out_channels=out_channels,
                     kernel_size=kernel_size, conv_dim=conv_dim,
                     hidden_sizes=hidden_sizes, acts=acts, pre_act=pre_act
                     )
                )
        self.conv_kwargs = conv_kwargs

    def set_param2zero(self):
        for net in self:
            for param in net.parameters():
                torch.nn.init.zeros_(param)

    def _outdated_transfer(self, scale_factor=1, **extra):
        # Outdated: must be updated and ...
        """
        Returns a copy of the current module if scale_factor is 1.
        Otherwise, uses the input scale_factor to resize the kernel size.
        """
        if scale_factor == 1:
            return copy.deepcopy(self)
        else:
            pass  # change the kernel size as below

        ksize = self.conv_kwargs['kernel_size']  # original kernel size
        ksize = 1 + 2 * round((ksize - 1) * scale_factor/2)  # new kernel size

        conv_kwargs = dict(**self.conv_kwargs)
        conv_kwargs['kernel_size'] = ksize

        new_size = [ksize] * conv_kwargs['conv_dim']
        resize = lambda p: torch.nn.functional.interpolate(p, size=new_size)

        state_dict_conv = {key: resize(value)
                for key, value in self.net[::2].state_dict().items()
                }

        state_dict_acts = {key: value
                for key, value in self.net[1::2].state_dict().items()
                }

        state_dict = dict(**state_dict_conv, **state_dict_acts)

        new_net = self.__class__(**conv_kwargs)
        new_net.net.load_state_dict(state_dict)

        return new_net


class LinearAct(torch.nn.Sequential):
    """
    As an extension to torch.nn.Linear, this network is a sequence of linear
    layers with possible hidden layers and activations.

    As an option, one can provide a list/tuple for `hidden_sizes`. Then, one
    must also provide another list/tuple for activations using the option
    `acts`; the lenght of `acts` must be equal to the lenght of `hidden_sizes`
    plus 1 (for the output layer).
    There is also another option for pre-activation of the input: `pre_act`.

    The axes of the input and output tensors are treated as
    :math:`tensor(..., f)`, where `...` stands for any number of dimensions
    and `f` for the features axis.

    Parameters
    ----------
    in_features (int):
        Number of features in the input data
    out_features (int):
        Number of features in the output data
    hidden_sizes (list/tuple of int, optional):
        Sizes of hidden layers (default is [])
    acts (list/tuple of str or None, optional):
        Activations after each layer (default is None)
    pre_act (str or None, optional):
        A possible activation layer before the rest (default is None)
    """
    def __init__(self,
            in_features: int,
            out_features: int,
            hidden_sizes = [],
            acts = [None],
            pre_act = None,
            final_bias = False,  # e.g., can be used with 'abs' activation
            features_axis = -1,
            **linear_kwargs  # all other kwargs to pass to torch.nn.Linear
            ):

        Linear = torch.nn.Linear
        sizes = [in_features, *hidden_sizes, out_features]
        assert len(acts) == len(hidden_sizes) + 1

        nets = [] if pre_act is None else [ACTIVATIONS[pre_act]]

        for i, act in enumerate(acts):
            nets.append(Linear(sizes[i], sizes[i+1], **linear_kwargs))
            if act is not None:
                nets.append(ACTIVATIONS[act])

        if final_bias:
            nets.append(PlusBias(out_features))

        super().__init__(*nets)

        # save all inputs so that the can be used later for transfer learning
        linear_kwargs.update(
                dict(in_features=in_features, out_features=out_features,
                     hidden_sizes=hidden_sizes, acts=acts, pre_act=pre_act,
                     final_bias=final_bias, features_axis=features_axis
                     )
                )
        self.linear_kwargs = linear_kwargs

    def forward(self, x):
        features_axis = self.linear_kwargs['features_axis']
        if features_axis == -1:
            return super().forward(x)
        else:
            x = torch.movedim(x, features_axis, -1)
            x = super().forward(x)
            return torch.movedim(x, -1, features_axis)

    def set_param2zero(self):
        for net in self:
            for param in net.parameters():
                torch.nn.init.zeros_(param)


class SplineNet(torch.nn.Module):
    """
    Return a neural network for spline interpolation/extrapolation.
    The input `knots_len` specifies the number of knots of the spline.
    In general, the first knot is always at (xlim[0], ylim[0]) and the last
    knot is always at (xlim[1], ylim[1]) and the coordintes of other knots are
    network parameters to be trained, unless one explicitely provides
    `knots_x` and/or `knots_y`.
    Assuming `knots_x` is None, one needs `(knots_len - 1)` parameters to
    specify the `x` position of the knots (with softmax);
    similarly for the `y` position.
    There will be additional `knots_len` parameters to specify the derivatives
    at knots unless `smooth == True`.

    Note that `knots_len` must be at least equal 2. Also note that

        SplineNet(2, smooth=True)

    is basically an identity net (although it has two dummy parameters!)

    Can be used as a probability distribution convertor for variables with
    nonzero probability in [0, 1].

    Parameters
    ----------
    knots_len : int
        number of knots of the spline.
    xlim & ylim : array-like
        the min and max values for `x` & `y` of the knots.
    knots_x & knots_y & knots_d : None or tensors, optional
        fix corresponding tensors to the input if provided.
     spline_shape : array-like  # (Question: Is this USED at all?)
        specifies number of splines organized as a tensor
        (default is [], indicating there is only one spline).
     knots_axis : int
        relevant only if spline_shape is not empty list (default value is -1).
    """

    softmax = torch.nn.Softmax(dim=0)
    softplus = torch.nn.Softplus(beta=np.log(2))
    # we set the beta of Softplus to log(2) so that self.softplust(0) is 1.
    # Then, it would be easy to set the derivatives to 1 (with zero inputs).

    def __init__(self, knots_len, xlim=(0, 1), ylim=(0, 1),
            knots_x=None, knots_y=None, knots_d=None,
            spline_shape=[], knots_axis=-1,
            smooth=False, Spline=RQSpline, label='spline', **spline_kwargs
            ):
        super().__init__()
        self.label = label

        # knots_len and spline_shape are relevant only if flag is True
        flag = (knots_x is None) or (knots_y is None) or (knots_d is None)

        assert not (flag and knots_len < 2), "oops: knots_len < 2 for splines"

        self.knots_len = knots_len
        self.knots_x = knots_x
        self.knots_y = knots_y
        self.knots_d = knots_d
        self.spline_shape = spline_shape
        self.knots_axis = knots_axis

        self.Spline = Spline
        self.spline_kwargs = spline_kwargs

        init = lambda n: torch.zeros(*spline_shape, n)

        if knots_x is None:
            self.xlim, self.xwidth = xlim, xlim[1] - xlim[0]
            self.weights_x = torch.nn.Parameter(init(knots_len - 1))

        if knots_y is None:
            self.ylim, self.ywidth = ylim, ylim[1] - ylim[0]
            self.weights_y = torch.nn.Parameter(init(knots_len - 1))

        if knots_d is None:
            self.weights_d = None if smooth else torch.nn.Parameter(init(knots_len))

    def forward(self, x):
        spline = self.make_spline()
        if len(self.spline_shape) > 0:
            return spline(x)
        else:
            return spline(x.ravel()).reshape(x.shape)

    def backward(self, x):
        spline = self.make_spline()
        if len(self.spline_shape) > 0:
            return spline.backward(x)
        else:
            return spline.backward(x.ravel()).reshape(x.shape)

    def make_spline(self):
        dim = self.knots_axis
        zero_shape = (*self.spline_shape, 1)
        zero = lambda w: torch.zeros(zero_shape, device=w.device)
        cumsumsoftmax = lambda w: torch.cumsum(self.softmax(w), dim=dim)
        to_coord = lambda w: torch.cat((zero(w), cumsumsoftmax(w)), dim=dim)
        to_deriv = lambda d: self.softplus(d) if d is not None else None

        knots_x = self.knots_x
        if knots_x is None:
            knots_x = to_coord(self.weights_x) * self.xwidth + self.xlim[0]

        knots_y = self.knots_y
        if knots_y is None:
            knots_y = to_coord(self.weights_y) * self.ywidth + self.ylim[0]

        knots_d = self.knots_d
        if knots_d is None:
            knots_d = to_deriv(self.weights_d)

        mydict = {'knots_x': knots_x, 'knots_y': knots_y, 'knots_d': knots_d}

        return self.Spline(**mydict, **self.spline_kwargs)
