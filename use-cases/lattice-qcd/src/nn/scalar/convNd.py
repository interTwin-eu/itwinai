# Copyright (c) 2023 Javad Komijani, Elias Nyholm

import torch
from typing import Tuple, Union


class ConvNd(torch.nn.Module):
    """Implements an N dimentional CNN using (N-1) dimensional CNN."""

    def __init__(self,
                    in_channels: int,
                    out_channels: int,
                    kernel_size: Union[Tuple[int, ...], int],
                    *,
                    conv_ndim: int,
                    stride: int = 1,
                    padding: Union[Tuple[int, ...], int, str] = 'same',
                    padding_mode: str = 'circular',
                    dilation: int = 1,
                    groups: int = 1,
                    bias: bool = True,
                    device = None,
                    dtype = None
                    ):

        super().__init__()

        assert conv_ndim > 1, "conv_ndim must be larger than 1."
        assert stride == 1, "Strides other than 1 not yet implemented!"
        assert dilation == 1, "Dilation rate other than 1 not yet implemented!"
        assert groups == 1, "Groups other than 1 not yet implemented!"
        assert padding_mode in ['circular'], \
            "Padding modes other than circular not yet implemented!"
        
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * conv_ndim

        if isinstance(padding, int):
            padding = tuple([padding]* conv_ndim)
        if isinstance(padding, tuple):
            assert padding[0] == kernel_size[0] // 2, \
                "Only padding mode 'same' is implemented in the first dimension!"
            padding_lower_dim = padding[1:]
        elif isinstance(padding, str):
            assert padding == 'same', \
                "Only padding mode 'same' is implemented in the first dimension!"
            padding_lower_dim = padding

        self.conv_ndim = conv_ndim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        kernel_size_lower_dim = kernel_size[1:]
        out_channels_lower_dim = out_channels * kernel_size[0]

        kwargs = dict(in_channels = in_channels,
                        out_channels = out_channels_lower_dim,
                        kernel_size = kernel_size_lower_dim,
                        stride = stride,
                        padding = padding_lower_dim,
                        padding_mode = padding_mode,
                        dilation = dilation,
                        groups = groups,
                        bias = False,  # bias added seperately
                        device = device,
                        dtype = dtype
                       )

        if conv_ndim > 4:
            self._conv_lower_dim = ConvNd(conv_ndim - 1, **kwargs)
        elif conv_ndim == 4:
            self._conv_lower_dim = torch.nn.Conv3d(**kwargs)
        elif conv_ndim == 3:
            self._conv_lower_dim = torch.nn.Conv2d(**kwargs)
        elif conv_ndim == 2:
            self._conv_lower_dim = torch.nn.Conv1d(**kwargs)

        if bias:
            self.bias = torch.nn.Parameter(
                    torch.randn(out_channels, dtype=dtype, device=device)
                    )
        else:
            self.bias = None

    def forward(self, input):

        if len(input.shape) == self.conv_ndim + 1:
            input = input.unsqueeze(0)  # introduce batch dimension

        assert len(input.shape) == self.conv_ndim + 2, "Inconsistant input shape"

        in_shape = input.shape

        batch_size, in_channels = in_shape[0], in_shape[1]
        tensor_size_0, tensor_size_rest = in_shape[2], in_shape[3:]
        kernel_size_0 = self.kernel_size[0]

        assert self.in_channels == in_channels, "Inconsistant input shape"

        big_batch_size = batch_size * tensor_size_0

        # swap dims 1&2 and reshape to make it ready to pass to _conv_lower_dim
        input = input.movedim(1, 2).reshape(big_batch_size, in_channels, *tensor_size_rest)

        out_lower_dim = self._conv_lower_dim(input)

        out_lower_dim = out_lower_dim.reshape(
                batch_size, tensor_size_0, self.out_channels, kernel_size_0,
                *tensor_size_rest
                )

        big_out = torch.empty_like(out_lower_dim)

        for i in range(kernel_size_0):
            j = (kernel_size_0 - 1) // 2 - i
            big_out[:, :, :, i] = out_lower_dim[:, :, :, i].roll(j, dims=1)

        # sum over the remaining kernel dimension & undo swaping dims 1&2
        out = torch.sum(big_out, dim=3).movedim(2, 1).contiguous()

        if self.bias is not None:
            bias_new_shape = (1, self.out_channels, *[1]*(len(out.shape[2:])))
            out = out + self.bias.reshape(*bias_new_shape)

        return out

    @property
    def weight(self):
        return self._to_standard_weight_shape(self._conv_lower_dim.weight)

    def _to_standard_weight_shape(self, weight):

        in_channels = self.in_channels
        out_channels = self.out_channels
        kernel_size = self.kernel_size
        # b: basic, i: interim, s: standard
        b_shape = (out_channels * kernel_size[0], in_channels, *kernel_size[1:])
        i_shape = (out_channels, kernel_size[0], in_channels, *kernel_size[1:])
        s_shape = (out_channels, in_channels, *kernel_size)

        return weight.reshape(*i_shape).movedim(1, 2).reshape(s_shape)


# =============================================================================
class Conv4d(ConvNd):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, conv_ndim=4, **kwargs)


# =============================================================================
def sanity_check(
        conv_ndim=3, sizes=[4, 6, 3], dtype=torch.float64, device=None
        ):
    """For 2 and 3 dimensional CNN compares ConvNd with torch.nn.Conv2d & 3d"""
    # sizes = [in_channels, out_channels, kernel_size]
    if conv_ndim == 2:
        CNN = torch.nn.Conv2d
    elif conv_ndim == 3:
        CNN = torch.nn.Conv3d
    else:
        raise Exception("Not implemented")

    kwargs = dict(bias=True, dtype=dtype, device=device)
    net_0 = ConvNd(*sizes, conv_ndim=conv_ndim, **kwargs)

    net_1 = CNN(*sizes, padding='same', padding_mode='circular', **kwargs)
    net_1.bias = net_0.bias
    net_1.weight.data = net_0.weight

    x = torch.randn(1000, sizes[0], *[10]*conv_ndim, dtype=dtype, device=device)
    delta = torch.mean(torch.abs(net_0(x) - net_1(x)))

    print("sanity check: " + "OK" if delta < 1e-12 else "failed")
