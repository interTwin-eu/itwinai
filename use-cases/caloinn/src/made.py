# based on https://github.com/bayesiains/nflows/tree/master/nflows/transforms/made.py
# and https://github.com/bayesiains/nflows/tree/master/nflows/transforms/autoregressive.py

from scipy.stats import special_ortho_group
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from FrEIA.modules import InvertibleModule
from .cubic import unconstrained_cubic_spline


def tile(x, n):
    if not (isinstance(n, int) and n > 0):
        raise TypeError("Argument 'n' must be a positive integer.")
    x_ = x.reshape(-1)
    x_ = x_.repeat(n)
    x_ = x_.reshape(n, -1)
    x_ = x_.transpose(1, 0)
    x_ = x_.reshape(-1)
    return x_


def _get_input_degrees(in_features):
    """Returns the degrees an input to MADE should have."""
    return torch.arange(1, in_features + 1)


class MaskedLinear(nn.Linear):
    """A linear module with a masked weight matrix."""

    def __init__(
        self,
        in_degrees,
        out_features,
        autoregressive_features,
        random_mask,
        is_output,
        bias=True,
    ):
        super().__init__(
            in_features=len(in_degrees), out_features=out_features, bias=bias
        )
        mask, degrees = self._get_mask_and_degrees(
            in_degrees=in_degrees,
            out_features=out_features,
            autoregressive_features=autoregressive_features,
            random_mask=random_mask,
            is_output=is_output,
        )
        self.register_buffer("mask", mask)
        self.register_buffer("degrees", degrees)
        if is_output:
            self.weight.data.fill_(0.0)
            self.bias.data.fill_(0.0)

    @classmethod
    def _get_mask_and_degrees(
        cls, in_degrees, out_features, autoregressive_features, random_mask, is_output
    ):
        if is_output:
            out_degrees = tile(
                _get_input_degrees(autoregressive_features),
                out_features // autoregressive_features,
            )
            mask = (out_degrees[..., None] > in_degrees).float()

        else:
            if random_mask:
                min_in_degree = torch.min(in_degrees).item()
                min_in_degree = min(min_in_degree, autoregressive_features - 1)
                out_degrees = torch.randint(
                    low=min_in_degree,
                    high=autoregressive_features,
                    size=[out_features],
                    dtype=torch.long,
                )
            else:
                max_ = max(1, autoregressive_features - 1)
                min_ = min(1, autoregressive_features - 1)
                out_degrees = torch.arange(out_features) % max_ + min_
            mask = (out_degrees[..., None] >= in_degrees).float()

        return mask, out_degrees

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)


class MaskedFeedforwardBlock(nn.Module):
    """A feedforward block based on a masked linear module.

    NOTE: In this implementation, the number of output features is taken to be equal to
    the number of input features.
    """

    def __init__(
        self,
        in_degrees,
        autoregressive_features,
        context_features=None,
        random_mask=False,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
    ):
        super().__init__()
        features = len(in_degrees)

        # Batch norm.
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(features, eps=1e-3)
        else:
            self.batch_norm = None

        # Masked linear.
        self.linear = MaskedLinear(
            in_degrees=in_degrees,
            out_features=features,
            autoregressive_features=autoregressive_features,
            random_mask=random_mask,
            is_output=False,
        )
        self.degrees = self.linear.degrees

        # Activation and dropout.
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(self, inputs, context=None):
        if self.batch_norm:
            temps = self.batch_norm(inputs)
        else:
            temps = inputs
        temps = self.linear(temps)
        temps = self.activation(temps)
        outputs = self.dropout(temps)
        return outputs


class MADE(nn.Module):
    """Implementation of MADE.

    Optionally, it can use batch norm or dropout within blocks (default is no).
    """

    def __init__(
        self,
        features,
        hidden_features,
        context_features=None,
        num_blocks=2,
        output_multiplier=1,
        random_mask=False,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
    ):
        super().__init__()

        # Initial layer.
        self.initial_layer = MaskedLinear(
            in_degrees=_get_input_degrees(features),
            out_features=hidden_features,
            autoregressive_features=features,
            random_mask=random_mask,
            is_output=False,
        )

        if context_features is not None:
            self.context_layer = nn.Linear(context_features, hidden_features)

        self.activation = activation
        blocks = []
        prev_out_degrees = self.initial_layer.degrees
        for _ in range(num_blocks):
            blocks.append(
                MaskedFeedforwardBlock(
                    in_degrees=prev_out_degrees,
                    autoregressive_features=features,
                    context_features=context_features,
                    random_mask=random_mask,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm,
                )
            )
            prev_out_degrees = blocks[-1].degrees
        self.blocks = nn.ModuleList(blocks)

        # Final layer.
        self.final_layer = MaskedLinear(
            in_degrees=prev_out_degrees,
            out_features=features * output_multiplier,
            autoregressive_features=features,
            random_mask=random_mask,
            is_output=True,
        )

    def forward(self, inputs, context=None):
        temps = self.initial_layer(inputs)
        if context is not None:
            temps += self.activation(self.context_layer(context))
        temps = self.activation(temps)
        for block in self.blocks:
            temps = block(temps, context)
        outputs = self.final_layer(temps)
        return outputs


class MaskedPiecewiseCubicAutoregressiveTransform(InvertibleModule):
    def __init__(
        self,
        dims_in,
        dims_c=[],
        num_bins: int = 10,
        bounds_init: float = 10.0,
        hidden_features: int = 512,
        permute_soft: bool = False,
        num_blocks=1,
        random_mask=False,
        activation=F.relu,
        dropout=0.0,
        use_batch_norm=False,
    ):
        super().__init__(dims_in, dims_c)

        self.num_bins = num_bins
        self.features = dims_in[0][0]
        self.tail_bound = bounds_init
        self.input_rank = len(dims_in[0]) - 1
        self.sum_dims = tuple(range(1, 2 + self.input_rank))
        if len(dims_c) == 0:
            self.conditional = False
            self.context_features = None
        else:
            assert tuple(dims_c[0][1:]) == tuple(
                dims_in[0][1:]
            ), f"Dimensions of input and condition don't agree: {dims_c} vs {dims_in}."
            self.conditional = True
            self.context_features = sum(dc[0] for dc in dims_c)
        self.autoregressive_net = MADE(
            features=self.features,
            hidden_features=hidden_features,
            context_features=self.context_features,
            num_blocks=num_blocks,
            output_multiplier=self._output_dim_multiplier(),
            random_mask=random_mask,
            activation=activation,
            dropout_probability=dropout,
            use_batch_norm=use_batch_norm,
        )
        try:
            self.permute_function = {
                0: F.linear,
                1: F.conv1d,
                2: F.conv2d,
                3: F.conv3d,
            }[self.input_rank]
        except KeyError:
            raise ValueError(f"Data is {1 + self.input_rank}D. Must be 1D-4D.")
        if permute_soft:
            w = special_ortho_group.rvs(self.features)
        else:
            w = torch.zeros((self.features, self.features))
            for i, j in enumerate(np.random.permutation(self.features)):
                w[i, self.features - i - 1] = 1.0
        self.w_perm = nn.Parameter(
            torch.FloatTensor(w).view(
                self.features, self.features, *([1] * self.input_rank)
            ),
            requires_grad=False,
        )
        self.w_perm_inv = nn.Parameter(
            torch.FloatTensor(w.T).view(
                self.features, self.features, *([1] * self.input_rank)
            ),
            requires_grad=False,
        )

    def forward(self, x, c=[], rev=False, jac=True):
        """See base class docstring"""
        (inputs,) = x
        if self.conditional:
            (context,) = c
        else:
            context = None
        if rev:
            inputs = self.permute_function(inputs, self.w_perm_inv)
            num_inputs = np.prod(inputs.shape[1:])
            outputs = torch.zeros_like(inputs)
            logabsdet = None
            for _ in range(num_inputs):
                autoregressive_params = self.autoregressive_net(outputs, context)
                outputs, logabsdet = self._elementwise(
                    inputs, autoregressive_params, True
                )
        else:
            autoregressive_params = self.autoregressive_net(inputs, context)
            outputs, logabsdet = self._elementwise(inputs, autoregressive_params)
            outputs = self.permute_function(outputs, self.w_perm)
        return (outputs,), logabsdet

    def _output_dim_multiplier(self):
        return self.num_bins * 2 + 2

    def _elementwise(self, inputs, autoregressive_params, inverse=False):
        batch_size = inputs.shape[0]

        transform_params = autoregressive_params.view(
            batch_size, self.features, self.num_bins * 2 + 2
        )

        unnormalized_widths = transform_params[..., : self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins : 2 * self.num_bins]
        derivatives = transform_params[..., 2 * self.num_bins :]
        unnorm_derivatives_left = derivatives[..., 0][..., None]
        unnorm_derivatives_right = derivatives[..., 1][..., None]

        if hasattr(self.autoregressive_net, "hidden_features"):
            unnormalized_widths /= np.sqrt(self.autoregressive_net.hidden_features)
            unnormalized_heights /= np.sqrt(self.autoregressive_net.hidden_features)

        outputs, logabsdet = unconstrained_cubic_spline(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnorm_derivatives_left=unnorm_derivatives_left,
            unnorm_derivatives_right=unnorm_derivatives_right,
            inverse=inverse,
            tail_bound=self.tail_bound,
        )
        return outputs, torch.sum(logabsdet, dim=self.sum_dims)

    def output_dims(self, input_dims):
        return input_dims
