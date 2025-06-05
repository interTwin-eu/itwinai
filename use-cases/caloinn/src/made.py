# based on https://github.com/bayesiains/nflows/tree/master/nflows/transforms/made.py
# and https://github.com/bayesiains/nflows/tree/master/nflows/transforms/autoregressive.py

from scipy.stats import special_ortho_group
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from typing import List, Optional, Tuple, Callable

from FrEIA.modules import InvertibleModule
from .cubic import unconstrained_cubic_spline


def tile(x: torch.Tensor, n: int) -> torch.Tensor:
    """Tile the input tensor `n` times along a new dimension and flatten the result.

    Args:
        x (torch.Tensor): Input tensor.
        n (int): Number of repetitions. Must be a positive integer.

    Returns:
        torch.Tensor: Tiled and reshaped tensor.

    Raises:
        TypeError: If `n` is not a positive integer.
    """
    if not (isinstance(n, int) and n > 0):
        raise TypeError("Argument 'n' must be a positive integer.")
    x_ = x.reshape(-1)
    x_ = x_.repeat(n)
    x_ = x_.reshape(n, -1)
    x_ = x_.transpose(1, 0)
    x_ = x_.reshape(-1)
    return x_


def _get_input_degrees(in_features: int) -> torch.Tensor:
    """Generate degrees for input features in MADE.

    Args:
        in_features (int): Number of input features.

    Returns:
        torch.Tensor: Tensor of degrees from 1 to `in_features`.
    """
    return torch.arange(1, in_features + 1)


class MaskedLinear(nn.Linear):
    """A linear module with an autoregressive mask on the weight matrix."""

    def __init__(
        self,
        in_degrees: torch.Tensor,
        out_features: int,
        autoregressive_features: int,
        random_mask: bool,
        is_output: bool,
        bias: bool = True,
    ) -> None:
        """Initializes a MaskedLinear layer.

        Args:
            in_degrees (torch.Tensor): Degrees of the input nodes.
            out_features (int): Number of output features.
            autoregressive_features (int): Number of autoregressive features.
            random_mask (bool): Whether to use a random mask.
            is_output (bool): Whether this is the output layer.
            bias (bool, optional): Whether to include bias. Defaults to True.
        """
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
        cls,
        in_degrees: torch.Tensor,
        out_features: int,
        autoregressive_features: int,
        random_mask: bool,
        is_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate the mask and degrees for the layer.

        Args:
            in_degrees (torch.Tensor): Degrees of the input features.
            out_features (int): Number of output features.
            autoregressive_features (int): Number of autoregressive features.
            random_mask (bool): Whether to use a random mask.
            is_output (bool): Whether this is the output layer.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The mask and the degrees.
        """
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
        in_degrees: torch.Tensor,
        autoregressive_features: int,
        context_features: Optional[int] = None,
        random_mask: bool = False,
        activation: callable = F.relu,
        dropout_probability: float = 0.0,
        use_batch_norm: bool = False,
    ) -> None:
        """Initializes a MaskedFeedforwardBlock.

        Args:
            in_degrees (torch.Tensor): Degrees of the input features.
            autoregressive_features (int): Number of autoregressive features.
            context_features (Optional[int], optional): Context size. Defaults to None.
            random_mask (bool, optional): Whether to use a random mask. Defaults to False.
            activation (callable, optional): Activation function. Defaults to F.relu.
            dropout_probability (float, optional): Dropout rate. Defaults to 0.0.
            use_batch_norm (bool, optional): Whether to apply batch normalization. Defaults to False.
        """
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

    def forward(
        self,
        inputs: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the block to inputs (and optionally context).

        Args:
            inputs (torch.Tensor): Input tensor.
            context (Optional[torch.Tensor], optional): Context tensor. Defaults to None.

        Returns:
            torch.Tensor: Output tensor.
        """
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
        features: int,
        hidden_features: int,
        context_features: Optional[int] = None,
        num_blocks: int = 2,
        output_multiplier: int = 1,
        random_mask: bool = False,
        activation: callable = F.relu,
        dropout_probability: float = 0.0,
        use_batch_norm: bool = False,
    ) -> None:
        """Initialize the MADE network.

        Args:
            features (int): Number of input features.
            hidden_features (int): Number of hidden units.
            context_features (Optional[int], optional): Context feature size. Defaults to None.
            num_blocks (int, optional): Number of feedforward blocks. Defaults to 2.
            output_multiplier (int, optional): Output dimension multiplier. Defaults to 1.
            random_mask (bool, optional): Use random connectivity mask. Defaults to False.
            activation (callable, optional): Activation function. Defaults to F.relu.
            dropout_probability (float, optional): Dropout rate. Defaults to 0.0.
            use_batch_norm (bool, optional): Whether to use batch normalization. Defaults to False.
        """
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

    def forward(
        self, inputs: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply MADE to the input and context.

        Args:
            inputs (torch.Tensor): Input tensor.
            context (Optional[torch.Tensor], optional): Context tensor. Defaults to None.

        Returns:
            torch.Tensor: Output tensor.
        """
        temps = self.initial_layer(inputs)
        if context is not None:
            temps += self.activation(self.context_layer(context))
        temps = self.activation(temps)
        for block in self.blocks:
            temps = block(temps, context)
        outputs = self.final_layer(temps)
        return outputs


class MaskedPiecewiseCubicAutoregressiveTransform(InvertibleModule):
    """An autoregressive transform based on piecewise cubic splines with masking.'"""
    def __init__(
        self,
        dims_in: List[Tuple[int, ...]],
        dims_c: List[Tuple[int, ...]] = [],
        num_bins: int = 10,
        bounds_init: float = 10.0,
        hidden_features: int = 512,
        permute_soft: bool = False,
        num_blocks: int = 1,
        random_mask: bool = False,
        activation: Callable = F.relu,
        dropout: float = 0.0,
        use_batch_norm: bool = False,
    ) -> None:
        """Initializes the transform module.

        Args:
            dims_in (List[Tuple[int, ...]]): Input tensor dimensions.
            dims_c (List[Tuple[int, ...]], optional): Context tensor dimensions.
            num_bins (int, optional): Number of spline bins. Defaults to 10.
            bounds_init (float, optional): Initial spline boundary. Defaults to 10.0.
            hidden_features (int, optional): Number of hidden features. Defaults to 512.
            permute_soft (bool, optional): Whether to use a soft permutation. Defaults to False.
            num_blocks (int, optional): Number of MADE blocks. Defaults to 1.
            random_mask (bool, optional): Whether to use random masks. Defaults to False.
            activation (Callable, optional): Activation function. Defaults to F.relu.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
            use_batch_norm (bool, optional): Whether to use batch normalization. Defaults to False.
        
        Raises:
            ValueError: If data dimensionality is not supported.
            AssertionError: If context and input shapes are incompatible.
        """
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

    def forward(
        self,
        x: Tuple[torch.Tensor],
        c: List[torch.Tensor] = [],
        rev: bool = False,
        jac: bool = True,
    ) -> Tuple[Tuple[torch.Tensor], Optional[torch.Tensor]]:
        """Applies the transform (or its inverse) to the input tensor.

        Args:
            x (Tuple[torch.Tensor]): Input tensor, shape (B, F, ...).
            c (List[torch.Tensor], optional): Context tensors. Defaults to [].
            rev (bool, optional): Whether to apply the inverse. Defaults to False.
            jac (bool, optional): Whether to compute the log-determinant. Defaults to True.

        Returns:
            Tuple[Tuple[torch.Tensor], Optional[torch.Tensor]]: 
                - Transformed output tensor.
                - Log absolute determinant of the Jacobian, if `jac` is True.
        """
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

    def _output_dim_multiplier(self) -> int:
        """Computes the number of output channels required by the autoregressive net.

        Returns:
            int: Output multiplier (number of channels per input feature).
        """
        return self.num_bins * 2 + 2

    def _elementwise(
        self,
        inputs: torch.Tensor,
        autoregressive_params: torch.Tensor,
        inverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies the element-wise spline transform.

        Args:
            inputs (torch.Tensor): Input tensor.
            autoregressive_params (torch.Tensor): Spline parameters from MADE.
            inverse (bool, optional): Whether to apply the inverse transformation. Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Transformed tensor and log-determinant.
        """
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
