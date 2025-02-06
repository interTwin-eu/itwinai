import math
from typing import Callable

import numpy as np
from scipy.stats import special_ortho_group
import torch
import torch.nn as nn
import torch.nn.functional as F
import FrEIA.modules as fm

import matplotlib.pyplot as plt
import numpy as np

class CubicSplineBlock(fm.InvertibleModule):

    DEFAULT_MIN_BIN_WIDTH = 1e-3
    DEFAULT_MIN_BIN_HEIGHT = 1e-3
    DEFAULT_EPS = 1e-5
    DEFAULT_QUADRATIC_THRESHOLD = 1e-3

    def __init__(self, dims_in, dims_c=[],
                 subnet_constructor: Callable = None,
                 num_bins: int = 10,
                 bounds_init: float = 1.,
                 permute_soft: bool = False,
                 tails='linear',
                 bounds_type="SOFTPLUS"):

        super().__init__(dims_in, dims_c)
        channels = dims_in[0][0]
        # rank of the tensors means 1d, 2d, 3d tensor etc.
        self.input_rank = len(dims_in[0]) - 1
        # tuple containing all dims except for batch-dim (used at various points)
        self.sum_dims = tuple(range(1, 2 + self.input_rank))
        if len(dims_c) == 0:
            self.conditional = False
            self.condition_channels = 0
        else:
            assert tuple(dims_c[0][1:]) == tuple(dims_in[0][1:]), \
                F"Dimensions of input and condition don't agree: {dims_c} vs {dims_in}."
            self.conditional = True
            self.condition_channels = sum(dc[0] for dc in dims_c)

        split_len1 = channels - channels // 2
        split_len2 = channels // 2
        self.splits = [split_len1, split_len2]
        self.num_bins = num_bins
        if self.DEFAULT_MIN_BIN_WIDTH * self.num_bins > 1.0:
            raise ValueError('Minimal bin width too large for the number of bins')
        if self.DEFAULT_MIN_BIN_HEIGHT * self.num_bins > 1.0:
            raise ValueError('Minimal bin height too large for the number of bins')

        try:
            self.permute_function = {0: F.linear,
                                     1: F.conv1d,
                                     2: F.conv2d,
                                     3: F.conv3d}[self.input_rank]
        except KeyError:
            raise ValueError(f"Data is {1 + self.input_rank}D. Must be 1D-4D.")

        if bounds_type == 'SIGMOID':
            bounds = 2. - np.log(10. / bounds_init - 1.)
            self.bounds_activation = (lambda a: 10 * torch.sigmoid(a - 2.))
        elif bounds_type == 'SOFTPLUS':
            bounds = 2. * np.log(np.exp(0.5 * 10. * bounds_init) - 1)
            self.softplus = nn.Softplus(beta=0.5)
            self.bounds_activation = (lambda a: 0.1 * self.softplus(a))
        elif bounds_type == 'EXP':
            bounds = np.log(bounds_init)
            self.bounds_activation = (lambda a: torch.exp(a))
        else:
            raise ValueError('Global affine activation must be "SIGMOID", "SOFTPLUS" or "EXP"')

        self.in_channels = channels
        #self.bounds = nn.Parameter(torch.ones(1, self.splits[1], *([1] * self.input_rank)) * float(bounds))
        self.bounds =  self.bounds_activation(torch.ones(1, self.splits[1], *([1] * self.input_rank)) * float(bounds))
        self.tails = tails

        if permute_soft:
            w = special_ortho_group.rvs(channels)
            w = torch.tensor(w, dtype=torch.get_default_dtype())
        else:
            w = torch.zeros((channels, channels))
            for i, j in enumerate(np.random.permutation(channels)):
                w[i, j] = 1.

        self.w_perm = nn.Parameter(w.view(channels, channels, *([1] * self.input_rank)),
                                   requires_grad=False)
        self.w_perm_inv = nn.Parameter(w.T.view(channels, channels, *([1] * self.input_rank)),
                                       requires_grad=False)

        if subnet_constructor is None:
            raise ValueError("Please supply a callable subnet_constructor"
                             "function or object (see docstring)")
        self.subnet = subnet_constructor(self.splits[0] + self.condition_channels, (2 * self.num_bins + 2) * self.splits[1])
        self.last_jac = None

    def _unconstrained_cubic_spline(self,
                                   inputs,
                                   theta,
                                   rev=False):


        inside_interval_mask = torch.all((inputs >= -self.bounds) & (inputs <= self.bounds),
                                         dim = -1)
        outside_interval_mask = ~inside_interval_mask

        masked_outputs = torch.zeros_like(inputs)
        masked_logabsdet = torch.zeros(inputs.shape[0], device=inputs.device)

        if self.tails == 'linear':
            masked_outputs[outside_interval_mask] = inputs[outside_interval_mask]
            masked_logabsdet[outside_interval_mask] = 0
        else:
            raise RuntimeError('{} tails are not implemented.'.format(self.tails))

        inputs = inputs[inside_interval_mask]
        theta = theta[inside_interval_mask]

        min_bin_width=self.DEFAULT_MIN_BIN_WIDTH
        min_bin_height=self.DEFAULT_MIN_BIN_HEIGHT
        eps=self.DEFAULT_EPS
        quadratic_threshold=self.DEFAULT_QUADRATIC_THRESHOLD

        bound = torch.min(self.bounds)
        left = -bound
        right = bound
        bottom = -bound
        top = bound

        unnormalized_widths = theta[...,:self.num_bins]
        unnormalized_heights = theta[...,self.num_bins:self.num_bins*2]
        unnorm_derivatives_left = theta[...,[-2]]
        unnorm_derivatives_right = theta[...,[-1]]

        if rev:
            inputs = (inputs - bottom) / (top - bottom)
        else:
            inputs = (inputs - left) / (right - left)

        widths = F.softmax(unnormalized_widths, dim=-1)
        widths = min_bin_width + (1 - min_bin_width * self.num_bins) * widths

        cumwidths = torch.cumsum(widths, dim=-1)
        cumwidths[..., -1] = 1
        cumwidths = F.pad(cumwidths, pad=(1, 0), mode='constant', value=0.0)

        heights = F.softmax(unnormalized_heights, dim=-1)
        heights = min_bin_height + (1 - min_bin_height * self.num_bins) * heights

        cumheights = torch.cumsum(heights, dim=-1)
        cumheights[..., -1] = 1
        cumheights = F.pad(cumheights, pad=(1, 0), mode='constant', value=0.0)

        slopes = heights / widths
        min_something_1 = torch.min(torch.abs(slopes[..., :-1]),
                                    torch.abs(slopes[..., 1:]))
        min_something_2 = (
                0.5 * (widths[..., 1:] * slopes[..., :-1] + widths[..., :-1] * slopes[..., 1:])
                / (widths[..., :-1] + widths[..., 1:])
        )
        min_something = torch.min(min_something_1, min_something_2)

        derivatives_left = min_bin_width + torch.sigmoid(unnorm_derivatives_left) * (3 * slopes[..., [0]] - 2*min_bin_width)
        derivatives_right = min_bin_width + torch.sigmoid(unnorm_derivatives_right) * (3 * slopes[..., [-1]] - 2*min_bin_width)

        derivatives = min_something * (torch.sign(slopes[..., :-1]) + torch.sign(slopes[..., 1:]))
        derivatives = torch.cat([derivatives_left,
                                 derivatives,
                                 derivatives_right], dim=-1)
        a = (derivatives[..., :-1] + derivatives[..., 1:] - 2 * slopes) / widths.pow(2)
        b = (3 * slopes - 2 * derivatives[..., :-1] - derivatives[..., 1:]) / widths
        c = derivatives[..., :-1]
        d = cumheights[..., :-1]

        if rev:
            bin_idx = self.searchsorted(cumheights, inputs)[..., None]
        else:
            bin_idx = self.searchsorted(cumwidths, inputs)[..., None]

        inputs_a = a.gather(-1, bin_idx)[..., 0]
        inputs_b = b.gather(-1, bin_idx)[..., 0]
        inputs_c = c.gather(-1, bin_idx)[..., 0]
        inputs_d = d.gather(-1, bin_idx)[..., 0]

        input_left_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
        input_right_cumwidths = cumwidths.gather(-1, bin_idx + 1)[..., 0]

        if rev:
            # Modified coefficients for solving the cubic.
            inputs_b_ = (inputs_b / inputs_a) / 3.
            inputs_c_ = (inputs_c / inputs_a) / 3.
            inputs_d_ = (inputs_d - inputs) / inputs_a

            delta_1 = -inputs_b_.pow(2) + inputs_c_
            delta_2 = -inputs_c_ * inputs_b_ + inputs_d_
            delta_3 = inputs_b_ * inputs_d_ - inputs_c_.pow(2)

            discriminant = 4. * delta_1 * delta_3 - delta_2.pow(2)

            depressed_1 = -2. * inputs_b_ * delta_1 + delta_2
            depressed_2 = delta_1

            three_roots_mask = discriminant >= 0  # Discriminant == 0 might be a problem in practice.
            one_root_mask = discriminant < 0

            outputs = torch.zeros_like(inputs)

            # Deal with one root cases.

            p = self.cbrt((-depressed_1[one_root_mask] + torch.sqrt(-discriminant[one_root_mask])) / 2.)
            q = self.cbrt((-depressed_1[one_root_mask] - torch.sqrt(-discriminant[one_root_mask])) / 2.)

            outputs[one_root_mask] = ((p + q)
                                      - inputs_b_[one_root_mask]
                                      + input_left_cumwidths[one_root_mask])

            # Deal with three root cases.

            theta = torch.atan2(torch.sqrt(discriminant[three_roots_mask]), -depressed_1[three_roots_mask])
            theta /= 3.

            cubic_root_1 = torch.cos(theta)
            cubic_root_2 = torch.sin(theta)

            root_1 = cubic_root_1
            root_2 = -0.5 * cubic_root_1 - 0.5 * math.sqrt(3) * cubic_root_2
            root_3 = -0.5 * cubic_root_1 + 0.5 * math.sqrt(3) * cubic_root_2

            root_scale = 2 * torch.sqrt(-depressed_2[three_roots_mask])
            root_shift = (-inputs_b_[three_roots_mask] + input_left_cumwidths[three_roots_mask])

            root_1 = root_1 * root_scale + root_shift
            root_2 = root_2 * root_scale + root_shift
            root_3 = root_3 * root_scale + root_shift

            root1_mask = ((input_left_cumwidths[three_roots_mask] - eps) < root_1)
            root1_mask &= (root_1 < (input_right_cumwidths[three_roots_mask] + eps))

            root2_mask = ((input_left_cumwidths[three_roots_mask] - eps) < root_2)
            root2_mask &= (root_2 < (input_right_cumwidths[three_roots_mask] + eps))

            root3_mask = ((input_left_cumwidths[three_roots_mask] - eps) < root_3)
            root3_mask &= (root_3 < (input_right_cumwidths[three_roots_mask] + eps))

            roots = torch.stack([root_1, root_2, root_3], dim=-1)
            masks = torch.stack([root1_mask, root2_mask, root3_mask], dim=-1)
            mask_index = torch.argsort(1.*masks, dim=-1, descending=True)[..., [0]]
            outputs[three_roots_mask] = torch.gather(roots, dim=-1, index=mask_index).view(-1)

            # Deal with a -> 0 (almost quadratic) cases.

            quadratic_mask = inputs_a.abs() < quadratic_threshold
            a = inputs_b[quadratic_mask]
            b = inputs_c[quadratic_mask]
            c = (inputs_d[quadratic_mask] - inputs[quadratic_mask])
            alpha = (-b + torch.sqrt(b.pow(2) - 4*a*c)) / (2*a)
            outputs[quadratic_mask] = alpha + input_left_cumwidths[quadratic_mask]

            shifted_outputs = (outputs - input_left_cumwidths)
            logabsdet = -torch.log((3 * inputs_a * shifted_outputs.pow(2) +
                                    2 * inputs_b * shifted_outputs +
                                    inputs_c))
        else:
            shifted_inputs = (inputs - input_left_cumwidths)
            outputs = (inputs_a * shifted_inputs.pow(3) +
                       inputs_b * shifted_inputs.pow(2) +
                       inputs_c * shifted_inputs +
                       inputs_d)

            logabsdet = torch.log((3 * inputs_a * shifted_inputs.pow(2) +
                                   2 * inputs_b * shifted_inputs +
                                   inputs_c))

        logabsdet = torch.sum(logabsdet, dim=1)

        if rev:
            outputs = outputs * (right - left) + left
            logabsdet = logabsdet - math.log(top - bottom) + math.log(right - left)
        else:
            outputs = outputs * (top - bottom) + bottom
            logabsdet = logabsdet + math.log(top - bottom) - math.log(right - left)
        masked_outputs[inside_interval_mask], masked_logabsdet[inside_interval_mask] = outputs, logabsdet

        return masked_outputs, masked_logabsdet

    def searchsorted(self, bin_locations, inputs, eps=1e-6):
        bin_locations[..., -1] += eps
        return torch.sum(
            inputs[..., None] >= bin_locations,
            dim=-1
        ) - 1

    def cbrt(self, x):
        """Cube root. Equivalent to torch.pow(x, 1/3), but numerically stable."""
        return torch.sign(x) * torch.exp(torch.log(torch.abs(x)) / 3.0)


    def _permute(self, x, rev=False):
        '''Performs the permutation and scaling after the coupling operation.
        Returns transformed outputs and the LogJacDet of the scaling operation.'''

        scale = torch.ones(x.shape[-1]).to(x.device)
        perm_log_jac = torch.sum(-torch.log(scale))

        if rev:
            return (self.permute_function(x * scale, self.w_perm_inv),
                    perm_log_jac)
        else:
            return (self.permute_function(x, self.w_perm) / scale,
                    perm_log_jac)

    def forward(self, x, c=[], rev=False, jac=True):
        '''See base class docstring'''
        self.bounds = self.bounds.to(x[0].device)
        if rev:
            x, global_scaling_jac = self._permute(x[0], rev=True)
            x = (x,)
        x1, x2 = torch.split(x[0], self.splits, dim=1)

        if self.conditional:
            x1c = torch.cat([x1, *c], 1)
        else:
            x1c = x1

        if not rev:
            theta = self.subnet(x1c).reshape(x1c.shape[0], self.splits[1], 2*self.num_bins + 2)
            x2, j2 = self._unconstrained_cubic_spline(x2, theta, rev=False)
        else:
            theta = self.subnet(x1c).reshape(x1c.shape[0], self.splits[1], 2*self.num_bins + 2)
            x2, j2 = self._unconstrained_cubic_spline(x2, theta, rev=True)

        log_jac_det = j2
        x_out = torch.cat((x1, x2), 1)
        if not rev:
            x_out, global_scaling_jac = self._permute(x_out, rev=False)

        # add the global scaling Jacobian to the total.
        # trick to get the total number of non-channel dimensions:
        # number of elements of the first channel of the first batch member
        n_pixels = x_out[0, :1].numel()
        log_jac_det += (-1)**rev * n_pixels * global_scaling_jac
        return (x_out,), log_jac_det

    def output_dims(self, input_dims):
        return input_dims


class RationalQuadraticSplineBlock(fm.InvertibleModule):

    DEFAULT_MIN_BIN_WIDTH = 1e-3
    DEFAULT_MIN_BIN_HEIGHT = 1e-3
    DEFAULT_MIN_DERIVATIVE = 1e-3

    def __init__(self, dims_in, dims_c=[],
                 subnet_constructor: Callable = None,
                 num_bins: int = 10,
                 bounds_init: float = 1.,
                 permute_soft: bool = False,
                 tails='linear',
                 bounds_type="SOFTPLUS"):

        super().__init__(dims_in, dims_c)
        channels = dims_in[0][0]
        # rank of the tensors means 1d, 2d, 3d tensor etc.
        self.input_rank = len(dims_in[0]) - 1
        # tuple containing all dims except for batch-dim (used at various points)
        self.sum_dims = tuple(range(1, 2 + self.input_rank))
        if len(dims_c) == 0:
            self.conditional = False
            self.condition_channels = 0
        else:
            assert tuple(dims_c[0][1:]) == tuple(dims_in[0][1:]), \
                F"Dimensions of input and condition don't agree: {dims_c} vs {dims_in}."
            self.conditional = True
            self.condition_channels = sum(dc[0] for dc in dims_c)

        split_len1 = channels - channels // 2
        split_len2 = channels // 2
        self.splits = [split_len1, split_len2]
        self.num_bins = num_bins
        if self.DEFAULT_MIN_BIN_WIDTH * self.num_bins > 1.0:
            raise ValueError('Minimal bin width too large for the number of bins')
        if self.DEFAULT_MIN_BIN_HEIGHT * self.num_bins > 1.0:
            raise ValueError('Minimal bin height too large for the number of bins')

        try:
            self.permute_function = {0: F.linear,
                                     1: F.conv1d,
                                     2: F.conv2d,
                                     3: F.conv3d}[self.input_rank]
        except KeyError:
            raise ValueError(f"Data is {1 + self.input_rank}D. Must be 1D-4D.")



        if bounds_type == 'SIGMOID':
            bounds = 2. - np.log(10. / bounds_init - 1.)
            self.bounds_activation = (lambda a: 10 * torch.sigmoid(a - 2.))
        elif bounds_type == 'SOFTPLUS':
            bounds = 2. * np.log(np.exp(0.5 * 10. * bounds_init) - 1)
            self.softplus = nn.Softplus(beta=0.5)
            self.bounds_activation = (lambda a: 0.1 * self.softplus(a))
        elif bounds_type == 'EXP':
            bounds = np.log(bounds_init)
            self.bounds_activation = (lambda a: torch.exp(a))
        elif bounds_type == 'LIN':
            bounds = bounds_init
            self.bounds_activation = (lambda a: a)
        else:
            raise ValueError('Global affine activation must be "SIGMOID", "SOFTPLUS" or "EXP"')

        self.in_channels         = channels
        self.bounds = self.bounds_activation(torch.ones(1, self.splits[1], *([1] * self.input_rank)) * float(bounds))
        self.tails = tails

        if permute_soft:
            w = special_ortho_group.rvs(channels)
        else:
            w = np.zeros((channels, channels))
            for i, j in enumerate(np.random.permutation(channels)):
                w[i, j] = 1.

        # self.w_perm = nn.Parameter(torch.FloatTensor(w).view(channels, channels, *([1] * self.input_rank)),
        #                            requires_grad=False)
        # self.w_perm_inv = nn.Parameter(torch.FloatTensor(w.T).view(channels, channels, *([1] * self.input_rank)),
        #                                requires_grad=False)
        self.w_perm = nn.Parameter(torch.Tensor(w).view(channels, channels, *([1] * self.input_rank)),
                                   requires_grad=False)
        self.w_perm_inv = nn.Parameter(torch.Tensor(w.T).view(channels, channels, *([1] * self.input_rank)),
                                       requires_grad=False)
        
        if subnet_constructor is None:
            raise ValueError("Please supply a callable subnet_constructor"
                             "function or object (see docstring)")
        self.subnet = subnet_constructor(self.splits[0] + self.condition_channels, (3 * self.num_bins - 1) * self.splits[1])
        self.last_jac = None

    def _unconstrained_rational_quadratic_spline(self,
                                   inputs,
                                   theta,
                                   rev=False):

        inside_interval_mask = torch.all((inputs >= -self.bounds) & (inputs <= self.bounds),
                                         dim = -1)
        outside_interval_mask = ~inside_interval_mask

        masked_outputs = torch.zeros_like(inputs)
        masked_logabsdet = torch.zeros(inputs.shape[0], dtype=inputs.dtype).to(inputs.device)

        min_bin_width=self.DEFAULT_MIN_BIN_WIDTH
        min_bin_height=self.DEFAULT_MIN_BIN_HEIGHT
        min_derivative=self.DEFAULT_MIN_DERIVATIVE


        if self.tails == 'linear':
            masked_outputs[outside_interval_mask] = inputs[outside_interval_mask]
            masked_logabsdet[outside_interval_mask] = 0

        else:
            raise RuntimeError('{} tails are not implemented.'.format(self.tails))
        inputs = inputs[inside_interval_mask]
        theta = theta[inside_interval_mask, :]
        bound = torch.min(self.bounds)

        left = -bound
        right = bound
        bottom = -bound
        top = bound

        #if not rev and (torch.min(inputs) < left or torch.max(inputs) > right):
        #    raise ValueError("Spline Block inputs are not within boundaries")
        #elif rev and (torch.min(inputs) < bottom or torch.max(inputs) > top):
        #    raise ValueError("Spline Block inputs are not within boundaries")

        unnormalized_widths = theta[...,:self.num_bins]
        unnormalized_heights = theta[...,self.num_bins:self.num_bins*2]
        unnormalized_derivatives = theta[...,self.num_bins*2:]

        unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
        constant = np.log(np.exp(1 - min_derivative) - 1)
        unnormalized_derivatives[..., 0] = constant
        unnormalized_derivatives[..., -1] = constant



        widths = F.softmax(unnormalized_widths, dim=-1)
        widths = min_bin_width + (1 - min_bin_width * self.num_bins) * widths
        cumwidths = torch.cumsum(widths, dim=-1)
        cumwidths = F.pad(cumwidths, pad=(1, 0), mode='constant', value=0.0)
        cumwidths = (right - left) * cumwidths + left
        cumwidths[..., 0] = left
        cumwidths[..., -1] = right
        widths = cumwidths[..., 1:] - cumwidths[..., :-1]

        derivatives = min_derivative + F.softplus(unnormalized_derivatives)

        heights = F.softmax(unnormalized_heights, dim=-1)
        heights = min_bin_height + (1 - min_bin_height * self.num_bins) * heights
        cumheights = torch.cumsum(heights, dim=-1)
        cumheights = F.pad(cumheights, pad=(1, 0), mode='constant', value=0.0)
        cumheights = (top - bottom) * cumheights + bottom
        cumheights[..., 0] = bottom
        cumheights[..., -1] = top
        heights = cumheights[..., 1:] - cumheights[..., :-1]

        if rev:
            bin_idx = self.searchsorted(cumheights, inputs)[..., None]
        else:
            bin_idx = self.searchsorted(cumwidths, inputs)[..., None]

        input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
        input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

        input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
        delta = heights / widths
        input_delta = delta.gather(-1, bin_idx)[..., 0]

        input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
        input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

        input_heights = heights.gather(-1, bin_idx)[..., 0]

        if rev:
            a = (((inputs - input_cumheights) * (input_derivatives
                                                 + input_derivatives_plus_one
                                                 - 2 * input_delta)
                  + input_heights * (input_delta - input_derivatives)))
            b = (input_heights * input_derivatives
                 - (inputs - input_cumheights) * (input_derivatives
                                                  + input_derivatives_plus_one
                                                  - 2 * input_delta))
            c = - input_delta * (inputs - input_cumheights)

            discriminant = b.pow(2) - 4 * a * c
            
            ######################################################################
            # Mini-Debug terminal
            if not (discriminant >= 0).all():
                print(f"{discriminant=}, \n {a=}, \n {b=}, \n {c=}, \n {theta=}")
                while True:
                    inp = input()
                    print(inp)
                    if inp=="break":
                        break
                    try:
                        print(eval(inp), flush=True)
                    except:
                        print("Cannot do this", flush=True)
            #######################################################################
            
            assert (discriminant >= 0).all()

            root = (2 * c) / (-b - torch.sqrt(discriminant))
            outputs = root * input_bin_widths + input_cumwidths

            theta_one_minus_theta = root * (1 - root)
            denominator = input_delta + ((input_derivatives + input_derivatives_plus_one - 2 * input_delta)
                                         * theta_one_minus_theta)
            derivative_numerator = input_delta.pow(2) * (input_derivatives_plus_one * root.pow(2)
                                                         + 2 * input_delta * theta_one_minus_theta
                                                         + input_derivatives * (1 - root).pow(2))
            logabsdet = - torch.log(derivative_numerator) + 2 * torch.log(denominator)

        else:
            theta = (inputs - input_cumwidths) / input_bin_widths
            theta_one_minus_theta = theta * (1 - theta)

            numerator = input_heights * (input_delta * theta.pow(2)
                                         + input_derivatives * theta_one_minus_theta)
            denominator = input_delta + ((input_derivatives + input_derivatives_plus_one - 2 * input_delta)
                                         * theta_one_minus_theta)
            outputs = input_cumheights + numerator / denominator

            derivative_numerator = input_delta.pow(2) * (input_derivatives_plus_one * theta.pow(2)
                                                         + 2 * input_delta * theta_one_minus_theta
                                                         + input_derivatives * (1 - theta).pow(2))
            logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        logabsdet = torch.sum(logabsdet, dim=1)

        masked_outputs[inside_interval_mask], masked_logabsdet[inside_interval_mask] = outputs, logabsdet

        return masked_outputs, masked_logabsdet

    def searchsorted(self, bin_locations, inputs, eps=1e-6):
        bin_locations[..., -1] += eps
        return torch.sum(
            inputs[..., None] >= bin_locations,
            dim=-1
        ) - 1

    def _permute(self, x, rev=False):
        '''Performs the permutation and scaling after the coupling operation.
        Returns transformed outputs and the LogJacDet of the scaling operation.'''

        scale = torch.ones(x.shape[-1]).to(x.device)
        perm_log_jac = torch.sum(-torch.log(scale))
        if rev:
            return (self.permute_function(x * scale, self.w_perm_inv),
                    perm_log_jac)
        else:
            return (self.permute_function(x, self.w_perm) / scale,
                    perm_log_jac)

    def forward(self, x, c=[], rev=False, jac=True):
        '''See base class docstring'''
        self.bounds = self.bounds.to(x[0].device)
        
        # For debugging
        # print(np.exp(c[0].cpu().numpy()))
        self.cond = torch.exp(c[0])
        self.data = x[0]

        if rev:
            x, global_scaling_jac = self._permute(x[0], rev=True)
            x = (x,)

        x1, x2 = torch.split(x[0], self.splits, dim=1)

        if self.conditional:
            x1c = torch.cat([x1, *c], 1)
        else:
            x1c = x1

        if not rev:
            theta = self.subnet(x1c).reshape(x1c.shape[0], self.splits[1], 3*self.num_bins - 1)
            x2, j2 = self._unconstrained_rational_quadratic_spline(x2, theta, rev=False)
        else:
            theta = self.subnet(x1c).reshape(x1c.shape[0], self.splits[1], 3*self.num_bins - 1)
            x2, j2 = self._unconstrained_rational_quadratic_spline(x2, theta, rev=True)
        log_jac_det = j2
        x_out = torch.cat((x1, x2), 1)

        if not rev:
            x_out, global_scaling_jac = self._permute(x_out, rev=False)
        # add the global scaling Jacobian to the total.
        # trick to get the total number of non-channel dimensions:
        # number of elements of the first channel of the first batch member
        n_pixels = x_out[0, :1].numel()
        log_jac_det += (-1)**rev * n_pixels * global_scaling_jac
        return (x_out,), log_jac_det

    def output_dims(self, input_dims):
        return input_dims
