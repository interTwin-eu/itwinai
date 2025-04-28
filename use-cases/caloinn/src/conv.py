import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import FrEIA.modules as fm

# import torchutils
# import n_checks as check


class myConv(fm.InvertibleModule):
    def __init__(self, dims_in, dims_c=None):
        super().__init__(dims_in, dims_c)
        weight = torch.randn(dims_in[0][0], dims_in[0][0])
        # use the Q matrix from QR decompisition as the initial weight to make sure it's invertible
        q, _ = torch.qr(weight)
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight)

    def forward(self, x, rev=False, jac=True):
        (x,) = x
        x = x.unsqueeze(2).unsqueeze(3)
        _, _, height, width = x.shape

        # You can also use torch.slogdet(self.weight)[1] to summarize the operations below\n",
        dlogdet = (
            height * width * torch.log(torch.abs(torch.det(self.weight.squeeze())))
        )

        if not rev:
            out = F.conv2d(x, self.weight)
            logdet = dlogdet

        else:
            out = F.conv2d(x, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))
            logdet = -dlogdet
        return (out.squeeze(),), logdet

    def output_dims(self, input_dims):
        return input_dims


class Transform(nn.Module):
    """Base class for all transform objects."""

    def forward(self, inputs, context=None):
        raise NotImplementedError()

    def inverse(self, inputs, context=None):
        raise InverseNotAvailable()


class Linear(Transform):
    """Abstract base class for linear transforms that parameterize a weight matrix."""

    def __init__(self, features, using_cache=False):
        if not check.is_positive_int(features):
            raise TypeError("Number of features must be a positive integer.")
        super().__init__()

        self.features = features
        self.bias = nn.Parameter(torch.zeros(features))

        # Caching flag and values.
        self.using_cache = using_cache
        self.cache = LinearCache()

    def forward(self, inputs, context=None):
        if not self.training and self.using_cache:
            self._check_forward_cache()
            outputs = F.linear(inputs, self.cache.weight, self.bias)
            logabsdet = self.cache.logabsdet * outputs.new_ones(outputs.shape[0])
            return outputs, logabsdet
        else:
            return self.forward_no_cache(inputs)

    def _check_forward_cache(self):
        if self.cache.weight is None and self.cache.logabsdet is None:
            self.cache.weight, self.cache.logabsdet = self.weight_and_logabsdet()

        elif self.cache.weight is None:
            self.cache.weight = self.weight()

        elif self.cache.logabsdet is None:
            self.cache.logabsdet = self.logabsdet()

    def inverse(self, inputs, context=None):
        if not self.training and self.using_cache:
            self._check_inverse_cache()
            outputs = F.linear(inputs - self.bias, self.cache.inverse)
            logabsdet = (-self.cache.logabsdet) * outputs.new_ones(outputs.shape[0])
            return outputs, logabsdet
        else:
            return self.inverse_no_cache(inputs)

    def _check_inverse_cache(self):
        if self.cache.inverse is None and self.cache.logabsdet is None:
            (
                self.cache.inverse,
                self.cache.logabsdet,
            ) = self.weight_inverse_and_logabsdet()

        elif self.cache.inverse is None:
            self.cache.inverse = self.weight_inverse()

        elif self.cache.logabsdet is None:
            self.cache.logabsdet = self.logabsdet()

    def train(self, mode=True):
        if mode:
            # If training again, invalidate cache.
            self.cache.invalidate()
        return super().train(mode)

    def use_cache(self, mode=True):
        if not check.is_bool(mode):
            raise TypeError("Mode must be boolean.")
        self.using_cache = mode

    def weight_and_logabsdet(self):
        # To be overridden by subclasses if it is more efficient to compute the weight matrix
        # and its logabsdet together.
        return self.weight(), self.logabsdet()

    def weight_inverse_and_logabsdet(self):
        # To be overridden by subclasses if it is more efficient to compute the weight matrix
        # inverse and weight matrix logabsdet together.
        return self.weight_inverse(), self.logabsdet()

    def forward_no_cache(self, inputs):
        """Applies `forward` method without using the cache."""
        raise NotImplementedError()

    def inverse_no_cache(self, inputs):
        """Applies `inverse` method without using the cache."""
        raise NotImplementedError()

    def weight(self):
        """Returns the weight matrix."""
        raise NotImplementedError()

    def weight_inverse(self):
        """Returns the inverse weight matrix."""
        raise NotImplementedError()

    def logabsdet(self):
        """Returns the log absolute determinant of the weight matrix."""
        raise NotImplementedError()


class LULinear(Linear):
    """A linear transform where we parameterize the LU decomposition of the weights."""

    def __init__(self, features, using_cache=False, identity_init=True, eps=1e-3):
        super().__init__(features, using_cache)

        self.eps = eps

        self.lower_indices = np.tril_indices(features, k=-1)
        self.upper_indices = np.triu_indices(features, k=1)
        self.diag_indices = np.diag_indices(features)

        n_triangular_entries = ((features - 1) * features) // 2

        self.lower_entries = nn.Parameter(torch.zeros(n_triangular_entries))
        self.upper_entries = nn.Parameter(torch.zeros(n_triangular_entries))
        self.unconstrained_upper_diag = nn.Parameter(torch.zeros(features))

        self._initialize(identity_init)

    def _initialize(self, identity_init):
        init.zeros_(self.bias)

        if identity_init:
            init.zeros_(self.lower_entries)
            init.zeros_(self.upper_entries)
            constant = np.log(np.exp(1 - self.eps) - 1)
            init.constant_(self.unconstrained_upper_diag, constant)
        else:
            stdv = 1.0 / np.sqrt(self.features)
            init.uniform_(self.lower_entries, -stdv, stdv)
            init.uniform_(self.upper_entries, -stdv, stdv)
            init.uniform_(self.unconstrained_upper_diag, -stdv, stdv)

    def _create_lower_upper(self):
        lower = self.lower_entries.new_zeros(self.features, self.features)
        lower[self.lower_indices[0], self.lower_indices[1]] = self.lower_entries
        # The diagonal of L is taken to be all-ones without loss of generality.
        lower[self.diag_indices[0], self.diag_indices[1]] = 1.0

        upper = self.upper_entries.new_zeros(self.features, self.features)
        upper[self.upper_indices[0], self.upper_indices[1]] = self.upper_entries
        upper[self.diag_indices[0], self.diag_indices[1]] = self.upper_diag

        return lower, upper

    def forward_no_cache(self, inputs):
        """Cost:
            output = O(D^2N)
            logabsdet = O(D)
        where:
            D = num of features
            N = num of inputs
        """
        lower, upper = self._create_lower_upper()
        outputs = F.linear(inputs, upper)
        outputs = F.linear(outputs, lower, self.bias)
        logabsdet = self.logabsdet() * inputs.new_ones(outputs.shape[0])
        return outputs, logabsdet

    def inverse_no_cache(self, inputs):
        """Cost:
            output = O(D^2N)
            logabsdet = O(D)
        where:
            D = num of features
            N = num of inputs
        """
        lower, upper = self._create_lower_upper()
        outputs = inputs - self.bias
        outputs, _ = torch.triangular_solve(
            outputs.t(), lower, upper=False, unitriangular=True
        )
        outputs, _ = torch.triangular_solve(
            outputs, upper, upper=True, unitriangular=False
        )
        outputs = outputs.t()

        logabsdet = -self.logabsdet()
        logabsdet = logabsdet * inputs.new_ones(outputs.shape[0])

        return outputs, logabsdet

    def weight(self):
        """Cost:
            weight = O(D^3)
        where:
            D = num of features
        """
        lower, upper = self._create_lower_upper()
        return lower @ upper

    def weight_inverse(self):
        """Cost:
            inverse = O(D^3)
        where:
            D = num of features
        """
        lower, upper = self._create_lower_upper()
        identity = torch.eye(self.features, self.features)
        lower_inverse, _ = torch.triangular_solve(
            identity, lower, upper=False, unitriangular=True
        )
        weight_inverse, _ = torch.triangular_solve(
            lower_inverse, upper, upper=True, unitriangular=False
        )
        return weight_inverse

    @property
    def upper_diag(self):
        return F.softplus(self.unconstrained_upper_diag) + self.eps

    def logabsdet(self):
        """Cost:
            logabsdet = O(D)
        where:
            D = num of features
        """
        return torch.sum(torch.log(self.upper_diag))


class OneByOneConvolution(LULinear):
    """An invertible 1x1 convolution with a fixed permutation, as introduced in the Glow paper.
    Reference:
    > D. Kingma et. al., Glow: Generative flow with invertible 1x1 convolutions, NeurIPS 2018.
    """

    def __init__(self, num_channels, using_cache=False, identity_init=True):
        super().__init__(num_channels, using_cache, identity_init)
        self.permutation = RandomPermutation(num_channels, dim=1)

    def _lu_forward_inverse(self, inputs, inverse=False):
        b, c, h, w = inputs.shape
        inputs = inputs.permute(0, 2, 3, 1).reshape(b * h * w, c)

        if inverse:
            outputs, logabsdet = super().inverse(inputs)
        else:
            outputs, logabsdet = super().forward(inputs)

        outputs = outputs.reshape(b, h, w, c).permute(0, 3, 1, 2)
        logabsdet = logabsdet.reshape(b, h, w)

        return outputs, torchutils.sum_except_batch(logabsdet)

    def forward(self, inputs, context=None):
        if inputs.dim() != 4:
            raise ValueError("Inputs must be a 4D tensor.")

        inputs, _ = self.permutation(inputs)

        return self._lu_forward_inverse(inputs, inverse=False)

    def inverse(self, inputs, context=None):
        if inputs.dim() != 4:
            raise ValueError("Inputs must be a 4D tensor.")

        outputs, logabsdet = self._lu_forward_inverse(inputs, inverse=True)

        outputs, _ = self.permutation.inverse(outputs)

        return outputs, logabsdet
