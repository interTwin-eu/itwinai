"""Various PyTorch utility functions."""

import n_checks as check
import numpy as np
import torch


def tile(x: torch.Tensor, n: int) -> torch.Tensor:
    """Tile a 1D tensor by repeating its elements in a specific pattern.

    Args:
        x (torch.Tensor): Input tensor to be tiled.
        n (int): Number of times to repeat each element. Must be positive.

    Returns:
        torch.Tensor: The tiled tensor.

    Raises:
        TypeError: If `n` is not a positive integer.
    """

    if not check.is_positive_int(n):
        raise TypeError("Argument 'n' must be a positive integer.")
    x_ = x.reshape(-1)
    x_ = x_.repeat(n)
    x_ = x_.reshape(n, -1)
    x_ = x_.transpose(1, 0)
    x_ = x_.reshape(-1)
    return x_


def sum_except_batch(x: torch.Tensor, num_batch_dims: int = 1) -> torch.Tensor:
    """Sums all elements of `x` except for the first `num_batch_dims` dimensions."""
    if not check.is_nonnegative_int(num_batch_dims):
        raise TypeError("Number of batch dimensions must be a non-negative integer.")
    reduce_dims = list(range(num_batch_dims, x.ndimension()))
    return torch.sum(x, dim=reduce_dims)


def split_leading_dim(x: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
    """Reshapes the leading dim of `x` to have the given shape."""
    new_shape = torch.Size(shape) + x.shape[1:]
    return torch.reshape(x, new_shape)


def merge_leading_dims(x: torch.Tensor, num_dims: int) -> torch.Tensor:
    """Reshapes the tensor `x` such that the first `num_dims` dimensions are merged to one."""
    if not check.is_positive_int(num_dims):
        raise TypeError("Number of leading dims must be a positive integer.")
    if num_dims > x.dim():
        raise ValueError(
            "Number of leading dims can't be greater than total number of dims."
        )
    new_shape = torch.Size([-1]) + x.shape[num_dims:]
    return torch.reshape(x, new_shape)


def repeat_rows(x: torch.Tensor, num_reps: int) -> torch.Tensor:
    """Each row of tensor `x` is repeated `num_reps` times along leading dimension."""
    if not check.is_positive_int(num_reps):
        raise TypeError("Number of repetitions must be a positive integer.")
    shape = x.shape
    x = x.unsqueeze(1)
    x = x.expand(shape[0], num_reps, *shape[1:])
    return merge_leading_dims(x, num_dims=2)


def tensor2numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def logabsdet(x: torch.Tensor) -> np.ndarray:
    """Returns the log absolute determinant of square matrix x."""
    # Note: torch.logdet() only works for positive determinant.
    _, res = torch.slogdet(x)
    return res


def random_orthogonal(size: int) -> torch.Tensor:
    """
    Returns a random orthogonal matrix as a 2-dim tensor of shape [size, size].
    """

    # Use the QR decomposition of a random Gaussian matrix.
    x = torch.randn(size, size)
    q, _ = torch.qr(x)
    return q


def get_num_parameters(model: nn.Module) -> int:
    """
    Returns the number of trainable parameters in a model of type nets.Module
    :param model: nets.Module containing trainable parameters
    :return: number of trainable parameters in model
    """
    num_parameters = 0
    for parameter in model.parameters():
        num_parameters += torch.numel(parameter)
    return num_parameters


def create_alternating_binary_mask(features: int, even: bool = True) -> torch.Tensor:
    """
    Creates a binary mask of a given dimension which alternates its masking.
    :param features: Dimension of mask.
    :param even: If True, even values are assigned 1s, odd 0s. If False, vice versa.
    :return: Alternating binary mask of type torch.Tensor.
    """
    mask = torch.zeros(features).byte()
    start = 0 if even else 1
    mask[start::2] += 1
    return mask


def create_mid_split_binary_mask(features: int) -> torch.Tensor:
    """
    Creates a binary mask of a given dimension which splits its masking at the midpoint.
    :param features: Dimension of mask.
    :return: Binary mask split at midpoint of type torch.Tensor
    """
    mask = torch.zeros(features).byte()
    midpoint = features // 2 if features % 2 == 0 else features // 2 + 1
    mask[:midpoint] += 1
    return mask


def create_random_binary_mask(features: int) -> torch.Tensor:
    """
    Creates a random binary mask of a given dimension with half of its entries
    randomly set to 1s.
    :param features: Dimension of mask.
    :return: Binary mask with half of its entries set to 1s, of type torch.Tensor.
    """
    mask = torch.zeros(features).byte()
    weights = torch.ones(features).float()
    num_samples = features // 2 if features % 2 == 0 else features // 2 + 1
    indices = torch.multinomial(
        input=weights, num_samples=num_samples, replacement=False
    )
    mask[indices] += 1
    return mask


def searchsorted(
    bin_locations: torch.Tensor, 
    inputs: torch.Tensor, 
    eps: float = 1e-6
) -> torch.Tensor:
    """Find the index of the bin each input value falls into."""
    bin_locations[..., -1] += eps
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1


def cbrt(x: torch.Tensor) -> torch.Tensor:
    """Cube root. Equivalent to torch.pow(x, 1/3), but numerically stable."""
    return torch.sign(x) * torch.exp(torch.log(torch.abs(x)) / 3.0)


def get_temperature(max_value: float, bound: float = 1 - 1e-3) -> torch.Tensor:
    """
    For a dataset with max value 'max_value', returns the temperature such that
        sigmoid(temperature * max_value) = bound.
    If temperature is greater than 1, returns 1.
    :param max_value:
    :param bound:
    :return:
    """
    max_value = torch.Tensor([max_value])
    bound = torch.Tensor([bound])
    temperature = min(-(1 / max_value) * (torch.log1p(-bound) - torch.log(bound)), 1)
    return temperature


def gaussian_kde_log_eval(
    samples: torch.Tensor, 
    query: torch.Tensor
) -> torch.Tensor:
    """Evaluate the log-density of a Gaussian kernel density estimate at query points.

    Args:
        samples (torch.Tensor): Sample points of shape `[N, D]`.
        query (torch.Tensor): Query points of shape `[M, D]`.

    Returns:
        torch.Tensor: Tensor of shape `[M]` containing the log-density estimates at each query point.
    """    
    N, D = samples.shape[0], samples.shape[-1]
    std = N ** (-1 / (D + 4))
    precision = (1 / (std**2)) * torch.eye(D)
    a = query - samples
    b = a @ precision
    c = -0.5 * torch.sum(a * b, dim=-1)
    d = -np.log(N) - (D / 2) * np.log(2 * np.pi) - D * np.log(std)
    c += d
    return torch.logsumexp(c, dim=-1)
