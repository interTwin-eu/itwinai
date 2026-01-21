import torch

from typing import Callable


def f_except(f: Callable[..., torch.Tensor], x: torch.Tensor, *dim: int, **kwargs: Any) -> torch.Tensor:
    """Apply f on all dimensions except those specified in dim"""
    result = x
    dimensions = [d for d in range(x.dim()) if d not in dim]

    if not dimensions:
        raise ValueError(
            f"Cannot exclude dims {dim} from x with shape {x.shape}: No dimensions left."
        )

    return f(result, dim=dimensions, **kwargs)


def sum_except(x: torch.Tensor, *dim: int) -> torch.Tensor:
    """Sum all dimensions of x except the ones specified in dim"""
    return f_except(torch.sum, x, *dim)


def sum_except_batch(x: torch.Tensor) -> torch.Tensor:
    """Sum all dimensions of x except the first (batch) dimension"""
    return sum_except(x, 0)
