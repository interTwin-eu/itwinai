from typing import Dict, Tuple

import torch

from .binned import BinnedSpline


class LinearSpline(BinnedSpline):
    """A spline module that uses piecewise linear interpolation."""

    def __init__(self, *args, bins: int = 10, **kwargs) -> None:
        """Initializes the LinearSpline with a specified number of bins.

        Args:
            *args: Positional arguments passed to the superclass.
            bins (int): Number of bins to use for the spline. Defaults to 10.
            **kwargs: Keyword arguments passed to the superclass.
        """
        super().__init__(*args, **kwargs, bins=bins, parameter_counts=None)

    def _spline1(
        self, x: torch.Tensor, parameters: Dict[str, torch.Tensor], rev: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        left, right, bottom, top = (
            parameters["left"],
            parameters["right"],
            parameters["bottom"],
            parameters["top"],
        )
        return linear_spline(x, left, right, bottom, top, rev=rev)

    def _spline2(
        self, x: torch.Tensor, parameters: Dict[str, torch.Tensor], rev: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        left, right, bottom, top = (
            parameters["left"],
            parameters["right"],
            parameters["bottom"],
            parameters["top"],
        )
        return linear_spline(x, left, right, bottom, top, rev=rev)


def linear_spline(
    x: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    bottom: torch.Tensor,
    top: torch.Tensor,
    rev: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes a linear spline, connecting each knot with a straight line
    Args:
        x: input tensor
        left: left bin edges for inputs inside the spline domain
        right: same as left
        bottom: same as left
        top: same as left
        rev: whether to run in reverse

    Returns:
        splined output tensor
        log jacobian matrix entries
    """

    dx = right - left
    dy = top - bottom

    scale = dy / dx
    shift = bottom - scale * left

    if not rev:
        out = scale * x + shift
    else:
        out = (x - shift) / scale

    log_jac = torch.log(dy) - torch.log(dx)

    return out, log_jac
