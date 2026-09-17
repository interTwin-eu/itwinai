"""Loss used by the original script. Not among TrainingConfiguration losses."""

import torch


def relative_l2(prediction, target):
    """Relative L2 error, the standard metric for neural operators."""
    num = torch.linalg.vector_norm(prediction - target, dim=(1, 2))
    den = torch.linalg.vector_norm(target, dim=(1, 2))
    return torch.mean(num / den)
