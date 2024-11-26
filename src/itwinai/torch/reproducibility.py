# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

"""This module provides the tools to support reproducible execution of torch scripts."""

import random
from typing import Optional

import numpy as np
import torch


def seed_worker(worker_id):
    """Seed DataLoader worker."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(rnd_seed: Optional[int], deterministic_cudnn: bool = True) -> torch.Generator:
    """Set torch random seed and return a PRNG object.

    Args:
        rnd_seed (Optional[int]): random seed. If None, the seed is not set.
        deterministic_cudnn (bool): if True, sets
            ``torch.backends.cudnn.benchmark = False``, which may affect
            performances.

    Returns:
        torch.Generator: PRNG object.
    """
    g = torch.Generator()
    if rnd_seed is not None:
        # Deterministic execution
        np.random.seed(rnd_seed)
        random.seed(rnd_seed)
        torch.manual_seed(rnd_seed)
        g.manual_seed(rnd_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(rnd_seed)
            torch.cuda.manual_seed_all(rnd_seed)
        if deterministic_cudnn:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    return g
