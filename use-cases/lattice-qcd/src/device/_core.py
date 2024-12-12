# Copyright (c) 2023 Javad Komijani, Elias Nyholm

import torch
import torch.distributed as dist
import numpy as np
import os
import warnings

from functools import partial
from torch.distributed import init_process_group, destroy_process_group
from torch.multiprocessing.spawn import ProcessException


# =============================================================================
class DDP(torch.nn.parallel.DistributedDataParallel):
    # After wrapping a Module with DistributedDataParallel, the attributes of
    # the module (e.g. custom methods) became inaccessible. To access them,
    # a workaround is to use a subclass of DistributedDataParallel as here.
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


# =============================================================================
class ModelDeviceHandler:
    def __init__(self, model):
        self._model = model
        self.nranks = dist.get_world_size()
        self.rank = dist.get_rank()
        self.local_rank = dist.get_rank()%torch.cuda.device_count()

        self.seed = None

    def set_seed(self, seed):
        self.seed = seed
        torch.manual_seed(self.seed)
        print(f"Seed set for rank {self.rank}: {self.seed}")

    def to(self, *args, **kwargs):
        self._model.net_.to(*args, **kwargs)
        self._model.prior.to(*args, **kwargs)

    def ddp_wrapper(self):
        # Detect device
        device = torch.device(f"cuda:{self.local_rank}")

        # First, move the model (prior and net_) to the specific GPU
        self._model.prior.to(device=device, dtype=None, non_blocking=False)
        self._model.net_.to(device=device, dtype=None, non_blocking=False)

        # Second, wrap the net_ with DDP class
        self._model.net_ = DDP(self._model.net_, \
                device_ids=[device], output_device=device)

    def all_gather_into_tensor(self, x):
        if self.nranks == 1:
            return x
        else:
            out_shape = list(x.shape)
            out_shape[0] *= self.nranks
            out = torch.zeros(*out_shape, dtype=x.dtype, device=x.device)
            torch.distributed.all_gather_into_tensor(out, x)
            return out


class DistributedFunc:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, model, *args, **kwargs):
        out = self.fn(model, *args, **kwargs) # call function
        destroy_process_group()  # clean-up NCCL process
        return out


