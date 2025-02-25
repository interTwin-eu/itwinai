# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

import logging
import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest
import torch

from itwinai.torch.distributed import (
    DeepSpeedStrategy,
    HorovodStrategy,
    TorchDDPStrategy,
    TorchDistributedStrategy,
)


@pytest.fixture(scope="package")
def ddp_strategy() -> Generator[TorchDistributedStrategy, None, None]:
    """Instantiate Torch's DistributedDataParallel strategy."""
    strategy = TorchDDPStrategy(backend="nccl" if torch.cuda.is_available() else "gloo")
    strategy.init()
    yield strategy
    strategy.clean_up()


@pytest.fixture(scope="package")
def deepspeed_strategy() -> Generator[DeepSpeedStrategy, None, None]:
    """Instantiate DeepSpeed strategy."""
    strategy = DeepSpeedStrategy(backend="nccl" if torch.cuda.is_available() else "gloo")
    strategy.init()
    yield strategy
    strategy.clean_up()


@pytest.fixture(scope="package")
def horovod_strategy() -> Generator[HorovodStrategy, None, None]:
    """Instantiate Horovod strategy."""
    strategy = HorovodStrategy()
    strategy.init()
    yield strategy
    strategy.clean_up()


@pytest.fixture(scope="function")
def shared_tmp_path():
    """Return the Path to a shared filesystem that all nodes can access.
    /tmp location is usually a local filesystem. Uses as a prefix SHARED_FS_PATH
    env variable, but if that's not set it falls back to /tmp.
    """
    if not os.environ.get("SHARED_FS_PATH"):
        logging.warning(
            "SHARED_FS_PATH env var not set! Falling back to /tmp, but this could cause "
            "problems as this fixture should return the path to a location reachable by all "
            "nodes (/tmp usually isn't)."
        )
    else:
        Path(os.environ.get("SHARED_FS_PATH")).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=os.environ.get("SHARED_FS_PATH", "/tmp")) as tmp_path:
        yield Path(tmp_path)
