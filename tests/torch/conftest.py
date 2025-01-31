# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

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
