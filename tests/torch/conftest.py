import pytest
import torch
from typing import Generator
from itwinai.torch.distributed import (
    TorchDistributedStrategy,
    TorchDDPStrategy,
    DeepSpeedStrategy,
    HorovodStrategy
)


@pytest.fixture(scope='package')
def ddp_strategy() -> Generator[TorchDistributedStrategy, None, None]:
    """Instantiate Torch's DistributedDataParallel strategy."""
    strategy = TorchDDPStrategy(backend='nccl' if torch.cuda.is_available() else 'gloo')
    strategy.init()
    yield strategy
    strategy.clean_up()


@pytest.fixture(scope='package')
def deepspeed_strategy() -> Generator[DeepSpeedStrategy, None, None]:
    """Instantiate DeepSpeed strategy."""
    strategy = DeepSpeedStrategy(backend='nccl' if torch.cuda.is_available() else 'gloo')
    strategy.init()
    yield strategy
    strategy.clean_up()


@pytest.fixture(scope='package')
def horovod_strategy() -> Generator[HorovodStrategy, None, None]:
    """Instantiate Horovod strategy."""
    strategy = HorovodStrategy()
    strategy.init()
    yield strategy
    strategy.clean_up()
