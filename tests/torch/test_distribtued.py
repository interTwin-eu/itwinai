"""Test distributed training strategies."""

from torch.optim import SGD
from torch.nn import Linear
import os
import subprocess
import pytest
import torch
from itwinai.torch.distributed import (
    TorchDistributedStrategy,
    TorchDDPStrategy,
    DeepSpeedStrategy,
    HorovodStrategy
)


def test_strategy(strategy: TorchDistributedStrategy):
    strategy.init()
    lrank = strategy.local_rank()
    grank = strategy.global_rank()
    nnodes = strategy.global_world_size() // strategy.local_world_size()

    # Gather local ranks
    if strategy.is_main_worker:
        lranks = strategy.gather_obj(lrank, dst_rank=0)
    else:
        strategy.gather_obj(lrank, dst_rank=0)
    assert len(lranks) == strategy.global_world_size()
    assert sum(lranks) == sum(range(strategy.local_world_size())) * nnodes

    # Gather global ranks
    if strategy.is_main_worker:
        granks = strategy.gather_obj(grank, dst_rank=0)
    else:
        strategy.gather_obj(grank, dst_rank=0)
    assert len(granks) == strategy.global_world_size()
    assert sum(granks) == sum(range(strategy.global_world_size()))

    # # Gather tensor from CPU
    # my_tensor = torch.ones(10) * strategy.global_rank()
    # if strategy.is_main_worker:
    #     tensors = strategy.gather(my_tensor, dst_rank=0)
    # else:
    #     strategy.gather(my_tensor, dst_rank=0)
    # assert torch.stack(tensors).sum() == sum(range(strategy.global_world_size())) * 10

    # TODO: test model and optim distribution


@pytest.mark.hpc
@pytest.mark.torchrun
def test_ddp_strategy():
    """Test TorchDDPStrategy class"""
    strategy = TorchDDPStrategy(backend='nccl')
    test_strategy(strategy)
    strategy.clean_up()


@pytest.mark.hpc
@pytest.mark.torchrun
def test_deepspeed_strategy():
    """Test DeepSpeedStrategy class"""
    strategy = DeepSpeedStrategy(backend='nccl')
    test_strategy(strategy)
    strategy.clean_up()


@pytest.mark.hpc
@pytest.mark.mpirun
def test_horovod_strategy():
    """Test HorovodStrategy class"""
    strategy = HorovodStrategy()
    test_strategy(strategy)
    strategy.clean_up()


@pytest.mark.skip(reason="Decorator is not really used atm.")
@pytest.mark.hpc
def test_distributed_decorator(torch_env):
    """Test function decorator. Needs torchrun cmd."""
    cmd = (f"{torch_env}/bin/torchrun "
           " --nnodes=1 --nproc_per_node=2 --rdzv_id=100 "
           "--rdzv_backend=c10d --rdzv_endpoint=localhost:29400 "
           "tests/torch/distribtued_decorator.py")
    subprocess.run(cmd.split(), check=True)


@pytest.mark.skip(reason="avoid nested torchrun")
@pytest.mark.hpc
def test_distributed_trainer(torch_env):
    """Test vanilla torch distributed trainer. Needs torchrun cmd."""
    cmd = (f"{torch_env}/bin/torchrun "
           "--nnodes=1 --nproc_per_node=2 --rdzv_id=100 "
           "--rdzv_backend=c10d --rdzv_endpoint=localhost:29400 "
           "tests/torch/torch_dist_trainer.py")
    subprocess.run(cmd.split(), check=True)


# Utility functions for distributed testing

def setup_distributed_env():
    """Setup distributed environment variables for testing"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    if 'RANK' not in os.environ:
        os.environ['RANK'] = '0'
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = '0'
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = '1'


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, size=100):
        self.size = size
        self.data = torch.randn(size, 10)
        self.targets = torch.randint(0, 2, (size,))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


@pytest.fixture
def simple_model():
    return Linear(10, 2)


@pytest.fixture
def optimizer(simple_model):
    return SGD(simple_model.parameters(), lr=0.01)


@pytest.fixture
def dataset():
    return DummyDataset()


@pytest.mark.hpc
@pytest.mark.torchrun
class TestTorchDDPStrategy:
    @pytest.fixture
    def ddp_strategy(self):
        from itwinai.torch.distributed import TorchDDPStrategy
        strategy = TorchDDPStrategy(backend='nccl' if torch.cuda.is_available() else 'gloo')
        setup_distributed_env()
        strategy.init()
        yield strategy
        strategy.clean_up()

    def test_init(self, ddp_strategy):
        assert ddp_strategy.is_initialized
        assert ddp_strategy.backend in ['nccl', 'gloo']
        assert ddp_strategy.global_world_size() >= 1
        assert ddp_strategy.local_world_size() >= 1
        assert ddp_strategy.global_rank() >= 0
        assert ddp_strategy.local_rank() >= 0

    def test_distributed_model(self, ddp_strategy, simple_model, optimizer):
        dist_model, dist_optimizer, _ = ddp_strategy.distributed(simple_model, optimizer)
        assert hasattr(dist_model, 'module') or isinstance(dist_model, torch.nn.Module)
        assert isinstance(dist_optimizer, torch.optim.Optimizer)

    def test_dataloader(self, ddp_strategy, dataset):
        dataloader = ddp_strategy.create_dataloader(
            dataset,
            batch_size=16,
            shuffle=True
        )
        assert isinstance(dataloader, torch.utils.data.DataLoader)
        assert isinstance(dataloader.sampler, torch.utils.data.distributed.DistributedSampler)

    def test_gather_operations(self, ddp_strategy):
        # Test tensor gather
        local_tensor = torch.tensor([ddp_strategy.global_rank()], device=ddp_strategy.device())
        gathered = ddp_strategy.gather(local_tensor)
        if ddp_strategy.is_main_worker:
            assert len(gathered) == ddp_strategy.global_world_size()

        # Test object gather
        local_obj = {"rank": ddp_strategy.global_rank()}
        gathered_obj = ddp_strategy.gather_obj(local_obj)
        if ddp_strategy.is_main_worker:
            assert len(gathered_obj) == ddp_strategy.global_world_size()

        # Test allgather
        all_gathered = ddp_strategy.allgather_obj(local_obj)
        assert len(all_gathered) == ddp_strategy.global_world_size()


@pytest.mark.hpc
@pytest.mark.torchrun
class TestDeepSpeedStrategy:
    @pytest.fixture
    def ds_strategy(self):
        from itwinai.torch.distributed import DeepSpeedStrategy
        strategy = DeepSpeedStrategy(backend='nccl' if torch.cuda.is_available() else 'gloo')
        setup_distributed_env()
        strategy.init()
        yield strategy
        strategy.clean_up()

    def test_init(self, ds_strategy):
        assert ds_strategy.is_initialized
        assert ds_strategy.backend in ['nccl', 'gloo']
        assert hasattr(ds_strategy, 'deepspeed')

    def test_distributed_model(self, ds_strategy, simple_model, optimizer):
        ds_config = {
            "train_batch_size": 16,
            "fp16": {"enabled": False},
            "optimizer": {
                "type": "SGD",
                "params": {
                    "lr": 0.001
                }
            }
        }

        dist_model, dist_optimizer, _ = ds_strategy.distributed(
            simple_model,
            optimizer=optimizer,
            config=ds_config
        )
        assert hasattr(dist_model, 'module')
        assert dist_optimizer is not None


@pytest.mark.hpc
@pytest.mark.mpirun
class TestHorovodStrategy:
    @pytest.fixture
    def hvd_strategy(self):
        from itwinai.torch.distributed import HorovodStrategy
        strategy = HorovodStrategy()
        strategy.init()
        yield strategy
        strategy.clean_up()

    def test_init(self, hvd_strategy):
        assert hvd_strategy.is_initialized
        assert hasattr(hvd_strategy, 'hvd')

    def test_distributed_model(self, hvd_strategy, simple_model, optimizer):
        dist_model, dist_optimizer, _ = hvd_strategy.distributed(
            simple_model,
            optimizer=optimizer,
            op=hvd_strategy.hvd.Average
        )
        assert isinstance(dist_model, torch.nn.Module)
        assert hasattr(dist_optimizer, '_allreduce_grads')

    def test_gather_operations(self, hvd_strategy):
        local_tensor = torch.tensor([hvd_strategy.global_rank()], device=hvd_strategy.device())
        gathered = hvd_strategy.gather(local_tensor)
        assert len(gathered) == hvd_strategy.global_world_size()

        local_obj = {"rank": hvd_strategy.global_rank()}
        all_gathered = hvd_strategy.allgather_obj(local_obj)
        assert len(all_gathered) == hvd_strategy.global_world_size()

# Conftest.py content for distributed test configuration


@pytest.fixture(autouse=True)
def cuda_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "distributed: mark test to run only in distributed environment"
    )


def pytest_collection_modifyitems(config, items):
    if "WORLD_SIZE" not in os.environ or os.environ["WORLD_SIZE"] == "1":
        skip_distributed = pytest.mark.skip(reason="need distributed environment to run")
        for item in items:
            if "distributed" in item.keywords:
                item.add_marker(skip_distributed)
