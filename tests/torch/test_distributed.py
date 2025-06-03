# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

"""Test distributed training strategies."""

from pathlib import Path
from typing import Any, Literal
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from ray.tune import RunConfig
from torch.nn import Linear
from torch.optim import SGD, Optimizer
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from itwinai.distributed import get_adaptive_ray_scaling_config, ray_cluster_is_running
from itwinai.torch.distributed import (
    DeepSpeedStrategy,
    HorovodStrategy,
    RayDDPStrategy,
    RayDeepSpeedStrategy,
    RayTorchDistributedStrategy,
    TorchDDPStrategy,
    TorchDistributedStrategy,
)
from itwinai.torch.type import DistributedStrategyError, UninitializedStrategyError


class DummyDataset(Dataset):
    def __init__(self, size=100):
        self.size = size
        self.data = torch.randn(size, 10)
        self.targets = torch.randint(0, 2, (size,))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class BaseTestDistributedStrategy:
    @pytest.fixture(scope="module")
    def strategy(self):
        """To be overridden by subclasses with the specific strategy."""
        raise NotImplementedError

    @pytest.fixture
    def simple_model(self):
        return Linear(10, 2)

    @pytest.fixture
    def optimizer(self, simple_model: nn.Module):
        return SGD(simple_model.parameters(), lr=0.01)

    @pytest.fixture
    def dataset(self):
        return DummyDataset()

    def test_cluster_properties(self, strategy: TorchDistributedStrategy):
        """Test that ranks and other simple properties are computed correctly."""
        assert strategy.is_initialized
        assert isinstance(strategy.device(), str)
        assert isinstance(strategy.is_main_worker, bool)
        assert strategy.global_world_size() >= 1
        assert strategy.local_world_size() >= 1
        assert strategy.global_rank() >= 0
        assert strategy.local_rank() >= 0

    def test_init_exceptions(
        self, strategy: TorchDistributedStrategy, simple_model: nn.Module, optimizer: Any
    ):
        """Check that the init method cannot be called twice and that the other methods raise
        and exception if called when the strategy is not initialized."""
        # Test re-initialization
        with pytest.raises(DistributedStrategyError) as init_exc:
            strategy.init()
            assert "already initialized" in init_exc.value

        # Test initialized flag
        strategy.is_initialized = False

        with pytest.raises(UninitializedStrategyError):
            strategy.distributed(simple_model, optimizer)
        with pytest.raises(UninitializedStrategyError):
            strategy.global_rank()
        with pytest.raises(UninitializedStrategyError):
            strategy.local_rank()
        with pytest.raises(UninitializedStrategyError):
            strategy.local_world_size()
        with pytest.raises(UninitializedStrategyError):
            strategy.global_world_size()
        with pytest.raises(UninitializedStrategyError):
            strategy.device()
        with pytest.raises(UninitializedStrategyError):
            strategy.is_main_worker
        x = torch.ones(2)
        with pytest.raises(UninitializedStrategyError):
            strategy.gather(x)
        with pytest.raises(UninitializedStrategyError):
            strategy.allgather_obj(x)
        with pytest.raises(UninitializedStrategyError):
            strategy.gather_obj(x)
        with pytest.raises(UninitializedStrategyError):
            strategy.clean_up()

        strategy.is_initialized = True

    def test_dataloader(self, strategy: TorchDistributedStrategy, dataset: Dataset):
        """Check that the dataloader is distributed correctly."""
        dataloader = strategy.create_dataloader(dataset, batch_size=16, shuffle=True)
        assert isinstance(dataloader, DataLoader)
        assert isinstance(dataloader.sampler, DistributedSampler)
        assert dataloader.sampler.rank == strategy.global_rank()
        assert dataloader.sampler.num_replicas == strategy.global_world_size()

    def test_gather_operations(self, strategy: TorchDistributedStrategy):
        """Test collective operations."""
        # Test tensor gather
        local_tensor = torch.tensor([strategy.global_rank()], device=strategy.device())
        gathered = strategy.gather(local_tensor)
        if strategy.is_main_worker:
            assert len(gathered) == strategy.global_world_size()
        else:
            assert gathered is None

        # Test tensor gather from CPU
        my_tensor = torch.ones(10) * strategy.global_rank()
        tensors = strategy.gather(my_tensor, dst_rank=0)
        if strategy.is_main_worker:
            assert torch.stack(tensors).sum() == sum(range(strategy.global_world_size())) * 10
        else:
            assert tensors is None

        # Test object gather
        local_obj = {"rank": strategy.global_rank()}
        gathered_obj = strategy.gather_obj(local_obj)
        if strategy.is_main_worker:
            assert len(gathered_obj) == strategy.global_world_size()
        else:
            assert gathered_obj is None

        # Test allgather
        all_gathered = strategy.allgather_obj(local_obj)
        assert len(all_gathered) == strategy.global_world_size()

    def test_ranks(self, strategy):
        """Check that the values of the ranks are as expected."""
        nnodes = strategy.global_world_size() // strategy.local_world_size()

        # Test local ranks
        lranks = strategy.gather_obj(strategy.local_rank(), dst_rank=0)
        if strategy.is_main_worker:
            assert len(lranks) == strategy.global_world_size()
            assert sum(lranks) == sum(range(strategy.local_world_size())) * nnodes
        else:
            assert lranks is None

        # Test global ranks
        granks = strategy.gather_obj(strategy.global_rank(), dst_rank=0)
        if strategy.is_main_worker:
            assert len(granks) == strategy.global_world_size()
            assert sum(granks) == sum(range(strategy.global_world_size()))
        else:
            assert granks is None


@pytest.mark.hpc
@pytest.mark.torch_dist
class TestTorchDDPStrategy(BaseTestDistributedStrategy):
    @pytest.fixture(scope="module")
    def strategy(self, ddp_strategy: TorchDistributedStrategy) -> TorchDDPStrategy:
        return ddp_strategy

    def test_init(self, strategy: TorchDDPStrategy):
        """Test specific initialization of TorchDDPStrategy."""
        assert strategy.backend in ["nccl", "gloo"]

        # Test initialization
        init_path = "torch.distributed.init_process_group"
        with patch(init_path, autospec=True) as mock_init_torch:
            strategy = TorchDDPStrategy(
                backend="nccl" if torch.cuda.is_available() else "gloo"
            )
            strategy.init()
            mock_init_torch.assert_called_once()

    def test_distributed_model(
        self, strategy: TorchDDPStrategy, simple_model: nn.Module, optimizer: Optimizer
    ):
        """Test NN model distribution."""
        from torch.nn.parallel import DistributedDataParallel

        dist_model, dist_optimizer, _ = strategy.distributed(simple_model, optimizer)
        assert isinstance(dist_model, DistributedDataParallel)
        assert hasattr(dist_model, "module") or isinstance(dist_model, nn.Module)
        assert isinstance(dist_optimizer, Optimizer)


@pytest.mark.hpc
@pytest.mark.deepspeed_dist
class TestDeepSpeedStrategy(BaseTestDistributedStrategy):
    @pytest.fixture(scope="module")
    def strategy(self, deepspeed_strategy: DeepSpeedStrategy) -> DeepSpeedStrategy:
        return deepspeed_strategy

    def test_init(self, strategy: DeepSpeedStrategy):
        """Test specific initialization of DeepSpeedStrategy."""
        assert strategy.backend in ["nccl", "gloo", "mpi"]
        assert hasattr(strategy, "deepspeed"), (
            "Lazy import of deepspeed not found in DeepSpeedStrategy class."
        )

        # Test initialization
        init_path = "deepspeed.init_distributed"
        with patch(init_path, autospec=True) as mock_init_ds:
            strategy = DeepSpeedStrategy(
                backend="nccl" if torch.cuda.is_available() else "gloo"
            )
            strategy.init()
            mock_init_ds.assert_called_once()

    def test_distributed_model(
        self, strategy: DeepSpeedStrategy, simple_model: nn.Module, optimizer: Optimizer
    ):
        """Test NN model distribution."""
        from deepspeed.runtime.engine import DeepSpeedEngine

        ds_config = {
            "train_batch_size": 16,
            "fp16": {"enabled": False},
            "optimizer": {"type": "SGD", "params": {"lr": 0.001}},
        }

        dist_model, dist_optimizer, _ = strategy.distributed(
            simple_model, optimizer=optimizer, config=ds_config
        )
        assert hasattr(dist_model, "module")
        assert isinstance(dist_model, DeepSpeedEngine)
        assert dist_optimizer is not None
        assert isinstance(dist_optimizer, Optimizer)


@pytest.mark.hpc
@pytest.mark.horovod_dist
class TestHorovodStrategy(BaseTestDistributedStrategy):
    @pytest.fixture(scope="module")
    def strategy(self, horovod_strategy: HorovodStrategy) -> HorovodStrategy:
        return horovod_strategy

    def test_init(self, strategy: HorovodStrategy):
        assert strategy.is_initialized
        assert hasattr(strategy, "hvd"), (
            "Lazy import of horovod not found in HorovodStrategy class."
        )

        # Test initialization
        init_path = "horovod.torch.init"
        with patch(init_path, autospec=True) as mock_init_ds:
            strategy = HorovodStrategy()
            strategy.init()
            mock_init_ds.assert_called_once()

    def test_distributed_model(self, strategy: HorovodStrategy, simple_model, optimizer):
        dist_model, dist_optimizer, _ = strategy.distributed(
            simple_model, optimizer=optimizer, op=strategy.hvd.Average
        )
        assert isinstance(dist_model, nn.Module)
        assert isinstance(dist_optimizer, Optimizer)
        assert hasattr(dist_optimizer, "synchronize"), (
            "synchronize() method not found for Horovod optimizer"
        )


@pytest.mark.hpc
@pytest.mark.ray_dist
@pytest.mark.parametrize(
    "strategy_name",
    [
        pytest.param("ddp"),
        pytest.param("deepspeed"),
        pytest.param("horovod"),
    ],
)
def test_ray_distributed_strategy(
    strategy_name: Literal["ddp"] | Literal["deepspeed"] | Literal["horovod"],
    shared_tmp_path: Path,
):
    import ray  # needed here

    assert ray_cluster_is_running(), "Ray cluster not detected. Aborting tests"

    # The worker function must be declared inline, ohterwise the Ray workers will not find it
    def ray_tests(config, strategy: RayTorchDistributedStrategy):
        """The tests below are a flattened version of BaseTestDistributedStrategy"""
        strategy.init()

        simple_model = Linear(10, 2)
        simple_optimizer = SGD(simple_model.parameters(), lr=0.01)

        # Strategy-specific tests
        if isinstance(strategy, RayDDPStrategy):
            # Test init
            assert hasattr(strategy, "ray_train"), (
                "Lazy import of ray.train not found in RayDDPStrategy class."
            )
            # Test distribute model
            from torch.nn.parallel import DistributedDataParallel

            dist_model, dist_optimizer, _ = strategy.distributed(
                simple_model, simple_optimizer
            )
            assert isinstance(dist_model, DistributedDataParallel)
            assert hasattr(dist_model, "module") or isinstance(dist_model, nn.Module)
            assert isinstance(dist_optimizer, Optimizer)

        elif isinstance(strategy, RayDeepSpeedStrategy):
            assert strategy.backend in ["nccl", "gloo", "mpi"]
            assert hasattr(strategy, "deepspeed"), (
                "Lazy import of deepspeed not found in RayDeepSpeedStrategy class."
            )

            # Test initialization
            init_path = "deepspeed.init_distributed"
            with patch(init_path, autospec=True) as mock_init_ds:
                strategy = RayDeepSpeedStrategy(
                    backend="nccl" if torch.cuda.is_available() else "gloo"
                )
                strategy.init()
                mock_init_ds.assert_called_once()

            # Test distribute model
            from deepspeed.runtime.engine import DeepSpeedEngine

            ds_config = {
                "train_batch_size": 16,
                "fp16": {"enabled": False},
                "optimizer": {"type": "SGD", "params": {"lr": 0.001}},
            }

            dist_model, dist_optimizer, _ = strategy.distributed(
                simple_model, optimizer=simple_optimizer, config=ds_config
            )
            assert hasattr(dist_model, "module")
            assert isinstance(dist_model, DeepSpeedEngine)
            assert dist_optimizer is not None
            assert isinstance(dist_optimizer, Optimizer)

        else:
            raise ValueError("Unrecognized strategy type")

        # Test cluster properties
        assert strategy.is_initialized
        assert isinstance(strategy.device(), str)
        assert isinstance(strategy.is_main_worker, bool)
        assert strategy.global_world_size() >= 1
        assert strategy.local_world_size() >= 1
        assert strategy.global_rank() >= 0
        assert strategy.local_rank() >= 0

        # Test re-initialization
        with pytest.raises(DistributedStrategyError) as init_exc:
            strategy.init()
            assert "already initialized" in init_exc.value

        # Test initialized flag
        strategy.is_initialized = False

        with pytest.raises(UninitializedStrategyError):
            strategy.distributed(simple_model, simple_optimizer)
        with pytest.raises(UninitializedStrategyError):
            strategy.global_rank()
        with pytest.raises(UninitializedStrategyError):
            strategy.local_rank()
        with pytest.raises(UninitializedStrategyError):
            strategy.local_world_size()
        with pytest.raises(UninitializedStrategyError):
            strategy.global_world_size()
        with pytest.raises(UninitializedStrategyError):
            strategy.device()
        with pytest.raises(UninitializedStrategyError):
            strategy.is_main_worker
        x = torch.ones(2)
        with pytest.raises(UninitializedStrategyError):
            strategy.gather(x)
        with pytest.raises(UninitializedStrategyError):
            strategy.allgather_obj(x)
        with pytest.raises(UninitializedStrategyError):
            strategy.gather_obj(x)
        with pytest.raises(UninitializedStrategyError):
            strategy.clean_up()

        strategy.is_initialized = True

        # Test tensor gather
        local_tensor = torch.tensor([strategy.global_rank()], device=strategy.device())
        gathered = strategy.gather(local_tensor)
        if strategy.is_main_worker:
            assert len(gathered) == strategy.global_world_size()
        else:
            assert gathered is None

        # Test tensor gather from CPU
        my_tensor = torch.ones(10) * strategy.global_rank()
        tensors = strategy.gather(my_tensor, dst_rank=0)
        if strategy.is_main_worker:
            assert torch.stack(tensors).sum() == sum(range(strategy.global_world_size())) * 10
        else:
            assert tensors is None

        # Test object gather
        local_obj = {"rank": strategy.global_rank()}
        gathered_obj = strategy.gather_obj(local_obj)
        if strategy.is_main_worker:
            assert len(gathered_obj) == strategy.global_world_size()
        else:
            assert gathered_obj is None

        # Test allgather
        all_gathered = strategy.allgather_obj(local_obj)
        assert len(all_gathered) == strategy.global_world_size()

        # Check that the values of the ranks are as expected
        nnodes = strategy.global_world_size() // strategy.local_world_size()

        # Test local ranks
        lranks = strategy.gather_obj(strategy.local_rank(), dst_rank=0)
        if strategy.is_main_worker:
            assert len(lranks) == strategy.global_world_size()
            assert sum(lranks) == sum(range(strategy.local_world_size())) * nnodes
        else:
            assert lranks is None

        # Test global ranks
        granks = strategy.gather_obj(strategy.global_rank(), dst_rank=0)
        if strategy.is_main_worker:
            assert len(granks) == strategy.global_world_size()
            assert sum(granks) == sum(range(strategy.global_world_size()))
        else:
            assert granks is None

    # scaling_config = ScalingConfig(num_workers=2, use_gpu=False)
    scaling_config = get_adaptive_ray_scaling_config()
    run_config = RunConfig(storage_path=shared_tmp_path / "ray_checkpoints")

    match strategy_name:
        case "ddp":
            # This calls ray.init under the hood
            strategy = RayDDPStrategy()

            # Trainable
            test_function = ray.tune.with_parameters(ray_tests, strategy=strategy)

            # Create a trainer
            trainer = ray.train.torch.TorchTrainer(
                test_function,
                scaling_config=scaling_config,
                run_config=run_config,
            )
            trainer.fit()
        case "deepspeed":
            # This calls ray.init under the hood
            strategy = RayDeepSpeedStrategy(
                backend="nccl" if torch.cuda.is_available() else "gloo"
            )

            # Trainable
            test_function = ray.tune.with_parameters(ray_tests, strategy=strategy)

            # Create a trainer
            trainer = ray.train.torch.TorchTrainer(
                test_function,
                scaling_config=scaling_config,
                run_config=run_config,
            )
            trainer.fit()
        case _:
            raise ValueError("unrecognized strategy name")
