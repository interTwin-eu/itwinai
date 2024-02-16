"""Test distributed training strategies."""

import subprocess
import pytest


@pytest.mark.slurm
def test_distributed_decorator():
    """Test function decorator. Needs torchrun cmd."""
    cmd = ("micromamba run -p ./.venv-pytorch "
           "torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=100 "
           "--rdzv_backend=c10d --rdzv_endpoint=localhost:29400 "
           "tests/torch/distribtued_decorator.py")
    subprocess.run(cmd.split(), check=True)


@pytest.mark.skip(reason="TorchTrainer not implemented yet")
@pytest.mark.slurm
def test_distributed_trainer():
    """Test vanilla torch distributed trainer. Needs torchrun cmd."""
    cmd = ("micromamba run -p ./.venv-pytorch "
           "torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=100 "
           "--rdzv_backend=c10d --rdzv_endpoint=localhost:29400 "
           "tests/torch/torch_dist_trainer.py")
    subprocess.run(cmd.split(), check=True)
