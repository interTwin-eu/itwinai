"""Test distributed training strategies."""

import subprocess
import pytest


@pytest.mark.hpc
def test_distributed_decorator():
    """Test function decorator. Needs torchrun cmd."""
    cmd = ("micromamba run -p ./ai/.venv-pytorch "
           "torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=100 "
           "--rdzv_backend=c10d --rdzv_endpoint=localhost:29400 "
           "tests/backend/torch/distribtued_decorator.py")
    subprocess.run(cmd.split(), check=True)


@pytest.mark.hpc
def test_distributed_trainer():
    """Test vanilla torch distributed trainer. Needs torchrun cmd."""
    cmd = ("micromamba run -p ./ai/.venv-pytorch "
           "torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=100 "
           "--rdzv_backend=c10d --rdzv_endpoint=localhost:29400 "
           "tests/backend/torch/torch_dist_trainer.py")
    subprocess.run(cmd.split(), check=True)
