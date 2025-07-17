"""Some functions to test torch distributed setup"""

import socket
import subprocess
import sys
from pathlib import Path
from typing import Literal

import ray
import torch
import torch.distributed as dist
import typer
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchConfig, TorchTrainer


def test_cuda():
    """Check the configuration of CUDA and NCCL"""
    typer.echo("\nTesting the configuration of CUDA and NCCL:")
    typer.echo(f"PyTorch version: {torch.__version__}")
    typer.echo(f"Built with CUDA: {torch.version.cuda}")
    typer.echo(f"cuDNN version: {torch.backends.cudnn.version()}")
    cuda_available = torch.cuda.is_available()
    typer.echo(f"CUDA available: {cuda_available}")
    if not cuda_available:
        sys.exit("ERROR: CUDA not available!")

    for i in range(torch.cuda.device_count()):
        typer.echo(f" GPU {i}: {torch.cuda.get_device_name(i)}")

    # Check NCCL version if available
    try:
        nccl_ver = torch.cuda.nccl.version()
        ver_str = ".".join(map(str, nccl_ver))
        typer.echo(f"NCCL version: {ver_str}")
    except Exception:
        typer.echo("NCCL version info not available in this build.")


def test_rocm():
    """Check the configuration of HIP and ROCm"""
    typer.echo("\nTesting the configuration of HIP and ROCm:")
    typer.echo(f"PyTorch version: {torch.__version__}")
    typer.echo(f"HIP support: {torch.version.hip}")
    cuda_available = torch.cuda.is_available()
    typer.echo(f"CUDA available: {cuda_available}")
    if cuda_available:
        for i in range(torch.cuda.device_count()):
            typer.echo(f" GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        typer.echo("No GPU device was found.")

    # ----------------------------------------------------------------
    # HIPCC path → ROCm root/version
    try:
        hipcc_path_str = subprocess.check_output(["which", "hipcc"], text=True).strip()
        hipcc_path = Path(hipcc_path_str)
        rocm_root = hipcc_path.parent.parent
        rocm_version = rocm_root.name
        typer.echo(f"ROCm  root        : {rocm_root}")
        typer.echo(f"ROCm version      : {rocm_version}")
    except subprocess.CalledProcessError:
        typer.echo("Could not locate hipcc; ROCm root/version unknown")
    # ----------------------------------------------------------------
    # HIP compiler version
    try:
        hipcc_out = subprocess.check_output(
            ["hipcc", "--version"], stderr=subprocess.STDOUT, text=True
        )
        hip_version = next(
            (
                line.split("HIP version:")[1].strip()
                for line in hipcc_out.splitlines()
                if "HIP version" in line
            ),
            "n/a",
        )
        typer.echo(f"HIP version       : {hip_version}")
    except Exception:
        typer.echo("Could not parse HIP version from hipcc")
    # ----------------------------------------------------------------
    # Hostname
    typer.echo(f"Hostname          : {socket.gethostname()}")
    # ----------------------------------------------------------------
    # librccl location
    if "rocm_root" in locals():
        lib_dir = rocm_root / "lib"
        matches = list(lib_dir.glob("librccl.so*"))
        if matches:
            typer.echo(f"Librccl path      : {matches[0]}")
        else:
            typer.echo("Librccl not found under ROCm root")
    else:
        typer.echo("Skipping librccl lookup (no ROCm root)")


def test_all_reduce(backend: Literal["nccl", "gloo"]):
    """Test all-reduce connectivity using the specified backend."""
    typer.echo(f"\nTesting {backend.upper()} all-reduce connectivity:")

    # initialize
    dist.init_process_group(backend=backend, init_method="env://")
    rank = dist.get_rank()
    world = dist.get_world_size()

    # pick device
    if backend == "nccl":
        device_count = torch.cuda.device_count()
        if device_count == 0:
            sys.exit("ERROR: No CUDA devices available for NCCL backend.")
        torch.cuda.set_device(rank % device_count)
        device = torch.device("cuda")
    else:  # gloo
        device = torch.device("cpu")

    # build tensor and all-reduce
    x = torch.tensor([float(rank)], device=device)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)

    # only rank 0 reports
    if rank == 0:
        expected = float(sum(range(world)))
        typer.echo(f"SUM={x.item():.1f}, expected={expected:.1f}")

    # cleanup
    dist.destroy_process_group()
    typer.echo(f"{backend.upper()} test completed successfully.")


def test_ray(backend: Literal["nccl", "gloo"]):
    """Test Ray TorchTrainer distributed training with selectable backend."""
    typer.echo(f"\nTesting Ray TorchTrainer distributed training (backend={backend}):")
    typer.echo(f"Ray version: {ray.__version__}")

    # Connect to existing Ray cluster
    try:
        ray.init(address="auto")
        typer.echo("Connected to Ray cluster.")
    except Exception as e:
        sys.exit(f"ERROR: Failed to initialize Ray cluster: {e}")

    # Per-worker training loop
    def train_loop(config):
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from ray import train

        ctx = train.get_context()
        rank = ctx.get_local_rank()
        device = (
            torch.device("cuda", rank % torch.cuda.device_count())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        # simple model
        model = nn.Linear(10, 1).to(device)
        model = train.torch.prepare_model(model)

        optimizer = optim.SGD(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()

        # synthetic data and one step
        x = torch.randn(32, 10, device=device)
        y = torch.randn(32, 1, device=device)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()

        train.report({"loss": loss.item()})

    # How many GPUs are available
    num_gpus = int(ray.cluster_resources().get("GPU", 0))
    use_gpu = num_gpus > 0
    if use_gpu:
        num_workers = num_gpus
        typer.echo(f"Found {num_gpus} GPUs, so Ray will spawn {num_workers} workers")
    else:
        # This is a CPU-only job
        num_workers = 4  # This is a magic number, could be improved
        typer.echo(f"Found NO GPUs, so Ray will spawn {num_workers} CPU-only workers")

    trainer = TorchTrainer(
        train_loop,
        scaling_config=ScalingConfig(
            num_workers=num_workers,
            use_gpu=use_gpu,
        ),
        torch_config=TorchConfig(backend=backend),
        run_config=RunConfig(name="ray-torch-backend-test"),
    )

    # Execute training
    result = trainer.fit()

    # Show per-worker losses
    typer.echo("\nFinal losses per worker:")
    typer.echo(result.metrics_dataframe.to_string(index=False))

    ray.shutdown()
    typer.echo("RAY test completed successfully.")
