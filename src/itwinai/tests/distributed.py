"""Some functions to test torch distributed setup"""

import glob
import os
import socket
import subprocess
import sys

import torch
import torch.distributed as dist
import typer


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
        hipcc_path = subprocess.check_output(["which", "hipcc"], text=True).strip()
        rocm_root = os.path.dirname(os.path.dirname(hipcc_path))
        rocm_version = os.path.basename(rocm_root)
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
        matches = glob.glob(os.path.join(rocm_root, "lib", "librccl.so*"))
        if matches:
            typer.echo(f"Librccl path      : {matches[0]}")
        else:
            typer.echo("Librccl not found under ROCm root")
    else:
        typer.echo("Skipping librccl lookup (no ROCm root)")


def test_nccl():
    """Test NCCL all-reduce connectivity (use with torchrun)."""
    typer.echo("\nTesting NCCL all-reduce connectivity:")
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world = dist.get_world_size()

    # Assign GPU based on rank
    device_count = torch.cuda.device_count()
    if device_count > 0:
        torch.cuda.set_device(rank % device_count)
        x = torch.tensor([float(rank)], device="cuda")
    else:
        sys.exit("ERROR: No CUDA devices for NCCL backend.")

    # Perform all-reduce sum
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    if rank == 0:
        expected = float(sum(range(world)))
        typer.echo(f"SUM={x.item():.1f}, expected={expected:.1f}")

    dist.destroy_process_group()
    typer.echo("NCCL test completed successfully.")


def test_gloo():
    """Test Gloo all-reduce connectivity (use with torchrun)."""
    typer.echo("\nTesting Gloo all-reduce connectivity:")
    dist.init_process_group(backend="gloo", init_method="env://")
    rank = dist.get_rank()
    world = dist.get_world_size()

    # CPU tensor
    x = torch.tensor([float(rank)])
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    if rank == 0:
        typer.echo(f"Gloo SUM OK: {x.item():.1f} (expected {sum(range(world)):.1f})")

    dist.destroy_process_group()
    typer.echo("Gloo test completed successfully.")
