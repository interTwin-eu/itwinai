# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

"""Show how to use DDP, Horovod and DeepSpeed strategies interchangeably
with a large neural network trained on Imagenet dataset, showing how
to use checkpoints.
"""

import argparse
import os
import sys
from timeit import default_timer as timer
from typing import Optional

import deepspeed
import horovod.torch as hvd
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils import imagenet_dataset

from itwinai.loggers import EpochTimeTracker
from itwinai.parser import ArgumentParser as ItwinaiArgParser
from itwinai.torch.distributed import (
    DeepSpeedStrategy,
    HorovodStrategy,
    TorchDDPStrategy,
    TorchDistributedStrategy,
)
from itwinai.torch.reproducibility import seed_worker, set_seed


def parse_params() -> argparse.Namespace:
    """
    Parse CLI args, which can also be loaded from a configuration file
    using the --config flag:

    >>> train.py --strategy ddp --config base-config.yaml --config foo.yaml
    """
    parser = ItwinaiArgParser(description="PyTorch Imagenet Example")

    # Distributed ML strategy
    parser.add_argument(
        "--strategy",
        "-s",
        type=str,
        choices=["ddp", "horovod", "deepspeed"],
        default="ddp",
    )

    # Data and logging
    parser.add_argument(
        "--data-dir",
        default="./",
        help=("location of the training dataset in the local " "filesystem"),
    )
    parser.add_argument(
        "--log-int", type=int, default=10, help="log interval per training"
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        help="Print parsed arguments",
    )
    parser.add_argument(
        "--nworker",
        type=int,
        default=0,
        help=("number of workers in DataLoader (default: 0 -" " only main)"),
    )
    parser.add_argument(
        "--prefetch",
        type=int,
        default=2,
        help="prefetch data in DataLoader (default: 2)",
    )

    # Model
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of epochs to train (default: 10)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        help="momentum in SGD optimizer (default: 0.5)",
    )
    parser.add_argument(
        "--shuff",
        action="store_true",
        default=False,
        help="shuffle dataset (default: False)",
    )

    # Reproducibility
    parser.add_argument(
        "--rnd-seed",
        type=Optional[int],
        default=None,
        help="seed integer for reproducibility (default: 0)",
    )

    # Distributed ML
    parser.add_argument(
        "--backend",
        type=str,
        default="nccl",
        help="backend for parrallelisation (default: nccl)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables GPGPUs"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local rank passed from distributed launcher",
    )

    # Horovod
    parser.add_argument(
        "--fp16-allreduce",
        action="store_true",
        default=False,
        help="use fp16 compression during allreduce",
    )
    parser.add_argument(
        "--use-adasum",
        action="store_true",
        default=False,
        help="use adasum algorithm to do reduction",
    )
    parser.add_argument(
        "--gradient-predivide-factor",
        type=float,
        default=1.0,
        help=("apply gradient pre-divide factor in optimizer " "(default: 1.0)"),
    )

    # DeepSpeed
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    if args.verbose:
        args_list = [f"{key}: {val}" for key, val in args.items()]
        print("PARSED ARGS:\n", "\n".join(args_list))

    return args


def train(
    model,
    device,
    train_loader,
    optimizer,
    epoch,
    strategy: TorchDistributedStrategy,
    args,
):
    """Training function, representing an epoch."""
    model.train()
    loss_acc = 0
    gwsize = strategy.global_world_size()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if (
            strategy.is_main_worker
            and args.log_int > 0
            and batch_idx % args.log_int == 0
        ):
            print(
                f"Train epoch: {epoch} "
                f"[{batch_idx * len(data)}/{len(train_loader.dataset)/gwsize} "
                f"({100.0 * batch_idx / len(train_loader):.0f}%)]\t\t"
                f"Loss: {loss.item():.6f}"
            )
        loss_acc += loss.item()
    return loss_acc


def main():
    # Parse CLI args
    args = parse_params()

    # Instantiate Strategy
    if args.strategy == "ddp":
        if not torch.cuda.is_available() or not torch.cuda.device_count() > 1:
            raise RuntimeError("Resources unavailable")

        strategy = TorchDDPStrategy(backend=args.backend)
        distribute_kwargs = {}
    elif args.strategy == "horovod":
        strategy = HorovodStrategy()
        distribute_kwargs = dict(
            compression=(
                hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
            ),
            op=hvd.Adasum if args.use_adasum else hvd.Average,
            gradient_predivide_factor=args.gradient_predivide_factor,
        )
    elif args.strategy == "deepspeed":
        strategy = DeepSpeedStrategy(backend=args.backend)
        distribute_kwargs = dict(
            config_params=dict(train_micro_batch_size_per_gpu=args.batch_size)
        )
    else:
        raise NotImplementedError(
            f"Strategy {args.strategy} is not recognized/implemented."
        )
    strategy.init()

    # Check resource availability
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    is_distributed = use_cuda and torch.cuda.device_count() > 0

    # Dataset
    subset_size = 5000
    train_dataset = imagenet_dataset(args.data_dir, subset_size=subset_size)

    # Set random seed for reproducibility
    torch_prng = set_seed(args.rnd_seed, deterministic_cudnn=False)

    # Get job rank info - rank==0 master gpu
    if is_distributed:
        # local world size - per node
        global_world_size = strategy.global_world_size()
        global_rank = strategy.global_rank()
    else:
        # Use a single worker (either on GPU or CPU)
        global_world_size = 1
        global_rank = 0

    # Encapsulate the model on the GPU assigned to the current process
    device = torch.device(strategy.device() if use_cuda else "cpu")

    if is_distributed:
        # Distributed sampler restricts data loading to a subset of the dataset
        # exclusive to the current process.
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=global_world_size,
            rank=global_rank,
            shuffle=(args.shuff and args.rnd_seed is None),
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.nworker,
            pin_memory=True,
            persistent_workers=(args.nworker > 1),
            prefetch_factor=args.prefetch,
            generator=torch_prng,
            worker_init_fn=seed_worker,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            generator=torch_prng,
            worker_init_fn=seed_worker,
        )

    # Create CNN model: resnet 50, resnet101, resnet152
    model = torchvision.models.resnet152()
    model.to(device)

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    if is_distributed:
        model, optimizer, _ = strategy.distributed(
            model, optimizer, lr_scheduler=None, **distribute_kwargs
        )

    if strategy.is_main_worker:
        num_nodes = os.environ.get("SLURM_NNODES", 1)
        strategy_name = f"{args.strategy}-it"
        epoch_time_tracker = EpochTimeTracker(
            strategy_name=strategy_name,
            save_path=f"epochtime_{strategy_name}_{num_nodes}N.csv",
            num_nodes=int(num_nodes),
        )

    start_time = timer()
    for epoch_idx in range(1, args.epochs + 1):
        epoch_start_time = timer()
        if is_distributed:
            train_sampler.set_epoch(epoch_idx)

        # Training
        train(
            model=model,
            device=device,
            train_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch_idx,
            strategy=strategy,
            args=args,
        )

        if strategy.is_main_worker:
            epoch_elapsed_time = timer() - epoch_start_time
            epoch_time_tracker.add_epoch_time(epoch_idx, epoch_elapsed_time)
            print(f"[{epoch_idx}/{args.epochs}] - time: {epoch_elapsed_time:.2f}s")

    if global_rank == 0:
        total_time = timer() - start_time
        print(f"Training finished - took {total_time:.2f}s")

    # Clean-up
    if is_distributed:
        strategy.clean_up()


if __name__ == "__main__":
    main()
    sys.exit()
