# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# --------------------------------------------------------------------------------------

"""Show how to use DDP, Horovod and DeepSpeed strategies interchangeably
with a large neural network trained on Imagenet dataset, showing how
to use checkpoints.
"""

import os
import sys
from pathlib import Path
from timeit import default_timer as timer

import horovod.torch as hvd
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils import get_parser, imagenet_dataset, train_epoch

from itwinai.loggers import EpochTimeTracker
from itwinai.torch.distributed import (
    DeepSpeedStrategy,
    HorovodStrategy,
    TorchDDPStrategy,
)
from itwinai.torch.reproducibility import seed_worker, set_seed


def main():
    # Parse CLI args
    parser = get_parser()
    args = parser.parse_args()

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
    epoch_time_save_dir = Path(args.epoch_time_directory)

    # Dataset
    train_dataset = imagenet_dataset(args.data_dir, subset_size=args.subset_size)

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
        save_path = epoch_time_save_dir / f"epochtime_{strategy_name}_{num_nodes}.csv"
        epoch_time_logger = EpochTimeTracker(
            strategy_name=strategy_name,
            save_path=save_path,
            num_nodes=int(num_nodes),
        )

    start_time = timer()
    for epoch_idx in range(1, args.epochs + 1):
        epoch_start_time = timer()
        if is_distributed:
            train_sampler.set_epoch(epoch_idx)

        # Training
        train_epoch(
            model=model,
            device=device,
            train_loader=train_loader,
            optimizer=optimizer,
        )

        if strategy.is_main_worker:
            epoch_elapsed_time = timer() - epoch_start_time
            epoch_time_logger.add_epoch_time(epoch_idx, epoch_elapsed_time)
            print(f"[{epoch_idx}/{args.epochs}] - time: {epoch_elapsed_time:.2f}s")

    if global_rank == 0:
        total_time = timer() - start_time
        print(f"Training finished - took {total_time:.2f}s")
        epoch_time_logger.save()

    # Clean-up
    if is_distributed:
        strategy.clean_up()


if __name__ == "__main__":
    main()
    sys.exit()
