# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# --------------------------------------------------------------------------------------


"""Scaling test of torch Distributed Data Parallel on Imagenet using Resnet."""

import os
from pathlib import Path
from timeit import default_timer as timer

import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils import get_parser, imagenet_dataset, train_epoch

from itwinai.loggers import EpochTimeTracker
from itwinai.torch.reproducibility import seed_worker, set_seed


def main():
    parser = get_parser()
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    is_distributed = use_cuda and torch.cuda.device_count() > 0
    torch_seed = set_seed(args.rnd_seed, deterministic_cudnn=False)
    shuffle: bool = args.shuff and args.rnd_seed is None
    persistent_workers = args.nworker > 1
    epoch_time_save_dir = Path(args.epoch_time_directory)

    train_dataset = imagenet_dataset(args.data_dir, subset_size=args.subset_size)
    train_sampler = None

    if is_distributed:
        # Initializing the distribution backend
        dist.init_process_group(backend=args.backend)

        local_world_size = torch.cuda.device_count()
        global_rank = dist.get_rank()
        local_rank = dist.get_rank() % local_world_size

        # Creating dataset and dataloader
        pin_memory = True
        train_sampler = DistributedSampler(train_dataset, shuffle=shuffle)
    else:
        local_world_size = 1
        global_rank = 0
        local_rank = 0
        pin_memory = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.nworker,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=args.prefetch,
        generator=torch_seed,
        worker_init_fn=seed_worker,
    )
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")

    model = torchvision.models.resnet152()
    model.to(device)

    # Distributing the model to the workers
    if is_distributed:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[device], output_device=device
        )

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    if global_rank == 0:
        num_nodes = os.environ.get("SLURM_NNODES", 1)
        save_path = epoch_time_save_dir / f"epochtime_ddp-bl_{num_nodes}.csv"
        epoch_time_logger = EpochTimeTracker(
            strategy_name="ddp-bl",
            save_path=save_path,
            num_nodes=int(num_nodes),
        )

    start_time = timer()
    for epoch_idx in range(1, args.epochs + 1):
        epoch_start_time = timer()

        if is_distributed:
            train_sampler.set_epoch(epoch_idx)

        train_epoch(
            model=model, device=device, train_loader=train_loader, optimizer=optimizer
        )

        if global_rank == 0:
            epoch_elapsed_time = timer() - epoch_start_time
            epoch_time_logger.add_epoch_time(epoch_idx, epoch_elapsed_time)
            print(f"[{epoch_idx}/{args.epochs}] - time: {epoch_elapsed_time:.2f}s")

    if global_rank == 0:
        total_time = timer() - start_time
        print(f"Training finished - took {total_time:.2f}s")
        epoch_time_logger.save()

    # Clean-up
    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
