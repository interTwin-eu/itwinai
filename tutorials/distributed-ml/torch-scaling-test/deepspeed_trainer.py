# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# --------------------------------------------------------------------------------------

"""Scaling test of Microsoft Deepspeed on Imagenet using Resnet."""

import os
from pathlib import Path
from timeit import default_timer as timer

import deepspeed
import torch
import torch.distributed as dist
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils import get_parser, imagenet_dataset, train_epoch

from itwinai.loggers import EpochTimeTracker
from itwinai.torch.reproducibility import seed_worker, set_seed


def main():
    # Parse CLI args
    parser = get_parser()
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # Check resources availability
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    is_distributed = use_cuda and torch.cuda.device_count() > 0
    torch_prng = set_seed(args.rnd_seed, deterministic_cudnn=False)

    shuffle = args.shuff and args.rnd_seed is None
    persistent_workers = args.nworker > 1
    epoch_time_save_dir = Path(args.epoch_time_directory)

    train_dataset = imagenet_dataset(args.data_dir, subset_size=args.subset_size)
    train_sampler = None

    if is_distributed:
        deepspeed.init_distributed(dist_backend=args.backend)

        local_world_size = torch.cuda.device_count()
        global_rank = dist.get_rank()
        local_rank = dist.get_rank() % local_world_size
        pin_memory = True
        train_sampler = DistributedSampler(train_dataset, shuffle=shuffle)

        # To fix problems with deepspeed OpenMPI rank conflicting with local rank
        os.environ["OMPI_COMM_WORLD_LOCAL_RANK"] = os.environ.get("LOCAL_RANK", "")
    else:
        # Use a single worker (either on GPU or CPU)
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
        generator=torch_prng,
        worker_init_fn=seed_worker,
    )
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")

    # Create CNN model
    model = torchvision.models.resnet152()
    model = model.to(device)

    # Initializing deepspeed distributed strategy
    deepspeed_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,  # redundant
        "optimizer": {
            "type": "SGD",
            "params": {"lr": args.lr, "momentum": args.momentum},
        },
        "fp16": {"enabled": False},
        "zero_optimization": False,
    }
    distrib_model, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        config_params=deepspeed_config,
    )

    # Start training loop
    if global_rank == 0:
        num_nodes = os.environ.get("SLURM_NNODES", "1")
        save_path = epoch_time_save_dir / f"epochtime_deepspeed-bl_{num_nodes}.csv"
        epoch_time_logger = EpochTimeTracker(
            strategy_name="deepspeed-bl",
            save_path=save_path,
            num_nodes=int(num_nodes),
        )

    start_time = timer()
    start_epoch = 1
    for epoch_idx in range(start_epoch, args.epochs + 1):
        epoch_start_time = timer()
        if is_distributed:
            train_sampler.set_epoch(epoch_idx)

        # Training
        train_epoch(
            model=distrib_model,
            device=device,
            train_loader=train_loader,
            optimizer=optimizer,
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
        deepspeed.sys.exit()


if __name__ == "__main__":
    main()
