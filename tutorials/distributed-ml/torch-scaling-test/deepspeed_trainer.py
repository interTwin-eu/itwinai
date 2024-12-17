# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------

"""Scaling test of Microsoft Deepspeed on Imagenet using Resnet."""

import os
from timeit import default_timer as timer


import deepspeed
import torch
import torch.distributed as dist
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils import imagenet_dataset, get_parser, train_epoch

from itwinai.loggers import EpochTimeTracker
from itwinai.torch.reproducibility import set_seed


def main():
    # Parse CLI args
    parser = get_parser()
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # Check resources availability
    subset_size = 5000 # limit number of examples from imagenet
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    is_distributed = use_cuda and torch.cuda.device_count() > 0
    torch_prng = set_seed(args.rnd_seed, deterministic_cudnn=False)

    st = timer()

    train_dataset = imagenet_dataset(args.data_dir, subset_size=subset_size)
    if is_distributed:
        deepspeed.init_distributed(dist_backend=args.backend)

        local_world_size = torch.cuda.device_count() 
        global_rank = dist.get_rank()
        local_rank = dist.get_rank() % local_world_size 

        shuffle = args.shuff and args.rnd_seed is None
        # pin_memory=True
        # persistent_workers = args.nworker > 1

        train_sampler = DistributedSampler(
            train_dataset,
            shuffle=shuffle,
        )
        # train_loader = DataLoader(
        #     train_dataset,
        #     batch_size=args.batch_size,
        #     sampler=train_sampler,
        #     num_workers=args.nworker,
        #     pin_memory=pin_memory,
        #     persistent_workers=persistent_workers,
        #     prefetch_factor=args.prefetch,
        #     generator=torch_prng,
        #     worker_init_fn=seed_worker,
        # )
    else:
        # Use a single worker (either on GPU or CPU)
        local_world_size = 1
        global_rank = 0
        local_rank = 0
        # train_loader = DataLoader(
        #     train_dataset,
        #     batch_size=args.batch_size,
        #     generator=torch_prng,
        #     worker_init_fn=seed_worker,
        # )

    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu", local_rank)

    # Create CNN model
    model = torchvision.models.resnet152().to(device)

    # Initialize DeepSpeed and get:
    # 1) Distributed model
    # 2) DeepSpeed optimizer
    # 3) Distributed data loader
    deepspeed_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,  # redundant
        "optimizer": {
            "type": "SGD",
            "params": {"lr": args.lr, "momentum": args.momentum},
        },
        "fp16": {"enabled": False},
        "zero_optimization": False,
    }
    distrib_model, optimizer, deepspeed_train_loader, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        config_params=deepspeed_config,
    )

    # Start training loop
    if global_rank == 0:
        num_nodes = os.environ.get("SLURM_NNODES", "1")
        epoch_time_tracker = EpochTimeTracker(
            strategy_name="deepspeed-bl",
            save_path=f"epochtime_deepspeed-bl_{num_nodes}N.csv",
            num_nodes=int(num_nodes),
        )

    start_epoch = 1
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = timer()
        if is_distributed:
            # Inform the sampler that a new epoch started: shuffle
            # may be needed
            train_sampler.set_epoch(epoch)

        # Training
        train_epoch(
            model=distrib_model, device=device, train_loader=deepspeed_train_loader, optimizer=optimizer
        )

        if global_rank == 0:
            epoch_elapsed_time = timer() - epoch_start_time
            epoch_time_tracker.add_epoch_time(epoch_idx, epoch_elapsed_time)
            print(f"[{epoch_idx+1}/{args.epochs+1}] - time: {epoch_elapsed_time:.2f}s")

    if is_distributed:
        dist.barrier()


    # Clean-up
    if is_distributed:
        deepspeed.sys.exit()


if __name__ == "__main__":
    main()
