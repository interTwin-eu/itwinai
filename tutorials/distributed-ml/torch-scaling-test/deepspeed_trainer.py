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
import sys
import time
from timeit import default_timer as timer

import deepspeed
import torch
import torch.distributed as dist
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils import imagenet_dataset, get_parser, train_epoch

from itwinai.loggers import EpochTimeTracker
from itwinai.torch.reproducibility import seed_worker, set_seed


def main():
    # Parse CLI args
    parser = get_parser()
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # Check resources availability
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    is_distributed = False
    if use_cuda and torch.cuda.device_count() > 0:
        is_distributed = True

    # Limit # of CPU threads to be used per worker
    # torch.set_num_threads(1)

    # Start the timer for profiling
    st = timer()

    # Set random seed for reproducibility
    torch_prng = set_seed(args.rnd_seed, deterministic_cudnn=False)

    if is_distributed:

        deepspeed.init_distributed(dist_backend=args.backend)
        # Get job rank info - rank==0 master gpu
        lwsize = torch.cuda.device_count()  # local world size - per node
        grank = dist.get_rank()  # global rank - assign per run
        lrank = dist.get_rank() % lwsize  # local rank - assign per node
    else:
        # Use a single worker (either on GPU or CPU)
        lwsize = 1
        grank = 0
        lrank = 0

    # Encapsulate the model on the GPU assigned to the current process
    if use_cuda:
        torch.cuda.set_device(lrank)

    # Read training dataset
    train_dataset = imagenet_dataset(args.data_dir)

    if is_distributed:
        # Distributed sampler restricts data loading to a subset of the dataset
        # exclusive to the current process.
        # `mun_replicas` and `rank` are automatically retrieved from
        # the current distributed group.
        train_sampler = DistributedSampler(
            train_dataset,  # num_replicas=gwsize, rank=grank,
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

    # Create CNN model
    model = torchvision.models.resnet152()

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
    if grank == 0:
        print("TIMER: broadcast:", timer() - st, "s")
        print("\nDEBUG: start training")
        print("--------------------------------------------------------")
        nnod = os.environ.get("SLURM_NNODES", "unk")
        epoch_time_tracker = EpochTimeTracker(
            strategy_name="deepspeed-bl",
            save_path=f"epochtime_deepspeed-bl_{nnod}N.csv",
            num_nodes=int(nnod),
        )

    et = timer()
    start_epoch = 1
    for epoch in range(start_epoch, args.epochs + 1):
        lt = timer()
        if is_distributed:
            # Inform the sampler that a new epoch started: shuffle
            # may be needed
            train_sampler.set_epoch(epoch)

        # Training
        train_epoch(
            model=distrib_model, device=device, train_loader=train_loader, optimizer=optimizer
        )

        # Save first epoch timer
        if epoch == start_epoch:
            first_ep_t = timer() - lt

        # Final epoch
        if epoch + 1 == args.epochs:
            train_loader.last_epoch = True

        if grank == 0:
            print("TIMER: epoch time:", timer() - lt, "s")
            epoch_time_tracker.add_epoch_time(epoch - 1, timer() - lt)

    if is_distributed:
        dist.barrier()

    if grank == 0:
        print("\n--------------------------------------------------------")
        print("DEBUG: results:\n")
        print("TIMER: first epoch time:", first_ep_t, " s")
        print("TIMER: last epoch time:", timer() - lt, " s")
        print("TIMER: average epoch time:", (timer() - et) / args.epochs, " s")
        print("TIMER: total epoch time:", timer() - et, " s")
        if epoch > 1:
            print("TIMER: total epoch-1 time:", timer() - et - first_ep_t, " s")
            print(
                "TIMER: average epoch-1 time:",
                (timer() - et - first_ep_t) / (args.epochs - 1),
                " s",
            )
        if use_cuda:
            print(
                "DEBUG: memory req:",
                int(torch.cuda.memory_reserved(lrank) / 1024 / 1024),
                "MB",
            )
            print("DEBUG: memory summary:\n\n", torch.cuda.memory_summary(0))
        print(f"TIMER: final time: {timer()-st} s\n")

    time.sleep(1)
    print(f"<Global rank: {grank}> - TRAINING FINISHED")

    # Clean-up
    if is_distributed:
        deepspeed.sys.exit()


if __name__ == "__main__":
    main()
    sys.exit()
