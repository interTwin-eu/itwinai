# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# --------------------------------------------------------------------------------------

"""Scaling test of Horovod on Imagenet using Resnet."""

import os
from pathlib import Path
from timeit import default_timer as timer

import horovod.torch as hvd
import torch
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils import get_parser, imagenet_dataset, train_epoch

from itwinai.loggers import EpochTimeTracker
from itwinai.torch.reproducibility import seed_worker, set_seed


def main():
    # Parse CLI args
    parser = get_parser()
    args = parser.parse_args()

    # Check resources availability
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    is_distributed = use_cuda and torch.cuda.device_count() > 0
    torch_seed = set_seed(args.rnd_seed, deterministic_cudnn=False)

    shuffle = args.shuff and args.rnd_seed is None
    persistent_workers = args.nworker > 1
    epoch_time_save_dir = Path(args.epoch_time_directory)

    train_dataset = imagenet_dataset(args.data_dir, subset_size=args.subset_size)
    train_sampler = None

    # Setting variables
    if is_distributed:
        hvd.init()

        local_rank = hvd.local_rank()
        global_rank = hvd.rank()
        global_world_size = hvd.size()

        # By default, Adasum doesn't need scaling up learning rate
        lr_scaler = hvd.size() if not args.use_adasum else 1

        # If using GPU Adasum allreduce, scale learning rate by local_size
        if args.use_adasum and hvd.nccl_built():
            lr_scaler = hvd.local_size()

        # Scale learning rate by lr_scaler
        args.lr *= lr_scaler
        pin_memory = True

        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=global_world_size,
            rank=global_rank,
            shuffle=shuffle,
        )
    else:
        # Use a single worker (either on GPU or CPU)
        local_rank = 0
        global_rank = 0
        global_world_size = 1
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

    # Create CNN model
    model = torchvision.models.resnet152()
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    if is_distributed:
        # Broadcast parameters & optimizer state
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        # Compression algorithm
        compression = (
            hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
        )

        # Wrap optimizer with DistributedOptimizer
        optimizer = hvd.DistributedOptimizer(
            optimizer,
            named_parameters=model.named_parameters(),
            compression=compression,
            op=hvd.Adasum if args.use_adasum else hvd.Average,
            gradient_predivide_factor=args.gradient_predivide_factor,
        )

    if global_rank == 0:
        num_nodes = os.environ.get("SLURM_NNODES", 1)
        save_path = epoch_time_save_dir / f"epochtime_horovod-bl_{num_nodes}.csv"
        epoch_time_logger = EpochTimeTracker(
            strategy_name="horovod-bl",
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

        # Final epoch
        if epoch_idx + 1 == args.epochs:
            train_loader.last_epoch = True

        if global_rank == 0:
            epoch_elapsed_time = timer() - epoch_start_time
            epoch_time_logger.add_epoch_time(epoch_idx, epoch_elapsed_time)
            print(f"[{epoch_idx}/{args.epochs}] - time: {epoch_elapsed_time:.2f}s")

    if global_rank == 0:
        total_time = timer() - start_time
        print(f"Training finished - took {total_time:.2f}s")
        epoch_time_logger.save()


if __name__ == "__main__":
    main()
