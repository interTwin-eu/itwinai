"""
Show how to use DDP, Horovod and DeepSpeed strategies interchangeably
with a large neural network trained on Imagenet dataset, showing how
to use checkpoints.
"""
from typing import Optional
import os
import argparse
import sys
from timeit import default_timer as timer
import time

import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import deepspeed
import horovod.torch as hvd

from itwinai.torch.distributed import (
    TorchDistributedStrategy,
    TorchDDPStrategy,
    HorovodStrategy,
    DeepSpeedStrategy,
)
from itwinai.parser import ArgumentParser as ItAIArgumentParser
from itwinai.loggers import EpochTimeTracker
from itwinai.torch.reproducibility import (
    seed_worker, set_seed
)

from utils import imagenet_dataset


def parse_params() -> argparse.Namespace:
    """
    Parse CLI args, which can also be loaded from a configuration file
    using the --config flag:

    >>> train.py --strategy ddp --config base-config.yaml --config foo.yaml
    """
    parser = ItAIArgumentParser(description='PyTorch Imagenet Example')

    # Distributed ML strategy
    parser.add_argument(
        "--strategy", "-s", type=str,
        choices=['ddp', 'horovod', 'deepspeed'],
        default='ddp'
    )

    # Data and logging
    parser.add_argument('--data-dir', default='./',
                        help=('location of the training dataset in the local '
                              'filesystem'))
    parser.add_argument('--log-int', type=int, default=10,
                        help='log interval per training')
    parser.add_argument('--verbose',
                        action=argparse.BooleanOptionalAction,
                        help='Print parsed arguments')
    parser.add_argument('--nworker', type=int, default=0,
                        help=('number of workers in DataLoader (default: 0 -'
                              ' only main)'))
    parser.add_argument('--prefetch', type=int, default=2,
                        help='prefetch data in DataLoader (default: 2)')

    # Model
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='momentum in SGD optimizer (default: 0.5)')
    parser.add_argument('--shuff', action='store_true', default=False,
                        help='shuffle dataset (default: False)')

    # Reproducibility
    parser.add_argument('--rnd-seed', type=Optional[int], default=None,
                        help='seed integer for reproducibility (default: 0)')

    # Distributed ML
    parser.add_argument('--backend', type=str, default='nccl',
                        help='backend for parrallelisation (default: nccl)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables GPGPUs')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')

    # Horovod
    parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                        help='use fp16 compression during allreduce')
    parser.add_argument('--use-adasum', action='store_true', default=False,
                        help='use adasum algorithm to do reduction')
    parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
                        help=('apply gradient pre-divide factor in optimizer '
                              '(default: 1.0)'))

    # DeepSpeed
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    if args.verbose:
        args_list = [f"{key}: {val}" for key, val in args.items()]
        print("PARSED ARGS:\n", '\n'.join(args_list))

    return args


def train(
    model, device, train_loader, optimizer, epoch,
    strategy: TorchDistributedStrategy, args
):
    """
    Training function, representing an epoch.
    """
    model.train()
    t_list = []
    loss_acc = 0
    gwsize = strategy.global_world_size()
    if strategy.is_main_worker:
        print("\n")
    for batch_idx, (data, target) in enumerate(train_loader):
        t = timer()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if (strategy.is_main_worker and args.log_int > 0
                and batch_idx % args.log_int == 0):
            print(
                f'Train epoch: {epoch} '
                f'[{batch_idx * len(data)}/{len(train_loader.dataset)/gwsize} '
                f'({100.0 * batch_idx / len(train_loader):.0f}%)]\t\t'
                f'Loss: {loss.item():.6f}')
        t_list.append(timer() - t)
        loss_acc += loss.item()
    if strategy.is_main_worker:
        print('TIMER: train time', sum(t_list) / len(t_list), 's')
    return loss_acc


def main():
    # Parse CLI args
    args = parse_params()

    # Instantiate Strategy
    if args.strategy == 'ddp':
        if (not torch.cuda.is_available()
                or not torch.cuda.device_count() > 1):
            raise RuntimeError('Resources unavailable')

        strategy = TorchDDPStrategy(backend=args.backend)
        distribute_kwargs = {}
    elif args.strategy == 'horovod':
        strategy = HorovodStrategy()
        distribute_kwargs = dict(
            compression=(
                hvd.Compression.fp16 if args.fp16_allreduce
                else hvd.Compression.none
            ),
            op=hvd.Adasum if args.use_adasum else hvd.Average,
            gradient_predivide_factor=args.gradient_predivide_factor
        )
    elif args.strategy == 'deepspeed':
        strategy = DeepSpeedStrategy(backend=args.backend)
        distribute_kwargs = dict(
            config_params=dict(train_micro_batch_size_per_gpu=args.batch_size)
        )
    else:
        raise NotImplementedError(
            f"Strategy {args.strategy} is not recognized/implemented.")
    strategy.init()

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

    # Get job rank info - rank==0 master gpu
    if is_distributed:
        # local world size - per node
        lwsize = strategy.local_world_size()   # local world size - per run
        gwsize = strategy.global_world_size()   # global world size - per run
        grank = strategy.global_rank()     # global rank - assign per run
        lrank = strategy.local_rank()     # local rank - assign per node
    else:
        # Use a single worker (either on GPU or CPU)
        lwsize = 1
        gwsize = 1
        grank = 0
        lrank = 0

    if strategy.is_main_worker:
        print('TIMER: initialise:', timer()-st, 's')
        print('DEBUG: local ranks:', lwsize, '/ global ranks:', gwsize)
        print('DEBUG: sys.version:', sys.version)
        print('DEBUG: args.data_dir:', args.data_dir)
        print('DEBUG: args.log_int:', args.log_int)
        print('DEBUG: args.nworker:', args.nworker)
        print('DEBUG: args.prefetch:', args.prefetch)
        print('DEBUG: args.batch_size:', args.batch_size)
        print('DEBUG: args.epochs:', args.epochs)
        print('DEBUG: args.lr:', args.lr)
        print('DEBUG: args.momentum:', args.momentum)
        print('DEBUG: args.shuff:', args.shuff)
        print('DEBUG: args.rnd_seed:', args.rnd_seed)
        print('DEBUG: args.backend:', args.backend)
        print('DEBUG: args.no_cuda:', args.no_cuda, '\n')

    # Encapsulate the model on the GPU assigned to the current process
    device = torch.device(
        strategy.device() if use_cuda
        else 'cpu')
    if use_cuda:
        torch.cuda.set_device(lrank)

    # Dataset
    train_dataset = imagenet_dataset(args.data_dir)

    if is_distributed:
        # Distributed sampler restricts data loading to a subset of the dataset
        # exclusive to the current process.
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=gwsize, rank=grank,
            shuffle=(args.shuff and args.rnd_seed is None)
        )

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size,
            sampler=train_sampler, num_workers=args.nworker, pin_memory=True,
            persistent_workers=(args.nworker > 1),
            prefetch_factor=args.prefetch, generator=torch_prng,
            worker_init_fn=seed_worker
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, generator=torch_prng,
            worker_init_fn=seed_worker
        )

    # Create CNN model: resnet 50, resnet101, resnet152
    model = torchvision.models.resnet152()

    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum)

    if is_distributed:
        distrib_model, optimizer, _ = strategy.distributed(
            model, optimizer, lr_scheduler=None, **distribute_kwargs
        )

    # Start training loop
    if strategy.is_main_worker:
        print('TIMER: broadcast:', timer()-st, 's')
        print('\nDEBUG: start training')
        print('--------------------------------------------------------')
        nnod = os.environ.get('SLURM_NNODES', 'unk')
        s_name = f"{args.strategy}-it"
        epoch_time_tracker = EpochTimeTracker(
            series_name=s_name,
            csv_file=f"epochtime_{s_name}_{nnod}N.csv"
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
        train(
            model=distrib_model,
            device=device,
            train_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            strategy=strategy,
            args=args
        )

        # Save first epoch timer
        if epoch == start_epoch:
            first_ep_t = timer()-lt

        # Final epoch
        if epoch + 1 == args.epochs:
            train_loader.last_epoch = True

        if strategy.is_main_worker:
            print('TIMER: epoch time:', timer()-lt, 's')
            epoch_time_tracker.add_epoch_time(epoch-1, timer()-lt)

    if strategy.is_main_worker:
        print('\n--------------------------------------------------------')
        print('DEBUG: training results:\n')
        print('TIMER: first epoch time:', first_ep_t, ' s')
        print('TIMER: last epoch time:', timer()-lt, ' s')
        print('TIMER: average epoch time:', (timer()-et)/args.epochs, ' s')
        print('TIMER: total epoch time:', timer()-et, ' s')
        if epoch > 1:
            print('TIMER: total epoch-1 time:',
                  timer()-et-first_ep_t, ' s')
            print('TIMER: average epoch-1 time:',
                  (timer()-et-first_ep_t)/(args.epochs-1), ' s')
        if use_cuda:
            print('DEBUG: memory req:',
                  int(torch.cuda.memory_reserved(lrank)/1024/1024), 'MB')
            print('DEBUG: memory summary:\n\n',
                  torch.cuda.memory_summary(0))

        print(f'TIMER: final time: {timer()-st} s\n')

    time.sleep(1)
    print(f"<Global rank: {strategy.global_rank()}> - TRAINING FINISHED")

    # Clean-up
    if is_distributed:
        strategy.clean_up()


if __name__ == "__main__":
    main()
    sys.exit()
