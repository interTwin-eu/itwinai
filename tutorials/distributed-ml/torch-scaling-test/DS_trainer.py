"""
Scaling test of Microsoft Deepspeed on Imagenet using Resnet.
"""
from typing import Optional
import argparse
import sys
import os
from timeit import default_timer as timer
import time
import deepspeed

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision

from itwinai.parser import ArgumentParser as ItAIArgumentParser
from itwinai.loggers import EpochTimeTracker

from utils import imagenet_dataset


def parse_params():
    parser = ItAIArgumentParser(description='PyTorch Imagenet scaling test')

    # Data and logging
    parser.add_argument('--data-dir', default='./',
                        help=('location of the training dataset in the '
                              'local filesystem'))
    parser.add_argument('--log-int', type=int, default=10,
                        help='log interval per training. Disabled if < 0.')
    parser.add_argument('--verbose',
                        action=argparse.BooleanOptionalAction,
                        help='Print parsed arguments')
    parser.add_argument('--nworker', type=int, default=0,
                        help=('number of workers in DataLoader '
                              '(default: 0 - only main)'))

    # Model
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')

    # Reproducibility
    parser.add_argument('--rnd-seed', type=Optional[int], default=None,
                        help='seed integer for reproducibility (default: 0)')

    # Distributed ML
    parser.add_argument('--backend', type=str, default='nccl', metavar='N',
                        help='backend for parallelization (default: nccl)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables GPGPUs')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')

    # parse to deepspeed
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    if args.verbose:
        args_list = [f"{key}: {val}" for key, val in args.items()]
        print("PARSED ARGS:\n", '\n'.join(args_list))

    return args


def train(args, model, train_loader, optimizer, epoch, grank, gwsize):
    device = model.local_rank
    t_list = []
    loss_acc = 0
    if grank == 0:
        print("\n")
    for batch_idx, (data, target) in enumerate(train_loader):
        # if grank == 0:
        #     print(f"BS == DATA: {data.shape}, TARGET: {target.shape}")
        t = timer()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if args.log_int > 0 and batch_idx % args.log_int == 0 and grank == 0:
            print(
                f'Train epoch: {epoch} [{batch_idx * len(data)}/'
                f'{len(train_loader.dataset)/gwsize} '
                f'({100.0 * batch_idx *len(data) / len(train_loader):.0f}%)]'
                f'\t\tLoss: {loss.item():.6f}')
        t_list.append(timer() - t)
        loss_acc += loss.item()
    if grank == 0:
        print('TIMER: train time', sum(t_list) / len(t_list), 's')
    return loss_acc


def main():
    # Parse CLI args
    args = parse_params()

    # Check resources availability
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    is_distributed = False
    if use_cuda and torch.cuda.device_count() > 0:
        is_distributed = True

    # Limit # of CPU threads to be used per worker
    # torch.set_num_threads(1)

    # Start the timer for profiling
    st = timer()

    # Initializes the distributed backend
    if is_distributed:
        deepspeed.init_distributed(dist_backend=args.backend)

    if args.rnd_seed is not None:
        # Deterministic execution
        torch.manual_seed(args.rnd_seed)

    if is_distributed:
        # Get job rank info - rank==0 master gpu
        gwsize = dist.get_world_size()     # global world size - per run
        lwsize = torch.cuda.device_count()  # local world size - per node
        grank = dist.get_rank()            # global rank - assign per run
        lrank = dist.get_rank() % lwsize     # local rank - assign per node
    else:
        # Use a single worker (either on GPU or CPU)
        lwsize = 1
        gwsize = 1
        grank = 0
        lrank = 0

    # some debug
    if grank == 0:
        print('TIMER: initialise:', timer()-st, 's')
        print('DEBUG: local ranks:', lwsize, '/ global ranks:', gwsize)
        print('DEBUG: sys.version:', sys.version)
        print('DEBUG: args.data_dir:', args.data_dir)
        print('DEBUG: args.log_int:', args.log_int)
        print('DEBUG: args.nworker:', args.nworker)
        print('DEBUG: args.batch_size:', args.batch_size)
        print('DEBUG: args.epochs:', args.epochs)
        print('DEBUG: args.lr:', args.lr)
        print('DEBUG: args.rnd_seed:', args.rnd_seed)
        print('DEBUG: args.backend:', args.backend)
        print('DEBUG: args.local_rank:', args.local_rank)
        print('DEBUG: args.no_cuda:', args.no_cuda, '\n')

    # Encapsulate the model on the GPU assigned to the current process
    if use_cuda:
        torch.cuda.set_device(lrank)

    # Read training dataset
    train_dataset = imagenet_dataset(args.data_dir)

    # Create CNN model
    model = torchvision.models.resnet152()

    # Initialize DeepSpeed to use the following features
    # 1) Distributed model
    # 2) DeepSpeed optimizer
    # 3) Distributed data loader
    deepspeed_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "optimizer": {
            "type": "SGD",
            "params": {
                "lr": args.lr,
                "momentum": 0.5
            }
        },
        "fp16": {
            "enabled": False
        },
        "zero_optimization": False
    }
    distrib_model, optimizer, train_loader, _ = deepspeed.initialize(
        args=args, model=model, model_parameters=model.parameters(),
        training_data=train_dataset, config_params=deepspeed_config)

    # Start training loop
    if grank == 0:
        print('TIMER: broadcast:', timer()-st, 's')
        print('\nDEBUG: start training')
        print('--------------------------------------------------------')
        nnod = os.environ.get('SLURM_NNODES', 'unk')
        epoch_time_tracker = EpochTimeTracker(
            series_name="deepspeed-bl",
            csv_file=f"epochtime_deepspeed-bl_{nnod}N.csv"
        )

    et = timer()
    start_epoch = 1
    for epoch in range(start_epoch, args.epochs + 1):
        lt = timer()
        # Training
        train(args, distrib_model, train_loader,
              optimizer, epoch, grank, gwsize)

        # Save first epoch timer
        if epoch == start_epoch:
            first_ep_t = timer()-lt

        # Final epoch
        if epoch + 1 == args.epochs:
            train_loader.last_epoch = True

        if grank == 0:
            print('TIMER: epoch time:', timer()-lt, 's')
            epoch_time_tracker.add_epoch_time(epoch-1, timer()-lt)

    if torch.cuda.is_available():
        dist.barrier()

    if grank == 0:
        print('\n--------------------------------------------------------')
        print('DEBUG: results:\n')
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
    print(f"<Global rank: {grank}> - TRAINING FINISHED")

    # Clean-up
    if is_distributed:
        deepspeed.sys.exit()


if __name__ == "__main__":
    main()
    sys.exit()
