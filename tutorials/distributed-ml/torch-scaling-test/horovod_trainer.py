"""
Scaling test of Horovod on Imagenet using Resnet.
"""
from typing import Optional
import argparse
import os
import sys
from timeit import default_timer as timer
import time

import torch
# import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import horovod.torch as hvd
import torchvision

from itwinai.parser import ArgumentParser as ItAIArgumentParser
from itwinai.loggers import EpochTimeTracker
from itwinai.torch.reproducibility import (
    seed_worker, set_seed
)

from utils import imagenet_dataset


def parse_params():
    parser = ItAIArgumentParser(description='PyTorch Imagenet Example')

    # Data and logging
    parser.add_argument('--data-dir', default='./',
                        help=('location of the training dataset in the '
                              'local filesystem'))
    parser.add_argument('--log-int', type=int, default=100,
                        help=('#batches to wait before logging training '
                              'status. Disabled if < 0.'))
    parser.add_argument('--verbose',
                        action=argparse.BooleanOptionalAction,
                        help='Print parsed arguments')
    parser.add_argument('--nworker', type=int, default=0,
                        help=('number of workers in DataLoader '
                              '(default: 0 - only main)'))
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
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--shuff', action='store_true', default=False,
                        help='shuffle dataset (default: False)')

    # Reproducibility
    parser.add_argument('--rnd-seed', type=Optional[int], default=None,
                        help='seed integer for reproducibility (default: 0)')

    # Distributed ML
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                        help='use fp16 compression during allreduce')
    parser.add_argument('--use-adasum', action='store_true', default=False,
                        help='use adasum algorithm to do reduction')
    parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
                        help=('apply gradient pre-divide factor in optimizer '
                              '(default: 1.0)'))

    args = parser.parse_args()
    if args.verbose:
        args_list = [f"{key}: {val}" for key, val in args.items()]
        print("PARSED ARGS:\n", '\n'.join(args_list))

    return args


def train(
        model, optimizer, train_sampler, train_loader,
        args, use_cuda, epoch, grank
):
    model.train()
    t_list = []
    loss_acc = 0
    if grank == 0:
        print("\n")
    for batch_idx, (data, target) in enumerate(train_loader):
        # if hvd.local_rank() == 0 and hvd.rank() == 0:
        #     print(f"BS == DATA: {data.shape}, TARGET: {target.shape}")
        t = timer()
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if grank == 0 and args.log_int > 0 and batch_idx % args.log_int == 0:
            # Use train_sampler to determine the number of examples in
            # this worker's partition
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_sampler),
                100. * batch_idx / len(train_loader), loss.item()))
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

    # Start the time.time for profiling
    st = timer()

    if is_distributed:
        # Initializes the distributed backend which will
        # take care of synchronizing the workers (nodes/GPUs)
        hvd.init()

    # Set random seed for reproducibility
    torch_prng = set_seed(args.rnd_seed, deterministic_cudnn=False)

    # is_main_worker = True
    # if is_distributed and (hvd.rank() != 0 or hvd.local_rank() != 0):
    #     is_main_worker = False

    # Get local rank
    if is_distributed:
        lrank = hvd.local_rank()
        grank = hvd.rank()
        gwsize = hvd.size()
        lwsize = torch.cuda.device_count()
    else:
        # Use a single worker (either on GPU or CPU)
        lrank = 0
        grank = 0
        gwsize = 1
        lwsize = 1

    if grank == 0:
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
        print('DEBUG: args.no_cuda:', args.no_cuda)
        print('DEBUG: args.fp16_allreduce:', args.fp16_allreduce)
        print('DEBUG: args.use_adasum:', args.use_adasum)
        print('DEBUG: args.gradient_predivide_factor:',
              args.gradient_predivide_factor)
        if use_cuda:
            print('DEBUG: torch.cuda.is_available():',
                  torch.cuda.is_available())
            print('DEBUG: torch.cuda.current_device():',
                  torch.cuda.current_device())
            print('DEBUG: torch.cuda.device_count():',
                  torch.cuda.device_count())
            print('DEBUG: torch.cuda.get_device_properties(hvd.local_rank()):',
                  torch.cuda.get_device_properties(hvd.local_rank()))

    if use_cuda:
        # Pin GPU to local rank
        torch.cuda.set_device(lrank)

    # Limit # of CPU threads to be used per worker
    # torch.set_num_threads(1)

    # Dataset
    train_dataset = imagenet_dataset(args.data_dir)

    # kwargs = {}
    # # When supported, use 'forkserver' to spawn dataloader workers instead...
    # # issues with Infiniband implementations that are not fork-safe
    # if (args.nworker > 0 and hasattr(mp, '_supports_context')
    #     and
    #         mp._supports_context and
    #         'forkserver' in mp.get_all_start_methods()):
    #     kwargs['multiprocessing_context'] = 'forkserver'

    if is_distributed:
        # Use DistributedSampler to partition the training data
        # Since Horovod is not based on torch.distributed,
        # `num_replicas` and `rank` cannot be retrieved from the
        # current distributed group, thus they need to be provided explicitly.
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
        )  # , **kwargs)
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, generator=torch_prng,
            worker_init_fn=seed_worker
        )  # , **kwargs)

    # Create CNN model
    model = torchvision.models.resnet152()

    if use_cuda:
        model.cuda()

    if is_distributed:
        # By default, Adasum doesn't need scaling up learning rate
        lr_scaler = hvd.size() if not args.use_adasum else 1
        # If using GPU Adasum allreduce, scale learning rate by local_size
        if args.use_adasum and hvd.nccl_built():
            lr_scaler = hvd.local_size()
        # Scale learning rate by lr_scaler
        args.lr *= lr_scaler

    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum)

    if is_distributed:
        # Broadcast parameters & optimizer state
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        # Compression algorithm
        compression = (
            hvd.Compression.fp16 if args.fp16_allreduce
            else hvd.Compression.none
        )

        # Wrap optimizer with DistributedOptimizer
        optimizer = hvd.DistributedOptimizer(
            optimizer,
            named_parameters=model.named_parameters(),
            compression=compression,
            op=hvd.Adasum if args.use_adasum else hvd.Average,
            gradient_predivide_factor=args.gradient_predivide_factor)

    if grank == 0:
        print('TIMER: broadcast:', timer()-st, 's')
        print('\nDEBUG: start training')
        print('--------------------------------------------------------')
        nnod = os.environ.get('SLURM_NNODES', 'unk')
        epoch_time_tracker = EpochTimeTracker(
            series_name="horovod-bl",
            csv_file=f"epochtime_horovod-bl_{nnod}N.csv"
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
        train(model, optimizer, train_sampler,
              train_loader, args, use_cuda, epoch, grank)

        # Save first epoch timer
        if epoch == start_epoch:
            first_ep_t = timer()-lt

        # Final epoch
        if epoch + 1 == args.epochs:
            train_loader.last_epoch = True

        if grank == 0:
            print('TIMER: epoch time:', timer()-lt, 's')
            epoch_time_tracker.add_epoch_time(epoch-1, timer()-lt)

    if grank == 0:
        print('\n--------------------------------------------------------')
        print('DEBUG: training results:\n')
        print('TIMER: first epoch time:', first_ep_t, ' s')
        print('TIMER: last epoch time:', timer()-lt, 's')
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
    print(f"<Hvd rank: {hvd.rank()}> - TRAINING FINISHED")


if __name__ == "__main__":
    main()
    sys.exit()
