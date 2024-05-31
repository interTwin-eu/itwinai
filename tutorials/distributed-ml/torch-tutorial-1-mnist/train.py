"""
Show how to use DDP, Horovod and DeepSpeed strategies interchangeably
with a simple neural network trained on MNIST dataset.
"""
from typing import Tuple
import argparse
import sys
import time
from timeit import default_timer as timer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset

import horovod.torch as hvd

import deepspeed

from itwinai.torch.distributed import (
    distributed_resources_available,
    TorchDistributedStrategy,
    TorchDDPStrategy,
    HorovodStrategy,
    DeepSpeedStrategy,
    NonDistributedStrategy
)
from itwinai.parser import ArgumentParser as ItAIArgumentParser
from itwinai.torch.reproducibility import (
    seed_worker, set_seed
)


def parse_params() -> argparse.Namespace:
    """
    Parse CLI args, which can also be loaded from a configuration file
    using the --config flag:

    >>> train.py --strategy ddp --config config.yaml
    """
    parser = ItAIArgumentParser(description='PyTorch MNIST Example')

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
    parser.add_argument('--restart-int', type=int, default=10,
                        help='restart interval per epoch (default: 10)')
    parser.add_argument('--download-only',
                        action=argparse.BooleanOptionalAction,
                        help='Download dataset and exit')
    parser.add_argument('--dataset-replication', type=int, default=100,
                        help='concatenate MNIST to this factor (default: 100)')
    parser.add_argument('--shuff', action='store_true', default=False,
                        help='shuffle dataset (default: False)')
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

    # Reproducibility
    parser.add_argument('--rnd-seed', type=int, default=0,
                        help='seed integer for reproducibility (default: 0)')

    # Distributed ML
    parser.add_argument('--backend', type=str, default='nccl',
                        help='backend for parrallelisation (default: nccl)')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')

    # Horovod: ignored when not using Horovod
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


class Net(nn.Module):
    """
    Simple neural network classifier for MNIST images.
    """

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


def train(
    model, train_loader, optimizer, epoch,
    strategy: TorchDistributedStrategy, args
):
    """
    Training function, representing an epoch.
    """
    model.train()
    t_list = []
    loss_acc = 0
    if strategy.is_main_worker:
        print("\n")
    for batch_idx, (data, target) in enumerate(train_loader):
        t = timer()
        data = data.to(strategy.device())
        target = target.to(strategy.device())
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if (strategy.is_main_worker and args.log_int > 0
                and batch_idx % args.log_int == 0):
            dl_size = len(train_loader.dataset)//strategy.global_world_size()
            print(
                f'Train epoch: {epoch} '
                f'[{batch_idx * len(data)}/{dl_size} '
                f'({100.0 * batch_idx / len(train_loader):.0f}%)]\t\t'
                f'Loss: {loss.item():.6f}')
        t_list.append(timer() - t)
        loss_acc += loss.item()
    if strategy.is_main_worker:
        print('TIMER: train time', sum(t_list) / len(t_list), 's')
    return loss_acc


def test(model, test_loader, strategy: TorchDistributedStrategy):
    """
    Model validation.
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(strategy.device())
            target = target.to(strategy.device())
            output = model(data)
            # Sum up batch loss
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            # Get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    if strategy.is_main_worker:
        dl_size = len(test_loader.dataset)//strategy.global_world_size()
        print(
            f'Test set: average loss: {test_loss:.4f}\t'
            f'accurate samples: {correct}/{dl_size}')
    acc_test = (
        100.0 * correct * strategy.global_world_size()
        / len(test_loader.dataset)
    )
    return acc_test


def download_mnist():
    """
    Use built-in torch datasets functions to pull MNIST dataset.
    """

    _ = datasets.MNIST(
        args.data_dir, train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
    _ = datasets.MNIST(
        args.data_dir, train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))


def mnist_dataset(dataset_replication: int = 1) -> Tuple[Dataset, Dataset]:
    """Load MNIST train and test datasets, replicating them.

    Args:
        dataset_replication (int): dataset replication factor. Default 1.

    Returns:
        Tuple[Dataset, Dataset]: train dataset and test dataset.
    """
    replicated_data = [
        datasets.MNIST(args.data_dir, train=True, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
        for _ in range(dataset_replication)
    ]
    train_dataset = torch.utils.data.ConcatDataset(replicated_data)

    replicated_data = [
        datasets.MNIST(args.data_dir, train=False, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
        for _ in range(dataset_replication)
    ]
    test_dataset = torch.utils.data.ConcatDataset(replicated_data)
    return train_dataset, test_dataset


if __name__ == "__main__":

    args = parse_params()

    if args.download_only:
        # Download datasets from a location with internet access and exit.
        # This is convenient when submitting training jobs to
        # a batch system where worker nodes have no internet
        # access, like in some HPCs.
        download_mnist()
        sys.exit()

    # Instantiate Strategy
    if not distributed_resources_available():
        print("WARNING: falling back to non-distributed strategy.")
        strategy = NonDistributedStrategy()
        distribute_kwargs = {}
    elif args.strategy == 'ddp':
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

    # Initialize strategy
    strategy.init()

    # Start the timer for profiling
    st = timer()

    # Set random seed for reproducibility
    torch_prng = set_seed(args.rnd_seed)

    if strategy.is_main_worker:
        print('TIMER: initialise:', timer()-st, 's')
        print('DEBUG: local ranks:', strategy.local_world_size(),
              '/ global ranks:', strategy.global_world_size())
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

    # Dataset
    train_dataset, test_dataset = mnist_dataset(args.dataset_replication)
    # Distributed dataloaders
    train_loader = strategy.create_dataloader(
        train_dataset, batch_size=args.batch_size,
        num_workers=args.nworker, pin_memory=True,
        persistent_workers=(args.nworker > 1),
        prefetch_factor=args.prefetch, generator=torch_prng,
        worker_init_fn=seed_worker
    )
    test_loader = strategy.create_dataloader(
        test_dataset, batch_size=args.batch_size,
        num_workers=args.nworker, pin_memory=True,
        persistent_workers=(args.nworker > 1),
        prefetch_factor=args.prefetch, generator=torch_prng,
        worker_init_fn=seed_worker
    )

    if strategy.is_main_worker:
        print('TIMER: read and concat data:', timer()-st, 's')

    # Create CNN model
    model = Net().to(strategy.device())

    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum)

    # Distributed model, optimizer, and scheduler
    model, optimizer, _ = strategy.distributed(
        model, optimizer, lr_scheduler=None, **distribute_kwargs
    )

    # Start training and test loop
    if strategy.is_main_worker:
        print('TIMER: broadcast:', timer()-st, 's')
        print('\nDEBUG: start training')
        print('--------------------------------------------------------')

    et = timer()
    start_epoch = 1
    for epoch in range(start_epoch, args.epochs + 1):
        lt = timer()
        if strategy.is_distributed:
            # Inform the sampler that a new epoch started: shuffle
            # may be needed
            train_loader.sampler.set_epoch(epoch)
            test_loader.sampler.set_epoch(epoch)

        # Training
        loss_acc = train(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            strategy=strategy,
            args=args
        )

        # Testing
        acc_test = test(
            model=model,
            test_loader=test_loader,
            strategy=strategy
        )

        # Save first epoch timer
        if epoch == start_epoch:
            first_ep_t = timer()-lt

        # Final epoch
        if epoch + 1 == args.epochs:
            train_loader.last_epoch = True
            test_loader.last_epoch = True

        if strategy.is_main_worker:
            print('TIMER: epoch time:', timer()-lt, 's')
            print('DEBUG: accuracy:', acc_test, '%')

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
        print('DEBUG: last accuracy:', acc_test, '%')
        if torch.cuda.is_available():
            print('DEBUG: memory req:',
                  int(torch.cuda.memory_reserved(
                      strategy.local_rank())/1024/1024),
                  'MB')
            print('DEBUG: memory summary:\n\n',
                  torch.cuda.memory_summary(0))

        print(f'TIMER: final time: {timer()-st} s\n')

    time.sleep(1)
    print(f"<Global rank: {strategy.global_rank()}> - TRAINING FINISHED")

    # Clean-up
    strategy.clean_up()
    sys.exit()
