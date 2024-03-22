"""
Show how to use DDP, Horovod and DeepSpeed strategies interchangeably
with a large neural network trained on Imagenet dataset, showing how
to use checkpoints.
"""
import os
import argparse
import sys
import time
import numpy as np
import random

import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, DistributedSampler

import deepspeed

from itwinai.torch.distributed import (
    TorchDistributedStrategy,
    DDPDistributedStrategy,
    HVDDistributedStrategy,
    DSDistributedStrategy,
)
from itwinai.parser import ArgumentParser as ItAIArgumentParser


def parse_args() -> argparse.Namespace:
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

    # IO parsers
    parser.add_argument('--data-dir', default='./',
                        help=('location of the training dataset in the local '
                              'filesystem'))
    parser.add_argument('--restart-int', type=int, default=10,
                        help='restart interval per epoch (default: 10)')
    parser.add_argument('--verbose',
                        action=argparse.BooleanOptionalAction,
                        help='Print parsed arguments')

    # model parsers
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
    parser.add_argument('--num-classes', type=int, default=1000,
                        help='number of classes in dataset')

    # debug parsers
    parser.add_argument('--testrun', action='store_true', default=False,
                        help='do a test run with seed (default: False)')
    parser.add_argument('--nseed', type=int, default=0,
                        help='seed integer for reproducibility (default: 0)')
    parser.add_argument('--log-int', type=int, default=10,
                        help='log interval per training')

    # parallel parsers
    parser.add_argument('--backend', type=str, default='nccl',
                        help='backend for parrallelisation (default: nccl)')
    parser.add_argument('--nworker', type=int, default=0,
                        help=('number of workers in DataLoader (default: 0 -'
                              ' only main)'))
    parser.add_argument('--prefetch', type=int, default=2,
                        help='prefetch data in DataLoader (default: 2)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables GPGPUs')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')

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
    gwsize = strategy.dist_gwsize()
    if strategy.is_main_worker():
        print("\n")
    for batch_idx, (data, target) in enumerate(train_loader):
        t = time.perf_counter()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_int == 0 and strategy.is_main_worker():
            print(
                f'Train epoch: {epoch} '
                f'[{batch_idx * len(data)}/{len(train_loader.dataset)/gwsize} '
                f'({100.0 * batch_idx / len(train_loader):.0f}%)]\t\t'
                f'Loss: {loss.item():.6f}')
        t_list.append(time.perf_counter() - t)
        loss_acc += loss.item()
    if strategy.is_main_worker():
        print('TIMER: train time', sum(t_list) / len(t_list), 's')
    return loss_acc


def test(model, device, test_loader, strategy: TorchDistributedStrategy):
    """
    Model validation.
    """
    model.eval()
    test_loss = 0
    correct = 0
    gwsize = strategy.dist_gwsize()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    if strategy.is_main_worker():
        print(
            f'Test set: average loss: {test_loss:.4f}\t'
            f'accurate samples: {correct}/{len(test_loader.dataset)/gwsize}')
    acc_test = 100.0 * correct * gwsize / len(test_loader.dataset)
    return acc_test


def save_state(
        epoch, distrib_model, loss_acc, optimizer,
        res_name, is_best, strategy: TorchDistributedStrategy
):
    """
    Save training state.
    """
    grank = strategy.dist_grank()
    rt = time.time()
    # find if is_best happened in any worker
    if torch.cuda.is_available():
        is_best_m = strategy.par_allgather_obj(is_best)

    if torch.cuda.is_available():
        if any(is_best_m):
            # find which rank is_best happened - select first rank if multiple
            is_best_rank = np.where(np.array(is_best_m))[0][0]

            # collect state
            state = {'epoch': epoch + 1,
                     'state_dict': distrib_model.state_dict(),
                     'best_acc': loss_acc,
                     'optimizer': optimizer.state_dict()}

            # write on worker with is_best
            if grank == is_best_rank:
                torch.save(state, './'+res_name)
                print(
                    f'DEBUG: state in {grank} is saved on epoch:{epoch} '
                    f'in {time.time()-rt} s')
    else:
        # collect state
        state = {'epoch': epoch + 1,
                 'state_dict': distrib_model.state_dict(),
                 'best_acc': loss_acc,
                 'optimizer': optimizer.state_dict()}

        torch.save(state, './'+res_name)
        print(
            f'DEBUG: state in {grank} is saved on epoch:{epoch} in '
            f'{time.time()-rt} s')


def seed_worker(worker_id):
    """
    Seed dataloader worker.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == "__main__":

    args = parse_args()

    # Instantiate Strategy
    if args.strategy == 'ddp':
        if (not torch.cuda.is_available()
                or not torch.cuda.device_count() > 1):
            raise RuntimeError('Resources unavailable')

        strategy = DDPDistributedStrategy(backend=args.backend)
    elif args.strategy == 'horovod':
        strategy = HVDDistributedStrategy()
    elif args.strategy == 'deepspeed':
        strategy = DSDistributedStrategy(backend=args.backend)
    else:
        raise NotImplementedError(
            f"Strategy {args.strategy} is not recognized/implemented.")
    strategy.init()

    # check CUDA availability
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # limit # of CPU threads to be used per worker
    torch.set_num_threads(1)

    # get directory
    program_dir = os.getcwd()

    # start the time.time for profiling
    st = time.time()

    # deterministic testrun
    if args.testrun:
        torch.manual_seed(args.nseed)
        g = torch.Generator()
        g.manual_seed(args.nseed)

    # get job rank info - rank==0 master gpu
    if torch.cuda.is_available():
        # local world size - per node
        lwsize = strategy.dist_lwsize() if args.cuda else 0
        gwsize = strategy.dist_gwsize()   # global world size - per run
        grank = strategy.dist_grank()     # global rank - assign per run
        lrank = strategy.dist_lrank()     # local rank - assign per node
    else:
        gwsize = 1
        grank = 0

    # some debug
    if strategy.is_main_worker():
        print('TIMER: initialise:', time.time()-st, 's')

    # move the model on the GPU assigned to the current process
    device = torch.device(
        strategy.dist_device() if args.cuda and torch.cuda.is_available()
        else 'cpu')
    if args.cuda:
        torch.cuda.set_device(lrank)
        # deterministic testrun
        if args.testrun:
            torch.cuda.manual_seed(args.nseed)

    # dataset
    # Initialize transformations for data augmentation
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=45),
        transforms.ColorJitter(
            brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load the ImageNet Object Localization Challenge dataset
    train_dataset = torchvision.datasets.ImageFolder(
        root=args.data_dir,
        transform=transform
    )
    # test_dataset = ...

    # restricts data loading to a subset of the dataset exclusive to the
    # current process
    args.shuff = args.shuff and not args.testrun
    if torch.cuda.is_available():
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=gwsize, rank=grank, shuffle=args.shuff)
        # test_sampler = DistributedSampler(
        #     test_dataset, num_replicas=gwsize, rank=grank,
        # shuffle=args.shuff)
    # distribute dataset to workers
    # persistent workers is not possible for nworker=0
    pers_w = True if args.nworker > 1 else False

    # deterministic testrun - the same dataset each run
    kwargs = {'worker_init_fn': seed_worker,
              'generator': g} if args.testrun else {}

    if torch.cuda.is_available():
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size,
            sampler=train_sampler, num_workers=args.nworker, pin_memory=True,
            persistent_workers=pers_w, prefetch_factor=args.prefetch, **kwargs
        )
        # test_loader = DataLoader(
        #     test_dataset, batch_size=args.batch_size,
        #     sampler=test_sampler, num_workers=args.nworker, pin_memory=True,
        #     persistent_workers=pers_w, prefetch_factor=args.prefetch,
        # **kwargs
        # )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size)
        # test_loader = DataLoader(
        #     test_dataset, batch_size=args.batch_size)

    if strategy.is_main_worker():
        print('TIMER: read and concat data:', time.time()-st, 's')

    # create CNN model
    model = torchvision.models.resnet101()
    model.fc = nn.Linear(2048, args.num_classes)

    # optimizer
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum)

    deepspeed_config = dict(train_batch_size=args.batch_size)
    # 'config_params' key is ignored if strategy != DSDistributedStrategy
    distrib_model, optimizer, _ = strategy.distributed(
        model, optimizer, lr_scheduler=None, config_params=deepspeed_config
    )

    # resume state
    start_epoch = 1
    best_acc = np.Inf
    res_name = f'{args.strategy}-checkpoint.pth.tar'
    if os.path.isfile(res_name):
        try:
            if torch.cuda.is_available():
                dist.barrier()
                # Map model to be loaded to specified single gpu.
                loc = {'cuda:%d' % 0: 'cuda:%d' % lrank} if args.cuda else {
                    'cpu:%d' % 0: 'cpu:%d' % lrank}
                checkpoint = torch.load(
                    program_dir+'/'+res_name, map_location=loc)
            else:
                checkpoint = torch.load(program_dir+'/'+res_name)
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            distrib_model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if torch.cuda.is_available():
                if strategy.is_main_worker():
                    print(f'WARNING: restarting from {start_epoch} epoch')
            else:
                print(f'WARNING: restarting from {start_epoch} epoch')
        except Exception:
            if torch.cuda.is_available():
                if strategy.is_main_worker():
                    print('WARNING: restart file cannot be loaded, '
                          'restarting!')
            else:
                print('WARNING: restart file cannot be loaded, restarting!')

    if start_epoch > args.epochs:
        if torch.cuda.is_available():
            if strategy.is_main_worker():
                print('WARNING: given epochs are less than the one in the '
                      'restart file!\n'
                      'WARNING: SYS.EXIT is issued')

            strategy.clean_up()
            sys.exit()
        else:
            print('WARNING: given epochs are less than the one in '
                  'the restart file!\n'
                  'WARNING: SYS.EXIT is issued')
            sys.exit()

    # start trainin/testing loop
    if strategy.is_main_worker():
        print('TIMER: broadcast:', time.time()-st, 's')
        print('\nDEBUG: start training')
        print('--------------------------------------------------------')

    et = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        lt = time.time()
        # training
        loss_acc = train(
            model=distrib_model,
            device=device,
            train_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            strategy=strategy,
            args=args
        )

        # # testing
        # acc_test = test(
        #     model=distrib_model,
        #     device=device,
        #     test_loader=test_loader,
        #     strategy=strategy
        # )

        # save first epoch timer
        if epoch == start_epoch:
            first_ep_t = time.time()-lt

        # final epoch
        if epoch + 1 == args.epochs:
            train_loader.last_epoch = True
            # test_loader.last_epoch = True

        if strategy.is_main_worker():
            print('TIMER: epoch time:', time.time()-lt, 's')
            # print('DEBUG: accuracy:', acc_test, '%')

        # save state if found a better state
        is_best = loss_acc < best_acc
        if epoch % args.restart_int == 0:
            save_state(
                epoch=epoch,
                distrib_model=distrib_model,
                loss_acc=loss_acc,
                optimizer=optimizer,
                res_name=res_name,
                is_best=is_best,
                strategy=strategy
            )
            # reset best_acc
            best_acc = min(loss_acc, best_acc)

    # finalise
    # save final state
    save_state(
        epoch=epoch,
        distrib_model=distrib_model,
        loss_acc=loss_acc,
        optimizer=optimizer,
        res_name=res_name,
        is_best=True,
        strategy=strategy
    )

    # some debug
    if strategy.is_main_worker():
        print('\n--------------------------------------------------------')
        print('DEBUG: training results:\n')
        print('TIMER: first epoch time:', first_ep_t, ' s')
        print('TIMER: last epoch time:', time.time()-lt, ' s')
        print('TIMER: average epoch time:', (time.time()-et)/args.epochs, ' s')
        print('TIMER: total epoch time:', time.time()-et, ' s')
        if epoch > 1:
            print('TIMER: total epoch-1 time:',
                  time.time()-et-first_ep_t, ' s')
            print('TIMER: average epoch-1 time:',
                  (time.time()-et-first_ep_t)/(args.epochs-1), ' s')
        # print('DEBUG: last accuracy:', acc_test, '%')
        print('DEBUG: memory req:',
              int(torch.cuda.memory_reserved(lrank)/1024/1024), 'MB') \
            if args.cuda else 'DEBUG: memory req: - MB'
        print('DEBUG: memory summary:\n\n',
              torch.cuda.memory_summary(0)) if args.cuda else ''

    if strategy.is_main_worker():
        print(f'TIMER: final time: {time.time()-st} s\n')

    print(f"<Global rank: {strategy.dist_grank()}> - TRAINING FINISHED")
    strategy.clean_up()
    sys.exit()
