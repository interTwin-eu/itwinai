#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author: RS, adapted from https://gitlab.jsc.fz-juelich.de/CoE-RAISE/FZJ/ai4hpc
# version: 211029a

# std libs
from typing import Any, Union
import argparse
import sys
import os
import time
import numpy as np
import random
import abc

# ml libs
import deepspeed
import torch
import torch.distributed as dist
import horovod.torch as hvd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# from itwinai.types import MLModel, MLDevice

# parsed settings


def pars_ini():
    global args
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

    # IO parsers
    parser.add_argument('--data-dir', default='./',
                        help='location of the training dataset in the local filesystem')
    parser.add_argument('--restart-int', type=int, default=10,
                        help='restart interval per epoch (default: 10)')

    # model parsers
    parser.add_argument('--strategy', type=str, default='DDP',
                        help='strategy for parallelization (DDP, HVD, DS)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--concM', type=int, default=100,
                        help='conc MNIST to this factor (default: 100)')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='momentum in SGD optimizer (default: 0.5)')
    parser.add_argument('--shuff', action='store_true', default=False,
                        help='shuffle dataset (default: False)')

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
                        help='number of workers in DataLoader (default: 0 - only main)')
    parser.add_argument('--prefetch', type=int, default=2,
                        help='prefetch data in DataLoader (default: 2)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables GPGPUs')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')

    try:
        parser = deepspeed.add_config_arguments(parser)
    except:
        pass

    args = parser.parse_args()


class DistributedStrategy(abc.ABC):
    """Abstract class to define the distributed backend methods"""
    @abc.abstractmethod
    def init_backend(self) -> None:
        """Initializes the chosen distributed backend"""

    @abc.abstractmethod
    def distribute_model(self, model: Any, device: Union[int, str]) -> Any:
        """Distributes a machine learning model.

        Args:
            model (Any): a generic ML model to be distributed.
            device (Union[int, str]): device on which the model is run.

        Returns:
            Any: distributed model instance.
        """

    @abc.abstractmethod
    def broadcast_params(self, distrib_model, optimizer) -> None:
        """
        HVD: broadcasts variables from root rank to all other processes
        DDP and DS: pass
        """

    @abc.abstractmethod
    def distribute_optimizer(self, optimizer, distrib_model) -> Any:
        """
        HVD: construct a new DistributedOptimizer, which uses another optimizer
        under the hood for computing single-process gradient values and
        applying gradient updates after the gradient values have been
        combined across all the Horovod ranks.
        DDP and DS: returns the same optimizer passed in argument
        """

    @abc.abstractmethod
    def dist_gwsize(self) -> int:
        """Returns the number of processes"""

    @abc.abstractmethod
    def dist_lwsize(self) -> int:
        """Returns the number of GPUs available"""

    @abc.abstractmethod
    def dist_grank(self) -> int:
        """
        Returns the rank of the current process.
        Rank ranges from 0 to world_size
        """

    @abc.abstractmethod
    def dist_lrank(self) -> int:
        """Returns the local rank of the current process."""

    @abc.abstractmethod
    def clean_up(self) -> None:
        """Destroys the current process group."""

    @abc.abstractmethod
    def par_allgather_obj(self, obj, gwsize) -> list:
        """
        Gathers any object from the whole group 
        in a list (to all workers)
        """


class DDPDistributedTrainer(DistributedStrategy):
    """PyTorch DDP distributed training class"""

    def init_backend(self) -> None:
        """Initializes the distributed process group and the distributed package"""
        if torch.cuda.is_available():
            dist.init_process_group(backend=args.backend)

    def distribute_model(self, model, device) -> nn.Module:
        """
        Achieves data parallelism by synchronising the gradients across 
        each model replica located in each available computing device. 
        """
        if torch.cuda.is_available():
            dist_model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[device],
                output_device=device
            )
        else:
            dist_model = model

        return dist_model

    def broadcast_params(self, distrib_model, optimizer) -> None:
        """Only applicable for Horovod, else pass"""
        pass

    def distribute_optimizer(
        self,
        optimizer,
        distrib_model
    ) -> optim.Optimizer:
        """Only applicable for Horovod, else returns the optimizer from the argument"""
        return optimizer

    def dist_gwsize(self) -> int:
        """Returns the number of processes"""
        return dist.get_world_size()

    def dist_lwsize(self) -> int:
        """Returns the number of GPUs available"""
        return torch.cuda.device_count()

    def dist_grank(self) -> int:
        """
        Returns the rank of the current process.
        Rank ranges from 0 to world_size
        """
        return dist.get_rank()

    def dist_lrank(self) -> int:
        """Returns the local rank of the current process."""
        return dist.get_rank() % torch.cuda.device_count()

    def clean_up(self) -> None:
        """Destroys the current process group."""
        if torch.cuda.is_available():
            dist.barrier()
            dist.destroy_process_group()

    def par_allgather_obj(self, obj, gwsize) -> list:
        """
        Gathers any object from the whole group 
        in a list (to all workers)
        """
        res = [None]*gwsize
        dist.all_gather_object(res, obj)
        return res


class DSDistributedTrainer(DistributedStrategy):
    """DeepSpeed distributed training class"""

    def init_backend(self) -> None:
        """Initializes the distributed process group and the distributed package"""
        deepspeed.init_distributed(dist_backend=args.backend)

    def distribute_model(self, model, device) -> nn.Module:
        """
        Achieves data parallelism by synchronising the gradients across 
        each model replica located in each available computing device. 
        """
        distrib_model, __, __, __ = deepspeed.initialize(
            args=args, model=model, model_parameters=model.parameters(), dist_init_required=True)
        return distrib_model

    def broadcast_params(self, distrib_model, optimizer) -> None:
        """Only applicable for Horovod, else pass"""
        pass

    def distribute_optimizer(self, optimizer, distrib_model) -> optim.Optimizer:
        """Only applicable for Horovod, else returns the optimizer from the argument"""
        return optimizer

    def dist_gwsize(self) -> int:
        """Returns the number of processes"""
        return dist.get_world_size()

    def dist_lwsize(self) -> int:
        """Returns the number of GPUs available"""
        return torch.cuda.device_count()

    def dist_grank(self) -> int:
        """
        Returns the rank of the current process.
        Rank ranges from 0 to world_size
        """
        return dist.get_rank()

    def dist_lrank(self) -> int:
        """Returns the local rank of the current process."""
        return dist.get_rank() % torch.cuda.device_count()

    def clean_up(self) -> None:
        """Destroys the current process group."""
        deepspeed.sys.exit()

    def par_allgather_obj(self, obj, gwsize) -> list:
        """
        Gathers any object from the whole group 
        in a list (to all workers)
        """
        res = [None]*gwsize
        dist.all_gather_object(res, obj)
        return res


class HVDDistributedTrainer(DistributedStrategy):
    """Horovod distributed training class"""

    def init_backend(self) -> None:
        """Initializes the Horovod distributed backend"""
        hvd.init()

    def distribute_model(self, model, device) -> nn.Module:
        """For Horovod, returns the same model passed as argument"""
        distrib_model = model
        return distrib_model

    def broadcast_params(self, distrib_model, optimizer) -> None:
        """Broadcasts variables from root rank to all other processes"""
        hvd.broadcast_parameters(distrib_model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=-0)

    def distribute_optimizer(self, optimizer, distrib_model) -> optim.Optimizer:
        """
        Construct a new DistributedOptimizer, which uses another optimizer 
        under the hood for computing single-process gradient values and 
        applying gradient updates after the gradient values have been 
        combined across all the Horovod ranks.
        """
        distOptimizer = hvd.DistributedOptimizer(optimizer,
                                                 named_parameters=distrib_model.named_parameters(),
                                                 op=hvd.Average)
        return distOptimizer

    def dist_gwsize(self) -> int:
        """Returns the number of processes"""
        return hvd.size()

    def dist_lwsize(self) -> int:
        """Returns the number of GPUs available"""
        return hvd.local_size()

    def dist_grank(self) -> int:
        """
        Returns the rank of the current process.
        Rank ranges from 0 to world_size
        """
        return hvd.rank()

    def dist_lrank(self) -> int:
        """Returns the local rank of the current process."""
        return hvd.local_rank()

    def clean_up(self) -> None:
        """Shuts Horovod down."""
        hvd.shutdown()

    def par_allgather_obj(self, obj, gwsize) -> list:
        """
        Gathers scalar objects across 
        all workers to a list with size(\#worker)   
        uses horovod communicator
        @param obj object in a worker
        @param gwsize global world size

        @return gathered list with size(#worker)
        """
        return hvd.allgather_object(obj)


class Net(nn.Module):
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
        return F.log_softmax(x)

# train loop


def train(model, device, train_loader, optimizer, epoch, grank, gwsize, args):
    model.train()
    t_list = []
    loss_acc = 0
    if grank == 0:
        print("\n")
    for batch_idx, (data, target) in enumerate(train_loader):
        t = time.perf_counter()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_int == 0 and grank == 0:
            print(
                f'Train epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)/gwsize} '
                f'({100.0 * batch_idx / len(train_loader):.0f}%)]\t\tLoss: {loss.item():.6f}')
        t_list.append(time.perf_counter() - t)
        loss_acc += loss.item()
    if grank == 0:
        print('TIMER: train time', sum(t_list) / len(t_list), 's')
    return loss_acc

# test loop


def test(model, device, test_loader, grank, gwsize, args):
    model.eval()
    test_loss = 0
    correct = 0
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
    if grank == 0:
        print(
            f'Test set: average loss: {test_loss:.4f}\t'
            f'accurate samples: {correct}/{len(test_loader.dataset)/gwsize}')
    acc_test = 100.0 * correct * gwsize / len(test_loader.dataset)
    return acc_test


# save state of the training
def save_state(epoch, distrib_model, loss_acc, optimizer, res_name, grank, gwsize, is_best, my_trainer):
    rt = time.time()
    # find if is_best happened in any worker
    if torch.cuda.is_available():
        is_best_m = my_trainer.par_allgather_obj(is_best, gwsize)

    if torch.cuda.is_available():
        if any(is_best_m):
            # find which rank is_best happened - select first rank if multiple
            is_best_rank = np.where(np.array(is_best_m) == True)[0][0]

            # collect state
            state = {'epoch': epoch + 1,
                     'state_dict': distrib_model.state_dict(),
                     'best_acc': loss_acc,
                     'optimizer': optimizer.state_dict()}

            # write on worker with is_best
            if grank == is_best_rank:
                torch.save(state, './'+res_name)
                print(
                    f'DEBUG: state in {grank} is saved on epoch:{epoch} in {time.time()-rt} s')
    else:
        # collect state
        state = {'epoch': epoch + 1,
                 'state_dict': distrib_model.state_dict(),
                 'best_acc': loss_acc,
                 'optimizer': optimizer.state_dict()}

        torch.save(state, './'+res_name)
        print(
            f'DEBUG: state in {grank} is saved on epoch:{epoch} in {time.time()-rt} s')


# deterministic dataloader
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


#
#
# MAIN
#
#
def main():
    # get parse args
    pars_ini()

    # check CUDA availibility
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Strategy for distributed training
    if args.strategy == 'DDP':
        my_trainer = DDPDistributedTrainer()

    elif args.strategy == 'DS':
        my_trainer = DSDistributedTrainer()

    elif args.strategy == 'HVD':
        my_trainer = HVDDistributedTrainer()

    # limit # of CPU threads to be used per worker
    torch.set_num_threads(1)

    # get directory
    program_dir = os.getcwd()

    # start the time.time for profiling
    st = time.time()

    # initializes the distributed backend which will take care of sychronizing nodes/GPUs
    my_trainer.init_backend()

# deterministic testrun
    if args.testrun:
        torch.manual_seed(args.nseed)
        g = torch.Generator()
        g.manual_seed(args.nseed)

    # get job rank info - rank==0 master gpu
    if torch.cuda.is_available():
        lwsize = my_trainer.dist_lwsize() if args.cuda else 0  # local world size - per node
        gwsize = my_trainer.dist_gwsize()     # global world size - per run
        grank = my_trainer.dist_grank()            # global rank - assign per run
        lrank = my_trainer.dist_lrank()     # local rank - assign per node
    else:
        gwsize = 1
        grank = 0

    # some debug
    if grank == 0:
        print('TIMER: initialise:', time.time()-st, 's')
        print('DEBUG: local ranks:', lwsize, '/ global ranks:', gwsize)
        print('DEBUG: sys.version:', sys.version, '\n')

        print('DEBUG: IO parsers:')
        print('DEBUG: args.data_dir:', args.data_dir)
        print('DEBUG: args.restart_int:', args.restart_int, '\n')

        print('DEBUG: model parsers:')
        print('DEBUG: args.batch_size:', args.batch_size)
        print('DEBUG: args.epochs:', args.epochs)
        print('DEBUG: args.lr:', args.lr)
        print('DEBUG: args.concM:', args.concM)
        print('DEBUG: args.momentum:', args.momentum)
        print('DEBUG: args.shuff:', args.shuff, '\n')

        print('DEBUG: debug parsers:')
        print('DEBUG: args.testrun:', args.testrun)
        print('DEBUG: args.nseed:', args.nseed)
        print('DEBUG: args.log_int:', args.log_int, '\n')

        print('DEBUG: parallel parsers:')
        print('DEBUG: args.backend:', args.backend)
        print('DEBUG: args.nworker:', args.nworker)
        print('DEBUG: args.prefetch:', args.prefetch)
        print('DEBUG: args.cuda:', args.cuda, '\n')

    # encapsulate the model on the GPU assigned to the current process
    device = torch.device(
        'cuda' if args.cuda and torch.cuda.is_available() else 'cpu', lrank)
    if args.cuda:
        torch.cuda.set_device(lrank)
        # deterministic testrun
        if args.testrun:
            torch.cuda.manual_seed(args.nseed)

# read data
    data_dir = args.data_dir
    mnist_scale = args.concM
    largeData = []
    for i in range(mnist_scale):
        largeData.append(
            datasets.MNIST(data_dir, train=True, download=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))
        )

    # concat data
    train_dataset = torch.utils.data.ConcatDataset(largeData)

    mnist_scale = args.concM
    largeData = []
    for i in range(mnist_scale):
        largeData.append(
            datasets.MNIST(data_dir, train=False, download=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))
        )

    # concat data
    test_dataset = torch.utils.data.ConcatDataset(largeData)

    # restricts data loading to a subset of the dataset exclusive to the current process
    args.shuff = args.shuff and not args.testrun
    if torch.cuda.is_available():
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=gwsize, rank=grank, shuffle=args.shuff)
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset, num_replicas=gwsize, rank=grank, shuffle=args.shuff)

# distribute dataset to workers
    # persistent workers is not possible for nworker=0
    pers_w = True if args.nworker > 1 else False

    # deterministic testrun - the same dataset each run
    kwargs = {'worker_init_fn': seed_worker,
              'generator': g} if args.testrun else {}

    if torch.cuda.is_available():
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                   sampler=train_sampler, num_workers=args.nworker, pin_memory=True,
                                                   persistent_workers=pers_w, prefetch_factor=args.prefetch, **kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                  sampler=test_sampler, num_workers=args.nworker, pin_memory=True,
                                                  persistent_workers=pers_w, prefetch_factor=args.prefetch, **kwargs)
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size)

    if grank == 0:
        print('TIMER: read and concat data:', time.time()-st, 's')

    # create CNN model
    model = Net().to(device)

    # distribute model to workers
    distrib_model = my_trainer.distribute_model(model, device)

    # optimizer
    optimizer = torch.optim.SGD(
        distrib_model.parameters(), lr=args.lr, momentum=args.momentum)

    my_trainer.broadcast_params(distrib_model, optimizer)

    optimizer = my_trainer.distribute_optimizer(optimizer, distrib_model)

# resume state
    start_epoch = 1
    best_acc = np.Inf
    res_name = 'checkpoint.pth.tar'
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
                if grank == 0:
                    print(f'WARNING: restarting from {start_epoch} epoch')
            else:
                print(f'WARNING: restarting from {start_epoch} epoch')
        except:
            if torch.cuda.is_available():
                if grank == 0:
                    print(f'WARNING: restart file cannot be loaded, restarting!')
            else:
                print(f'WARNING: restart file cannot be loaded, restarting!')

    if start_epoch > args.epochs:
        if torch.cuda.is_available():
            if grank == 0:
                print(f'WARNING: given epochs are less than the one in the restart file!\n'
                      f'WARNING: SYS.EXIT is issued')

            my_trainer.clean_up()
            sys.exit()
        else:
            print(f'WARNING: given epochs are less than the one in the restart file!\n'
                  f'WARNING: SYS.EXIT is issued')
            sys.exit()

# start trainin/testing loop
    if grank == 0:
        print('TIMER: broadcast:', time.time()-st, 's')
        print(f'\nDEBUG: start training')
        print(f'--------------------------------------------------------')

    et = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        lt = time.time()
        # training
        loss_acc = train(distrib_model, device, train_loader,
                         optimizer, epoch, grank, gwsize, args)

        # testing
        acc_test = test(distrib_model, device,
                        test_loader, grank, gwsize, args)

        # save first epoch timer
        if epoch == start_epoch:
            first_ep_t = time.time()-lt

        # final epoch
        if epoch + 1 == args.epochs:
            train_loader.last_epoch = True
            test_loader.last_epoch = True

        if grank == 0:
            print('TIMER: epoch time:', time.time()-lt, 's')
            print('DEBUG: accuracy:', acc_test, '%')

        # save state if found a better state
        is_best = loss_acc < best_acc
        if epoch % args.restart_int == 0:
            save_state(epoch, distrib_model, loss_acc, optimizer,
                       res_name, grank, gwsize, is_best, my_trainer)
            # reset best_acc
            best_acc = min(loss_acc, best_acc)

# finalise
    # save final state
    save_state(epoch, distrib_model, loss_acc, optimizer,
               res_name, grank, gwsize, True, my_trainer)
    # if torch.cuda.is_available():
    #    dist.barrier()

    # some debug
    if grank == 0:
        print(f'\n--------------------------------------------------------')
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
        print('DEBUG: last accuracy:', acc_test, '%')
        print('DEBUG: memory req:', int(torch.cuda.memory_reserved(lrank)/1024/1024), 'MB') \
            if args.cuda else 'DEBUG: memory req: - MB'
        print('DEBUG: memory summary:\n\n',
              torch.cuda.memory_summary(0)) if args.cuda else ''

    if grank == 0:
        print(f'TIMER: final time: {time.time()-st} s\n')

    my_trainer.clean_up()


if __name__ == "__main__":
    main()
    sys.exit()

# eof
