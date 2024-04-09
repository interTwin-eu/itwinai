"""
Scaling test of torch Distributed Data Parallel on Imagenet using Resnet.
"""
import argparse
import sys
import os
import time
import random
import numpy as np
import logging

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms

import argparse

#from itwinai.parser import ArgumentParser as ItAIArgumentParser
#from itwinai.loggers import EpochTimeTracker


def pars_ini():
    parser = argparse.ArgumentParser(description='itwinai - parsed arguments')

    # IO parsers
    parser.add_argument('--data-dir', default='./',
                        help=('location of the training dataset in the '
                              'local filesystem'))
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

    # debug parsers
    parser.add_argument('--testrun', action='store_true', default=False,
                        help='do a test run with seed (default: False)')
    parser.add_argument('--nseed', type=int, default=0,
                        help='seed integer for reproducibility (default: 0)')
    parser.add_argument('--log-int', type=int, default=10,
                        help='log interval per training')
    parser.add_argument('--benchrun',
                        action='store_true', default=True)

    # parallel parsers
    parser.add_argument('--backend', type=str, default='nccl',
                        help='backend for parrallelisation (default: nccl)')
    parser.add_argument('--nworker', type=int, default=0,
                        help=('number of workers in DataLoader '
                              '(default: 0 - only main)'))
    parser.add_argument('--prefetch', type=int, default=2,
                        help='prefetch data in DataLoader (default: 2)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables GPGPUs')

    args = parser.parse_args()

    if args.verbose:
        args_list = [f"{key}: {val}" for key, val in args.items()]
        print("PARSED ARGS:\n", '\n'.join(args_list))
    return args


def train(model, device, train_loader, optimizer, epoch, grank, gwsize, args):
    model.train()
    lt_1 = time.perf_counter()
    loss_acc = 0
    if grank == 0:
        print("\n")
    for batch_idx, (data, target) in enumerate(train_loader):
        # if grank == 0:
        #     print(f"BS == DATA: {data.shape}, TARGET: {target.shape}")
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_int == 0 and grank == 0:
            print(
                f'Train epoch: {epoch} [{batch_idx * len(data)}/'
                f'{len(train_loader.dataset)/gwsize} '
                f'({100.0 * batch_idx / len(train_loader):.0f}%)]\t\tLoss: '
                f'{loss.item():.6f}')
        
        loss_acc += loss.item()
    if grank == 0:
        logging.info('epoch time: {:.2f}'.format(time.perf_counter()-lt_1)+' s')
    return loss_acc


def test(model, device, test_loader, grank, gwsize):
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


def save_state(
        epoch, distrib_model, loss_acc,
        optimizer, res_name, grank, gwsize, is_best
):
    """Save training state."""
    rt = time.time()
    # find if is_best happened in any worker
    if torch.cuda.is_available():
        is_best_m = par_allgather_obj(is_best, gwsize)

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
            f'DEBUG: state in {grank} is saved on epoch:{epoch} '
            f'in {time.time()-rt} s')


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def par_allgather_obj(obj, gwsize):
    """Gathers any object from the whole group in a list (to all workers)"""
    res = [None]*gwsize
    dist.all_gather_object(res, obj, group=None)
    return res


def main():
    # get parse args
    args = pars_ini()

    # check CUDA availibility
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # get directory
    program_dir = os.getcwd()

    # start the time.time for profiling
    st = time.time()

    # initializes the distributed backend which will take care of synchronizing
    # nodes/GPUs
    if torch.cuda.is_available():
        dist.init_process_group(backend=args.backend)

    # deterministic testrun
    if args.testrun:
        torch.manual_seed(args.nseed)
        g = torch.Generator()
        g.manual_seed(args.nseed)

    # get job rank info - rank==0 master gpu
    if torch.cuda.is_available():
        # local world size - per node
        lwsize = torch.cuda.device_count() if args.cuda else 0
        gwsize = dist.get_world_size()     # global world size - per run
        grank = dist.get_rank()            # global rank - assign per run
        lrank = dist.get_rank() % lwsize     # local rank - assign per node
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
        print('DEBUG: args.cuda:', args.cuda)
        print('DEBUG: args.benchrun:', args.benchrun, '\n')

    # encapsulate the model on the GPU assigned to the current process
    device = torch.device(
        'cuda' if args.cuda and torch.cuda.is_available() else 'cpu', lrank)
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
    train_dataset = datasets.ImageFolder(
        root=args.data_dir,
        transform=transform
    )
    # test_dataset = ...

    # restricts data loading to a subset of the dataset exclusive to the
    # current process
    args.shuff = args.shuff and not args.testrun
    if torch.cuda.is_available():
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=gwsize, rank=grank, shuffle=args.shuff)
        # test_sampler = torch.utils.data.distributed.DistributedSampler(
        #     test_dataset, num_replicas=gwsize, rank=grank,
        # shuffle=args.shuff)

    # distribute dataset to workers
    # persistent workers is not possible for nworker=0
    pers_w = True if args.nworker > 1 else False

    # deterministic testrun - the same dataset each run
    kwargs = {'worker_init_fn': seed_worker,
              'generator': g} if args.testrun else {}

    if torch.cuda.is_available():
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size,
            sampler=train_sampler, num_workers=args.nworker, pin_memory=True,
            persistent_workers=pers_w, prefetch_factor=args.prefetch, **kwargs)
        # test_loader = torch.utils.data.DataLoader(
        #     test_dataset, batch_size=args.batch_size,
        #     sampler=test_sampler, num_workers=args.nworker, pin_memory=True,
        #     persistent_workers=pers_w, prefetch_factor=args.prefetch,
        # **kwargs)
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size)
        # test_loader = torch.utils.data.DataLoader(
        #     test_dataset, batch_size=args.batch_size)

    if grank == 0:
        print('TIMER: read and concat data:', time.time()-st, 's')

    # create CNN model
    model = torchvision.models.resnet152().to(device)

    # distribute model to workers
    if torch.cuda.is_available():
        distrib_model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device],
            output_device=device)
    else:
        distrib_model = model

    # optimizer
    # optimizer = torch.optim.Adam(distrib_model.parameters(), lr=args.lr)
    optimizer = torch.optim.SGD(
        distrib_model.parameters(), lr=args.lr, momentum=args.momentum)

    # resume state
    start_epoch = 1
    best_acc = np.Inf
    nnod = os.environ.get('SLURM_NNODES', 'unk')
    res_name = f'ddp-{nnod}N-checkpoint.pth.tar'
    if os.path.isfile(res_name) and not args.benchrun:
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
        except Exception:
            if torch.cuda.is_available():
                if grank == 0:
                    print('WARNING: restart file cannot '
                          'be loaded, restarting!')
            else:
                print('WARNING: restart file cannot be loaded, restarting!')

    if start_epoch >= args.epochs:
        if torch.cuda.is_available():
            if grank == 0:
                print('WARNING: given epochs are less than the one in the'
                      ' restart file!\n'
                      'WARNING: SYS.EXIT is issued')
            dist.barrier()
            dist.destroy_process_group()
            sys.exit()
        else:
            print('WARNING: given epochs are less than the one in the '
                  'restart file!\n'
                  'WARNING: SYS.EXIT is issued')
            sys.exit()

    # start trainin/testing loop
    if grank == 0:
        print('TIMER: broadcast:', time.time()-st, 's')
        print('\nDEBUG: start training')
        print('--------------------------------------------------------')
        #epoch_time_tracker = EpochTimeTracker(series_name="ddp-bl")

    et = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        lt = time.time()
        # training
        if args.benchrun and epoch == args.epochs:
            # profiling (done on last epoch - slower!)
            with torch.autograd.profiler.profile(use_cuda=args.cuda,
                                                 profile_memory=True) as prof:
                loss_acc = train(distrib_model, device, train_loader,
                                 optimizer, epoch, grank, gwsize, args)
        else:
            loss_acc = train(distrib_model, device, train_loader,
                             optimizer, epoch, grank, gwsize, args)

        # # testing
        # acc_test = test(distrib_model, device,
        #                 test_loader, grank, gwsize, args)

        # save first epoch timer
        if epoch == start_epoch:
            first_ep_t = time.time()-lt

        # final epoch
        if epoch + 1 == args.epochs:
            train_loader.last_epoch = True
            # test_loader.last_epoch = True

        if grank == 0:
            print('TIMER: epoch time:', time.time()-lt, 's')
            #epoch_time_tracker.add_epoch_time(epoch-1, time.time()-lt)
            # print('DEBUG: accuracy:', acc_test, '%')
            if args.benchrun and epoch == args.epochs:
                print('\n----------------------------------------------------')
                print('DEBUG: benchmark of last epoch:\n')
                what1 = 'cuda' if args.cuda else 'cpu'
                print(prof.key_averages().table(
                    sort_by='self_'+str(what1)+'_time_total'))

        # save state if found a better state
        is_best = loss_acc < best_acc
        if epoch % args.restart_int == 0 and not args.benchrun:
            save_state(epoch, distrib_model, loss_acc, optimizer,
                       res_name, grank, gwsize, is_best)
            # reset best_acc
            best_acc = min(loss_acc, best_acc)

    # finalise
    # save final state
    if not args.benchrun:
        save_state(epoch, distrib_model, loss_acc,
                   optimizer, res_name, grank, gwsize, True)
    if torch.cuda.is_available():
        dist.barrier()

    # some debug
    if grank==0:
        print(f'\n--------------------------------------------------------')
        logging.info('training results:')
        logging.info('first epoch time: {:.2f}'.format(first_ep_t)+' s')
        logging.info('last epoch time: {:.2f}'.format(time.time()-lt)+' s')
        logging.info('total epoch time: {:.2f}'.format(time.time()-et)+' s')
        logging.info('average epoch time: {:.2f}'.format((time.time()-et)/done_epochs)+' s')
        if epoch>1:
            logging.info('total epoch-1 time: {:.2f}'.format(time.time()-et-first_ep_t)+' s')
            logging.info('average epoch-1 time: {:.2f}'.format((time.time()-et-first_ep_t)/(args.epochs-1))+' s')
        if args.benchrun:
            tot_ep_tm2 = tot_ep_t - first_ep_t - last_ep_t
            logging.info('total epoch-2 time: {:.2f}'.format(lt-first_ep_t)+' s')
            logging.info('average epoch-2 time: {:.2f}'.format((lt-first_ep_t)/(args.epochs-2))+' s')
        # memory on worker 0
        if args.cuda:
            logging.info('memory req: '+str(int(torch.cuda.max_memory_reserved(0)/1024/1024))+' MB')
            logging.info('memory summary:\n'+str(torch.cuda.memory_summary(0)))

    # timer for current epoch
    if grank==0:
        logging.info('epoch time: {:.2f}'.format(time.perf_counter()-lt_1)+' s')

    if grank == 0:
        print(f'TIMER: final time: {time.time()-st} s\n')
        nnod = os.environ.get('SLURM_NNODES', 'unk')
        #epoch_time_tracker.save(
        #    csv_file=f"epochtime_ddp-bl_{nnod}N.csv")

    print(f"<Global rank: {grank}> - TRAINING FINISHED")

    # clean-up
    if torch.cuda.is_available():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
    sys.exit()
