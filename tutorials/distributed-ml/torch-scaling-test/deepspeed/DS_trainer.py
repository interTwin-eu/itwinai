"""
Scaling test of Microsoft Deepspeed on Imagenet using Resnet.
"""
import argparse
import sys
import os
import time
import random
import numpy as np
import deepspeed

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms

from itwinai.parser import ArgumentParser as ItAIArgumentParser
from itwinai.loggers import EpochTimeTracker


def parsIni():
    parser = ItAIArgumentParser(
        description='PyTorch Imagenet scaling test')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--log-int', type=int, default=100, metavar='N',
                        help=(
                            'how many batches to wait before logging '
                            'training status'))
    parser.add_argument('--data-dir', default='./',
                        help=('location of the training dataset in the local '
                              'filesystem'))
    parser.add_argument('--backend', type=str, default='nccl', metavar='N',
                        help='backend for parrallelisation (default: nccl)')
    parser.add_argument('--restart-int', type=int, default=10, metavar='N',
                        help='restart int per epoch (default: 10)')
    parser.add_argument('--testrun', action='store_true', default=False,
                        help='do a test run (default: False)')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument('--nworker', type=int, default=0,
                        help=('number of workers in DataLoader '
                              '(default: 0 - only main)'))
    parser.add_argument('--verbose',
                        action=argparse.BooleanOptionalAction,
                        help='Print parsed arguments')
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
        t = time.perf_counter()
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
                f'({100.0 * batch_idx *len(data) / len(train_loader):.0f}%)]'
                '\t\tLoss: {loss.item():.6f}')
        t_list.append(time.perf_counter() - t)
        loss_acc += loss.item()
    if grank == 0:
        print('TIMER: train time', sum(t_list) / len(t_list), 's')
    return loss_acc


def test(model, test_loader, grank, gwsize):
    device = model.local_rank
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
    args = parsIni()

    # limit # of CPU threads to be used per worker
    torch.set_num_threads(1)

    # get directory
    program_dir = os.getcwd()

    # start the time.time for profiling
    st = time.time()

    # initializes the distributed backend
    deepspeed.init_distributed(dist_backend=args.backend)

    # get job rank info - rank==0 master gpu
    gwsize = dist.get_world_size()     # global world size - per run
    lwsize = torch.cuda.device_count()  # local world size - per node
    grank = dist.get_rank()            # global rank - assign per run
    lrank = dist.get_rank() % lwsize     # local rank - assign per node

    # some debug
    if grank == 0:
        print('TIMER: initialise:', time.time()-st, 's')
        print('DEBUG: local ranks:', lwsize, '/ global ranks:', gwsize)
        print('DEBUG: sys.version:', sys.version)
        print('DEBUG: args.data_dir:', args.data_dir)
        print('DEBUG: args.batch_size:', args.batch_size)
        print('DEBUG: args.epochs:', args.epochs)
        print('DEBUG: args.lr:', args.lr)
        print('DEBUG: args.backend:', args.backend)
        print('DEBUG: args.log_int:', args.log_int)
        print('DEBUG: args.restart_int:', args.restart_int)
        print('DEBUG: args.testrun:', args.testrun, '\n')

    # encapsulate the model on the GPU assigned to the current process
    torch.cuda.set_device(lrank)

    # read training dataset
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

    # # distribute test dataset
    # test_sampler = torch.utils.data.distributed.DistributedSampler(
    #     test_dataset, num_replicas=gwsize, rank=grank)
    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset, batch_size=args.batch_size,
    #     sampler=test_sampler, num_workers=0, pin_memory=True, shuffle=False)

    if grank == 0:
        print('TIMER: read and concat data:', time.time()-st, 's')

    # create CNN model
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

    # optimizer
    # optimizer = torch.optim.Adam(distrib_model.parameters(), lr=args.lr)
    # optimizer = torch.optim.SGD(
    #     distrib_model.parameters(), lr=args.lr, momentum=0.5)

    # resume state
    start_epoch = 1
    best_acc = np.Inf
    res_name = 'ds-checkpoint.pth.tar'
    if os.path.isfile(res_name):
        try:
            dist.barrier()
            # Map model to be loaded to specified single gpu.
            loc = {'cuda:%d' % 0: 'cuda:%d' % lrank}
            checkpoint = torch.load(program_dir+'/'+res_name, map_location=loc)
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            distrib_model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if grank == 0:
                print(f'WARNING: restarting from {start_epoch} epoch')
        except Exception:
            if grank == 0:
                print('WARNING: restart file cannot be loaded, restarting!')

    if start_epoch >= args.epochs+1:
        if grank == 0:
            print('WARNING: given epochs are less than the '
                  'one in the restart file!\n'
                  'WARNING: SYS.EXIT is issued')
        deepspeed.sys.exit()
        sys.exit()

    # start trainin/testing loop
    if grank == 0:
        print('TIMER: broadcast:', time.time()-st, 's')
        print('\nDEBUG: start training')
        print('--------------------------------------------------------')
        epoch_time_tracker = EpochTimeTracker(series_name="deepspeed-bl")

    et = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        lt = time.time()
        # training
        loss_acc = train(args, distrib_model, train_loader,
                         optimizer, epoch, grank, gwsize)

        # testing
        # acc_test = test(distrib_model, test_loader, grank, gwsize)

        # save state if found a better state
        is_best = loss_acc < best_acc
        if epoch % args.restart_int == 0:
            save_state(epoch, distrib_model, loss_acc, optimizer,
                       res_name, grank, gwsize, is_best)
            # reset best_acc
            best_acc = min(loss_acc, best_acc)

        if grank == 0:
            print('TIMER: epoch time:', time.time()-lt, 's')
            epoch_time_tracker.add_epoch_time(epoch-1, time.time()-lt)
            # print('DEBUG: accuracy:', acc_test, '%')

    # finalise
    # save final state
    save_state(epoch, distrib_model, loss_acc,
               optimizer, res_name, grank, gwsize, True)
    dist.barrier()

    # some debug
    if grank == 0:
        print('\n--------------------------------------------------------')
        print('DEBUG: results:\n')
        print('TIMER: last epoch time:', time.time()-lt, 's')
        print('TIMER: total epoch time:', time.time()-et, 's')
        # print('DEBUG: last accuracy:', acc_test, '%')
        print('DEBUG: memory req:', int(
            torch.cuda.memory_reserved(lrank)/1024/1024), 'MB')

    if grank == 0:
        print(f'TIMER: final time: {time.time()-st} s\n')
        nnod = os.environ.get('SLURM_NNODES', 'unk')
        epoch_time_tracker.save(
            csv_file=f"epochtime_deepspeed-bl_{nnod}N.csv")

    print(f"<Global rank: {grank}> - TRAINING FINISHED")

    # clean-up
    deepspeed.sys.exit()


if __name__ == "__main__":
    main()
    sys.exit()
