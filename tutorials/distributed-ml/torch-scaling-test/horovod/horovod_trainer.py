"""
Scaling test of Horovod on Imagenet using Resnet.
"""
import argparse
import os
import sys
from timeit import default_timer as timer

import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
import horovod.torch as hvd
import torchvision
from torchvision import datasets, transforms

from itwinai.parser import ArgumentParser as ItAIArgumentParser
from itwinai.loggers import EpochTimeTracker


def parsIni():
    parser = ItAIArgumentParser(description='PyTorch Imagenet Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='#batches to wait before logging training status')
    parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                        help='use fp16 compression during allreduce')
    parser.add_argument('--use-adasum', action='store_true', default=False,
                        help='use adasum algorithm to do reduction')
    parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
                        help=('apply gradient predivide factor in optimizer '
                              '(default: 1.0)'))
    parser.add_argument('--data-dir', default='./',
                        help=('location of the training dataset in the '
                              'local filesystem'))
    parser.add_argument('--verbose',
                        action=argparse.BooleanOptionalAction,
                        help='Print parsed arguments')
    parser.add_argument('--nworker', type=int, default=0,
                        help=('number of workers in DataLoader '
                              '(default: 0 - only main)'))

    args = parser.parse_args()
    if args.verbose:
        args_list = [f"{key}: {val}" for key, val in args.items()]
        print("PARSED ARGS:\n", '\n'.join(args_list))

    return args


def train(epoch):
    model.train()
    # Horovod: set epoch to sampler for shuffling
    train_sampler.set_epoch(epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            # Horovod: use train_sampler to determine the number of examples in
            # this worker's partition
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_sampler),
                100. * batch_idx / len(train_loader), loss.item()))


def metric_average(val, namegiv):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=namegiv)
    return avg_tensor.item()


# def test():
#     model.eval()
#     test_loss = 0.
#     test_accuracy = 0.
#     for data, target in test_loader:
#         if args.cuda:
#             data, target = data.cuda(), target.cuda()
#         output = model(data)
#         # sum up batch loss
#         test_loss += F.nll_loss(output, target, size_average=False).item()
#         # get the index of the max log-probability
#         pred = output.data.max(1, keepdim=True)[1]
#         test_accuracy += \
#           pred.eq(target.data.view_as(pred)).cpu().float().sum()

#     # Horovod: use test_sampler to determine the number of examples in
#     # this worker's partition
#     test_loss /= len(test_sampler)
#     test_accuracy /= len(test_sampler)

#     # Horovod: average metric values across workers
#     test_loss = metric_average(test_loss, 'avg_loss')
#     test_accuracy = metric_average(test_accuracy, 'avg_accuracy')

#     # Horovod: print output only on first rank
#     if hvd.rank() == 0:
#         print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
#             test_loss, 100. * test_accuracy))


if __name__ == '__main__':
    # get parse args
    args = parsIni()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Horovod: init
    st = timer()
    hvd.init()
    torch.manual_seed(args.seed)

    # some debug
    if hvd.rank() == 0 and hvd.local_rank() == 0:
        print('DEBUG: sys.version:', sys.version)
        print('DEBUG: torch.cuda.is_available():', torch.cuda.is_available())
        print('DEBUG: torch.cuda.current_device():',
              torch.cuda.current_device())
        print('DEBUG: torch.cuda.device_count():', torch.cuda.device_count())
        print('DEBUG: torch.cuda.get_device_properties(hvd.local_rank()):',
              torch.cuda.get_device_properties(hvd.local_rank()))
        print('DEBUG: args.data_dir:', args.data_dir)
        print('DEBUG: args.batch_size:', args.batch_size)
        print('DEBUG: args.epochs:', args.epochs)

    if hvd.rank() == 0 and hvd.local_rank() == 0:
        print('TIMER: initialise:', timer()-st, 's')

    if args.cuda:
        # Horovod: pin GPU to local rank
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)

    # Horovod: limit # of CPU threads to be used per worker
    torch.set_num_threads(1)

    # kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    kwargs = {'num_workers': args.nworker,
              'pin_memory': True} if args.cuda else {}
    # When supported, use 'forkserver' to spawn dataloader workers instead...
    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context')
        and
            mp._supports_context and
            'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'

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

    # Horovod: use DistributedSampler to partition the training data
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=train_sampler, **kwargs)

    # create CNN model
    model = torchvision.models.resnet152()

    # by default, Adasum doesn't need scaling up learning rate
    lr_scaler = hvd.size() if not args.use_adasum else 1

    if args.cuda:
        # move model to GPU.
        model.cuda()
        # if using GPU Adasum allreduce, scale learning rate by local_size
        if args.use_adasum and hvd.nccl_built():
            lr_scaler = hvd.local_size()

    # Horovod: scale learning rate by lr_scaler
    optimizer = optim.SGD(model.parameters(), lr=args.lr * lr_scaler,
                          momentum=args.momentum)

    # Horovod: broadcast parameters & optimizer state
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # Horovod: (optional) compression algorithm
    compression = (
        hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
    )

    # Horovod: wrap optimizer with DistributedOptimizer
    optimizer = hvd.DistributedOptimizer(
        optimizer,
        named_parameters=model.named_parameters(),
        compression=compression,
        op=hvd.Adasum if args.use_adasum else hvd.Average,
        gradient_predivide_factor=args.gradient_predivide_factor)

    if hvd.rank() == 0 and hvd.local_rank() == 0:
        print('TIMER: broadcast:', timer()-st, 's')
        epoch_time_tracker = EpochTimeTracker(series_name="horovod-bl")

    et = timer()
    for epoch in range(1, args.epochs + 1):
        lt = timer()
        train(epoch)
        # test()
        print('TIMER: hvd.rank():', hvd.rank(),
              'hvd.local_rank():', hvd.local_rank(),
              ', epoch time:', timer()-lt, 's')

        if hvd.rank() == 0 and hvd.local_rank() == 0:
            epoch_time_tracker.add_epoch_time(epoch-1, timer()-lt)

    print('TIMER: last epoch time:', timer()-lt, 's')
    print('TIMER: total epoch time:', timer()-et, 's')

    if hvd.rank() == 0 and hvd.local_rank() == 0:
        print('\n', torch.cuda.memory_summary(0), '\n')
        nnod = os.environ.get('SLURM_NNODES', 'unk')
        epoch_time_tracker.save(
            csv_file=f"epochtime_horovod-bl_{nnod}N.csv")

    print('DEBUG: hvd.rank():', hvd.rank(),
          'hvd.local_rank():', hvd.local_rank(),
          ', torch.cuda.memory_reserved():',
          int(torch.cuda.memory_reserved(hvd.local_rank())/1024/1024), 'MB')

    if hvd.rank() == 0 and hvd.local_rank() == 0:
        print('DEBUG: memory req:',
              int(torch.cuda.memory_reserved(hvd.local_rank())/1024/1024),
              'MB')

    print(f"<Hvd rank: {hvd.rank()}> - TRAINING FINISHED")
