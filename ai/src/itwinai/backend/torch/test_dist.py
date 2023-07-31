# std libs
import argparse
import sys
import os
import time
from typing import Optional
from itwinai.backend.torch.trainer import (
    TorchDistributedBackend,
    TorchDistributedStrategy
)


# ml libs
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from .exmptrainer import Net, train, test
from .utils import save_state, seed_worker
from .trainer import TorchTrainer


def pars_ini():
    global args
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

    # IO parsers
    parser.add_argument('--data-dir', default='./',
                        help='location of the training dataset in the local filesystem')

    # model parsers
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--concM', type=int, default=1,
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
    parser.add_argument('--benchrun', action='store_true', default=False,
                        help='do a bench run w/o IO (default: False)')

    args = parser.parse_args()

    # set minimum of 3 epochs when benchmarking (last epoch produces logs)
    args.epochs = 3 if args.epochs < 3 and args.benchrun else args.epochs


class MNISTTrainer(TorchTrainer):
    def __init__(
        self,
        model: Module,
        epochs: int,
        testrun: bool = False,
        shuffle_data: bool = False,
        seed: int = None,
        log_int: int = 10,
        strategy: TorchDistributedStrategy = None,
        backend: TorchDistributedBackend = 'nccl',
        use_cuda: bool = True,
        benchrun: bool = False
    ) -> None:
        super().__init__(model, epochs, testrun, shuffle_data,
                         seed, log_int, strategy, backend, use_cuda, benchrun)

        self.optim = torch.optim.SGD(
            self.model.parameters(),
            lr=args.lr,
            momentum=args.momentum
        )

    # def configure_optimizers(self) -> torch.optim.SGD:
    #     return torch.optim.SGD(
    #         self.model.parameters(),
    #         lr=args.lr,
    #         momentum=args.momentum
    #     )

    def training_step(self, batch, batch_idx):
        # optim = self.optimizers()
        pass

    def validation_step(self, batch, batch_idx):
        pass


if __name__ == '__main__':
    pass
