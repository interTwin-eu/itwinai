#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author: RS, adapted from https://gitlab.jsc.fz-juelich.de/CoE-RAISE/FZJ/ai4hpc
# version: 211029a

# std libs
import argparse
import sys
import os
import time
import numpy as np
import random

# ml libs
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# network


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
