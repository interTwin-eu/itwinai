# std libs
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


def save_state(epoch, distrib_model, loss_acc, optimizer, res_name, grank, gwsize, is_best):
    """save state of the training"""
    rt = time.time()
    # find if is_best happened in any worker
    if torch.cuda.is_available():
        is_best_m = par_allgather_obj(is_best, gwsize)

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


def seed_worker(worker_id):
    """deterministic dataloader"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def par_allgather_obj(obj, gwsize):
    """gathers any object from the whole group in a list (to all workers)"""
    res = [None]*gwsize
    dist.all_gather_object(res, obj, group=None)
    return res
