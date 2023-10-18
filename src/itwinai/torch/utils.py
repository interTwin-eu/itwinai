from typing import Hashable, Dict
import time
import numpy as np
import random

import torch
import torch.distributed as dist


def save_state(
    epoch, distrib_model, loss_val, optimizer, res_name, grank, gwsize,
    is_best, distributed: bool = True
):
    """Save training state"""
    rt = time.time()
    # find if is_best happened in any worker
    if torch.cuda.is_available() and distributed:
        is_best_m = par_allgather_obj(is_best, gwsize)

    if torch.cuda.is_available() and distributed:
        if any(is_best_m):
            # find which rank is_best happened - select first rank if multiple
            is_best_rank = np.where(np.array(is_best_m))[0][0]

            # collect state
            state = {'epoch': epoch + 1,
                     'state_dict': distrib_model.state_dict(),
                     'best_loss': loss_val,
                     'optimizer': optimizer.state_dict()}

            # write on worker with is_best
            if grank == is_best_rank:
                torch.save(state, './'+res_name)
                print(f'DEBUG: state in {grank} is saved on '
                      f'epoch:{epoch} in {time.time()-rt} s')
    else:
        # collect state
        state = {'epoch': epoch + 1,
                 'state_dict': distrib_model.state_dict(),
                 'best_loss': loss_val,
                 'optimizer': optimizer.state_dict()}

        torch.save(state, './'+res_name)
        print(
            f'DEBUG: state in {grank} is saved on epoch:{epoch} '
            f'in {time.time()-rt} s')


def seed_worker(worker_id):
    """deterministic dataloader"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def par_allgather_obj(obj, gwsize):
    """gathers any object from the whole group in a list (to all workers)"""
    res = [None]*gwsize
    dist.all_gather_object(res, obj, group=None)
    # print(f'ALLGATHER: {res}')
    return res


def clear_key(
        my_dict: Dict,
        dict_name: str,
        key: Hashable,
        complain: bool = True
) -> Dict:
    """Remove key from dictionary if present and complain.

    Args:
        my_dict (Dict): Dictionary.
        dict_name (str): name of the dictionary.
        key (Hashable): Key to remove.
    """
    if key in my_dict:
        if complain:
            print(
                f"Field '{key}' should not be present "
                f"in dictionary '{dict_name}'"
            )
        del my_dict[key]
    return my_dict
