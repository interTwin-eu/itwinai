#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""! @brief HPO """ 

##
# @mainpage HPO
#
# @section description_main Description
# Hyperparameter optimization of neural networks with Ray Tune library.
#
#
#
# @section notes_main Notes
# - The data directory of the CIFAR-10 dataset has the be specified in the startscript
#
# Copyright (c) 2023 RAISE, All rights reserved.


##
# @file cifar_tune_asha.py
#
# @brief Optimizing the hyperparameters of a ResNet18 trained on the CIFAR-10 dataset with Ray Tune libray and the ASHA algorithm.
#
# @section description_cifar_tune_asha description
# A standard ResNet18 model is trained on the CIFAR-10 vision dataset. To optimize the performance, multiple 
# training runs (trials) with different hyperparameters (chagend learning rate and batch size) are performed using 
# the Ray Tune library. The overall hyperparameter optimization process, as well as the single training runs can be 
# parallelized across multiple GPUs. Trials with low performance (in terms of test set acuracy) are terminated early 
# with the ASHA aglorithm.
# 
#
# @section libraries_main Libraries/Modules
# - argparse standard library (https://docs.python.org/3/library/argparse.html)
#   - Parse command-line options 
# - sys standard library (https://docs.python.org/3/library/sys.html)
#   - System commands
# - os standard library (https://docs.python.org/3/library/os.html)
#   - OS commands 
# - time standard library (https://docs.python.org/3/library/time.html)
#   - Access timers for profilers 
# - numpy library (https://numpy.org/)
#   - Access numpy functions
# - random standard library (https://docs.python.org/3/library/time.html)
#   - Generate random numbers
# - matplotlib library (https://matplotlib.org/)
#   - Post-process data for validation 
# - torch library (https://pytorch.org/)
#   - ML framework
# - torchvision library (https://pypi.org/project/torchvision/)
#   - Torch library additions for popular datasets and their transformations
# - ray libray (https://www.ray.io/)
#   - Framework for distributed computing with a focus on hyperparameter optimization
#
# @section notes_doxygen_example Notes
# - None.
#
# @section todo TODO
# - None.
#
# @section author Author(s)
# - Created by MA on 04/05/2023.
# - Modified by 
#
# Copyright (c) 2023 RAISE, All rights reserved.

# load general modules
import argparse
import os
import time
import numpy as np

# load torch and torchvision modules 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torchvision
from torchvision import datasets, transforms, models

# load ray modules
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.air import session, RunConfig
import ray.train as train
from ray.train.torch import TorchTrainer
from ray.air.config import ScalingConfig
from ray.tune.tuner import Tuner, TuneConfig
from ray.train import Checkpoint

def parsIni():
    parser = argparse.ArgumentParser(description='Ray Tune Cifar-10 Example')
    parser.add_argument('--num-samples', type=int, default=24, metavar='N',
                    help='number of samples to train (default: 24)')
    parser.add_argument('--max-iterations', type=int, default=10, metavar='N',
                    help='maximum iterations to train (default: 10)')
    parser.add_argument('--ngpus', type=int, default=1, metavar='N',
                    help='parallel gpu workers to train on a single trial (default: 1)')
    parser.add_argument('--scheduler', type=str, default='RAND',
                    help='scheduler for tuning (default: RandomSearch)')
    parser.add_argument('--data-dir', type=str, default='',
                    help='data directory for cifar-10 dataset')
    parser.add_argument('--nworker', type=int, default=0,
                    help='number of workers in DataLoader (default: 0 - only main)')
    return parser

def accuracy(output, target):
    """! function that computes the accuracy of an output and target vector 
    @param output vector that the model predicted
    @param target actual  vector
    
    @return correct number of correct predictions
    @return total number of total elements
    """
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    
    # count correct classifications
    correct = pred.eq(target.view_as(pred)).cpu().float().sum()
    
    # count total samples
    total = target.size(0)
    return correct, total

def par_sum(field):
    """! function that sums a field across all workers to a worker
    @param field field in worker that should be summed up
    
    @return sum of all fields
    """
    # convert field to tensor
    res = torch.Tensor([field])
    
    # move field to GPU/worker
    res = res.cuda()
    
    # AllReduce operation
    dist.all_reduce(res,op=dist.ReduceOp.SUM,group=None,async_op=True).wait()
    
    return res

def load_data(data_dir=None):
    """! function that loads training and test set of cifar-10
    @param data_dir directory where the data is stored
    
    @return train_set training set of cifar-10
    @return test_set test set of cifar-10
    """
    # vision preprocessing values
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    
    # transformations for the training set 
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # transformations for the testset
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # load the cifar-10 dataset from directory
    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=False, transform=transform_train)

    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=False, transform=transform_test)

    return train_set, test_set
    
def train_cifar(config):
    """! function to train a ResNet on cifar-10 with different hyperparameters
    @param config hyperparameter search space
    """    

    # load a ResNet model
    model = models.resnet18()
    
    # prepare the model for Ray Tune
    model = train.torch.prepare_model(model)
        
    # define optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["lr"]*dist.get_world_size())

    current_epoch = 0
    
    # Load existing model and optimizer checkpoint through `get_checkpoint()` API.
    if train.get_checkpoint():
        loaded_checkpoint = train.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            # Load checkpoint
            checkpoint = torch.load(os.path.join(loaded_checkpoint_dir, "checkpoint.pt"))

            # Restore epoch value, model state, and optimizer state
            current_epoch = checkpoint['current_epoch']
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # load the training and test data
    train_set, test_set = load_data(str(config["data_dir"]))
    
    # define the train and test dataloader
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=4)
    
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=4)

    # prepare the dataloaders for Ray Tune
    train_loader = train.torch.prepare_data_loader(train_loader)
    test_loader = train.torch.prepare_data_loader(test_loader)
        
    # prepare metrics
    train_acc = 0
    train_correct = 0
    train_total = 0
    
    test_acc = 0
    test_correct = 0
    test_total = 0
    
    # training and testing loop
    for epoch in range(current_epoch, 100):
        # prepare model for training and loop over training dataset
        model.train()
        for i, (images, target) in enumerate(train_loader):

            # compute output
            optimizer.zero_grad()
            output = model(images)

            # compute loss
            loss = criterion(output, target)

            # count correct classifications
            tmp_correct, tmp_total = accuracy(output, target)    
            train_correct +=tmp_correct
            train_total +=tmp_total

            # backpropagation and optimization step
            loss.backward() 
            optimizer.step()

        # average the train metrics over all workers
        train_correct = par_sum(train_correct)
        train_total = par_sum(train_total)

        # compute final training accuracy
        train_acc = train_correct/train_total
        
        # only perform the testing loop every 10 epochs
        if ((epoch+1)%10 == 0):
            # prepare model for testing and loop over test dataset
            model.eval()
            with torch.no_grad():
                for i, (images, target) in enumerate(test_loader): 
                    
                    # compute output
                    output = model(images)
                    
                    # count correct classifications
                    tmp_correct, tmp_total = accuracy(output, target)
                    test_correct +=tmp_correct
                    test_total +=tmp_total    

                # average the test metrics over all workers
                test_correct = par_sum(test_correct)
                test_total = par_sum(test_total)
                
                # compute final test accuracy
                test_acc = test_correct/test_total
            
            # Save current state of model and optimizer
            os.makedirs("tune_model", exist_ok=True)
            torch.save({
                'current_epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, "tune_model/checkpoint.pt")

            checkpoint = Checkpoint.from_directory("tune_model")

            # report the training and testing accuracy back to the head node of Ray Tune
            session.report({"train_acc": train_acc.item(), "test_acc": test_acc.item()}, checkpoint = checkpoint)

def main(args):
    """! main function
    @param args input arguments
    """    
    
    # initalize Ray with the correct adress and node ip adress
    ray.init(address=os.environ['ip_head'], _node_ip_address=os.environ["head_node_ip"])  
    
    # define the hyperparameter search space 
    config = {
        "batch_size": tune.choice([64, 128, 256, 512]),
        "lr": tune.loguniform(10e-5, 1),
        "data_dir": tune.choice([args.data_dir]),
    }
    
    # select a hyperparameter optimization algorithm
    if (args.scheduler == "ASHA"):
        # Asynchronous Successive Halving Algorithm
        scheduler = ASHAScheduler(
               # the number of iterations to allow the trials to run at max 
               max_t=args.max_iterations,
               # how many iterations before a bad trials get terminated 
               grace_period=2,
               # which percentage of trials to terminate
               reduction_factor=3)
        
        # set search algorithm
        search_alg = None
        
    if (args.scheduler == "RAND"):
        # random scheduler
        scheduler = None
        search_alg = None
        
    # define a reporter/logger to specifify which metrics to print out during the optimization process    
    reporter = CLIReporter(
        metric_columns=["train_acc", "test_acc", "training_iteration", "time_this_iter_s", "time_total_s"],
        max_report_frequency=60)
    
    # define the general RunConfig of Ray Tune
    run_config = RunConfig(
        # name of the training run (directory name).
        name="cifar_test_training",
        # directory to store the ray tune results in .
        storage_path=os.path.join(os.path.abspath(os.getcwd()), "ray_results"),
        # logger
        progress_reporter=reporter,
        # stopping criterion when to end the optimization process
        stop={"training_iteration": args.max_iterations}
    )
    
    # wrapping the torch training function inside a TorchTrainer logic
    trainer = TorchTrainer(
        # torch training function
        train_loop_per_worker=train_cifar,
        # default hyperparameters for the function
        train_loop_config={"batch_size": 64, "lr": 0.1, "data_dir": "/"},
        # setting the default resources/workers to use for the training function, including the number of CPUs and GPUs
        scaling_config=ScalingConfig(num_workers=args.ngpus, use_gpu=True, resources_per_worker={"CPU": 4, "GPU": 1}),
    )
    
    # defining the hyperparameter tuner 
    tuner = Tuner(
        # function to tune
        trainer,
        # hyperparameter search space
        param_space={"train_loop_config": config},
        # the tuning configuration
        tune_config=TuneConfig(
           # define how many trials to evaluate 
           num_samples=args.num_samples,
           # define which metric to use for measuring the performance of the trials
           metric="test_acc",
           # if the metric should be maximized or minimized 
           mode="max",
           # define which scheduler to use 
           scheduler=scheduler,
            # define which search algorithm to use
           search_alg=search_alg,
           ),
        run_config=run_config
    )
    
    # measure the total runtime
    start_time = time.time()
    
    # start the optimization process
    result = tuner.fit()
    
    runtime = time.time() - start_time
    
    # print total runtime
    print("Total runtime: ", runtime)

    # print metrics of the best trial
    best_result = result.get_best_result(metric="test_acc", mode="max")    
    
    print("Best result metrics: ", best_result) 

    # print results dataframe
    print("Result dataframe: ")
    print(result.get_dataframe().sort_values("test_acc", ascending=False))

if __name__ == "__main__":
    # get custom arguments from parser
    parser = parsIni()
    args = parser.parse_args()
    
    # call the main function to launch Ray
    main(args)

# eof
