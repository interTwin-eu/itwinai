"""
Adapted from: https://github.com/pytorch/examples/blob/main/mnist/main.py
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
import torchvision
from dataset import imagenet_dataset
from net import Net
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import Subset

from itwinai.loggers import ConsoleLogger
from itwinai.torch.config import TrainingConfiguration
from itwinai.torch.trainer import TorchTrainer


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--data-dir",
        default="/p/scratch/intertwin/datasets/imagenet/ILSVRC2012/train/",
        help=("location of the training dataset in the local filesystem"),
    )
    parser.add_argument(
        "--epochs", type=int, default=2, help="number of epochs to train (default: 14)"
    )
    parser.add_argument(
        "--strategy", type=str, default="ddp", help="distributed strategy (default=ddp)"
    )
    args = parser.parse_args()
    train_dataset = imagenet_dataset(args.data_dir)

    # Only consider the first 1000 elements
    indices = list(range(4000))
    train_dataset = Subset(train_dataset, indices)

    model = torchvision.models.resnet152()

    training_config = TrainingConfiguration(
        batch_size=args.batch_size,
        optimizer="sgd",
        loss="nllloss",
        num_workers_dataloader=1,
    )
    logger = ConsoleLogger()

    profiler = profile(
        activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
        with_modules=True,
    )

    trainer = TorchTrainer(
        config=training_config,
        model=model,
        strategy=args.strategy,
        epochs=args.epochs,
        logger=logger,
    )

    strategy = trainer.strategy

    profiler.start()
    trainer.execute(train_dataset)
    profiler.stop()

    # key_averages = profiler.key_averages(group_by_stack_n=0)
    key_averages = profiler.key_averages()
    print(key_averages.table(sort_by="cuda_time_total"))

    profiling_data = []
    for event in key_averages:
        profiling_data.append(
            {
                "name": event.key,
                "node_id": event.node_id,
                "self_cpu_time_total": event.self_cpu_time_total,
                "cpu_time_total": event.cpu_time_total,
                "cpu_time_total_str": event.cpu_time_total_str,
                "self_cuda_time_total": event.self_cuda_time_total,
                "cuda_time_total": event.cuda_time_total,
                "cuda_time_total_str": event.cuda_time_total_str,
                "calls": event.count,
            }
        )
    grank = strategy.global_rank()
    num_gpus = strategy.global_world_size()
    df = pd.DataFrame(profiling_data)

    output_dir = Path("logs")
    output_path = output_dir / f"profile_{args.strategy}_{num_gpus}_{grank}.csv"

    print(f"Saving the dataframe to {output_path}", force=True)
    df.to_csv(output_path, index=False)
    strategy.clean_up()


if __name__ == "__main__":
    main()
