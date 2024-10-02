"""
Adapted from: https://github.com/pytorch/examples/blob/main/mnist/main.py
"""

import argparse
from pathlib import Path
import torch
import pandas as pd

from torchvision import datasets, transforms

from itwinai.loggers import ConsoleLogger
from itwinai.torch.config import TrainingConfiguration
from itwinai.torch.trainer import TorchTrainer

from net import Net
from torch.profiler import ProfilerActivity, profile



def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs", type=int, default=2, help="number of epochs to train (default: 14)"
    )
    parser.add_argument(
        "--strategy", type=str, default="ddp", help="distributed strategy (default=ddp)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="learning rate (default: 1.0)"
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument(
        "--ckpt-interval",
        type=int,
        default=10,
        help="how many batches to wait before logging training status",
    )
    args = parser.parse_args()

    # Dataset creation
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    root_dir = Path("data")
    train_dataset = datasets.MNIST(
        str(root_dir), train=True, download=True, transform=transform
    )
    validation_dataset = datasets.MNIST(str(root_dir), train=False, transform=transform)

    # Neural network to train
    model = Net()

    training_config = TrainingConfiguration(
        batch_size=args.batch_size,
        optim_lr=args.lr,
        optimizer="adadelta",
        loss="cross_entropy",
        num_workers_dataloader=1
    )
    logger = ConsoleLogger()

    profiler = profile(
        activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], 
        # on_trace_ready=torch.profiler.tensorboard_trace_handler("./logs"),
        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
        record_shapes=True,
        # profile_memory=True,
    )

    trainer = TorchTrainer(
        config=training_config,
        model=model,
        strategy=args.strategy,
        epochs=args.epochs,
        random_seed=args.seed,
        checkpoint_every=args.ckpt_interval,
        logger=logger
    )

    profiler.start()
    trainer.execute(train_dataset, validation_dataset, profiler=profiler)
    profiler.stop()

    print(f"Running this!")

    profiling_data = []
    for event in profiler.key_averages(): 
        profiling_data.append({
        "name": event.key,
        "cpu_time": event.cpu_time_total,
        "self_cpu_time": event.self_cpu_time_total,
        "gpu_time": event.cuda_time_total,
        "self_gpu_time": event.self_cuda_time_total,
        "calls": event.count,
    })

    df = pd.DataFrame(profiling_data)
    df.to_csv("logs/dataframe.csv")


if __name__ == "__main__":
    main()
