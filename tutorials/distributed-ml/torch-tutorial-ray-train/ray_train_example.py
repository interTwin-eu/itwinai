import os
import tempfile
import sys
import argparse
import pathlib

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor, Normalize, Compose

import ray.train.torch
import ray.tune as tune


def train_func(config):
    # Model, Loss, Optimizer
    model = resnet18(num_classes=10)
    model.conv1 = torch.nn.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )
    # [1] Prepare model.
    model = ray.train.torch.prepare_model(model)
    # model.to("cuda")  # This is done by `prepare_model`
    criterion = CrossEntropyLoss()

    # HPs
    lr = config["learning_rate"]
    batch_size = config["batch_size"]

    optimizer = Adam(model.parameters(), lr=lr)

    # Data
    try:
        transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
        data_dir = config["data_path"]
        print(data_dir)
        train_data = FashionMNIST(root=data_dir, train=True,
                                  download=False, transform=transform)
        print("Successfully loaded data!")
    except Exception as e:
        print(e)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    # [2] Prepare dataloader.
    train_loader = ray.train.torch.prepare_data_loader(train_loader)

    # Training
    for epoch in range(3):
        if ray.train.get_context().get_world_size() > 1:
            train_loader.sampler.set_epoch(epoch)

        for images, labels in train_loader:
            # This is done by `prepare_data_loader`!
            # images, labels = images.to("cuda"), labels.to("cuda")
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # [3] Report metrics and checkpoint.
        metrics = {"loss": loss.item(), "epoch": epoch}
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            torch.save(
                model.module.state_dict(),
                os.path.join(temp_checkpoint_dir, "model.pt")
            )
            ray.train.report(
                metrics,
                checkpoint=ray.train.Checkpoint.from_directory(temp_checkpoint_dir),
            )
        if ray.train.get_context().get_world_rank() == 0:
            print(metrics)


if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Ray Train on MNIST Dataset')
    parser.add_argument(
        '--download_only', type=bool,
        default=False
    )
    parser.add_argument(
        '--ngpus',
        type=int
    )
    parser.add_argument(
        '--ncpus',
        type=int
    ),
    parser.add_argument(
        '--num_samples',
        type=int
    )
    args = parser.parse_args()

    data_dir = pathlib.Path("data")

    if args.download_only:
        # Data
        transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
        train_data = FashionMNIST(root=data_dir, train=True,
                                  download=True, transform=transform)
        sys.exit()

    # [4] Configure scaling and resource requirements.
    scaling_config = ray.train.ScalingConfig(
        num_workers=args.ngpus,
        use_gpu=True,
        resources_per_worker={"CPU": args.ncpus // args.ngpus - 1,
                              "GPU": 1},  # -1 to avoid blocking
    )
    # shared_checkpointing_path = pathlib.Path("persistent_store").absolute()
    # run_config = ray.train.RunConfig(storage_path=shared_checkpointing_path)
    # [5] Launch distributed training job.

    # Tuning parameters (hyperparameters to be tuned by Ray Tune)
    param_space = {
        "train_loop_config": {
            'learning_rate': tune.uniform(1e-5, 1e-3),  # Hyperparameter: Learning rate
            'batch_size': tune.choice([32, 64, 128]),  # Hyperparameter: Batch size
            'data_path': data_dir.absolute()
        }
    }

    ray.init(
        address=os.environ["ip_head"],
        _node_ip_address=os.environ["head_node_ip"],
    )

    trainer = ray.train.torch.TorchTrainer(
        train_func,
        scaling_config=scaling_config
        # [5a] If running in a multi-node cluster, this is where you
        # should configure the run's persistent storage that is accessible
        # across all worker nodes.
        # run_config=ray.train.RunConfig(storage_path="s3://..."),
    )

    tuner = tune.Tuner(
        trainer,
        param_space=param_space,
        tune_config=tune.TuneConfig(
            num_samples=args.num_samples,
            metric="loss"
        )
    )

    result_grid = tuner.fit()

    result_df = result_grid.get_dataframe()
    print(result_df)
    print(result_df.columns)
