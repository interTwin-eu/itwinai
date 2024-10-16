import argparse
import time
from pathlib import Path
from threading import Thread
from typing import Dict

import GPUtil
from net import Net
from torchvision import datasets, transforms

from itwinai.loggers import ConsoleLogger
from itwinai.torch.config import TrainingConfiguration
from itwinai.torch.distributed import TorchDistributedStrategy
from itwinai.torch.trainer import TorchTrainer


class Monitor(Thread):
    """Class for monitoring GPU utilization using a different thread than the main one.
    Works by sleeping for a certain amount of time, specified by `delay`, allowing
    the GIL in Python to run other threads in the mean time.
    """

    def __init__(
        self, delay: int, strategy: TorchDistributedStrategy, global_rank: int
    ) -> None:
        super(Monitor, self).__init__()
        self.stopped = False
        self.delay = delay
        self.strategy: TorchDistributedStrategy = strategy
        self.start_time = time.time()
        self.monitoring_log: Dict = {}
        self.global_rank = global_rank
        self.start()

    def run(self) -> None:
        if not self.strategy.global_rank() == self.global_rank:
            return

        local_gpus = list(range(self.strategy.local_world_size()))
        print(f"local_gpus: {local_gpus}")

        while not self.stopped:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                if gpu.id not in local_gpus:
                    continue

                empty_gpu_log = {"load": [], "memory": []}
                gpu_stats = self.monitoring_log.get(gpu.id, empty_gpu_log)

                gpu_stats["load"].append(gpu.load)
                gpu_stats["memory"].append(gpu.memoryUtil)
                self.monitoring_log[gpu.id] = gpu_stats

            time.sleep(self.delay)

    def stop(self):
        self.stopped = True


class GPUMonitorTrainer(TorchTrainer):

    def execute(self, train_dataset, validation_dataset=None, test_dataset=None):
        """Prepares distributed environment and data structures
        for the actual training.

        Args:
            train_dataset (Dataset): training dataset.
            validation_dataset (Optional[Dataset], optional): validation
                dataset. Defaults to None.
            test_dataset (Optional[Dataset], optional): test dataset.
                Defaults to None.

        Returns:
            Tuple[Dataset, Dataset, Dataset, Any]: training dataset,
            validation dataset, test dataset, trained model.
        """
        self._init_distributed_strategy()
        self._setup_metrics()

        self.create_dataloaders(
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            test_dataset=test_dataset,
        )
        self.create_model_loss_optimizer()

        if self.logger:
            self.logger.create_logger_context(rank=self.strategy.global_rank())
            hparams = self.config.model_dump()
            hparams["distributed_strategy"] = self.strategy.__class__.__name__
            self.logger.save_hyperparameters(hparams)

        self.train()

        if self.logger:
            self.logger.destroy_logger_context()
        # self.strategy.clean_up()
        return train_dataset, validation_dataset, test_dataset, self.model


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
        num_workers_dataloader=1,
    )
    logger = ConsoleLogger()

    trainer = GPUMonitorTrainer(
        config=training_config,
        model=model,
        strategy=args.strategy,
        epochs=args.epochs,
        random_seed=args.seed,
        checkpoint_every=args.ckpt_interval,
        logger=logger,
    )

    monitor = None
    time_delay = 3
    strategy = trainer.strategy
    strategy.init()
    if strategy.local_rank() == 0:
        monitor = Monitor(
            delay=time_delay, strategy=strategy, global_rank=strategy.global_rank()
        )

    trainer.execute(train_dataset, validation_dataset)

    if monitor is not None:
        monitor.stop()


if __name__ == "__main__":
    main()
