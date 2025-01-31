# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Henry Mutegeki
#
# Credit:
# - Henry Mutegeki <henry.mutegeki@cern.ch> - CERN
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# - Jarl Sondre Sæther <jarl.sondre.saether@cern.ch> - CERN
# --------------------------------------------------------------------------------------

import argparse
import os
from typing import Dict, Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

from itwinai.loggers import Logger, MLFlowLogger
from itwinai.torch.config import TrainingConfiguration
from itwinai.torch.distributed import DeepSpeedStrategy
from itwinai.torch.trainer import TorchTrainer
from itwinai.torch.type import Metric


class Generator(nn.Module):
    def __init__(self, z_dim, g_hidden, image_channel):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input layer
            nn.ConvTranspose2d(z_dim, g_hidden * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(g_hidden * 8),
            nn.ReLU(True),
            # 1st hidden layer
            nn.ConvTranspose2d(g_hidden * 8, g_hidden * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_hidden * 4),
            nn.ReLU(True),
            # 2nd hidden layer
            nn.ConvTranspose2d(g_hidden * 4, g_hidden * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_hidden * 2),
            nn.ReLU(True),
            # 3rd hidden layer
            nn.ConvTranspose2d(g_hidden * 2, g_hidden, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_hidden),
            nn.ReLU(True),
            # output layer
            nn.ConvTranspose2d(g_hidden, image_channel, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, d_hidden, image_channel):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 1st layer
            nn.Conv2d(image_channel, d_hidden, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 2nd layer
            nn.Conv2d(d_hidden, d_hidden * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_hidden * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 3rd layer
            nn.Conv2d(d_hidden * 2, d_hidden * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_hidden * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 4th layer
            nn.Conv2d(d_hidden * 4, d_hidden * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_hidden * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # output layer
            nn.Conv2d(d_hidden * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)


class GANTrainer(TorchTrainer):
    """Trainer class for GAN model using pytorch.

    Args:
        config (Union[Dict, TrainingConfiguration]): training configuration
            containing hyperparameters.
        epochs (int): number of training epochs.
        discriminator (nn.Module): pytorch discriminator model to train GAN.
        generator (nn.Module): pytorch generator model to train GAN.
        strategy (Literal['ddp', 'deepspeed', 'horovod'], optional):
            distributed strategy. Defaults to 'ddp'.
        validation_every (Optional[int], optional): run a validation epoch
            every ``validation_every`` epochs. Disabled if None. Defaults to 1.
        test_every (Optional[int], optional): run a test epoch
            every ``test_every`` epochs. Disabled if None. Defaults to None.
        random_seed (Optional[int], optional): set random seed for
            reproducibility. If None, the seed is not set. Defaults to None.
        logger (Optional[Logger], optional): logger for ML tracking.
            Defaults to None.
        metrics (Optional[Dict[str, Metric]], optional): map of torch metrics
            metrics. Defaults to None.
        checkpoints_location (str): path to checkpoints directory.
            Defaults to "checkpoints".
        checkpoint_every (Optional[int]): save a checkpoint every
            ``checkpoint_every`` epochs. Disabled if None. Defaults to None.
        name (Optional[str], optional): trainer custom name. Defaults to None.
    """

    def __init__(
        self,
        config: Union[Dict, TrainingConfiguration],
        epochs: int,
        discriminator: nn.Module,
        generator: nn.Module,
        strategy: Literal["ddp", "deepspeed"] = "ddp",
        validation_every: Optional[int] = 1,
        test_every: Optional[int] = None,
        random_seed: Optional[int] = None,
        logger: Optional[Logger] = None,
        metrics: Optional[Dict[str, Metric]] = None,
        checkpoints_location: str = "checkpoints",
        checkpoint_every: Optional[int] = None,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            config=config,
            epochs=epochs,
            model=None,
            strategy=strategy,
            validation_every=validation_every,
            test_every=test_every,
            random_seed=random_seed,
            logger=logger,
            metrics=metrics,
            checkpoints_location=checkpoints_location,
            checkpoint_every=checkpoint_every,
            name=name,
            **kwargs,
        )
        self.save_parameters(**self.locals2params(locals()))
        self.discriminator = discriminator
        self.generator = generator

    def create_model_loss_optimizer(self) -> None:
        self.optimizerD = optim.Adam(
            self.discriminator.parameters(), lr=self.config.lr, betas=(0.5, 0.999)
        )
        self.optimizerG = optim.Adam(
            self.generator.parameters(), lr=self.config.lr, betas=(0.5, 0.999)
        )
        self.criterion = nn.BCELoss()

        # https://stackoverflow.com/a/67437077
        self.discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.discriminator)
        self.generator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.generator)

        # First, define strategy-wise optional configurations
        if isinstance(self.strategy, DeepSpeedStrategy):
            # Batch size definition is not optional for DeepSpeedStrategy!
            distribute_kwargs = dict(
                config_params=dict(train_micro_batch_size_per_gpu=self.config.batch_size)
            )
        else:
            distribute_kwargs = {}
        # Distribute discriminator and its optimizer
        self.discriminator, self.optimizerD, _ = self.strategy.distributed(
            self.discriminator, self.optimizerD, **distribute_kwargs
        )
        self.generator, self.optimizerG, _ = self.strategy.distributed(
            self.generator, self.optimizerG, **distribute_kwargs
        )
        self.discriminator.to(self.device)
        self.generator.to(self.device)

    def train_epoch(self, epoch: int):
        self.discriminator.train()
        self.generator.train()
        gen_train_losses = []
        disc_train_losses = []
        disc_train_accuracy = []
        for batch_idx, (real_images, _) in enumerate(self.train_dataloader):
            lossG, lossD, accuracy_disc = self.train_step(real_images, batch_idx)
            gen_train_losses.append(lossG)
            disc_train_losses.append(lossD)
            disc_train_accuracy.append(accuracy_disc)

            self.train_glob_step += 1
        # Aggregate and log losses and accuracy
        avg_disc_accuracy = torch.mean(torch.stack(disc_train_accuracy))
        self.log(
            item=avg_disc_accuracy.item(),
            identifier="disc_train_accuracy_per_epoch",
            kind="metric",
            step=epoch,
        )
        avg_gen_loss = torch.mean(torch.stack(gen_train_losses))
        self.log(
            item=avg_gen_loss.item(),
            identifier="gen_train_loss_per_epoch",
            kind="metric",
            step=epoch,
        )

        avg_disc_loss = torch.mean(torch.stack(disc_train_losses))
        self.log(
            item=avg_disc_loss.item(),
            identifier="disc_train_loss_per_epoch",
            kind="metric",
            step=epoch,
        )

        self.save_fake_generator_images(epoch)

    def validation_epoch(self, epoch: int):
        gen_validation_losses = []
        gen_validation_accuracy = []
        disc_validation_losses = []
        disc_validation_accuracy = []
        self.discriminator.eval()
        self.generator.eval()
        for batch_idx, (real_images, _) in enumerate(self.validation_dataloader):
            loss_gen, accuracy_gen, loss_disc, accuracy_disc = self.validation_step(
                real_images, batch_idx
            )
            gen_validation_losses.append(loss_gen)
            gen_validation_accuracy.append(accuracy_gen)
            disc_validation_losses.append(loss_disc)
            disc_validation_accuracy.append(accuracy_disc)
            self.validation_glob_step += 1

        # Aggregate and log metrics
        disc_validation_loss = torch.mean(torch.stack(disc_validation_losses))
        self.log(
            item=disc_validation_loss.item(),
            identifier="disc_valid_loss_per_epoch",
            kind="metric",
            step=epoch,
        )
        disc_validation_accuracy = torch.mean(torch.stack(disc_validation_accuracy))
        self.log(
            item=disc_validation_accuracy.item(),
            identifier="disc_valid_accuracy_epoch",
            kind="metric",
            step=epoch,
        )
        gen_validation_loss = torch.mean(torch.stack(gen_validation_losses))
        self.log(
            item=gen_validation_loss.item(),
            identifier="gen_valid_loss_per_epoch",
            kind="metric",
            step=epoch,
        )
        gen_validation_accuracy = torch.mean(torch.stack(gen_validation_accuracy))
        self.log(
            item=gen_validation_accuracy.item(),
            identifier="gen_valid_accuracy_epoch",
            kind="metric",
            step=epoch,
        )

        return gen_validation_loss

    def train_step(self, real_images, batch_idx):
        real_images = real_images.to(self.device)
        batch_size = real_images.size(0)
        real_labels = torch.ones((batch_size,), dtype=torch.float, device=self.device)
        fake_labels = torch.zeros((batch_size,), dtype=torch.float, device=self.device)

        # Train Discriminator with real images
        output_real = self.discriminator(real_images)
        lossD_real = self.criterion(output_real, real_labels)
        # Generate fake images and train Discriminator
        noise = torch.randn(batch_size, self.config.z_dim, 1, 1, device=self.device)

        fake_images = self.generator(noise)
        output_fake = self.discriminator(fake_images.detach())
        lossD_fake = self.criterion(output_fake, fake_labels)

        lossD = (lossD_real + lossD_fake) / 2

        self.optimizerD.zero_grad()
        lossD.backward()
        self.optimizerD.step()

        accuracy = ((output_real > 0.5).float() == real_labels).float().mean() + (
            (output_fake < 0.5).float() == fake_labels
        ).float().mean()
        accuracy_disc = accuracy.mean()

        # Train Generator
        output_fake = self.discriminator(fake_images)
        lossG = self.criterion(output_fake, real_labels)
        self.optimizerG.zero_grad()
        lossG.backward()
        self.optimizerG.step()
        self.log(
            item=accuracy_disc,
            identifier="disc_train_accuracy_per_batch",
            kind="metric",
            step=self.train_glob_step,
            batch_idx=batch_idx,
        )
        self.log(
            item=lossG,
            identifier="gen_train_loss_per_batch",
            kind="metric",
            step=self.train_glob_step,
            batch_idx=batch_idx,
        )
        self.log(
            item=lossD,
            identifier="disc_train_loss_per_batch",
            kind="metric",
            step=self.train_glob_step,
            batch_idx=batch_idx,
        )

        return lossG, lossD, accuracy_disc

    def validation_step(self, real_images, batch_idx):
        real_images = real_images.to(self.device)
        batch_size = real_images.size(0)
        real_labels = torch.ones((batch_size,), dtype=torch.float, device=self.device)
        fake_labels = torch.zeros((batch_size,), dtype=torch.float, device=self.device)

        # Validate with real images
        output_real = self.discriminator(real_images)
        loss_real = self.criterion(output_real, real_labels)

        # Generate and validate fake images
        noise = torch.randn(batch_size, self.config.z_dim, 1, 1, device=self.device)

        with torch.no_grad():
            fake_images = self.generator(noise)
            output_fake = self.discriminator(fake_images.detach())
        loss_fake = self.criterion(output_fake, fake_labels)

        # Generator's attempt to fool the discriminator
        loss_gen = self.criterion(output_fake, real_labels)
        accuracy_gen = ((output_fake > 0.5).float() == real_labels).float().mean()

        # Calculate total discriminator loss and accuracy
        loss_disc = (loss_real + loss_fake) / 2
        accuracy = ((output_real > 0.5).float() == real_labels).float().mean() + (
            (output_fake < 0.5).float() == fake_labels
        ).float().mean()
        accuracy_disc = accuracy.mean()

        self.log(
            item=loss_gen.item(),
            identifier="gen_valid_loss_per_batch",
            kind="metric",
            step=self.validation_glob_step,
            batch_idx=batch_idx,
        )
        self.log(
            item=accuracy_gen.item(),
            identifier="gen_valid_accuracy_per_batch",
            kind="metric",
            step=self.validation_glob_step,
            batch_idx=batch_idx,
        )

        self.log(
            item=loss_disc.item(),
            identifier="disc_valid_loss_per_batch",
            kind="metric",
            step=self.validation_glob_step,
            batch_idx=batch_idx,
        )
        self.log(
            item=accuracy_disc,
            identifier="disc_valid_accuracy_per_batch",
            kind="metric",
            step=self.validation_glob_step,
            batch_idx=batch_idx,
        )
        return loss_gen, accuracy_gen, loss_disc, accuracy_disc

    def save_checkpoint(self, name, epoch, loss=None):
        """Save training checkpoint with both optimizers."""
        if not os.path.exists(self.checkpoints_location):
            os.makedirs(self.checkpoints_location)

        checkpoint_path = os.path.join(self.checkpoints_location, f"{name}")
        checkpoint = {
            "epoch": epoch,
            "loss": loss.item() if loss is not None else None,
            "discriminator_state_dict": self.discriminator.state_dict(),
            "generator_state_dict": self.generator.state_dict(),
            "optimizerD_state_dict": self.optimizerD.state_dict(),
            "optimizerG_state_dict": self.optimizerG.state_dict(),
            "lr_scheduler": (self.lr_scheduler.state_dict() if self.lr_scheduler else None),
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load models and optimizers from checkpoint."""
        checkpoint = torch.load(checkpoint_path)

        self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.optimizerD.load_state_dict(checkpoint["optimizerD_state_dict"])
        self.optimizerG.load_state_dict(checkpoint["optimizerG_state_dict"])

        if "lr_scheduler" in checkpoint and checkpoint["lr_scheduler"] is not None:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        print(f"Checkpoint loaded from {checkpoint_path}")

    def save_fake_generator_images(self, epoch):
        self.generator.eval()
        noise = torch.randn(64, self.config.z_dim, 1, 1, device=self.device)
        fake_images = self.generator(noise)
        fake_images_grid = torchvision.utils.make_grid(fake_images, normalize=True)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_axis_off()
        ax.set_title(f"Fake images for epoch {epoch}")
        ax.imshow(np.transpose(fake_images_grid.cpu().numpy(), (1, 2, 0)))
        self.log(
            item=fig,
            identifier=f"fake_images_epoch_{epoch}.png",
            kind="figure",
            step=epoch,
        )


def main():
    parser = argparse.ArgumentParser(description="PyTorch MNIST GAN Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="input batch size for training (default: 128)",
    )
    parser.add_argument(
        "--epochs", type=int, default=15, help="number of epochs to train (default: 15)"
    )
    parser.add_argument(
        "--strategy", type=str, default="ddp", help="distributed strategy (default=ddp)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="learning rate (default: 0.001)"
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument(
        "--ckpt-interval",
        type=int,
        default=2,
        help="how many batches to wait before logging training status",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Dataset
    transform = transforms.Compose(
        [
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    train_dataset = datasets.MNIST("../data", train=True, download=True, transform=transform)
    validation_dataset = datasets.MNIST("../data", train=False, transform=transform)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    # Models
    netG = Generator(z_dim=100, g_hidden=64, image_channel=1)
    netG.apply(weights_init)
    netD = Discriminator(d_hidden=64, image_channel=1)
    netD.apply(weights_init)

    # Training configuration
    training_config = TrainingConfiguration(
        batch_size=args.batch_size, lr=args.lr, epochs=args.epochs, z_dim=100
    )

    # Logger
    logger = MLFlowLogger(experiment_name="Distributed GAN MNIST", log_freq=10)

    # Trainer
    trainer = GANTrainer(
        config=training_config,
        discriminator=netD,
        generator=netG,
        strategy=args.strategy,
        epochs=args.epochs,
        random_seed=args.seed,
        logger=logger,
    )

    # Launch training
    _, _, _, trained_model = trainer.execute(train_dataset, validation_dataset, None)


if __name__ == "__main__":
    main()
