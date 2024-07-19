import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import torchvision
from torchvision import datasets, transforms
from itwinai.torch.trainer import TorchTrainer
from itwinai.torch.config import TrainingConfiguration
from itwinai.loggers import MLFlowLogger
from typing import (
    Optional, Dict, Union, Literal
)
from itwinai.loggers import Logger
from itwinai.torch.type import Metric
import matplotlib.pyplot as plt
import numpy as np
from itwinai.torch.distributed import (
    DeepSpeedStrategy
)

REAL_LABEL = 1
FAKE_LABEL = 0
Z_DIM = 100
G_HIDDEN = 64
IMAGE_CHANNEL = 1
D_HIDDEN = 64

NOISE_DIM = 100


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input layer
            nn.ConvTranspose2d(Z_DIM, G_HIDDEN * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(G_HIDDEN * 8),
            nn.ReLU(True),
            # 1st hidden layer
            nn.ConvTranspose2d(G_HIDDEN * 8, G_HIDDEN * \
                               4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(G_HIDDEN * 4),
            nn.ReLU(True),
            # 2nd hidden layer
            nn.ConvTranspose2d(G_HIDDEN * 4, G_HIDDEN * \
                               2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(G_HIDDEN * 2),
            nn.ReLU(True),
            # 3rd hidden layer
            nn.ConvTranspose2d(G_HIDDEN * 2, G_HIDDEN, 4, 2, 1, bias=False),
            nn.BatchNorm2d(G_HIDDEN),
            nn.ReLU(True),
            # output layer
            nn.ConvTranspose2d(G_HIDDEN, IMAGE_CHANNEL, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 1st layer
            nn.Conv2d(IMAGE_CHANNEL, D_HIDDEN, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 2nd layer
            nn.Conv2d(D_HIDDEN, D_HIDDEN * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(D_HIDDEN * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 3rd layer
            nn.Conv2d(D_HIDDEN * 2, D_HIDDEN * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(D_HIDDEN * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 4th layer
            nn.Conv2d(D_HIDDEN * 4, D_HIDDEN * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(D_HIDDEN * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # output layer
            nn.Conv2d(D_HIDDEN * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)


class GANTrainer(TorchTrainer):
    def __init__(
            self,
            config: Union[Dict, TrainingConfiguration],
            epochs: int, discriminator: nn.Module,
            generator: nn.Module,
            strategy: Literal["ddp", "deepspeed"] = 'ddp',
            validation_every: Optional[int] = 1,
            test_every: Optional[int] = None,
            random_seed: Optional[int] = None,
            logger: Optional[Logger] = None,
            log_all_workers: bool = False,
            metrics: Optional[Dict[str, Metric]] = None,
            checkpoints_location: str = "checkpoints",
            checkpoint_every: Optional[int] = None,
            name: Optional[str] = None, **kwargs) -> None:
        super().__init__(
            config=config,
            epochs=epochs,
            model=None,
            strategy=strategy,
            validation_every=validation_every,
            test_every=test_every,
            random_seed=random_seed,
            logger=logger,
            log_all_workers=log_all_workers,
            metrics=metrics,
            checkpoints_location=checkpoints_location,
            checkpoint_every=checkpoint_every,
            name=name,
            **kwargs)
        self.save_parameters(**self.locals2params(locals()))
        self.discriminator = discriminator
        self.generator = generator

    def create_model_loss_optimizer(self) -> None:
        self.optimizerD = optim.Adam(
            self.discriminator.parameters(), lr=self.config.lr,
            betas=(0.5, 0.999)
        )
        self.optimizerG = optim.Adam(
            self.generator.parameters(), lr=self.config.lr,
            betas=(0.5, 0.999))
        self.criterion = nn.BCELoss()

        # https://stackoverflow.com/a/67437077
        self.discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
            self.discriminator)
        self.generator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
            self.generator)

        # First, define strategy-wise optional configurations
        if isinstance(self.strategy, DeepSpeedStrategy):
            # Batch size definition is not optional for DeepSpeedStrategy!
            distribute_kwargs = dict(
                config_params=dict(
                    train_micro_batch_size_per_gpu=self.config.batch_size
                )
            )
        else:
            distribute_kwargs = {}
        # Distribute discriminator and its optimizer
        self.discriminator, self.optimizerD, _ = self.strategy.distributed(
            self.discriminator, self.optimizerD, **distribute_kwargs)
        self.generator, self.optimizerG, _ = self.strategy.distributed(
            self.generator, self.optimizerG, **distribute_kwargs)

    def train_epoch(self, epoch: int):
        self.discriminator.train()
        self.generator.train()
        g_train_losses = []
        d_train_losses = []
        for batch_idx, (real_images, _) in enumerate(self.train_dataloader):
            lossG, lossD = self.train_step(
                real_images, batch_idx)
            g_train_losses.append(lossG)
            d_train_losses.append(lossD)
            self.log(
                item=lossG,
                identifier='generator training loss per epoch',
                kind='metric',
                step=self.train_glob_step,
                batch_idx=batch_idx
            )
            self.log(
                item=lossD,
                identifier='discriminator training loss per epoch',
                kind='metric',
                step=self.train_glob_step,
                batch_idx=batch_idx
            )

            self.train_glob_step += 1
        # Aggregate and log losses
        avg_g_loss = torch.mean(torch.stack(g_train_losses))
        self.log(
            item=avg_g_loss.item(),
            identifier='g_train_loss_av per batch',
            kind='metric',
            step=epoch,
        )

        avg_d_loss = torch.mean(torch.stack(d_train_losses))
        self.log(
            item=avg_d_loss.item(),
            identifier='d_train_loss_av per batch',
            kind='metric',
            step=epoch,
        )

        self.save_plots_and_images(epoch)

    def validation_epoch(self, epoch: int):
        g_validation_losses = []
        g_validation_accuracy = []
        self.discriminator.eval()
        self.generator.eval()
        validation_loss = torch.tensor(0.0, device=self.device)
        for batch_idx, (real_images, _) in enumerate(
                self.validation_dataloader):
            loss_gen, accuracy_gen = self.validation_step(
                real_images, batch_idx)
            g_validation_losses.append(loss_gen)
            g_validation_accuracy.append(accuracy_gen)
            self.validation_glob_step += 1

        # Aggregate and log metrics
        validation_loss = torch.mean(torch.stack(g_validation_losses))
        self.log(
            item=validation_loss.item(),
            identifier='gen_validation_loss_epoch',
            kind='metric',
            step=self.validation_glob_step,
        )
        validation_accuracy = torch.mean(torch.stack(g_validation_accuracy))
        self.log(
            item=validation_accuracy.item(),
            identifier='gen_validation_accuracy_epoch',
            kind='metric',
            step=self.validation_glob_step,
        )

        return validation_loss

    def train_step(self, real_images, batch_idx):
        real_images = real_images.to(self.device)
        batch_size = real_images.size(0)
        real_labels = torch.full(
            (batch_size,), REAL_LABEL,
            dtype=torch.float, device=self.device)
        fake_labels = torch.full(
            (batch_size,), FAKE_LABEL,
            dtype=torch.float, device=self.device)

        # Train Discriminator with real images
        output_real = self.discriminator(real_images)
        lossD_real = self.criterion(output_real, real_labels)
        # Generate fake images and train Discriminator
        noise = torch.randn(batch_size, NOISE_DIM, 1, 1, device=self.device)

        fake_images = self.generator(noise)
        output_fake = self.discriminator(fake_images.detach())
        lossD_fake = self.criterion(output_fake, fake_labels)

        loss = (lossD_real+lossD_fake)/2

        self.optimizerD.zero_grad()
        loss.backward()
        self.optimizerD.step()

        # Train Generator
        output_fake = self.discriminator(fake_images)
        lossG = self.criterion(output_fake, real_labels)
        self.optimizerG.zero_grad()
        lossG.backward()
        self.optimizerG.step()

        return lossG, loss

    def validation_step(self, real_images, batch_idx):
        real_images = real_images.to(self.device)
        batch_size = real_images.size(0)
        real_labels = torch.full((batch_size,), REAL_LABEL,
                                 dtype=torch.float, device=self.device)
        fake_labels = torch.full((batch_size,), FAKE_LABEL,
                                 dtype=torch.float, device=self.device)

        # Validate with real images
        output_real = self.discriminator(real_images)
        loss_real = self.criterion(output_real, real_labels)

        # Generate and validate fake images
        noise = torch.randn(batch_size, NOISE_DIM, 1, 1, device=self.device)

        with torch.no_grad():
            fake_images = self.generator(noise)
            output_fake = self.discriminator(fake_images.detach())
        loss_fake = self.criterion(output_fake, fake_labels)

        # Generator's attempt to fool the discriminator
        loss_gen = self.criterion(output_fake, real_labels)
        accuracy_gen = (
            (output_fake > 0.5).float() == real_labels).float().mean()

        # Calculate total discriminator loss and accuracy
        d_total_loss = (loss_real + loss_fake) / 2
        accuracy = ((output_real > 0.4).float() == real_labels).float().mean(
        ) + ((output_fake < 0.5).float() == fake_labels).float().mean()
        d_accuracy = accuracy.item()/2

        self.log(
            item=loss_gen.item(),
            identifier='generator validation_loss',
            kind='metric',
            step=self.validation_glob_step,
            batch_idx=batch_idx
        )
        self.log(
            item=accuracy_gen.item(),
            identifier='generator validation_accuracy',
            kind='metric',
            step=self.validation_glob_step,
            batch_idx=batch_idx
        )

        self.log(
            item=d_total_loss.item(),
            identifier='discriminator validation_loss',
            kind='metric',
            step=self.validation_glob_step,
            batch_idx=batch_idx
        )
        self.log(
            item=d_accuracy,
            identifier='discriminator validation_accuracy',
            kind='metric',
            step=self.validation_glob_step,
            batch_idx=batch_idx
        )
        return loss_gen, accuracy_gen

    def configure_optimizers(self):
        return [self.optimizerD, self.optimizerG]

    def save_checkpoint(self, name, epoch, loss=None):
        """Save training checkpoint with both optimizers."""
        if not os.path.exists(self.checkpoints_location):
            os.makedirs(self.checkpoints_location)

        checkpoint_path = os.path.join(self.checkpoints_location, f"{name}")
        checkpoint = {
            'epoch': epoch,
            'loss': loss.item() if loss is not None else None,
            'discriminator_state_dict': self.discriminator.state_dict(),
            'generator_state_dict': self.generator.state_dict(),
            'optimizerD_state_dict': self.optimizerD.state_dict(),
            'optimizerG_state_dict': self.optimizerG.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict() if
            self.lr_scheduler else None
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load models and optimizers from checkpoint."""
        checkpoint = torch.load(checkpoint_path)

        self.discriminator.load_state_dict(
            checkpoint['discriminator_state_dict'])
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
        self.optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])

        if 'lr_scheduler' in checkpoint:
            if checkpoint['lr_scheduler'] is not None:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        print(f"Checkpoint loaded from {checkpoint_path}")

    def save_plots_and_images(self, epoch):
        self.generator.eval()
        noise = torch.randn(64, NOISE_DIM, 1, 1, device=self.device)
        fake_images = self.generator(noise)
        fake_images_grid = torchvision.utils.make_grid(
            fake_images, normalize=True)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_axis_off()
        ax.set_title(f'Fake images for epoch {epoch}')
        ax.imshow(np.transpose(fake_images_grid.cpu().numpy(), (1, 2, 0)))
        self.log(
            item=fig,
            identifier=f'fake_images_epoch_{epoch}.png',
            kind='figure',
            step=epoch,
        )


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST GAN Example')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=15,
                        help='number of epochs to train (default: 15)')
    parser.add_argument('--strategy', type=str, default='ddp',
                        help='distributed strategy (default=ddp)')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate (default: 1.0)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument(
        '--ckpt-interval', type=int, default=2,
        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Dataset
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(
        '../data', train=True, download=True, transform=transform)
    validation_dataset = datasets.MNIST(
        '../data', train=False, transform=transform)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    # Models
    netG = Generator().to(torch.device("cuda" if torch.cuda.is_available()
                                       else "cpu "))
    netG.apply(weights_init)
    netD = Discriminator().to(torch.device("cuda" if torch.cuda.is_available()
                                           else "cpu"))
    netD.apply(weights_init)

    # Training configuration
    training_config = TrainingConfiguration(
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        loss='cross_entropy'
    )

    # Logger
    logger = MLFlowLogger(experiment_name='Distributed GAN MNIST', log_freq=10)

    metrics = {
        'accuracy': torchmetrics.Accuracy(task='multiclass', num_classes=10),
        'precision': torchmetrics.Precision(task='multiclass', num_classes=10)
    }

    # Trainer
    trainer = GANTrainer(
        config=training_config,
        discriminator=netD,
        generator=netG,
        metrics=metrics,
        strategy=args.strategy,
        epochs=args.epochs,
        random_seed=args.seed,
        logger=logger
    )

    # Launch training
    _, _, _, trained_model = trainer.execute(
        train_dataset, validation_dataset, None)


if __name__ == '__main__':
    main()
