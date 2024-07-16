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

REAL_LABEL = 1
FAKE_LABEL = 0
Z_DIM = 100
G_HIDDEN = 64
IMAGE_CHANNEL = 1
D_HIDDEN = 64
G_losses = []
D_losses = []

# Define Generator


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

# Discriminator


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
    def __init__(self,
                 config: Union[Dict, TrainingConfiguration],
                 epochs: int, discriminator: nn.Module,
                 generator: nn.Module,
                 strategy: Literal["ddp", "deepspeed", "horovod"] = 'ddp',
                 validation_every: Optional[int] = 1,
                 test_every: Optional[int] = None,
                 random_seed: Optional[int] = None,
                 logger: Optional[Logger] = None,
                 log_all_workers: bool = False,
                 metrics: Optional[Dict[str, Metric]] = None,
                 checkpoints_location: str = "checkpoints",
                 checkpoint_every: Optional[int] = None,
                 name: Optional[str] = None, **kwargs) -> None:
        super().__init__(config=config,
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
            betas=(0.5, 0.999))
        self.optimizerG = optim.Adam(
            self.generator.parameters(), lr=self.config.lr, betas=(0.5, 0.999))
        self.criterion = nn.BCELoss()
        # Distribute discriminator and its optimizer
        self.discriminator, self.optimizerD, _ = self.strategy.distributed(
            self.discriminator, self.optimizerD)
        self.generator, self.optimizerG, _ = self.strategy.distributed(
            self.generator, self.optimizerG)

    def train_epoch(self):
        self.discriminator.train()
        self.generator.train()
        for batch_idx, (real_images, _) in enumerate(self.train_dataloader):
            lossG, lossD, fake_images = self.train_step(
                real_images, batch_idx)
        self.save_plots_and_images(
            batch_idx, lossD, lossG, real_images, fake_images)

    def validation_epoch(self):
        self.discriminator.eval()
        self.generator.eval()
        validation_loss = torch.tensor(0.0, device=self.device)
        for batch_idx, (real_images, _) in enumerate(
                self.validation_dataloader):
            validation_loss += self.validation_step(
                real_images, batch_idx).item()
        return validation_loss / len(self.validation_dataloader)

    def train_step(self, real_images, batch_idx):
        real_images = real_images.to(self.device)
        batch_size = real_images.size(0)
        label = torch.full((batch_size,), REAL_LABEL,
                           dtype=torch.float, device=self.device)

        # Train Discriminator with real images
        self.discriminator.zero_grad()
        output_real = self.discriminator(real_images)
        lossD_real = self.criterion(output_real, label)
        lossD_real.backward()

        # Generate fake images and train Discriminator
        noise = torch.randn(batch_size, Z_DIM, 1, 1, device=self.device)
        fake_images = self.generator(noise)
        label.fill_(FAKE_LABEL)
        output_fake = self.discriminator(fake_images.detach())
        lossD_fake = self.criterion(output_fake, label)
        lossD_fake.backward()
        self.optimizerD.step()

        # Train Generator
        self.generator.zero_grad()
        label.fill_(REAL_LABEL)
        output_fake = self.discriminator(fake_images)
        lossG = self.criterion(output_fake, label)
        lossG.backward()
        self.optimizerG.step()

        # Save Losses for plotting later
        G_losses.append(lossG.item())
        D_losses.append(lossD_real.item() +
                        lossD_fake.item())

        return G_losses, D_losses, fake_images

    def validation_step(self, real_images, batch_idx):
        real_images = real_images.to(self.device)
        batch_size = real_images.size(0)

        # Generate fake images
        noise = torch.randn(batch_size, Z_DIM, 1, 1, device=self.device)
        fake_images = self.generator(noise)
        with torch.no_grad():
            fake_preds = self.discriminator(fake_images)
        # fake_labels = torch.full(
        #     (batch_size,), FAKE_LABEL, dtype=torch.float, device=self.device)
        label = torch.full((batch_size,), REAL_LABEL,
                           dtype=torch.float, device=self.device)
        label.fill_(FAKE_LABEL)

        # Calculate loss on fake images
        loss = self.criterion(fake_preds, label)
        print(f'the validation loss is:{loss}')

        # Logging fake prediction accuracy
        pred = (fake_preds > 0.5).float()
        accuracy = (pred == label).float().mean()
        print(f'The validation accuracy is:{accuracy}')

        self.log(
            item=loss.item(),
            identifier='validation_loss',
            kind='metric',
            step=self.validation_glob_step,
            batch_idx=batch_idx
        )
        self.log(
            item=accuracy.item(),
            identifier='validation_accuracy',
            kind='metric',
            step=self.validation_glob_step,
            batch_idx=batch_idx
        )
        return loss

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

    def save_plots_and_images(self, epoch, lossG, lossD,
                              real_images, fake_images):
        """Save training plots and images generated by the GAN."""
        images_dir = os.path.join(self.checkpoints_location, 'images')
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)

        # Plot training losses
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(lossG, label="G")
        plt.plot(lossD, label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(images_dir, f'loss_epoch_{epoch}.png'))
        plt.close()

        # Save real images
        real_images_grid = torchvision.utils.make_grid(
            real_images[:64], normalize=True)
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title("Real Images")
        plt.imshow(np.transpose(real_images_grid.cpu().numpy(), (1, 2, 0)))
        plt.savefig(os.path.join(images_dir, f'real_images_epoch_{epoch}.png'))
        plt.close()

        # Save generated images
        fake_images_grid = torchvision.utils.make_grid(
            fake_images[:64], normalize=True)
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(np.transpose(fake_images_grid.cpu().numpy(), (1, 2, 0)))
        plt.savefig(os.path.join(images_dir, f'fake_images_epoch_{epoch}.png'))
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST GAN Example')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=6,
                        help='number of epochs to train (default: 6)')
    parser.add_argument('--strategy', type=str, default='ddp',
                        help='distributed strategy (default=ddp)')
    parser.add_argument('--lr', type=float, default=1.0,
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
        # optimizer='adam',
        loss='cross_entropy'
    )

    # Logger
    logger = MLFlowLogger(experiment_name='GAN MNIST Experiment', log_freq=10)

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
