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

import torch
import torch.nn as nn
from torchvision import datasets, transforms

from itwinai.loggers import MLFlowLogger
from itwinai.torch.gan import GANTrainer, GANTrainingConfiguration


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
    training_config = GANTrainingConfiguration(
        batch_size=args.batch_size,
        optim_generator_lr=args.lr,
        optim_discriminator_lr=args.lr,
        z_dim=100,
    )

    # Logger
    logger = MLFlowLogger(experiment_name="Distributed GAN MNIST", log_freq=10)

    # Trainer
    trainer = GANTrainer(
        config=training_config,
        epochs=args.epochs,
        discriminator=netD,
        generator=netG,
        strategy=args.strategy,
        random_seed=args.seed,
        logger=logger,
        checkpoint_every=args.ckpt_interval,
    )

    # Launch training
    _, _, _, trained_model = trainer.execute(train_dataset, validation_dataset, None)


if __name__ == "__main__":
    main()
