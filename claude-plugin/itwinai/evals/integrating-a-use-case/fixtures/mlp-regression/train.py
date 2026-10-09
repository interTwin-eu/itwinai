"""Fit a small MLP to a noisy sine wave.

A deliberately conventional script: plain nn.Module, MSE loss, Adam, a hand-written loop.
Data is synthetic, so no download is needed.

Usage:
    python train.py --epochs 50
"""

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split


class MLP(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


def make_dataset(n_samples, seed):
    generator = torch.Generator().manual_seed(seed)
    x = torch.rand(n_samples, 1, generator=generator) * 6 - 3
    y = torch.sin(x) + 0.1 * torch.randn(n_samples, 1, generator=generator)
    return TensorDataset(x, y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-samples", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dataset = make_dataset(args.n_samples, args.seed)
    n_val = len(dataset) // 5
    train_set, val_set = random_split(dataset, [len(dataset) - n_val, n_val])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    model = MLP(hidden=args.hidden)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x)
        train_loss /= len(train_set)

        model.eval()
        with torch.no_grad():
            val_loss = sum(loss_fn(model(x), y).item() * len(x) for x, y in val_loader)
        val_loss /= len(val_set)
        print(f"epoch {epoch + 1}/{args.epochs}  train {train_loss:.4f}  val {val_loss:.4f}")

    torch.save(model.state_dict(), "mlp.pth")


if __name__ == "__main__":
    main()
