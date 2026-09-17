"""Train a Fourier Neural Operator to emulate a 2D Darcy flow solver.

Learns the map from a permeability field a(x) to the pressure field u(x) solving

    -div(a grad u) = f,   u = 0 on the boundary

Data is generated on the fly, so no download is needed.

Usage:
    python train.py --epochs 20 --grid-size 32
"""

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def gaussian_random_field(n_samples, size, alpha=3.0, rng=None):
    """Sample smooth random fields by filtering white noise in Fourier space."""
    rng = rng or np.random.default_rng(0)
    noise = rng.normal(size=(n_samples, size, size))

    kx = np.fft.fftfreq(size) * size
    ky = np.fft.fftfreq(size) * size
    k2 = kx[:, None] ** 2 + ky[None, :] ** 2
    k2[0, 0] = 1.0
    spectrum = k2 ** (-alpha / 2.0)
    spectrum[0, 0] = 0.0

    field = np.fft.ifft2(np.fft.fft2(noise) * spectrum).real
    std = field.std(axis=(1, 2), keepdims=True)
    return field / np.where(std == 0, 1.0, std)


def apply_operator(a_right, a_left, a_up, a_down, u, h2):
    """Matrix-free -div(a grad u) with zero Dirichlet boundaries."""
    out = (a_right + a_left + a_up + a_down) * u
    out[:, :, :-1] -= a_right[:, :, :-1] * u[:, :, 1:]
    out[:, :, 1:] -= a_left[:, :, 1:] * u[:, :, :-1]
    out[:, :-1, :] -= a_down[:, :-1, :] * u[:, 1:, :]
    out[:, 1:, :] -= a_up[:, 1:, :] * u[:, :-1, :]
    return out / h2


def solve_darcy(a, tol=1e-8, max_iter=2000):
    """Solve the Darcy system for a batch of permeability fields with batched CG."""
    n_samples, size, _ = a.shape
    h2 = (1.0 / (size + 1)) ** 2

    # Harmonic means at the cell faces. The domain is padded by replication so that
    # boundary faces keep a non-zero coefficient: u vanishes outside, which is what
    # imposes the Dirichlet condition, but the conductivity there must not vanish too.
    a_pad = np.pad(a, ((0, 0), (1, 1), (1, 1)), mode="edge")

    def face(shifted):
        return 2.0 * a * shifted / np.maximum(a + shifted, 1e-12)

    a_right = face(a_pad[:, 1:-1, 2:])
    a_left = face(a_pad[:, 1:-1, :-2])
    a_down = face(a_pad[:, 2:, 1:-1])
    a_up = face(a_pad[:, :-2, 1:-1])

    b = np.ones((n_samples, size, size))
    u = np.zeros_like(b)
    r = b - apply_operator(a_right, a_left, a_up, a_down, u, h2)
    p = r.copy()
    rs_old = np.sum(r * r, axis=(1, 2))

    for _ in range(max_iter):
        ap = apply_operator(a_right, a_left, a_up, a_down, p, h2)
        denom = np.sum(p * ap, axis=(1, 2))
        alpha = rs_old / np.where(np.abs(denom) < 1e-30, 1e-30, denom)
        u += alpha[:, None, None] * p
        r -= alpha[:, None, None] * ap
        rs_new = np.sum(r * r, axis=(1, 2))
        if np.all(np.sqrt(rs_new) < tol):
            break
        p = r + (rs_new / np.where(rs_old == 0, 1e-30, rs_old))[:, None, None] * p
        rs_old = rs_new

    return u


def make_dataset(n_samples, size, seed):
    """Thresholded random fields give the usual piecewise-constant Darcy benchmark."""
    rng = np.random.default_rng(seed)
    psi = gaussian_random_field(n_samples, size, rng=rng)
    a = np.where(psi >= 0.0, 12.0, 3.0)
    u = solve_darcy(a)
    return a.astype(np.float32), u.astype(np.float32)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class SpectralConv2d(nn.Module):
    """Multiply the lowest Fourier modes by learned complex weights."""

    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        self.weight_low = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes, modes, dtype=torch.cfloat)
        )
        self.weight_high = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes, modes, dtype=torch.cfloat)
        )

    def forward(self, x):
        batch = x.shape[0]
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(
            batch,
            self.weight_low.shape[1],
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        m = self.modes
        out_ft[:, :, :m, :m] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, :m, :m], self.weight_low
        )
        out_ft[:, :, -m:, :m] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, -m:, :m], self.weight_high
        )
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class FNO2d(nn.Module):
    def __init__(self, modes=12, width=32, n_layers=4):
        super().__init__()
        self.lift = nn.Linear(3, width)
        self.spectral = nn.ModuleList(
            [SpectralConv2d(width, width, modes) for _ in range(n_layers)]
        )
        self.pointwise = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.project1 = nn.Linear(width, 128)
        self.project2 = nn.Linear(128, 1)

    def forward(self, a):
        # a: (B, H, W) -> stack the coordinate grid so the operator sees position
        batch, height, width_ = a.shape
        ys = torch.linspace(0, 1, height, device=a.device)
        xs = torch.linspace(0, 1, width_, device=a.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid_y = grid_y.expand(batch, -1, -1)
        grid_x = grid_x.expand(batch, -1, -1)

        x = torch.stack([a, grid_y, grid_x], dim=-1)
        x = self.lift(x).permute(0, 3, 1, 2)

        for spec, point in zip(self.spectral, self.pointwise, strict=True):
            x = F.gelu(spec(x) + point(x))

        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.project1(x))
        return self.project2(x).squeeze(-1)


def relative_l2(prediction, target):
    """Relative L2 error, the standard metric for neural operators."""
    num = torch.linalg.vector_norm(prediction - target, dim=(1, 2))
    den = torch.linalg.vector_norm(target, dim=(1, 2))
    return torch.mean(num / den)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--n-train", type=int, default=256)
    parser.add_argument("--n-val", type=int, default=64)
    parser.add_argument("--modes", type=int, default=12)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default="fno.pth")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Generating training data...")
    t0 = time.time()
    a_train, u_train = make_dataset(args.n_train, args.grid_size, args.seed)
    a_val, u_val = make_dataset(args.n_val, args.grid_size, args.seed + 1)
    print(f"Data generated in {time.time() - t0:.1f}s")

    # Normalise the inputs, otherwise the lift layer sees values around 12
    a_mean, a_std = a_train.mean(), a_train.std()
    a_train = (a_train - a_mean) / a_std
    a_val = (a_val - a_mean) / a_std

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(a_train), torch.from_numpy(u_train)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(a_val), torch.from_numpy(u_val)),
        batch_size=args.batch_size,
    )

    model = FNO2d(modes=args.modes, width=args.width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for a_batch, u_batch in train_loader:
            a_batch, u_batch = a_batch.to(device), u_batch.to(device)
            optimizer.zero_grad()
            loss = relative_l2(model(a_batch), u_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * a_batch.shape[0]
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for a_batch, u_batch in val_loader:
                a_batch, u_batch = a_batch.to(device), u_batch.to(device)
                val_loss += relative_l2(model(a_batch), u_batch).item() * a_batch.shape[0]
        val_loss /= len(val_loader.dataset)

        scheduler.step()
        print(f"epoch {epoch + 1}/{args.epochs}  train {train_loss:.4f}  val {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), args.checkpoint)

    print(f"Best validation relative L2: {best_val:.4f}")


if __name__ == "__main__":
    main()
