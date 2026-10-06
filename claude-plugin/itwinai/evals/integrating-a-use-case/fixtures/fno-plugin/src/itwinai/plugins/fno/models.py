"""FNO model, moved unchanged from train.py."""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
