"""Three surrogates for the map u0 -> u(T): an MLP, a Transformer, and a Fourier Neural Operator.

All take (batch, n) and return (batch, n). The FNO and the Transformer are resolution-agnostic
(they accept any grid size n at inference); the MLP is tied to its training grid.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, n: int, width: int = 512, depth: int = 3):
        super().__init__()
        layers, d = [], n
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.GELU()]
            d = width
        layers.append(nn.Linear(d, n))
        self.net = nn.Sequential(*layers)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.net(u)


def grid_features(n: int, n_freq: int, device) -> torch.Tensor:
    x = torch.arange(n, device=device) / n
    k = torch.arange(1, n_freq + 1, device=device)
    ang = 2 * math.pi * x[:, None] * k[None, :]
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # (n, 2*n_freq)


class TransformerOperator(nn.Module):
    """Grid points are tokens; token features are (u(x), Fourier features of x)."""

    def __init__(self, d_model: int = 64, n_heads: int = 4, n_layers: int = 4, d_ff: int = 128, n_freq: int = 8):
        super().__init__()
        self.n_freq = n_freq
        self.inp = nn.Linear(1 + 2 * n_freq, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        b, n = u.shape
        feats = grid_features(n, self.n_freq, u.device).expand(b, -1, -1)
        h = self.inp(torch.cat([u.unsqueeze(-1), feats], dim=-1))
        return self.out(self.encoder(h)).squeeze(-1)


class SpectralConv1d(nn.Module):
    def __init__(self, channels: int, modes: int):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (channels * channels)
        self.weight = nn.Parameter(scale * torch.randn(channels, channels, modes, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (b, c, n)
        n = x.shape[-1]
        xh = torch.fft.rfft(x, dim=-1)
        m = min(self.modes, xh.shape[-1])
        out = torch.zeros_like(xh)
        out[..., :m] = torch.einsum("bix,iox->box", xh[..., :m], self.weight[..., :m])
        return torch.fft.irfft(out, n=n, dim=-1)


class FNO1d(nn.Module):
    """Fourier Neural Operator (Li et al.): lift, [spectral conv + pointwise skip] x L, project."""

    def __init__(self, width: int = 32, modes: int = 16, n_layers: int = 4):
        super().__init__()
        self.lift = nn.Linear(2, width)  # inputs: u0(x) and the coordinate x
        self.spec = nn.ModuleList(SpectralConv1d(width, modes) for _ in range(n_layers))
        self.skip = nn.ModuleList(nn.Conv1d(width, width, 1) for _ in range(n_layers))
        self.proj = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        b, n = u.shape
        x = (torch.arange(n, device=u.device) / n).expand(b, -1)
        h = self.lift(torch.stack([u, x], dim=-1)).transpose(1, 2)  # (b, c, n)
        for i, (s, w) in enumerate(zip(self.spec, self.skip)):
            h = s(h) + w(h)
            if i < len(self.spec) - 1:
                h = torch.nn.functional.gelu(h)
        return self.proj(h.transpose(1, 2)).squeeze(-1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters())
