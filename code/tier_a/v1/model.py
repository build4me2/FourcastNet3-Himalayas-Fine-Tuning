"""Larger elevation-conditioned residual UNet for Tier-A v1-reg.

RRCA-FD Plan A / Leonard TIER_A_V1_SKETCH_CALL.md:
  - Frozen FCN3 crop → ERA5 interim residual (never backprop into FCN3).
  - NO diffusion in this cut (v1-diff deferred).
  - PhysicsNeMo's stock UNet is 3D (MaxPool3d) — not used for the 21×37 Nepal
    crop. This module is a 2D residual UNet with FiLM elev/lead conditioning.
  - Zero-init residual head so start ≈ identity (raw FCN3).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLM(nn.Module):
    """Zero-init feature-wise linear modulation: y = x * (1+γ) + β."""

    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.to_gb = nn.Linear(cond_dim, channels * 2)
        nn.init.zeros_(self.to_gb.weight)
        nn.init.zeros_(self.to_gb.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gb = self.to_gb(cond)
        gamma, beta = gb.chunk(2, dim=1)
        return x * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, cond_dim: int, dropout: float = 0.0):
        super().__init__()
        gn1 = min(8, cout)
        gn2 = min(8, cout)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.gn1 = nn.GroupNorm(gn1, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.gn2 = nn.GroupNorm(gn2, cout)
        self.film = FiLM(cond_dim, cout)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.gn1(self.conv1(x)))
        h = self.drop(h)
        h = self.gn2(self.conv2(h))
        h = self.film(h, cond)
        return F.gelu(h + self.skip(x))


def _match_hw(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if t.shape[-2:] != ref.shape[-2:]:
        t = F.interpolate(t, size=ref.shape[-2:], mode="bilinear", align_corners=False)
    return t


class ElevCondResidualUNet(nn.Module):
    """3-level elev/lead-FiLM residual UNet on Nepal crop (21×37).

    Input channels: FCN3 crop vars + elev_norm + lead_norm.
    Output: residual toward ERA5 on (t2m, u10m, v10m).
    FCN3 backbone is never in this graph (frozen upstream).
    """

    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 3,
        base: int = 32,
        depth: int = 3,
        cond_dim: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if depth < 2 or depth > 3:
            raise ValueError("v1 ElevCondResidualUNet supports depth 2 or 3")
        self.depth = depth
        self.in_channels = in_channels
        b = base
        self.enc1 = ConvBlock(in_channels, b, cond_dim, dropout)
        self.down1 = nn.Conv2d(b, b * 2, 3, stride=2, padding=1)
        self.enc2 = ConvBlock(b * 2, b * 2, cond_dim, dropout)
        if depth >= 3:
            self.down2 = nn.Conv2d(b * 2, b * 4, 3, stride=2, padding=1)
            self.enc3 = ConvBlock(b * 4, b * 4, cond_dim, dropout)
            self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
            self.dec2 = ConvBlock(b * 4, b * 2, cond_dim, dropout)
        self.up1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.dec1 = ConvBlock(b * 2, b, cond_dim, dropout)
        self.head = nn.Conv2d(b, out_channels, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _cond(self, x: torch.Tensor) -> torch.Tensor:
        """Condition from spatial elev_norm + lead_norm (last two input chans)."""
        if x.shape[1] < 2:
            return x.new_zeros((x.shape[0], 2))
        elev = x[:, -2].mean(dim=(1, 2))
        lead = x[:, -1].mean(dim=(1, 2))
        return torch.stack([lead, elev], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cond = self._cond(x)
        e1 = self.enc1(x, cond)
        e2 = self.enc2(self.down1(e1), cond)
        if self.depth >= 3:
            e3 = self.enc3(self.down2(e2), cond)
            u2 = _match_hw(self.up2(e3), e2)
            d2 = self.dec2(torch.cat([u2, e2], dim=1), cond)
            u1 = _match_hw(self.up1(d2), e1)
        else:
            u1 = _match_hw(self.up1(e2), e1)
        d1 = self.dec1(torch.cat([u1, e1], dim=1), cond)
        return self.head(d1)


def build_model(cfg: dict) -> nn.Module:
    m = cfg.get("model", {})
    name = str(m.get("name", "ElevCondResidualUNet"))
    if name in {"PhysicsNeMoUNet", "physicsnemo_unet"}:
        raise RuntimeError(
            "PhysicsNeMo stock UNet is 3D; v1-reg uses ElevCondResidualUNet "
            "for the 21×37 2D Nepal crop. Do not enable v1-diff here."
        )
    return ElevCondResidualUNet(
        in_channels=int(m.get("in_channels", 5)),
        out_channels=int(m.get("out_channels", 3)),
        base=int(m.get("base_channels", 32)),
        depth=int(m.get("depth", 3)),
        cond_dim=int(m.get("cond_dim", 2)),
        dropout=float(m.get("dropout", 0.1)),
    )
