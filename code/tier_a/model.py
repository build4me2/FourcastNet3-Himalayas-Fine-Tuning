"""Tiny elevation-aware residual UNet for Tier-A v0 (diagnostic, not FCN3 FT)."""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.GroupNorm(min(8, cout), cout),
            nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.GroupNorm(min(8, cout), cout),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class TinyElevResidualUNet(nn.Module):
    """Minimal 2-level UNet residual head on Nepal crop (21×37).

    Input:  FCN3 crop vars + elev_norm + lead_norm  → residual toward ERA5.
    FCN3 backbone is never in this graph (frozen upstream).
    """

    def __init__(self, in_channels: int = 5, out_channels: int = 3, base: int = 16, depth: int = 2):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base)
        self.down1 = nn.Conv2d(base, base * 2, 3, stride=2, padding=1)
        self.enc2 = ConvBlock(base * 2, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)
        self.head = nn.Conv2d(base, out_channels, 1)
        # zero-init residual head so start ≈ identity (raw FCN3)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        u = self.up1(e2)
        # pad/crop if odd spatial dims cause off-by-one
        if u.shape[-2:] != e1.shape[-2:]:
            u = torch.nn.functional.interpolate(
                u, size=e1.shape[-2:], mode="bilinear", align_corners=False
            )
        d1 = self.dec1(torch.cat([u, e1], dim=1))
        return self.head(d1)


def build_model(cfg: dict) -> nn.Module:
    m = cfg.get("model", {})
    return TinyElevResidualUNet(
        in_channels=int(m.get("in_channels", 5)),
        out_channels=int(m.get("out_channels", 3)),
        base=int(m.get("base_channels", 16)),
        depth=int(m.get("depth", 2)),
    )
