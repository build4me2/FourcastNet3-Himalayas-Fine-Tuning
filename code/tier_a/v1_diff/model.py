"""Custom 2D EDM residual diffuser for Tier-A v1-diff (CorrDiff-lite).

Architecture choice (LOCKED in config): custom 2D EDM — NOT PhysicsNeMo
(stock UNet is 3D / MaxPool3d), NOT DDPM. Nepal crop 21×37 fits full-field.
Target: leftover = ERA5 − (FCN3 + frozen residual mean).
Conditioner: FCN3 pred + elev_norm + lead_norm + frozen residual mean.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn_groups(channels: int, prefer: int = 8) -> int:
    for g in (prefer, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class SigmaEmbedding(nn.Module):
    """Fourier features of log(sigma) → MLP → embedding."""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, log_sigma: torch.Tensor) -> torch.Tensor:
        # log_sigma: (B,)
        half = self.dim // 2
        freqs = torch.arange(half, device=log_sigma.device, dtype=log_sigma.dtype)
        freqs = 2 * math.pi * (freqs / max(half - 1, 1))
        angles = log_sigma[:, None] * freqs[None, :]
        feat = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if feat.shape[-1] < self.dim:
            feat = F.pad(feat, (0, self.dim - feat.shape[-1]))
        return self.mlp(feat[:, : self.dim])


class FiLM(nn.Module):
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
        gn = _gn_groups(cout)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.gn1 = nn.GroupNorm(gn, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.gn2 = nn.GroupNorm(gn, cout)
        self.film = FiLM(cond_dim, cout)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.gn1(self.conv1(x)))
        h = self.drop(h)
        h = self.gn2(self.conv2(h))
        h = self.film(h, cond)
        return F.silu(h + self.skip(x))


def _match_hw(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if t.shape[-2:] != ref.shape[-2:]:
        t = F.interpolate(t, size=ref.shape[-2:], mode="bilinear", align_corners=False)
    return t


class ResidualEDM2D(nn.Module):
    """EDM-preconditioned 2D UNet denoiser for leftover residual fields.

    Network input = concat(c_in * x_noisy, spatial_cond).
    Predicts F; D = c_skip * x_noisy + c_out * F  (Karras EDM).
    """

    def __init__(
        self,
        target_channels: int = 3,
        cond_channels: int = 8,
        base: int = 24,
        depth: int = 2,
        dropout: float = 0.10,
        sigma_data: float = 1.0,
        emb_dim: int = 64,
    ):
        super().__init__()
        if depth not in (2, 3):
            raise ValueError("ResidualEDM2D supports depth 2 or 3")
        self.depth = depth
        self.target_channels = target_channels
        self.cond_channels = cond_channels
        self.sigma_data = float(sigma_data)
        self.emb = SigmaEmbedding(emb_dim)
        # cond vector = sigma_emb + global (mean elev, mean lead from cond maps)
        self.cond_dim = emb_dim + 2
        b = base
        cin = target_channels + cond_channels
        self.enc1 = ConvBlock(cin, b, self.cond_dim, dropout)
        self.down1 = nn.Conv2d(b, b * 2, 3, stride=2, padding=1)
        self.enc2 = ConvBlock(b * 2, b * 2, self.cond_dim, dropout)
        if depth >= 3:
            self.down2 = nn.Conv2d(b * 2, b * 4, 3, stride=2, padding=1)
            self.enc3 = ConvBlock(b * 4, b * 4, self.cond_dim, dropout)
            self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
            self.dec2 = ConvBlock(b * 4, b * 2, self.cond_dim, dropout)
        self.up1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.dec1 = ConvBlock(b * 2, b, self.cond_dim, dropout)
        self.head = nn.Conv2d(b, target_channels, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _precond_coeffs(self, sigma: torch.Tensor):
        sd = self.sigma_data
        sigma = sigma.view(-1)
        c_skip = sd**2 / (sigma**2 + sd**2)
        c_out = sigma * sd / torch.sqrt(sigma**2 + sd**2)
        c_in = 1.0 / torch.sqrt(sigma**2 + sd**2)
        c_noise = torch.log(sigma.clamp_min(1e-8)) / 4.0
        return c_skip, c_out, c_in, c_noise

    def _vec_cond(self, spatial_cond: torch.Tensor, c_noise: torch.Tensor) -> torch.Tensor:
        # spatial_cond channels: pred(3), elev, lead, res_mean(3) — use elev/lead means
        elev = spatial_cond[:, 3].mean(dim=(1, 2))
        lead = spatial_cond[:, 4].mean(dim=(1, 2))
        emb = self.emb(c_noise)
        return torch.cat([emb, lead[:, None], elev[:, None]], dim=1)

    def forward_F(
        self,
        x_noisy: torch.Tensor,
        sigma: torch.Tensor,
        spatial_cond: torch.Tensor,
    ) -> torch.Tensor:
        c_skip, c_out, c_in, c_noise = self._precond_coeffs(sigma)
        x_in = c_in[:, None, None, None] * x_noisy
        h_in = torch.cat([x_in, spatial_cond], dim=1)
        cond = self._vec_cond(spatial_cond, c_noise)
        e1 = self.enc1(h_in, cond)
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

    def denoise(
        self,
        x_noisy: torch.Tensor,
        sigma: torch.Tensor,
        spatial_cond: torch.Tensor,
    ) -> torch.Tensor:
        c_skip, c_out, c_in, c_noise = self._precond_coeffs(sigma)
        F_x = self.forward_F(x_noisy, sigma, spatial_cond)
        return c_skip[:, None, None, None] * x_noisy + c_out[:, None, None, None] * F_x

    def forward(
        self,
        x_noisy: torch.Tensor,
        sigma: torch.Tensor,
        spatial_cond: torch.Tensor,
    ) -> torch.Tensor:
        return self.denoise(x_noisy, sigma, spatial_cond)


def edm_loss_weight(sigma: torch.Tensor, sigma_data: float = 1.0) -> torch.Tensor:
    """lambda(sigma) = (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2"""
    sd = sigma_data
    return (sigma**2 + sd**2) / ((sigma * sd) ** 2 + 1e-12)


def sample_sigma(
    batch: int,
    device,
    dtype,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
) -> torch.Tensor:
    """Log-uniform-ish EDM sigma via rho-power schedule random draw."""
    u = torch.rand(batch, device=device, dtype=dtype)
    # Invert Karras sigma(t) with t~U(0,1)
    sigma = (
        sigma_max ** (1.0 / rho)
        + u * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))
    ) ** rho
    return sigma.clamp(min=sigma_min, max=sigma_max)


@torch.no_grad()
def edm_sample(
    model: ResidualEDM2D,
    spatial_cond: torch.Tensor,
    *,
    steps: int = 18,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    S_churn: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Heun EDM sampler (deterministic when S_churn=0). Returns leftover sample."""
    device = spatial_cond.device
    dtype = spatial_cond.dtype
    B = spatial_cond.shape[0]
    C = model.target_channels
    H, W = spatial_cond.shape[-2:]

    step_idx = torch.arange(steps, device=device, dtype=dtype)
    t = step_idx / max(steps - 1, 1)
    sigmas = (
        sigma_max ** (1.0 / rho)
        + t * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))
    ) ** rho
    sigmas = torch.cat([sigmas, torch.zeros(1, device=device, dtype=dtype)])

    x = sigmas[0] * torch.randn(
        B, C, H, W, device=device, dtype=dtype, generator=generator
    )
    for i in range(steps):
        sigma_cur = sigmas[i].expand(B)
        sigma_next = sigmas[i + 1].expand(B)
        # optional stochastic churn (off by default for mean-ish decode)
        gamma = min(S_churn / max(steps, 1), 2**0.5 - 1) if S_churn > 0 else 0.0
        if gamma > 0:
            eps = torch.randn(B, C, H, W, device=device, dtype=dtype, generator=generator)
            sigma_hat = sigma_cur * (1 + gamma)
            x = x + torch.sqrt((sigma_hat**2 - sigma_cur**2).clamp_min(0))[:, None, None, None] * eps
        else:
            sigma_hat = sigma_cur

        denoised = model.denoise(x, sigma_hat, spatial_cond)
        d = (x - denoised) / sigma_hat[:, None, None, None].clamp_min(1e-8)
        dt = (sigma_next - sigma_hat)[:, None, None, None]
        x_euler = x + d * dt
        if float(sigmas[i + 1].item()) == 0:
            x = x_euler
        else:
            denoised2 = model.denoise(x_euler, sigma_next, spatial_cond)
            d2 = (x_euler - denoised2) / sigma_next[:, None, None, None].clamp_min(1e-8)
            x = x + 0.5 * dt * (d + d2)
    return x



@torch.no_grad()
def one_step_mean_leftover(
    model: ResidualEDM2D,
    spatial_cond: torch.Tensor,
    *,
    sigma: float = 0.002,
) -> torch.Tensor:
    """Near-zero-noise deterministic decode: denoise x~N(0,sigma^2) at sigma (K=1)."""
    B = spatial_cond.shape[0]
    H, W = spatial_cond.shape[-2:]
    device, dtype = spatial_cond.device, spatial_cond.dtype
    C = model.target_channels
    sig = torch.full((B,), float(sigma), device=device, dtype=dtype)
    x = sig[:, None, None, None] * torch.randn(B, C, H, W, device=device, dtype=dtype)
    return model.denoise(x, sig, spatial_cond)

@torch.no_grad()
def ens_mean_leftover(
    model: ResidualEDM2D,
    spatial_cond: torch.Tensor,
    *,
    K: int = 4,
    steps: int = 18,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
) -> torch.Tensor:
    """K-member ens-mean leftover (default decode for gates)."""
    acc = None
    for k in range(K):
        torch.manual_seed(10_000 + k)
        if spatial_cond.is_cuda:
            torch.cuda.manual_seed_all(10_000 + k)
        s = edm_sample(
            model,
            spatial_cond,
            steps=steps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=rho,
            S_churn=0.0,
            generator=None,
        )
        acc = s if acc is None else acc + s
    return acc / float(K)


def build_model(cfg: dict) -> nn.Module:
    m = cfg.get("model", {})
    name = str(m.get("name", "ResidualEDM2D"))
    if name in {"PhysicsNeMoUNet", "physicsnemo_unet"} or m.get("physicsnemo"):
        raise RuntimeError(
            "PhysicsNeMo stock UNet is 3D; v1-diff uses custom ResidualEDM2D "
            "for the 21×37 Nepal crop."
        )
    if str(m.get("diffusion_backend", "edm")).lower() not in {"edm", "edm_custom_2d"}:
        raise RuntimeError(
            f"v1-diff locked to EDM custom 2D; got {m.get('diffusion_backend')}"
        )
    return ResidualEDM2D(
        target_channels=int(m.get("in_channels_target", 3)),
        cond_channels=int(m.get("cond_channels", 8)),
        base=int(m.get("base_channels", 24)),
        depth=int(m.get("depth", 2)),
        dropout=float(m.get("dropout", 0.10)),
        sigma_data=float(m.get("sigma_data", 1.0)),
    )


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))
