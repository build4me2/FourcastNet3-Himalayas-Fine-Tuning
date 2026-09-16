"""Tier-A v1-diff dataset: leftover = ERA5 − (FCN3 + frozen residual mean)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None
    Dataset = object  # type: ignore

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LIVING_DIR = ROOT / "code" / "tier_a" / "v1_2b_thick2_train"


def _load_living_module(name: str):
    path = LIVING_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"living_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load living module {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"living_v1_2b_{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


_living_dataset = _load_living_module("dataset")
_living_model = _load_living_module("model")

expand = _living_dataset.expand
load_elevation = _living_dataset.load_elevation
load_split_ids = _living_dataset.load_split_ids
load_pair_samples = _living_dataset.load_pair_samples


def load_frozen_residual(cfg: dict, device: str = "cpu"):
    """Load ElevCondResidualUNet from living residual ckpt — FROZEN, eval-only."""
    if torch is None:
        raise RuntimeError("torch required")
    ckpt_path = expand(cfg["data"]["living_residual_ckpt"])
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"living residual ckpt missing: {ckpt_path}")
    # Living residual model defaults match v1.2b thick-2 train config
    model = _living_model.ElevCondResidualUNet(
        in_channels=5,
        out_channels=3,
        base=16,
        depth=3,
        cond_dim=2,
        dropout=0.20,
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, str(ckpt_path)


def _residual_x_from_sample(s: dict[str, Any]) -> "torch.Tensor":
    """Build living residual input: pred(3)+elev_norm+lead_map."""
    lead_map = np.full_like(s["elev_norm"], float(s["lead_norm"]), dtype=np.float32)
    x = np.concatenate(
        [s["pred"], s["elev_norm"][None, ...], lead_map[None, ...]],
        axis=0,
    )
    return torch.from_numpy(x[None, ...])  # (1,5,H,W)


@torch.no_grad()
def attach_frozen_residual_mean(
    samples: list[dict[str, Any]],
    frozen_model,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """Add residual_mean and leftover target to each sample."""
    frozen_model = frozen_model.to(device)
    out: list[dict[str, Any]] = []
    for s in samples:
        x = _residual_x_from_sample(s).to(device)
        res_mean = frozen_model(x)[0].detach().cpu().numpy().astype(np.float32)
        # leftover = ERA5 − (FCN3 + residual_mean) = residual − residual_mean
        leftover = (s["residual"] - res_mean).astype(np.float32)
        mean_path = (s["pred_out"] + res_mean).astype(np.float32)
        s2 = dict(s)
        s2["residual_mean"] = res_mean
        s2["leftover"] = leftover
        s2["mean_path"] = mean_path
        out.append(s2)
    return out


class TierADiffDataset(Dataset):
    """Conditioned leftover diffusion samples."""

    def __init__(self, samples: list[dict[str, Any]]):
        if torch is None:
            raise RuntimeError("torch required")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        lead_map = np.full_like(s["elev_norm"], float(s["lead_norm"]), dtype=np.float32)
        # cond: pred(3) + elev + lead + residual_mean(3) = 8
        cond = np.concatenate(
            [
                s["pred"],
                s["elev_norm"][None, ...],
                lead_map[None, ...],
                s["residual_mean"],
            ],
            axis=0,
        )
        return {
            "cond": torch.from_numpy(cond.astype(np.float32)),
            "leftover": torch.from_numpy(s["leftover"]),
            "residual_mean": torch.from_numpy(s["residual_mean"]),
            "pred_out": torch.from_numpy(s["pred_out"]),
            "mean_path": torch.from_numpy(s["mean_path"]),
            "truth": torch.from_numpy(s["truth"]),
            "elev_m": torch.from_numpy(s["elev_m"]),
            "bin_id": torch.from_numpy(s["bin_id"].astype(np.int64)),
            "lead_h": s["lead_h"],
            "ic_id": s["ic_id"],
        }


def build_loaders(
    cfg: dict,
    splits: dict[str, list[str]],
    *,
    frozen_model=None,
    use_members_train: bool = False,
    device: str = "cpu",
):
    from torch.utils.data import DataLoader

    elev = load_elevation(expand(cfg["data"]["elevation_mask"]))
    pairs_dir = expand(cfg["data"]["pairs_dir"])
    vin = list(cfg["variables_in"])
    vout = list(cfg["variables_out"])
    leads = list(cfg["leads_h"])
    bs = int(cfg["train"]["batch_size"])

    if frozen_model is None:
        frozen_model, _ = load_frozen_residual(cfg, device=device)

    train_s = load_pair_samples(
        pairs_dir, splits["train"], vin, vout, leads, elev, use_members=use_members_train
    )
    val_s = load_pair_samples(
        pairs_dir, splits["val"], vin, vout, leads, elev, use_members=False
    )
    test_s = load_pair_samples(
        pairs_dir, splits["test"], vin, vout, leads, elev, use_members=False
    )

    train_s = attach_frozen_residual_mean(train_s, frozen_model, device=device)
    val_s = attach_frozen_residual_mean(val_s, frozen_model, device=device)
    test_s = attach_frozen_residual_mean(test_s, frozen_model, device=device)

    train_ds = TierADiffDataset(train_s)
    val_ds = TierADiffDataset(val_s)
    test_ds = TierADiffDataset(test_s)

    return {
        "train": DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=False),
        "val": DataLoader(val_ds, batch_size=max(bs, 1), shuffle=False),
        "test": DataLoader(test_ds, batch_size=max(bs, 1), shuffle=False),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "elev": elev,
        "samples": {"train": train_s, "val": val_s, "test": test_s},
        "frozen_model": frozen_model,
    }
