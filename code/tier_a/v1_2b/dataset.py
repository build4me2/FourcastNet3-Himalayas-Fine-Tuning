"""Tier-A dataset: load tier0_holdout FCN3-crop / ERA5 pairs + elevation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None
    Dataset = object  # type: ignore


def expand(p: str | Path) -> Path:
    return Path(str(p)).expanduser().resolve()


def load_elevation(mask_path: Path) -> dict[str, np.ndarray]:
    import netCDF4 as nc

    ds = nc.Dataset(mask_path)
    try:
        elev = np.asarray(ds.variables["elevation_m"][:], dtype=np.float32)
        bin_id = np.asarray(ds.variables["elev_bin_id"][:], dtype=np.int16)
        edges = np.asarray(ds.variables["elev_bin_edges_m"][:], dtype=np.float32)
    finally:
        ds.close()
    elev_norm = (elev - float(np.nanmean(elev))) / (float(np.nanstd(elev)) + 1e-6)
    return {
        "elev_m": elev,
        "elev_norm": elev_norm.astype(np.float32),
        "bin_id": bin_id,
        "edges_m": edges,
    }


def load_split_ids(manifest_path: Path) -> dict[str, list[str]]:
    man = json.loads(manifest_path.read_text())
    out: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for ic in man.get("ics", []):
        if ic.get("status") != "staged":
            continue
        split = ic.get("split", "train")
        if split in out:
            out[split].append(ic["id"])
    return out


def _var_index(variables: list[str], name: str) -> int:
    if name not in variables:
        raise KeyError(f"{name} not in {variables}")
    return variables.index(name)


def load_pair_samples(
    pairs_dir: Path,
    want_ids: list[str],
    variables_in: list[str],
    variables_out: list[str],
    leads_h: list[int],
    elev: dict[str, np.ndarray],
    use_members: bool = False,
) -> list[dict[str, Any]]:
    """Build flat sample list: one row per (ic, lead[, member])."""
    fc_dir = pairs_dir / "forecast"
    tg_dir = pairs_dir / "targets"
    want = set(want_ids)
    samples: list[dict[str, Any]] = []

    for fp in sorted(fc_dir.glob("*_crop.npz")):
        fz = np.load(fp, allow_pickle=True)
        ic_id = str(fz["ic_id"])
        if ic_id not in want:
            continue
        tg_candidates = list(tg_dir.glob(f"{ic_id}_*_era5.npz"))
        if not tg_candidates:
            raise FileNotFoundError(f"no ERA5 target for {ic_id} under {tg_dir}")
        tz = np.load(tg_candidates[0], allow_pickle=True)

        fc_vars = [str(v) for v in fz["variables"]]
        tg_vars = [str(v) for v in tz["variables"]]
        fc_leads = [int(x) for x in fz["leads_h"]]
        pred_mean = np.asarray(fz["ens_mean"], dtype=np.float32)  # (L,C,H,W)
        truth = np.asarray(tz["data"], dtype=np.float32)
        members = None
        if use_members and "members" in fz.files:
            members = np.asarray(fz["members"], dtype=np.float32)  # (M,L,C,H,W)

        for li, lh in enumerate(fc_leads):
            if lh not in leads_h:
                continue
            lead_norm = np.float32((lh - 24.0) / 96.0)  # 24→0, 120→1

            def pack(pred_lc: np.ndarray) -> dict[str, Any]:
                xin = []
                for v in variables_in:
                    xin.append(pred_lc[_var_index(fc_vars, v)])
                x_stack = np.stack(xin, axis=0)  # (Cin,H,W)
                y_res = []
                for v in variables_out:
                    pi = _var_index(fc_vars, v)
                    ti = _var_index(tg_vars, v)
                    y_res.append(truth[li, ti] - pred_lc[pi])
                y_stack = np.stack(y_res, axis=0)
                return {
                    "ic_id": ic_id,
                    "ic_time": str(fz["ic_time"]),
                    "lead_h": int(lh),
                    "lead_norm": lead_norm,
                    "pred": x_stack.astype(np.float32),
                    "residual": y_stack.astype(np.float32),
                    "truth": np.stack(
                        [truth[li, _var_index(tg_vars, v)] for v in variables_out],
                        axis=0,
                    ).astype(np.float32),
                    "pred_out": np.stack(
                        [pred_lc[_var_index(fc_vars, v)] for v in variables_out],
                        axis=0,
                    ).astype(np.float32),
                    "elev_norm": elev["elev_norm"],
                    "elev_m": elev["elev_m"],
                    "bin_id": elev["bin_id"],
                }

            samples.append(pack(pred_mean[li]))
            if members is not None:
                for m in range(members.shape[0]):
                    samples.append(pack(members[m, li]))

    if not samples:
        raise RuntimeError(f"no samples for ids={want_ids} under {pairs_dir}")
    return samples


class TierAResidualDataset(Dataset):
    """Torch dataset wrapping flat residual samples."""

    def __init__(self, samples: list[dict[str, Any]]):
        if torch is None:
            raise RuntimeError("torch required")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        # channels: pred vars + elev_norm + lead_norm map
        lead_map = np.full_like(s["elev_norm"], float(s["lead_norm"]), dtype=np.float32)
        x = np.concatenate(
            [s["pred"], s["elev_norm"][None, ...], lead_map[None, ...]],
            axis=0,
        )
        return {
            "x": torch.from_numpy(x),
            "y_residual": torch.from_numpy(s["residual"]),
            "pred_out": torch.from_numpy(s["pred_out"]),
            "truth": torch.from_numpy(s["truth"]),
            "elev_m": torch.from_numpy(s["elev_m"]),
            "bin_id": torch.from_numpy(s["bin_id"].astype(np.int64)),
            "lead_h": s["lead_h"],
            "ic_id": s["ic_id"],
        }


def build_loaders(cfg: dict, splits: dict[str, list[str]], *, use_members_train: bool = True):
    from torch.utils.data import DataLoader

    elev = load_elevation(expand(cfg["data"]["elevation_mask"]))
    pairs_dir = expand(cfg["data"]["pairs_dir"])
    vin = list(cfg["variables_in"])
    vout = list(cfg["variables_out"])
    leads = list(cfg["leads_h"])
    bs = int(cfg["train"]["batch_size"])

    train_s = load_pair_samples(
        pairs_dir, splits["train"], vin, vout, leads, elev, use_members=use_members_train
    )
    val_s = load_pair_samples(
        pairs_dir, splits["val"], vin, vout, leads, elev, use_members=False
    )
    test_s = load_pair_samples(
        pairs_dir, splits["test"], vin, vout, leads, elev, use_members=False
    )

    train_ds = TierAResidualDataset(train_s)
    val_ds = TierAResidualDataset(val_s)
    test_ds = TierAResidualDataset(test_s)

    return {
        "train": DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=False),
        "val": DataLoader(val_ds, batch_size=bs, shuffle=False),
        "test": DataLoader(test_ds, batch_size=bs, shuffle=False),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "elev": elev,
        "samples": {"train": train_s, "val": val_s, "test": test_s},
    }
