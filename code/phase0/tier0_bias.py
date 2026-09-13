#!/usr/bin/env python3
"""Tier-0 elevation-aware bias baseline (CPU-only plumbing).

Aligned to docs/research/GATE_RECIPE_TIER0_G0.md (Leonard freeze 2026-09-11).

What this does
--------------
- Crops package orography.nc / land_mask.nc to the locked Nepal box
- Builds elevation (m) = Z/g and elevation-bin mask w_R
- Fits a tiny elevation-binned linear bias per variable:
      y_hat = y_fcn3 + a_b + b_b * z
  (also reports per-bin constant bias)
- Writes runs/phase0/tier0/ artifacts + metrics JSON sidecar

What this does NOT do
---------------------
- No FCN3 inference, no CUDA allocation, no GPU use
- No G0 / G1 skill claims from Random-IC smoke crops
- Random-IC ensemble residuals are SMOKE ONLY (self-consistency plumbing);
  skill scoring needs global ERA5 ICs → frozen FCN3 crop + paired truth
  (see GATE_RECIPE §0, §1, §2)

Modes
-----
  --dry-run   Validate config + paths; print plan; exit 0 if scaffolding ready
  --smoke     Fit on synthetic elevation-correlated residual (NOT skill);
              write artifacts; exit 0
  --real      Fit on global-IC→crop FCN3 vs ERA5_interim pairs
              (claim_level=interim_era5; not_skill=false for plumbing;
              cannot claim G1)
  (default)   Prefer --real pairs if present; else smoke with not_skill=true

Usage
-----
  cd ~/fourcastnet
  ~/fcn3-venv/bin/python code/phase0/tier0_bias.py --help
  ~/fcn3-venv/bin/python code/phase0/tier0_bias.py --dry-run
  ~/fcn3-venv/bin/python code/phase0/tier0_bias.py --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None

G = 9.80665  # m s^-2 — convert surface geopotential Z -> elevation meters

# Leonard GATE_RECIPE §2.2 default bins (m)
DEFAULT_ELEV_EDGES_M = [0.0, 1500.0, 3000.0, 4500.0, 9000.0]

VAR_ALIASES = {
    "t2m": ("t2m",),
    "u10m": ("u10m", "u10"),
    "v10m": ("v10m", "v10"),
    "tcwv": ("tcwv",),
}


@dataclass
class Box:
    lon_west: float
    lon_east: float
    lat_south: float
    lat_north: float

    @classmethod
    def from_cfg(cls, cfg: dict) -> "Box":
        b = cfg["region"]["lat_lon_box"]
        return cls(
            lon_west=float(b["lon_west"]),
            lon_east=float(b["lon_east"]),
            lat_south=float(b["lat_south"]),
            lat_north=float(b["lat_north"]),
        )


def _require_yaml() -> None:
    if yaml is None:
        raise RuntimeError("PyYAML required; pip install pyyaml")


def load_config(path: Path) -> dict:
    _require_yaml()
    return yaml.safe_load(path.read_text())


def expand(p: str | Path) -> Path:
    return Path(str(p)).expanduser().resolve()


def crop_indices(lats, lons, box: Box):
    import numpy as np

    lat_idx = np.where((lats >= box.lat_south) & (lats <= box.lat_north))[0]
    lon_idx = np.where((lons >= box.lon_west) & (lons <= box.lon_east))[0]
    if lat_idx.size == 0 or lon_idx.size == 0:
        raise RuntimeError(
            f"empty crop for box={box} lat=[{lats.min()},{lats.max()}] "
            f"lon=[{lons.min()},{lons.max()}]"
        )
    return slice(int(lat_idx[0]), int(lat_idx[-1]) + 1), slice(int(lon_idx[0]), int(lon_idx[-1]) + 1)


def load_orography_crop(oro_path: Path, land_path: Path | None, box: Box) -> dict:
    """Crop global package orography to Nepal box. CPU / netCDF only."""
    import numpy as np
    import netCDF4 as nc

    ds = nc.Dataset(oro_path)
    try:
        lats = np.asarray(ds.variables["latitude"][:], dtype=np.float64)
        lons = np.asarray(ds.variables["longitude"][:], dtype=np.float64)
        Z = np.asarray(ds.variables["Z"][0], dtype=np.float32)  # (lat, lon)
    finally:
        ds.close()

    lat_sl, lon_sl = crop_indices(lats, lons, box)
    Z_c = Z[lat_sl, lon_sl]
    lat_c = lats[lat_sl]
    lon_c = lons[lon_sl]
    elev_m = (Z_c / G).astype(np.float32)

    land = None
    if land_path is not None and land_path.is_file():
        ds = nc.Dataset(land_path)
        try:
            land = np.asarray(ds.variables["LSM"][0], dtype=np.float32)[lat_sl, lon_sl]
        finally:
            ds.close()

    return {
        "lat": lat_c,
        "lon": lon_c,
        "Z": Z_c,
        "elev_m": elev_m,
        "land_mask": land,
        "shape": tuple(elev_m.shape),
        "oro_path": str(oro_path),
        "land_path": str(land_path) if land_path else None,
    }


def build_elevation_bins(elev_m, edges_m: list[float], land_mask=None) -> dict:
    """Build per-bin boolean masks w_R and integer bin id map.

    Bin k covers [edges[k], edges[k+1]) except the last which is closed on the right.
    Points outside [edges[0], edges[-1]] or over ocean (if land_mask given & <0.5)
    get bin_id = -1 (excluded from fit).
    """
    import numpy as np

    edges = np.asarray(edges_m, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("elevation_bins_m need ≥2 edges")
    h, w = elev_m.shape
    bin_id = np.full((h, w), -1, dtype=np.int16)
    masks = []
    labels = []
    for k in range(len(edges) - 1):
        lo, hi = edges[k], edges[k + 1]
        if k == len(edges) - 2:
            m = (elev_m >= lo) & (elev_m <= hi)
            label = f"[{lo:.0f},{hi:.0f}]m"
        else:
            m = (elev_m >= lo) & (elev_m < hi)
            label = f"[{lo:.0f},{hi:.0f})m"
        if land_mask is not None:
            m = m & (land_mask >= 0.5)
        bin_id[m] = k
        masks.append(m)
        labels.append(label)
    n_per_bin = [int(m.sum()) for m in masks]
    return {
        "edges_m": edges.tolist(),
        "labels": labels,
        "masks": masks,  # list of bool (H,W)
        "bin_id": bin_id,
        "n_per_bin": n_per_bin,
        "n_valid": int((bin_id >= 0).sum()),
        "n_excluded": int((bin_id < 0).sum()),
    }


def fit_binned_linear(y_pred, y_true, elev_m, bins: dict, var_name: str) -> dict:
    """Fit per-bin: residual r = y_true - y_pred ≈ a_b + b_b * z.

    Also records constant-bias-only (a_b with b_b=0) for the headline Tier-0.
    """
    import numpy as np

    r = (y_true - y_pred).astype(np.float64)
    z = elev_m.astype(np.float64)
    # flatten time dims if present: expect (... , H, W) with last two spatial
    if r.ndim == 2:
        r2, z2 = r, z
        pred2, true2 = y_pred, y_true
    else:
        h, w = elev_m.shape
        if r.shape[-2:] != (h, w):
            raise ValueError(f"{var_name}: residual spatial {r.shape[-2:]} != elev {elev_m.shape}")
        r2 = r.reshape(-1, h, w)
        pred2 = np.asarray(y_pred, dtype=np.float64).reshape(-1, h, w)
        true2 = np.asarray(y_true, dtype=np.float64).reshape(-1, h, w)
        z2 = z

    per_bin = []
    corrected_const = np.array(pred2, dtype=np.float64, copy=True)
    corrected_lin = np.array(pred2, dtype=np.float64, copy=True)

    for k, (mask, label) in enumerate(zip(bins["masks"], bins["labels"])):
        if r2.ndim == 2:
            rr = r2[mask]
            zz = z2[mask]
        else:
            # broadcast mask over time
            rr = r2[:, mask].ravel()
            zz = np.broadcast_to(z2[mask], r2[:, mask].shape).ravel()
        n = int(rr.size)
        entry: dict[str, Any] = {
            "bin": k,
            "label": label,
            "n": n,
            "n_grid": int(mask.sum()),
        }
        if n < 2:
            entry.update({"a_const": None, "a_lin": None, "b_lin": None, "rmse_raw": None})
            per_bin.append(entry)
            continue
        a_const = float(np.mean(rr))
        # linear: r ≈ a + b z  via least squares
        A = np.column_stack([np.ones(n), zz])
        try:
            coef, _, _, _ = np.linalg.lstsq(A, rr, rcond=None)
            a_lin, b_lin = float(coef[0]), float(coef[1])
        except Exception:
            a_lin, b_lin = a_const, 0.0
        rmse_raw = float(np.sqrt(np.mean(rr**2)))
        resid_const = rr - a_const
        resid_lin = rr - (a_lin + b_lin * zz)
        entry.update(
            {
                "a_const": a_const,
                "a_lin": a_lin,
                "b_lin": b_lin,
                "mean_elev_m": float(np.mean(zz)),
                "rmse_raw": rmse_raw,
                "rmse_after_const": float(np.sqrt(np.mean(resid_const**2))),
                "rmse_after_lin": float(np.sqrt(np.mean(resid_lin**2))),
                "mean_bias_raw": float(np.mean(rr)),
            }
        )
        per_bin.append(entry)
        # apply corrections on spatial mask
        if r2.ndim == 2:
            corrected_const[mask] = pred2[mask] + a_const
            corrected_lin[mask] = pred2[mask] + (a_lin + b_lin * z2[mask])
        else:
            corrected_const[:, mask] = pred2[:, mask] + a_const
            corrected_lin[:, mask] = pred2[:, mask] + (
                a_lin + b_lin * z2[mask]
            )

    def _rmse(a, b, mask_2d):
        if a.ndim == 2:
            d = (a - b)[mask_2d]
        else:
            d = (a - b)[:, mask_2d].ravel()
        if d.size == 0:
            return None
        return float(np.sqrt(np.mean(np.asarray(d, dtype=np.float64) ** 2)))

    valid = bins["bin_id"] >= 0
    true_ref = true2
    return {
        "variable": var_name,
        "model": "elevation_binned_linear",
        "headline": "per-bin constant bias (a_b); linear a_b+b_b*z reported alongside",
        "per_bin": per_bin,
        "rmse_raw_valid": _rmse(pred2, true_ref, valid),
        "rmse_after_const_valid": _rmse(corrected_const, true_ref, valid),
        "rmse_after_lin_valid": _rmse(corrected_lin, true_ref, valid),
        "n_samples_valid": int(valid.sum() if r2.ndim == 2 else valid.sum() * r2.shape[0]),
    }


def make_synthetic_pair(elev_m, land_mask, seed: int = 0) -> dict:
    """Synthetic residual with known elevation correlation — SMOKE ONLY.

    Simulates FCN3-like field and 'truth' = pred + elev-dependent bias + noise.
    NOT a skill claim. Documents plumbing of fit path.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    h, w = elev_m.shape
    z = elev_m.astype(np.float64)
    z_n = (z - z.mean()) / (z.std() + 1e-6)
    land = (land_mask >= 0.5) if land_mask is not None else np.ones_like(z, dtype=bool)

    # fake "FCN3" t2m around 280 K with weak elev lapse already partial
    t_pred = 295.0 - 0.004 * z + rng.normal(0, 0.5, size=z.shape)
    # planted bias: +2 K lowland, growing colder high (under-lapse), + noise
    planted = 1.5 - 0.003 * z + rng.normal(0, 0.3, size=z.shape)
    t_true = t_pred + planted
    t_pred = np.where(land, t_pred, t_pred).astype(np.float32)
    t_true = np.where(land, t_true, t_true).astype(np.float32)

    u_pred = rng.normal(0, 2, size=z.shape).astype(np.float32)
    u_true = (u_pred + 0.5 * z_n + rng.normal(0, 0.2, size=z.shape)).astype(np.float32)
    v_pred = rng.normal(0, 2, size=z.shape).astype(np.float32)
    v_true = (v_pred - 0.3 * z_n + rng.normal(0, 0.2, size=z.shape)).astype(np.float32)

    return {
        "mode": "synthetic_smoke",
        "not_skill": True,
        "disclaimer": (
            "Synthetic elevation-correlated residual for plumbing only. "
            "NOT a skill claim. G0/Tier-0 skill needs global ERA5 ICs + paired truth "
            "(GATE_RECIPE_TIER0_G0.md)."
        ),
        "seed": seed,
        "fields": {
            "t2m": {"pred": t_pred, "true": t_true},
            "u10m": {"pred": u_pred, "true": u_true},
            "v10m": {"pred": v_pred, "true": v_true},
        },
    }


def try_load_random_ic_self_consistency(infer_dir: Path, elev_m) -> dict | None:
    """Optional: ensemble-member residuals vs ensemble mean from Random-IC smoke.

    Still NOT skill — Random IC forbidden for gates (GATE_RECIPE §0).
    Used only to exercise real tensor I/O path when --allow-random-ic-plumbing.
    """
    import numpy as np

    npz = infer_dir / "ensemble_crop.npz"
    if not npz.is_file():
        return None
    ens = np.load(npz, allow_pickle=True)
    data = ens["data"]  # (M, T, C, H, W)
    variables = [str(v) for v in ens["variables"]]
    if data.shape[-2:] != elev_m.shape:
        return {
            "ok": False,
            "error": f"ensemble spatial {data.shape[-2:]} != elev {elev_m.shape}",
        }
    mean = data.mean(axis=0)  # (T, C, H, W)
    # use member 0 vs mean at lead index 4 (~24h if 6h steps) if available
    lead_i = min(4, data.shape[1] - 1)
    fields = {}
    for name, aliases in VAR_ALIASES.items():
        idx = None
        for a in aliases:
            if a in variables:
                idx = variables.index(a)
                break
        if idx is None:
            continue
        pred = mean[lead_i, idx]
        true = data[0, lead_i, idx]  # self-consistency target = one member
        fields[name] = {"pred": pred.astype(np.float32), "true": true.astype(np.float32)}
    if not fields:
        return None
    return {
        "mode": "random_ic_self_consistency_plumbing",
        "not_skill": True,
        "disclaimer": (
            "Random-IC ensemble self-consistency residual — plumbing only. "
            "FORBIDDEN for G0/G1/Tier-0 skill (GATE_RECIPE §0). "
            "Needs ≥8 global ERA5 ICs + g0_base_results.json before gate numbers."
        ),
        "infer_dir": str(infer_dir),
        "lead_index": lead_i,
        "ensemble_shape": list(data.shape),
        "ic_source": str(ens["ic_source"]) if "ic_source" in ens else "unknown",
        "fields": fields,
    }



def load_real_pairs(pairs_dir: Path, variables: list[str], leads_want: list[int], elev_shape) -> dict:
    """Load global-IC→crop forecasts paired with ERA5_interim targets.

    Files written by tier0_crop_rollout.py + tier0_era5_targets.py.
    Pair key = ic_id. Spatial grid must match elevation crop.
    """
    import numpy as np

    pairs_dir = Path(pairs_dir)
    fc_dir = pairs_dir / "forecast"
    tg_dir = pairs_dir / "targets"
    if not fc_dir.is_dir() or not tg_dir.is_dir():
        return {
            "ok": False,
            "error": f"missing forecast/ or targets/ under {pairs_dir}",
        }
    fc_files = sorted(fc_dir.glob("*_crop.npz"))
    tg_by_id = {}
    for pth in tg_dir.glob("*_era5.npz"):
        z = np.load(pth, allow_pickle=True)
        ic_id = str(z["ic_id"])
        tg_by_id[ic_id] = (pth, z)

    if not fc_files:
        return {"ok": False, "error": f"no *_crop.npz in {fc_dir}"}

    # accumulate per-variable lists of (n_ic, n_lead, H, W)
    per_ic = []
    pred_acc: dict[str, list] = {v: [] for v in variables}
    true_acc: dict[str, list] = {v: [] for v in variables}
    used_ics = []
    lat = lon = None
    leads_h = None

    for fp in fc_files:
        fz = np.load(fp, allow_pickle=True)
        ic_id = str(fz["ic_id"])
        if ic_id not in tg_by_id:
            return {"ok": False, "error": f"no ERA5 target for {ic_id}"}
        tp, tz = tg_by_id[ic_id]
        fc_vars = [str(v) for v in fz["variables"]]
        tg_vars = [str(v) for v in tz["variables"]]
        fc_leads = [int(x) for x in fz["leads_h"]]
        tg_leads = [int(x) for x in tz["leads_h"]]
        if fc_leads != tg_leads:
            return {"ok": False, "error": f"{ic_id} lead mismatch {fc_leads} vs {tg_leads}"}
        if leads_h is None:
            leads_h = fc_leads
        elif leads_h != fc_leads:
            return {"ok": False, "error": f"{ic_id} lead set {fc_leads} != {leads_h}"}
        # ens_mean (L, C, H, W); target data (L, C, H, W)
        pred = np.asarray(fz["ens_mean"], dtype=np.float32)
        truth = np.asarray(tz["data"], dtype=np.float32)
        if pred.shape[-2:] != tuple(elev_shape) or truth.shape[-2:] != tuple(elev_shape):
            return {
                "ok": False,
                "error": (
                    f"{ic_id} spatial pred={pred.shape[-2:]} true={truth.shape[-2:]} "
                    f"elev={elev_shape}"
                ),
            }
        if pred.shape[:2] != truth.shape[:2]:
            return {"ok": False, "error": f"{ic_id} pred {pred.shape} vs true {truth.shape}"}
        lat = np.asarray(fz["lat"], dtype=np.float64)
        lon = np.asarray(fz["lon"], dtype=np.float64)
        for v in variables:
            if v not in fc_vars or v not in tg_vars:
                continue
            pi, ti = fc_vars.index(v), tg_vars.index(v)
            pred_acc[v].append(pred[:, pi])
            true_acc[v].append(truth[:, ti])
        used_ics.append(
            {
                "ic_id": ic_id,
                "ic_time": str(fz["ic_time"]),
                "forecast": str(fp),
                "target": str(tp),
                "finite_pred": bool(np.isfinite(pred).all()),
                "finite_true": bool(np.isfinite(truth).all()),
            }
        )
        per_ic.append(ic_id)

    fields: dict[str, dict] = {}
    for v in variables:
        if not pred_acc[v]:
            continue
        # (n_ic, L, H, W)
        fields[v] = {
            "pred": np.stack(pred_acc[v], axis=0),
            "true": np.stack(true_acc[v], axis=0),
        }
    if not fields:
        return {"ok": False, "error": "no overlapping variables in pairs"}
    if leads_want and leads_h is not None:
        missing = [lh for lh in leads_want if lh not in leads_h]
        if missing:
            return {"ok": False, "error": f"pairs missing leads {missing}; have {leads_h}"}
    return {
        "ok": True,
        "mode": "global_ic_crop_vs_era5_interim",
        "not_skill": False,
        "claim_level": "interim_era5",
        "target": "ERA5_interim",
        "disclaimer": (
            "Real global-ERA5-IC → frozen FCN3 Nepal crop vs ARCO ERA5 at the "
            "same valid times. target=ERA5_interim — plumbing/skill-vs-ERA5 only; "
            "NOT obs-aware G1. Year-split not frozen; all staged ICs are 2018–2021 "
            "(suggested later: train 2018–21 / val 2022 / test 2023–24)."
        ),
        "ics": used_ics,
        "leads_h": leads_h,
        "lat": lat,
        "lon": lon,
        "fields": fields,
        "n_ics": len(used_ics),
        "pairs_dir": str(pairs_dir),
    }


def subset_lead(fields: dict, leads_h: list[int], lead: int) -> dict:
    """Slice pair fields to one lead (axis=1)."""
    if lead not in leads_h:
        raise KeyError(lead)
    i = leads_h.index(lead)
    out = {}
    for v, d in fields.items():
        out[v] = {"pred": d["pred"][:, i], "true": d["true"][:, i]}
    return out


def save_elevation_mask_nc(path: Path, oro: dict, bins: dict) -> None:
    import numpy as np
    import netCDF4 as nc

    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = oro["elev_m"].shape
    ds = nc.Dataset(path, "w", format="NETCDF4")
    try:
        ds.createDimension("latitude", h)
        ds.createDimension("longitude", w)
        ds.createDimension("bin", len(bins["labels"]))
        ds.createDimension("edge", len(bins["edges_m"]))
        lat = ds.createVariable("latitude", "f8", ("latitude",))
        lon = ds.createVariable("longitude", "f8", ("longitude",))
        elev = ds.createVariable("elevation_m", "f4", ("latitude", "longitude"))
        zvar = ds.createVariable("surface_geopotential", "f4", ("latitude", "longitude"))
        bid = ds.createVariable("elev_bin_id", "i2", ("latitude", "longitude"))
        edges = ds.createVariable("elev_bin_edges_m", "f4", ("edge",))
        nbin = ds.createVariable("n_per_bin", "i4", ("bin",))
        w_r = ds.createVariable("w_R", "f4", ("bin", "latitude", "longitude"))
        lat[:] = oro["lat"]
        lon[:] = oro["lon"]
        elev[:] = oro["elev_m"]
        zvar[:] = oro["Z"]
        bid[:] = bins["bin_id"]
        edges[:] = np.asarray(bins["edges_m"], dtype=np.float32)
        nbin[:] = np.asarray(bins["n_per_bin"], dtype=np.int32)
        for k, m in enumerate(bins["masks"]):
            w_r[k, :, :] = m.astype(np.float32)
        if oro["land_mask"] is not None:
            lsm = ds.createVariable("land_mask", "f4", ("latitude", "longitude"))
            lsm[:] = oro["land_mask"]
        ds.setncattr("title", "Nepal-box elevation mask w_R for Tier-0 bias")
        ds.setncattr("source_orography", oro["oro_path"])
        ds.setncattr("g", G)
        ds.setncattr("recipe", "GATE_RECIPE_TIER0_G0.md")
        ds.setncattr(
            "bin_labels",
            json.dumps(bins["labels"]),
        )
    finally:
        ds.close()


def save_bias_maps_nc(path: Path, oro: dict, bins: dict, fits: list[dict]) -> None:
    """Write per-variable constant-bias map (a_b stamped on grid) + linear coeffs."""
    import numpy as np
    import netCDF4 as nc

    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = oro["elev_m"].shape
    ds = nc.Dataset(path, "w", format="NETCDF4")
    try:
        ds.createDimension("latitude", h)
        ds.createDimension("longitude", w)
        ds.createDimension("bin", len(bins["labels"]))
        lat = ds.createVariable("latitude", "f8", ("latitude",))
        lon = ds.createVariable("longitude", "f8", ("longitude",))
        elev = ds.createVariable("elevation_m", "f4", ("latitude", "longitude"))
        lat[:] = oro["lat"]
        lon[:] = oro["lon"]
        elev[:] = oro["elev_m"]

        def _stamp(fit):
            a_map = np.full((h, w), np.nan, dtype=np.float32)
            b_map = np.full((h, w), np.nan, dtype=np.float32)
            for entry in fit["per_bin"]:
                k = entry["bin"]
                mask = bins["masks"][k]
                if entry.get("a_const") is not None:
                    a_map[mask] = entry["a_const"]
                if entry.get("b_lin") is not None:
                    b_map[mask] = entry["b_lin"]
            return a_map, b_map

        # Headline maps = pooled (or only) fit per variable; per-lead as _lXXX suffix
        written = set()
        headline = {}
        for fit in fits:
            scope = str(fit.get("scope", "pooled"))
            if scope.startswith("pooled") or scope in ("smoke", "t2m_only", "synthetic_smoke"):
                headline[fit["variable"]] = fit
        if not headline:
            # first occurrence of each var
            for fit in fits:
                headline.setdefault(fit["variable"], fit)
        for var, fit in headline.items():
            a_map, b_map = _stamp(fit)
            av = ds.createVariable(f"{var}_bias_const", "f4", ("latitude", "longitude"))
            bv = ds.createVariable(f"{var}_bias_lin_slope", "f4", ("latitude", "longitude"))
            av[:] = a_map
            bv[:] = b_map
            av.setncattr("long_name", f"{var} per-bin constant bias a_b (truth - pred); scope={fit.get('scope')}")
            bv.setncattr("long_name", f"{var} per-bin linear slope b_b in a_b + b_b*z")
            av.setncattr("scope", str(fit.get("scope", "")))
            written.add(f"{var}_bias_const")
        for fit in fits:
            lead = fit.get("lead_h")
            if lead in (None, "pooled", "smoke", "fallback"):
                continue
            try:
                suffix = f"l{int(lead):03d}"
            except (TypeError, ValueError):
                continue
            var = fit["variable"]
            a_name = f"{var}_{suffix}_bias_const"
            if a_name in written:
                continue
            a_map, b_map = _stamp(fit)
            av = ds.createVariable(a_name, "f4", ("latitude", "longitude"))
            bv = ds.createVariable(f"{var}_{suffix}_bias_lin_slope", "f4", ("latitude", "longitude"))
            av[:] = a_map
            bv[:] = b_map
            av.setncattr("lead_h", int(lead))
            written.add(a_name)
        ds.setncattr("title", "Tier-0 elevation-binned bias maps")
        ds.setncattr("recipe", "GATE_RECIPE_TIER0_G0.md")
        ds.setncattr("target", "ERA5_interim")
        ds.setncattr("claim_level", "interim_era5")
        ds.setncattr("g1_claimable", "false")
        ds.setncattr("not_skill", "false")
        ds.setncattr("year_split_frozen", "false")
    finally:
        ds.close()


def write_metrics_csv(path: Path, fits: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "lead_h,scope,variable,bin,label,n,a_const,a_lin,b_lin,rmse_raw,rmse_after_const,rmse_after_lin,mean_elev_m"
    ]
    for fit in fits:
        lead = fit.get("lead_h", "pooled")
        scope = fit.get("scope", "pooled")
        for e in fit["per_bin"]:
            lines.append(
                ",".join(
                    [
                        str(lead),
                        str(scope),
                        fit["variable"],
                        str(e["bin"]),
                        e["label"].replace(",", ";"),
                        str(e["n"]),
                        _fmt(e.get("a_const")),
                        _fmt(e.get("a_lin")),
                        _fmt(e.get("b_lin")),
                        _fmt(e.get("rmse_raw")),
                        _fmt(e.get("rmse_after_const")),
                        _fmt(e.get("rmse_after_lin")),
                        _fmt(e.get("mean_elev_m")),
                    ]
                )
            )
    path.write_text("\n".join(lines) + "\n")


def _fmt(x) -> str:
    if x is None:
        return ""
    return f"{float(x):.6g}"


def check_paths(cfg: dict) -> dict:
    data = cfg.get("data", {})
    out = {
        "orography": expand(data.get("orography", "~/fourcastnet/models/fourcastnet3/orography.nc")),
        "land_mask": expand(data.get("land_mask", "~/fourcastnet/models/fourcastnet3/land_mask.nc")),
        "output_dir": expand(data.get("output_dir", cfg.get("output_dir", "~/fourcastnet/runs/phase0/tier0"))),
        "elevation_mask_out": expand(
            data.get("elevation_mask", "~/fourcastnet/data/masks/w_R_elevation.nc")
        ),
        "infer_dir": expand(data.get("infer_dir", "~/fourcastnet/data/cache/phase0_infer")),
        "forecast_zarr": expand(data.get("forecast_zarr", "~/fourcastnet/data/cache/phase0_infer/forecast.zarr")),
        "target_zarr": expand(data.get("target_zarr", "~/fourcastnet/data/era5/target_crop.zarr")),
        "pairs_dir": expand(data.get("pairs_dir", "~/fourcastnet/runs/phase0/tier0/pairs")),
        "era5_raw_dir": expand(data.get("era5_raw_dir", "~/fourcastnet/data/era5/raw")),
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Tier-0 elevation-aware bias baseline (CPU-only; GATE_RECIPE aligned)"
    )
    ap.add_argument(
        "--config",
        type=Path,
        default=Path.home() / "fourcastnet" / "configs" / "tier0_bias.yaml",
    )
    ap.add_argument("--dry-run", action="store_true", help="Validate paths/config; no fit writes")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Fit on synthetic elev-correlated residual (NOT skill); write artifacts",
    )
    ap.add_argument(
        "--allow-random-ic-plumbing",
        action="store_true",
        help="Also exercise Random-IC ensemble self-consistency I/O (still not_skill)",
    )
    ap.add_argument(
        "--real",
        action="store_true",
        help="Fit on global-IC crop vs ERA5_interim pairs (claim_level=interim_era5)",
    )
    ap.add_argument(
        "--pairs-dir",
        type=Path,
        default=None,
        help="Override pairs dir (forecast/ + targets/)",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    report: dict[str, Any] = {
        "tier": "0",
        "script": "code/phase0/tier0_bias.py",
        "recipe": "docs/research/GATE_RECIPE_TIER0_G0.md",
        "cpu_only": True,
        "gpu_used": False,
        "fcn3_inference": False,
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dry_run": bool(args.dry_run),
        "smoke": bool(args.smoke),
        "ok": False,
    }

    if not args.config.is_file():
        report["error"] = f"missing config {args.config}"
        print(json.dumps(report, indent=2))
        return 1

    try:
        cfg = load_config(args.config)
    except Exception as e:
        report["error"] = f"config load failed: {e}"
        print(json.dumps(report, indent=2))
        return 1

    locks = cfg.get("locks", {})
    report["locks"] = locks
    report["config"] = str(args.config)
    box = Box.from_cfg(cfg)
    report["box"] = {
        "lon_west": box.lon_west,
        "lon_east": box.lon_east,
        "lat_south": box.lat_south,
        "lat_north": box.lat_north,
    }
    paths = check_paths(cfg)
    report["paths"] = {k: str(v) for k, v in paths.items()}
    report["path_exists"] = {
        "orography": paths["orography"].is_file(),
        "land_mask": paths["land_mask"].is_file(),
        "infer_dir": paths["infer_dir"].is_dir(),
        "ensemble_crop_npz": (paths["infer_dir"] / "ensemble_crop.npz").is_file(),
        "forecast_zarr": paths["forecast_zarr"].exists(),
        "target_zarr": paths["target_zarr"].exists(),
    }

    edges = list(cfg.get("elevation_bins_m", DEFAULT_ELEV_EDGES_M))
    # accept either edges or mid-style; if user gave old [0,500,1500,...] keep as edges
    report["elevation_bins_m"] = edges
    variables = list(cfg.get("variables", ["t2m", "u10m", "v10m"]))
    # drop tcwv from required smoke if present — still allowed
    report["variables"] = variables
    report["method"] = cfg.get("method", "elevation_binned_linear")
    report["gate_notes"] = {
        "random_ic_for_skill": False,
        "nepal_crop_as_ic": False,
        "g0_requires": "≥8 global ERA5 ICs (721×1440×72) + g0_base_results.json on frozen FCN3",
        "tier0_skill_requires": "global ERA5 IC → FCN3 crop forecasts paired with target; label ERA5 target as interim",
        "gpu_blocked_pending": "Manisha greenlight for next GPU job",
    }

    blockers = []
    if not locks.get("geo_box_frozen"):
        blockers.append("geo_box lock not frozen in config")
    # years_frozen is intentionally NOT a hard blocker — Manisha: freeze later
    if not paths["orography"].is_file():
        blockers.append(f"missing orography: {paths['orography']}")
    report["blockers_hard"] = blockers
    report["year_split_frozen"] = bool(locks.get("years_frozen", False))
    report["year_split_note"] = (
        "Not frozen. Suggested later: train 2018-2021 / val 2022 / test 2023-2024. "
        "This run uses all staged G0 ICs (2018-2021 only); no 2022+ holdout."
    )

    # Soft blockers for G1 / claimable skill
    soft = []
    pairs_dir = expand(args.pairs_dir) if args.pairs_dir else paths["pairs_dir"]
    report["pairs_dir"] = str(pairs_dir)
    report["path_exists"]["pairs_forecast"] = (pairs_dir / "forecast").is_dir() and any(
        (pairs_dir / "forecast").glob("*_crop.npz")
    )
    report["path_exists"]["pairs_targets"] = (pairs_dir / "targets").is_dir() and any(
        (pairs_dir / "targets").glob("*_era5.npz")
    )
    if not report["path_exists"]["pairs_forecast"]:
        soft.append("no global-IC→crop forecast npz yet (run tier0_crop_rollout.py)")
    if not report["path_exists"]["pairs_targets"]:
        soft.append("no ERA5_interim target npz yet (run tier0_era5_targets.py)")
    if not locks.get("years_frozen"):
        soft.append("year-split not frozen — cannot claim G1; interim table only")
    soft.append("target=ERA5_interim cannot be used as G1 obs-aware pass")
    report["blockers_skill"] = soft

    if blockers:
        report["status"] = "blocked"
        report["ok"] = False
        print(json.dumps(report, indent=2))
        return 1

    # --- dry-run: load oro crop briefly to prove path, no artifact requirement ---
    t0 = time.time()
    oro = load_orography_crop(paths["orography"], paths["land_mask"], box)
    bins = build_elevation_bins(oro["elev_m"], edges, oro["land_mask"])
    report["orography_crop"] = {
        "shape": list(oro["shape"]),
        "elev_m_min": float(oro["elev_m"].min()),
        "elev_m_max": float(oro["elev_m"].max()),
        "elev_m_mean": float(oro["elev_m"].mean()),
        "n_per_bin": bins["n_per_bin"],
        "bin_labels": bins["labels"],
        "n_valid": bins["n_valid"],
        "n_excluded": bins["n_excluded"],
    }
    report["load_oro_s"] = round(time.time() - t0, 3)

    if args.dry_run:
        report["status"] = "dry_run_ok"
        report["ok"] = True
        report["would_write"] = {
            "elevation_mask": str(paths["elevation_mask_out"]),
            "output_dir": str(paths["output_dir"]),
            "metrics_json": str(paths["output_dir"] / "tier0_metrics.json"),
            "metrics_csv": str(paths["output_dir"] / "tier0_metrics.csv"),
            "bias_maps": str(paths["output_dir"] / "tier0_bias_maps.nc"),
        }
        print(json.dumps(report, indent=2))
        return 0

    # Choose data pair
    pair = None
    want_real = bool(args.real) or (
        report["path_exists"].get("pairs_forecast")
        and report["path_exists"].get("pairs_targets")
        and not args.smoke
    )
    if want_real and not args.smoke:
        pair = load_real_pairs(
            pairs_dir,
            variables,
            list(cfg.get("leads_h", [24, 72, 120])),
            oro["elev_m"].shape,
        )
        if not pair.get("ok"):
            report["error"] = f"real pair load failed: {pair.get('error')}"
            report["status"] = "blocked_pairs"
            print(json.dumps(report, indent=2, default=str))
            return 1
    elif args.smoke or not want_real:
        pair = make_synthetic_pair(oro["elev_m"], oro["land_mask"], seed=args.seed)
        if args.allow_random_ic_plumbing:
            ric = try_load_random_ic_self_consistency(paths["infer_dir"], oro["elev_m"])
            report["random_ic_plumbing"] = {
                k: v for k, v in (ric or {"ok": False, "error": "no ensemble"}).items() if k != "fields"
            }
    else:
        report["error"] = "no pair source"
        print(json.dumps(report, indent=2))
        return 1

    report["pair_mode"] = pair["mode"]
    report["not_skill"] = pair["not_skill"]
    report["claim_level"] = pair.get("claim_level", "smoke" if pair["not_skill"] else "interim_era5")
    report["target"] = pair.get("target", "synthetic" if pair["not_skill"] else "ERA5_interim")
    report["disclaimer"] = pair["disclaimer"]
    if pair.get("ics"):
        report["pair_ics"] = pair["ics"]
        report["pair_leads_h"] = pair.get("leads_h")

    def _fit_fields(fields_map, lead_h, scope):
        out = []
        for vname, flds in fields_map.items():
            if variables and vname not in variables:
                continue
            fit = fit_binned_linear(flds["pred"], flds["true"], oro["elev_m"], bins, vname)
            fit["lead_h"] = lead_h
            fit["scope"] = scope
            out.append(fit)
        return out

    fits = []
    if pair.get("leads_h") and pair["mode"] == "global_ic_crop_vs_era5_interim":
        for lh in pair["leads_h"]:
            fits.extend(_fit_fields(subset_lead(pair["fields"], pair["leads_h"], lh), lh, f"lead_{lh}h"))
        fits.extend(_fit_fields(pair["fields"], "pooled", "pooled_all_leads"))
    else:
        fits.extend(_fit_fields(pair["fields"], "smoke", pair["mode"]))
    if not fits and "t2m" in pair["fields"]:
        fits.extend(_fit_fields({"t2m": pair["fields"]["t2m"]}, pair.get("leads_h", ["na"])[0] if False else "fallback", "t2m_only"))

    report["fits_summary"] = [
        {
            "variable": f["variable"],
            "rmse_raw_valid": f["rmse_raw_valid"],
            "rmse_after_const_valid": f["rmse_after_const_valid"],
            "rmse_after_lin_valid": f["rmse_after_lin_valid"],
            "n_bins_fit": sum(1 for e in f["per_bin"] if e.get("a_const") is not None),
        }
        for f in fits
    ]

    # Write artifacts
    out_dir = paths["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    paths["elevation_mask_out"].parent.mkdir(parents=True, exist_ok=True)
    save_elevation_mask_nc(paths["elevation_mask_out"], oro, bins)
    bias_maps = out_dir / "tier0_bias_maps.nc"
    save_bias_maps_nc(bias_maps, oro, bins, fits)
    metrics_csv = out_dir / "tier0_metrics.csv"
    write_metrics_csv(metrics_csv, fits)
    metrics_json = out_dir / "tier0_metrics.json"
    is_real = pair.get("mode") == "global_ic_crop_vs_era5_interim"
    metrics_body = {
        "tier": "0",
        "not_skill": False if is_real else True,
        "claim_level": "interim_era5" if is_real else "smoke_not_skill",
        "target": "ERA5_interim" if is_real else "synthetic_or_random_ic",
        "g1_claimable": False,
        "year_split_frozen": False,
        "year_split_note": report.get("year_split_note"),
        "pair_mode": pair["mode"],
        "disclaimer": pair["disclaimer"],
        "recipe": "GATE_RECIPE_TIER0_G0.md",
        "g0_base_call": "docs/research/G0_BASE_CALL.md (finite PASS; claimable G0 NOT YET)",
        "box": report["box"],
        "elevation_bins_m": edges,
        "bin_labels": bins["labels"],
        "n_per_bin": bins["n_per_bin"],
        "orography_crop": report["orography_crop"],
        "leads_h": pair.get("leads_h") or cfg.get("leads_h"),
        "n_ics": pair.get("n_ics"),
        "ics": pair.get("ics"),
        "fits": fits,
        "artifacts": {
            "elevation_mask": str(paths["elevation_mask_out"]),
            "bias_maps": str(bias_maps),
            "metrics_csv": str(metrics_csv),
            "metrics_json": str(metrics_json),
        },
        "cpu_only": True,
        "gpu_used": False,
        "fcn3_inference_this_script": False,
        "skill_blocked_on": report["blockers_skill"],
    }
    # per_bin masks are not JSON-serializable — fits already has numeric fields only
    metrics_json.write_text(json.dumps(metrics_body, indent=2, default=str))
    # also drop a small README in run dir
    (out_dir / "README.md").write_text(
        "\n".join(
            [
                "# Tier-0 run artifacts",
                "",
                f"claim_level={'interim_era5' if is_real else 'smoke_not_skill'}",
                f"target={'ERA5_interim' if is_real else 'synthetic'}",
                f"not_skill={False if is_real else True}",
                "g1_claimable=false (ERA5 target + year-split not frozen)",
                "",
                "GATE_RECIPE_TIER0_G0.md §2 + G0_BASE_CALL.md",
                "Nepal crop must never be used as FCN3 IC.",
                "G0 claimable CRPS/SSR/PSD still NOT YET (verifying ERA5 hook).",
                "",
            ]
        )
    )

    report["artifacts"] = metrics_body["artifacts"]
    report["status"] = "real_ok" if is_real else ("smoke_ok" if args.smoke or pair["not_skill"] else "fit_ok")
    report["ok"] = True
    report["wall_s"] = round(time.time() - t0, 3)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
