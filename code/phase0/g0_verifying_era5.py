#!/usr/bin/env python3
"""G0 verifying ERA5 → real CRPS / SSR / PSD (Leonard #1; claimable G0 blocker).

Fetches ARCO ERA5 preview vars at score leads, then streaming re-rolls frozen
FCN3 (preview_only — no 73GB buffers) and scores online.

Usage:
  cd ~/fourcastnet
  ~/fcn3-venv/bin/python code/phase0/g0_verifying_era5.py --dry-run
  ~/fcn3-venv/bin/python code/phase0/g0_verifying_era5.py --fetch-global --leads 120
  ~/fcn3-venv/bin/python code/phase0/g0_verifying_era5.py --smoke --allow-gpu
  ~/fcn3-venv/bin/python code/phase0/g0_verifying_era5.py --full --allow-gpu

Do NOT kill ERA5 regional PID 611595. Avoid double GPU jobs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # ~/fourcastnet
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from arco_direct_fetch import fetch_timestep_array, grid_lat_lon  # noqa: E402

PREVIEW_VARS = ["t2m", "u10m", "v10m", "tcwv", "z500", "t850"]
DEFAULT_LEADS = [120, 240, 360]  # hours: +5 / +10 / +15 d
EXPECTED_SHAPE = (72, 721, 1440)


def parse_ic_time(s: str) -> datetime:
    s = s.replace("Z", "")
    if "T" not in s:
        s += "T00:00:00"
    if len(s) == 16:
        s += ":00"
    return datetime.fromisoformat(s)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def expand(p: str | Path) -> Path:
    return Path(os.path.expanduser(str(p)))


def load_config(path: Path) -> dict:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def plan(manifest: dict, leads: list[int], variables: list[str]) -> dict:
    ics = [ic for ic in manifest.get("ics", []) if ic.get("status") == "staged"]
    frames = []
    for ic in ics:
        ic_dt = parse_ic_time(ic["time"])
        for lh in leads:
            frames.append(
                {
                    "ic_id": ic.get("id"),
                    "ic_time": ic["time"],
                    "lead_h": lh,
                    "valid_time": (ic_dt + timedelta(hours=int(lh))).isoformat(),
                }
            )
    return {
        "schema": "g0_verifying_era5_plan/v1",
        "n_ics": len(ics),
        "leads_h": leads,
        "variables": variables,
        "n_frames": len(frames),
        "frames": frames,
        "g0_preview_tensors_on_disk": False,
        "g0_storage_mode": "preview_only_discarded",
        "g0_pass_claimable": False,
        "how_to_score": (
            "Use --smoke/--full --allow-gpu for streaming re-roll + CRPS/SSR/PSD."
        ),
        "preferred_next": (
            "Fetch verifying ERA5 then streaming score re-roll "
            "(8 IC × 4 mem × 60 steps at +120/+240/+360 h)."
        ),
        "memory_note": (
            "Do not store 61×72×721×1440×M. Keep lead-only preview frames."
        ),
        "source": "arco_zarr_direct (not CDS; leave PID 611595 alone)",
        "target_label": "ERA5_interim_for_G0_metrics",
        "provisional_years": {
            "train": "2018-2021",
            "val": "2022",
            "test": "2023-2024",
            "note": "provisional; Tier-0 holdout re-score required",
        },
    }


def area_weights(lat: Any) -> Any:
    import numpy as np

    # cos(lat) weights, normalized to mean 1 over grid
    w = np.cos(np.deg2rad(lat)).astype(np.float64)
    w = np.clip(w, 0.0, None)
    w2 = np.broadcast_to(w[:, None], (lat.size, 1440))
    return (w2 / w2.mean()).astype(np.float32)


def midlat_mask(lat: Any, lo: float = 20.0, hi: float = 70.0) -> Any:
    import numpy as np

    abs_lat = np.abs(lat)
    m = (abs_lat >= lo) & (abs_lat <= hi)
    return np.broadcast_to(m[:, None], (lat.size, 1440))


def fair_crps_ensemble(ens: Any, obs: Any, weights: Any | None = None) -> float:
    """Fair ensemble CRPS, spatially averaged.

    ens: (M, H, W), obs: (H, W), weights: (H, W) optional.
    CRPS = E|X - y| - 0.5 E|X - X'|  (fair: M/(M-1) on pairwise term)
    """
    import numpy as np

    ens = np.asarray(ens, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    m = ens.shape[0]
    if m < 1:
        return float("nan")
    # E|X-y|
    term1 = np.mean(np.abs(ens - obs[None, ...]), axis=0)
    if m == 1:
        crps_map = term1
    else:
        # pairwise |Xi - Xj| mean; fair factor m/(m-1)
        # Use (2/(m(m-1))) * sum_{i<j} |Xi-Xj| = fair E|X-X'|
        # Efficient: sort along member axis
        s = np.sort(ens, axis=0)
        # sum_{i<j} |si - sj| = sum_k (2k - m - 1) * s_k
        k = np.arange(1, m + 1, dtype=np.float64)[:, None, None]
        coeff = 2.0 * k - m - 1.0
        pair_sum = np.sum(coeff * s, axis=0)  # sum_{i<j}|diff| * 2? wait
        # Correct identity: sum_{i=1..m} (2i - m - 1) * s_(i) = sum_{i<j} (s_j - s_i)
        # E|X-X'| unbiased (fair) = 2/(m(m-1)) * sum_{i<j}|Xi-Xj|
        pair_mean = (2.0 / (m * (m - 1))) * pair_sum
        crps_map = term1 - 0.5 * pair_mean
    if weights is None:
        return float(np.mean(crps_map))
    w = np.asarray(weights, dtype=np.float64)
    return float(np.sum(crps_map * w) / np.sum(w))


def ssr_spread_skill(ens: Any, obs: Any, weights: Any | None = None) -> dict:
    """SSR = spread / RMSE(ensemble mean). ens (M,H,W)."""
    import numpy as np

    ens = np.asarray(ens, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    m = ens.shape[0]
    mean = ens.mean(axis=0)
    # unbiased variance across members
    if m >= 2:
        var = ens.var(axis=0, ddof=1)
    else:
        var = np.zeros_like(mean)
    spread_map = np.sqrt(np.maximum(var, 0.0))
    err_map = np.abs(mean - obs)
    sq_err = (mean - obs) ** 2
    if weights is None:
        spread = float(np.mean(spread_map))
        rmse = float(np.sqrt(np.mean(sq_err)))
        mae = float(np.mean(err_map))
    else:
        w = np.asarray(weights, dtype=np.float64)
        ws = float(np.sum(w))
        spread = float(np.sum(spread_map * w) / ws)
        rmse = float(np.sqrt(np.sum(sq_err * w) / ws))
        mae = float(np.sum(err_map * w) / ws)
    ssr = float(spread / rmse) if rmse > 0 else float("nan")
    return {
        "ssr": ssr,
        "spread": spread,
        "rmse": rmse,
        "mae": mae,
        "n_members": int(m),
    }


def zonal_psd_ratio(fcst: Any, obs: Any, lat: Any, lat_band=(20.0, 70.0)) -> dict:
    """Crude zonal PSD energy ratio fcst/obs in synoptic band (k=4..20).

    fcst/obs: (H, W). Returns mean power ratio over midlat rows.
    """
    import numpy as np

    fcst = np.asarray(fcst, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    abs_lat = np.abs(lat)
    rows = np.where((abs_lat >= lat_band[0]) & (abs_lat <= lat_band[1]))[0]
    if rows.size == 0:
        return {"status": "empty_band", "ratio_synoptic": float("nan")}

    ratios = []
    power_f = []
    power_o = []
    for r in rows:
        a = fcst[r] - fcst[r].mean()
        b = obs[r] - obs[r].mean()
        fa = np.fft.rfft(a)
        fb = np.fft.rfft(b)
        pa = (np.abs(fa) ** 2).real
        pb = (np.abs(fb) ** 2).real
        # wavenumber indices 4..20 (synoptic / planetary-synoptic)
        lo, hi = 4, min(20, pa.size - 1)
        if hi <= lo:
            continue
        pf = float(pa[lo : hi + 1].sum())
        po = float(pb[lo : hi + 1].sum())
        power_f.append(pf)
        power_o.append(po)
        if po > 0:
            ratios.append(pf / po)
    if not ratios:
        return {"status": "no_ratios", "ratio_synoptic": float("nan")}
    ratio = float(np.mean(ratios))
    # Sanity: collapse if ratio << 1 (MSE blur); blow-up if >> 1
    return {
        "status": "ok",
        "ratio_synoptic": ratio,
        "power_fcst_mean": float(np.mean(power_f)),
        "power_obs_mean": float(np.mean(power_o)),
        "n_rows": int(rows.size),
        "k_band": [4, 20],
        "lat_band": list(lat_band),
        "sane": bool(0.25 <= ratio <= 4.0),
        "note": "ratio~1 matched; <<1 spectral collapse; >>1 excess small-scale",
    }


def score_lead_ensemble(
    ens_lead: Any,
    obs: Any,
    preview_names: list[str],
    lat: Any,
) -> dict:
    """ens_lead: (M, Vp, H, W), obs: (Vp, H, W)."""
    import numpy as np

    w_global = area_weights(lat)
    mask = midlat_mask(lat)
    w_mid = w_global * mask.astype(np.float32)
    # renormalize midlat weights
    if w_mid.sum() > 0:
        w_mid = w_mid * (w_mid.size / w_mid.sum())  # keep comparable magnitude? use mean-1 on mask
        w_mid = w_mid / (w_mid[mask].mean() if mask.any() else 1.0)

    per_var = {}
    for i, name in enumerate(preview_names):
        ens_v = ens_lead[:, i]
        obs_v = obs[i]
        crps_g = fair_crps_ensemble(ens_v, obs_v, w_global)
        crps_m = fair_crps_ensemble(ens_v, obs_v, w_mid)
        ssr_g = ssr_spread_skill(ens_v, obs_v, w_global)
        ssr_m = ssr_spread_skill(ens_v, obs_v, w_mid)
        mean = ens_v.mean(axis=0)
        psd = zonal_psd_ratio(mean, obs_v, lat)
        per_var[name] = {
            "crps_global": crps_g,
            "crps_midlat": crps_m,
            "ssr_global": ssr_g,
            "ssr_midlat": ssr_m,
            "psd_zonal": psd,
            "finite_ens": bool(np.isfinite(ens_v).all()),
            "finite_obs": bool(np.isfinite(obs_v).all()),
        }
    return {"variables": per_var}


def fetch_verify_frames(
    frames: list[dict],
    variables: list[str],
    out_dir: Path,
    skip_existing: bool = True,
) -> list[dict]:
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    recs = []
    for fr in frames:
        stem = f"{fr['ic_id']}_l{int(fr['lead_h']):03d}_era5_preview.npz"
        path = out_dir / stem
        if skip_existing and path.is_file():
            d = np.load(path)
            arr = d["data"]
            recs.append(
                {
                    **fr,
                    "path": str(path),
                    "shape": list(arr.shape),
                    "cached": True,
                    "finite": bool(np.isfinite(arr).all()),
                }
            )
            print(json.dumps({"cached": stem, "shape": list(arr.shape)}), flush=True)
            continue
        dt = parse_ic_time(fr["valid_time"])
        print(json.dumps({"fetch_global": fr}), flush=True)
        arr, meta = fetch_timestep_array(variables, dt, verbose=True)
        np.savez_compressed(
            path,
            data=arr,
            variables=np.array(variables),
            ic_id=np.array(fr["ic_id"]),
            ic_time=np.array(fr["ic_time"]),
            lead_h=np.array(fr["lead_h"]),
            valid_time=np.array(fr["valid_time"]),
        )
        recs.append(
            {
                **fr,
                "path": str(path),
                "shape": list(arr.shape),
                "fetch_s": meta.get("fetch_s"),
                "cached": False,
                "finite": bool(np.isfinite(arr).all()),
            }
        )
        print(
            json.dumps(
                {
                    "saved": stem,
                    "fetch_s": meta.get("fetch_s"),
                    "finite": bool(np.isfinite(arr).all()),
                }
            ),
            flush=True,
        )
    return recs


def load_obs_for_ic(
    verify_dir: Path,
    ic_id: str,
    leads: list[int],
    variables: list[str],
) -> dict[int, Any]:
    import numpy as np

    out = {}
    for lh in leads:
        path = verify_dir / f"{ic_id}_l{int(lh):03d}_era5_preview.npz"
        if not path.is_file():
            raise FileNotFoundError(f"missing verifying ERA5: {path}")
        d = np.load(path)
        arr = np.asarray(d["data"], dtype=np.float32)
        vars_on_disk = [str(x) for x in d["variables"].tolist()] if "variables" in d else list(variables)
        if list(vars_on_disk) != list(variables):
            # reorder if needed
            idx = [vars_on_disk.index(v) for v in variables]
            arr = arr[idx]
        out[int(lh)] = arr
    return out


def _peak_mem_gb() -> dict:
    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux: KB
    rss_gb = rss / 1e6
    out = {"rss_gb": round(rss_gb, 3)}
    try:
        import torch

        if torch.cuda.is_available():
            out["cuda_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
            out["cuda_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 3)
    except Exception:
        pass
    return out


def preview_indices(variables: Any, preview_names: list[str]) -> tuple[list[int], list[str]]:
    import numpy as np

    vars_arr = np.asarray(variables)
    idx: list[int] = []
    names_hit: list[str] = []
    for n in preview_names:
        hits = np.where(vars_arr == n)[0]
        if hits.size:
            idx.append(int(hits[0]))
            names_hit.append(n)
    return idx, names_hit


def build_ic_tensor(path: Path, ic_time: str, model, device):
    import numpy as np
    import torch

    arr = np.load(path)
    if tuple(arr.shape) != EXPECTED_SHAPE:
        raise RuntimeError(f"IC shape {arr.shape} != {EXPECTED_SHAPE}")
    if not np.isfinite(arr).all():
        raise RuntimeError(f"IC non-finite: {path}")
    x = torch.from_numpy(np.array(arr, copy=True)).to(device=device, dtype=torch.float32)
    x = x.view(1, 1, 1, *EXPECTED_SHAPE)
    t = np.datetime64(ic_time.replace("Z", ""))
    if "T" not in str(t):
        t = np.datetime64(str(t) + "T00:00")
    coords = OrderedDict(
        {
            "batch": np.array([0]),
            "time": np.array([t]),
            "lead_time": np.array([np.timedelta64(0, "h")]),
            "variable": np.array(model.variables),
            "lat": np.linspace(90.0, -90.0, 721),
            "lon": np.linspace(0, 360, 1440, endpoint=False),
        }
    )
    return x, coords


def load_fcn3(cfg: dict, report: dict):
    import torch
    from earth2studio.models.auto import Package
    from earth2studio.models.px.fcn3 import FCN3

    pkg_root = expand(cfg["model"]["package_root"])
    ckpt = expand(cfg["model"]["checkpoint"])
    report["package_root"] = str(pkg_root)
    report["checkpoint"] = str(ckpt)
    if not ckpt.is_file():
        raise FileNotFoundError(f"checkpoint missing: {ckpt}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report["device"] = str(device)
    report["cuda_available"] = torch.cuda.is_available()
    if device.type != "cuda":
        raise RuntimeError("G0 verifying GPU path requires CUDA")
    t0 = time.time()
    package = Package(str(pkg_root), cache=False)
    model = FCN3.load_model(package)
    model = model.to(device)
    model.eval()
    report["load_s"] = round(time.time() - t0, 2)
    report["n_params"] = int(sum(p.numel() for p in model.parameters()))
    report["n_variables"] = int(len(model.variables))
    report["torch"] = torch.__version__
    return model, device


def _squeeze_field(out) -> Any:
    t = out.detach()
    while t.ndim > 4 and t.shape[0] == 1:
        t = t.squeeze(0)
    while t.ndim > 4 and t.shape[0] == 1:
        t = t.squeeze(0)
    if t.ndim == 4 and t.shape[0] == 1:
        t = t[0]
    if t.ndim != 3:
        raise RuntimeError(f"unexpected out ndim={t.ndim} shape={tuple(t.shape)}")
    return t


def aggregate_gate(ic_results: list[dict], leads: list[int]) -> dict:
    """Decide g0_pass_claimable from real metrics (base: finite + SSR/PSD sane)."""
    import numpy as np

    if not ic_results:
        return {
            "g0_pass_claimable": False,
            "reason": "no_ic_results",
            "crps_ssr_psd": "missing",
        }

    all_finite = all(r.get("all_finite", False) for r in ic_results)
    # Collect CRPS @ +360 (or max lead) midlat for key vars
    key_vars = ["t2m", "z500", "t850"]
    lead_primary = 360 if 360 in leads else max(leads)
    crps_vals = {v: [] for v in key_vars}
    ssr_vals = {v: [] for v in key_vars}
    psd_sane = []
    ssr_sane = []

    for r in ic_results:
        scores = r.get("scores_by_lead", {})
        lead_rec = scores.get(str(lead_primary)) or scores.get(lead_primary)
        if not lead_rec:
            continue
        vars_ = lead_rec.get("variables", {})
        for v in key_vars:
            if v not in vars_:
                continue
            crps_vals[v].append(vars_[v]["crps_midlat"])
            ssr = vars_[v]["ssr_midlat"]["ssr"]
            ssr_vals[v].append(ssr)
            # SSR sane: not collapsed (<0.05) not blow-up (>5)
            ssr_sane.append(bool(np.isfinite(ssr) and 0.05 <= ssr <= 5.0))
            psd = vars_[v]["psd_zonal"]
            psd_sane.append(bool(psd.get("sane", False)))

    mean_crps = {
        v: (float(np.mean(crps_vals[v])) if crps_vals[v] else None) for v in key_vars
    }
    mean_ssr = {
        v: (float(np.mean(ssr_vals[v])) if ssr_vals[v] else None) for v in key_vars
    }

    metrics_real = all(
        mean_crps[v] is not None and np.isfinite(mean_crps[v]) for v in key_vars
    )
    ssr_ok = bool(ssr_sane) and all(ssr_sane)
    psd_ok = bool(psd_sane) and all(psd_sane)

    # Base G0 PASS: Path A metrics finite; PSD/SSR sane on all >=8 ICs (GATE §1.5).
    # Require primary lead +15 d (360 h) and full IC count for claimable.
    # ~5% CRPS rule applies to *adapter* vs this baseline — not to base itself.
    protocol_complete = (
        len(ic_results) >= 8
        and lead_primary == 360
        and 360 in leads
    )
    claimable = bool(
        protocol_complete and all_finite and metrics_real and ssr_ok and psd_ok
    )

    return {
        "g0_pass_claimable": claimable,
        "crps_ssr_psd": "real" if metrics_real else "partial",
        "all_finite": all_finite,
        "ssr_sane_all": ssr_ok,
        "psd_sane_all": psd_ok,
        "primary_lead_h": lead_primary,
        "mean_crps_midlat": mean_crps,
        "mean_ssr_midlat": mean_ssr,
        "n_ics_scored": len(ic_results),
        "protocol_complete": protocol_complete,
        "leonard_5pct_rule": (
            "Applies to adapter vs this baseline, not to base itself. "
            "Base claimable when metrics real + finite + SSR/PSD sane on >=8 ICs @ +15 d."
        ),
        "g0_pass_note": (
            "CLAIMABLE: real CRPS/SSR/PSD finite and sane on >=8 ICs @ +15 d"
            if claimable
            else (
                "NOT claimable yet: need >=8 ICs, lead +360 h, finite real metrics, SSR/PSD sane"
                if not protocol_complete
                else "NOT claimable: metrics/SSR/PSD failed sanity on scored ICs"
            )
        ),
    }


def run_streaming_score(
    cfg: dict,
    ics: list[dict],
    n_members: int,
    n_steps: int,
    seed0: int,
    leads: list[int],
    verify_dir: Path,
    out_dir: Path,
    report: dict,
    progress_every: int = 5,
) -> dict:
    """Re-roll FCN3 preview_only; keep only score-lead frames; score vs ERA5."""
    import numpy as np
    import torch

    model, device = load_fcn3(cfg, report)
    preview_cfg = list(cfg.get("g0_base", {}).get("preview_variables", PREVIEW_VARS))
    pidx, preview_names = preview_indices(model.variables, preview_cfg)
    if not pidx:
        raise RuntimeError(f"none of preview_variables {preview_cfg} in model")
    report["preview_variables"] = preview_names
    report["preview_indices"] = pidx
    report["storage_mode"] = "lead_only_preview"
    report["score_leads_h"] = leads

    # Map lead hours → step index (6h steps)
    lead_to_step = {lh: lh // 6 for lh in leads}
    for lh, st in lead_to_step.items():
        if st > n_steps:
            raise RuntimeError(f"lead {lh}h needs step {st} but n_steps={n_steps}")

    lat, _lon = grid_lat_lon()
    ic_results = []
    partial_dir = out_dir / "partial_verify"
    partial_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t_all0 = time.time()
    with torch.inference_mode():
        for ic in ics:
            path = Path(ic["path"]).expanduser()
            ic_time = ic["time"]
            ic_id = ic.get("id")
            print(
                json.dumps(
                    {
                        "ic_start": ic_id,
                        "time": ic_time,
                        "n_members": n_members,
                        "n_steps": n_steps,
                        "leads": leads,
                        "storage": "lead_only_preview",
                    }
                ),
                flush=True,
            )
            obs_by_lead = load_obs_for_ic(verify_dir, ic_id, leads, preview_names)
            x0, coords0 = build_ic_tensor(path, ic_time, model, device)

            # member_leads[m] = {lead_h: (Vp,H,W)}
            member_leads: list[dict[int, Any]] = []
            members_meta = []
            all_finite = True

            for m in range(n_members):
                seed = seed0 + m
                model.set_rng(seed=seed, reset=True)
                x = x0.clone()
                coords = OrderedDict(
                    (k, (v.copy() if hasattr(v, "copy") else v)) for k, v in coords0.items()
                )
                t_m0 = time.time()
                it = model.create_iterator(x, coords)
                kept: dict[int, Any] = {}
                member_finite = True
                for step in range(n_steps + 1):
                    t_step0 = time.time()
                    out, c = next(it)
                    field = _squeeze_field(out)
                    lead_h = 6 * step
                    if lead_h in lead_to_step and lead_to_step[lead_h] == step:
                        preview_t = field[pidx].float().cpu()
                        if not bool(torch.isfinite(preview_t).all().item()):
                            member_finite = False
                            all_finite = False
                        kept[lead_h] = preview_t.numpy()
                        del preview_t
                    del field, out
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    step_s = round(time.time() - t_step0, 2)
                    if step % progress_every == 0 or step == n_steps or lead_h in kept:
                        print(
                            json.dumps(
                                {
                                    "progress": {
                                        "ic": ic_id,
                                        "member": m,
                                        "step": step,
                                        "lead_h": lead_h,
                                        "step_s": step_s,
                                        "kept_leads": sorted(kept.keys()),
                                        "peak_mem": _peak_mem_gb(),
                                    }
                                }
                            ),
                            flush=True,
                        )
                member_leads.append(kept)
                m_info = {
                    "member": m,
                    "seed": seed,
                    "wall_s": round(time.time() - t_m0, 2),
                    "finite_preview_leads": member_finite,
                    "leads_kept": sorted(kept.keys()),
                }
                members_meta.append(m_info)
                print(json.dumps({"member_done": {**m_info, "ic": ic_id}}), flush=True)
                del kept, x, it, coords
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Score each lead
            scores_by_lead: dict[str, Any] = {}
            for lh in leads:
                ens = np.stack([member_leads[m][lh] for m in range(n_members)], axis=0)
                obs = obs_by_lead[lh]
                if obs.shape[0] != ens.shape[1]:
                    raise RuntimeError(
                        f"obs channels {obs.shape[0]} != ens Vp {ens.shape[1]}"
                    )
                scored = score_lead_ensemble(ens, obs, preview_names, lat)
                scored["ensemble_shape"] = list(ens.shape)
                scored["obs_shape"] = list(obs.shape)
                scores_by_lead[str(lh)] = scored
                # free ens
                del ens
            # free member tensors
            del member_leads, obs_by_lead

            ic_rec = {
                "id": ic_id,
                "time": ic_time,
                "path": str(path),
                "season": ic.get("season"),
                "region_class": ic.get("region_class"),
                "region_label": ic.get("region_label"),
                "members": members_meta,
                "scores_by_lead": scores_by_lead,
                "all_finite": all_finite,
                "metrics_kind": "real_crps_ssr_psd",
                "stub": False,
            }
            partial_path = partial_dir / f"{ic_id}.json"
            partial_path.write_text(json.dumps(ic_rec, indent=2, default=str))
            ic_rec["partial_json"] = str(partial_path)
            print(
                json.dumps(
                    {
                        "ic_done": ic_id,
                        "all_finite": all_finite,
                        "partial_json": str(partial_path),
                        "peak_mem": _peak_mem_gb(),
                    }
                ),
                flush=True,
            )
            ic_results.append(ic_rec)
            del x0, coords0
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    report["rollout_s"] = round(time.time() - t_all0, 2)
    report["peak_mem"] = _peak_mem_gb()
    report["ic_results"] = ic_results
    report["aggregate"] = aggregate_gate(ic_results, leads)
    report["aggregate"]["n_ics"] = len(ic_results)
    report["aggregate"]["storage"] = "lead_only_preview"
    return report


def write_sidecar_base_update(results: dict, base_path: Path) -> Path:
    """Write sidecar noting verifying metrics; do not overwrite stub base file."""
    sidecar = base_path.with_name("g0_base_results_verifying_sidecar.json")
    agg = results.get("aggregate", {})
    payload = {
        "schema": "g0_base_verifying_sidecar/v1",
        "created_utc": utc_now(),
        "base_results": str(base_path),
        "verifying_results": results.get("results_json"),
        "g0_pass_claimable": agg.get("g0_pass_claimable", False),
        "crps_ssr_psd": agg.get("crps_ssr_psd"),
        "mean_crps_midlat": agg.get("mean_crps_midlat"),
        "mean_ssr_midlat": agg.get("mean_ssr_midlat"),
        "note": (
            "Stub g0_base_results.json preserved as plumbing reference. "
            "Claimable G0 evaluated from verifying_results."
        ),
        "leonard_5pct_rule": agg.get("leonard_5pct_rule"),
    }
    sidecar.write_text(json.dumps(payload, indent=2, default=str))
    return sidecar


def main() -> int:
    ap = argparse.ArgumentParser(description="G0 verifying ERA5 + streaming CRPS/SSR/PSD")
    ap.add_argument(
        "--config",
        type=Path,
        default=Path.home() / "fourcastnet" / "configs" / "g0_base.yaml",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path.home() / "fourcastnet" / "data" / "g0_ics" / "g0_ic_manifest.json",
    )
    ap.add_argument("--leads", type=str, default="120,240,360")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fetch-global", action="store_true")
    ap.add_argument("--fetch-nepal-crop", action="store_true")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path.home() / "fourcastnet" / "data" / "g0_verify",
    )
    ap.add_argument("--smoke", action="store_true", help="1 IC, leads to +120h, 2 members")
    ap.add_argument("--full", action="store_true", help="8 IC × 4 mem × 60 steps")
    ap.add_argument("--allow-gpu", action="store_true")
    ap.add_argument("--max-ics", type=int, default=None)
    ap.add_argument("--members", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument(
        "--results-json",
        type=Path,
        default=Path.home() / "fourcastnet" / "runs" / "phase0" / "g0" / "g0_verifying_results.json",
    )
    ap.add_argument("--skip-fetch", action="store_true", help="Score only; ERA5 must exist")
    ap.add_argument("--force-refetch", action="store_true")
    args = ap.parse_args()

    leads = [int(x) for x in args.leads.split(",") if x.strip()]
    man = json.loads(Path(args.manifest).expanduser().read_text())
    report_plan = plan(man, leads, PREVIEW_VARS)
    report_plan["utc"] = utc_now()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    (out / "g0_verify_plan.json").write_text(json.dumps(report_plan, indent=2))

    # ---- plan / dry-run only ----
    scoring = args.smoke or args.full
    if args.dry_run or (
        not args.fetch_global
        and not args.fetch_nepal_crop
        and not scoring
    ):
        report_plan["ok"] = True
        report_plan["status"] = "plan_only"
        print(json.dumps(report_plan, indent=2))
        return 0

    if args.fetch_nepal_crop:
        from arco_direct_fetch import fetch_crop_array
        import numpy as np

        recs = []
        for fr in report_plan["frames"]:
            dt = parse_ic_time(fr["valid_time"])
            arr, meta = fetch_crop_array(PREVIEW_VARS, dt, verbose=True)
            recs.append({**fr, "shape": list(arr.shape), "fetch_s": meta.get("fetch_s")})
        report_plan["nepal_crop_fetches"] = recs
        report_plan["note"] = "Nepal crop verifying is NOT a G0 CRPS substitute."
        (out / "g0_verify_manifest.json").write_text(
            json.dumps(report_plan, indent=2, default=str)
        )
        print(json.dumps({"ok": True, "status": "nepal_crop_only"}, indent=2))
        if not scoring:
            return 0

    # ---- fetch global verifying fields ----
    ics_all = [ic for ic in man.get("ics", []) if ic.get("status") == "staged"]
    if args.smoke:
        max_ics = args.max_ics or 1
        # smoke default: score +120 only unless user overrode leads
        if args.leads == "120,240,360":
            leads = [120]
        n_members = args.members or 2
        n_steps = args.steps if args.steps is not None else max(leads) // 6
        mode = "smoke"
    elif args.full:
        max_ics = args.max_ics or 8
        n_members = args.members or 4
        n_steps = args.steps if args.steps is not None else 60
        mode = "full"
    else:
        max_ics = args.max_ics
        n_members = args.members or 4
        n_steps = args.steps if args.steps is not None else 60
        mode = "fetch" if args.fetch_global else "custom"

    ics = ics_all[:max_ics] if max_ics else ics_all
    frames = []
    for ic in ics:
        ic_dt = parse_ic_time(ic["time"])
        for lh in leads:
            frames.append(
                {
                    "ic_id": ic.get("id"),
                    "ic_time": ic["time"],
                    "lead_h": lh,
                    "valid_time": (ic_dt + timedelta(hours=int(lh))).isoformat(),
                }
            )

    if args.fetch_global or (scoring and not args.skip_fetch):
        recs = fetch_verify_frames(
            frames,
            PREVIEW_VARS,
            out,
            skip_existing=not args.force_refetch,
        )
        manifest = {
            **report_plan,
            "frames": frames,
            "global_fetches": recs,
            "n_ics_selected": len(ics),
            "leads_h": leads,
            "status": "verify_fields_ready",
            "utc": utc_now(),
            "ok": True,
        }
        (out / "g0_verify_manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str)
        )
        print(
            json.dumps(
                {
                    "ok": True,
                    "fetched_or_cached": len(recs),
                    "out": str(out),
                    "status": "verify_fields_ready",
                },
                indent=2,
            ),
            flush=True,
        )
        if not scoring:
            return 0

    # ---- streaming score ----
    if scoring:
        if not args.allow_gpu:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": "--allow-gpu required for smoke/full re-roll",
                    }
                ),
                flush=True,
            )
            return 2
        cfg = load_config(expand(args.config))
        results_path = expand(args.results_json)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        report: dict[str, Any] = {
            "schema": "g0_verifying_results/v1",
            "created_utc": utc_now(),
            "mode": mode,
            "allow_gpu": True,
            "leads_h": leads,
            "n_ics": len(ics),
            "ensemble_members": n_members,
            "steps": n_steps,
            "preview_variables": PREVIEW_VARS,
            "verify_dir": str(out),
            "results_json": str(results_path),
            "manifest": str(expand(args.manifest)),
            "config": str(expand(args.config)),
            "target_label": "ERA5_ARCO_verifying",
            "provisional_years": report_plan["provisional_years"],
            "tier0_interim_bar": {
                "t2m_rmse_K": 1.978,
                "note": "beat-this interim; holdout re-score required",
            },
            "era5_regional_pid_untouched": 611595,
            "path": "A",
            "frozen": True,
            "stub": False,
        }
        try:
            run_streaming_score(
                cfg=cfg,
                ics=ics,
                n_members=n_members,
                n_steps=n_steps,
                seed0=int(cfg.get("g0_base", {}).get("seed0", 333)),
                leads=leads,
                verify_dir=out,
                out_dir=results_path.parent,
                report=report,
            )
            report["ok"] = True
            report["status"] = f"{mode}_ok"
        except Exception as e:
            report["ok"] = False
            report["status"] = f"{mode}_failed"
            report["error"] = f"{type(e).__name__}: {e}"
            results_path.write_text(json.dumps(report, indent=2, default=str))
            print(json.dumps({"ok": False, "error": report["error"]}, indent=2), flush=True)
            raise

        results_path.write_text(json.dumps(report, indent=2, default=str))
        base_path = expand(
            cfg.get("g0_base", {}).get(
                "results_json",
                str(Path.home() / "fourcastnet/runs/phase0/g0/g0_base_results.json"),
            )
        )
        sidecar = write_sidecar_base_update(report, base_path)
        report["sidecar"] = str(sidecar)
        results_path.write_text(json.dumps(report, indent=2, default=str))
        print(
            json.dumps(
                {
                    "ok": True,
                    "status": report["status"],
                    "results_json": str(results_path),
                    "sidecar": str(sidecar),
                    "aggregate": report.get("aggregate"),
                    "peak_mem": report.get("peak_mem"),
                    "rollout_s": report.get("rollout_s"),
                },
                indent=2,
                default=str,
            ),
            flush=True,
        )
        return 0 if report.get("ok") else 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
