#!/usr/bin/env python3
"""Phase 0 — frozen FCN3 inference smoke (real Earth2Studio path).

Runs 4-member × 16-step Nepal-box crop rollout with frozen weights from the
local package at ~/fourcastnet/models/fourcastnet3 (best_ckpt_mp0.tar).
No train / no backprop.

IC: Earth2Studio Random datasource on the model 721×1440 grid. Regional ERA5
crops under data/era5/raw are Nepal-box only (21×37) and cannot fill the global
FCN3 state; documented in the JSON report.

Members run sequentially (batch=1) for Spark 128GB UMA safety.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import OrderedDict
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


def load_config(path: Path) -> dict:
    text = path.read_text()
    if yaml is None:
        raise RuntimeError("PyYAML required to load config; pip install pyyaml")
    return yaml.safe_load(text)


def _peak_mem_gb() -> dict:
    out = {"rss_gb": None, "cuda_alloc_gb": None, "cuda_reserved_gb": None}
    try:
        import resource

        # ru_maxrss is KB on Linux
        out["rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 3)
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            out["cuda_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
            out["cuda_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 3)
    except Exception:
        pass
    return out


def crop_nepal(x, coords, box: dict):
    """Crop tensor [... lat, lon] to lat_lon_box; return crop + lat/lon arrays."""
    import numpy as np

    lats = np.asarray(coords["lat"])
    lons = np.asarray(coords["lon"])
    lat_s, lat_n = float(box["lat_south"]), float(box["lat_north"])
    lon_w, lon_e = float(box["lon_west"]), float(box["lon_east"])
    # model lon is 0..360; box is 80..89E — same numeric range
    lat_idx = np.where((lats >= lat_s) & (lats <= lat_n))[0]
    lon_idx = np.where((lons >= lon_w) & (lons <= lon_e))[0]
    if lat_idx.size == 0 or lon_idx.size == 0:
        raise RuntimeError(f"empty crop for box={box} lat_range=[{lats.min()},{lats.max()}] lon_range=[{lons.min()},{lons.max()}]")
    # lat may be descending (90→-90); keep index order as in grid
    crop = x[..., lat_idx[0] : lat_idx[-1] + 1, lon_idx[0] : lon_idx[-1] + 1]
    return crop, lats[lat_idx[0] : lat_idx[-1] + 1], lons[lon_idx[0] : lon_idx[-1] + 1]


def preview_vars(variables, names: list[str]) -> list[int]:
    import numpy as np

    vars_arr = np.asarray(variables)
    idxs = []
    for n in names:
        hits = np.where(vars_arr == n)[0]
        if hits.size:
            idxs.append(int(hits[0]))
    return idxs


def run_inference(cfg: dict, report: dict) -> None:
    import numpy as np
    import torch
    from earth2studio.data import Random
    from earth2studio.models.auto import Package
    from earth2studio.models.px.fcn3 import FCN3
    from earth2studio.run import fetch_data

    smoke = cfg.get("inference_smoke", {})
    n_members = int(smoke.get("ensemble_members", 4))
    n_steps = int(smoke.get("steps", 16))
    out_dir = Path(smoke.get("output_dir", "~/fourcastnet/data/cache/phase0_infer")).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_names = list(smoke.get("variables_preview", ["t2m", "u10m", "v10m", "tcwv", "z500", "t850"]))
    box = cfg["region"]["lat_lon_box"]
    pkg_root = Path(cfg["model"]["package_root"]).expanduser()
    ckpt = Path(cfg["model"]["checkpoint"]).expanduser()

    report["package_root"] = str(pkg_root)
    report["checkpoint"] = str(ckpt)
    report["checkpoint_exists"] = ckpt.is_file()
    report["ensemble_members"] = n_members
    report["steps"] = n_steps
    report["output_dir"] = str(out_dir)
    report["box"] = box
    report["ic_source"] = "earth2studio.data.Random"
    report["ic_reason"] = (
        "ERA5 Nepal crops (data/era5/raw) are regional 21×37 only; "
        "FCN3 requires global 721×1440 IC. Random used for offline smoke."
    )
    report["frozen"] = True
    report["train"] = False

    if not ckpt.is_file():
        raise FileNotFoundError(f"checkpoint missing: {ckpt}")
    if not (pkg_root / "config.json").is_file():
        raise FileNotFoundError(f"package config.json missing under {pkg_root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report["device"] = str(device)
    report["torch"] = torch.__version__
    report["cuda_available"] = torch.cuda.is_available()

    t_load0 = time.time()
    package = Package(str(pkg_root), cache=False)
    model = FCN3.load_model(package)
    model = model.to(device)
    model.eval()
    report["load_s"] = round(time.time() - t_load0, 2)
    report["n_params"] = int(sum(p.numel() for p in model.parameters()))
    report["n_variables"] = int(len(model.variables))

    ic = model.input_coords()
    data = Random(OrderedDict({"lat": ic["lat"], "lon": ic["lon"]}))
    # Fixed IC time for reproducibility of the smoke (not a real analysis date claim)
    ic_time = np.datetime64("2020-01-01T00:00")
    report["ic_time"] = str(ic_time)

    x0, coords0 = fetch_data(
        source=data,
        time=[ic_time],
        variable=ic["variable"],
        lead_time=ic["lead_time"],
        device=device,
    )
    report["ic_shape"] = list(x0.shape)

    var_idx = preview_vars(model.variables, preview_names)
    report["preview_variables"] = [preview_names[i] for i in range(len(preview_names)) if preview_names[i] in set(model.variables.tolist())]
    report["preview_variable_indices"] = var_idx

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    member_meta = []
    # Store cropped rollout: [member, step+1(IC..), preview_vars|all_preview, lat, lon]
    # Save full 72-var crop as netCDF-friendly npz + a small preview tensor.
    all_crops = []  # list of np arrays per member: (steps+1, nvar, nlat, nlon)

    t_roll0 = time.time()
    with torch.inference_mode():
        for m in range(n_members):
            seed = 333 + m
            model.set_rng(seed=seed, reset=True)
            # fresh IC clone per member (Zero perturbation; diversity via FCN3 RNG)
            x = x0.clone()
            # CoordSystem must remain OrderedDict (batch_func uses move_to_end)
            coords = OrderedDict(
                (k, (v.copy() if hasattr(v, "copy") else v)) for k, v in coords0.items()
            )

            t_m0 = time.time()
            it = model.create_iterator(x, coords)
            steps_out = []
            lat_c = lon_c = None
            for step in range(n_steps + 1):  # includes IC at step 0
                out, c = next(it)
                crop, lat_c, lon_c = crop_nepal(out, c, box)
                steps_out.append(crop.detach().float().cpu())
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            # stack: (steps+1, ...var..., lat, lon) — drop leading singleton dims carefully
            stacked = torch.stack(steps_out, dim=0)  # (S, *out.shape)
            # out shape typically (batch?, time?, lead?, var, lat_c, lon_c) or without batch
            while stacked.ndim > 4 and stacked.shape[1] == 1:
                stacked = stacked.squeeze(1)
            # expect (S, var, lat, lon) or (S, 1, var, lat, lon)
            if stacked.ndim == 5:
                stacked = stacked[:, 0]
            if stacked.ndim != 4:
                raise RuntimeError(f"unexpected stacked crop ndim={stacked.ndim} shape={tuple(stacked.shape)}")

            mem_path = out_dir / f"member{m:02d}_crop.pt"
            torch.save(
                {
                    "tensor": stacked,  # (steps+1, nvar, nlat, nlon)
                    "variables": np.asarray(model.variables),
                    "lat": lat_c,
                    "lon": lon_c,
                    "lead_hours": [6 * s for s in range(n_steps + 1)],
                    "seed": seed,
                    "ic_time": str(ic_time),
                    "ic_source": "Random",
                },
                mem_path,
            )
            # preview subset
            if var_idx:
                prev = stacked[:, var_idx]
                prev_path = out_dir / f"member{m:02d}_preview.pt"
                torch.save(
                    {
                        "tensor": prev,
                        "variables": [np.asarray(model.variables)[i] for i in var_idx],
                        "lat": lat_c,
                        "lon": lon_c,
                        "lead_hours": [6 * s for s in range(n_steps + 1)],
                        "seed": seed,
                    },
                    prev_path,
                )

            all_crops.append(stacked.numpy())
            m_info = {
                "member": m,
                "seed": seed,
                "shape": list(stacked.shape),
                "path": str(mem_path),
                "wall_s": round(time.time() - t_m0, 2),
            }
            member_meta.append(m_info)
            print(json.dumps({"member_done": m_info}), flush=True)

            del steps_out, stacked, x, it
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    report["rollout_s"] = round(time.time() - t_roll0, 2)
    report["members"] = member_meta
    report["crop_lat"] = lat_c.tolist() if lat_c is not None else None
    report["crop_lon"] = lon_c.tolist() if lon_c is not None else None
    report["crop_shape_hw"] = [len(lat_c), len(lon_c)] if lat_c is not None else None
    report["peak_mem"] = _peak_mem_gb()

    # ensemble stack npz for convenience
    ens = np.stack(all_crops, axis=0)  # (M, S, V, H, W)
    ens_path = out_dir / "ensemble_crop.npz"
    np.savez_compressed(
        ens_path,
        data=ens,
        variables=np.asarray(model.variables),
        lat=lat_c,
        lon=lon_c,
        lead_hours=np.arange(0, (n_steps + 1) * 6, 6),
        seeds=np.array([333 + m for m in range(n_members)]),
        ic_time=str(ic_time),
        ic_source="Random",
    )
    report["ensemble_npz"] = str(ens_path)
    report["ensemble_shape"] = list(ens.shape)

    report["sidecar"] = str(out_dir / "smoke_report.json")


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 0 frozen FCN3 inference smoke")
    ap.add_argument(
        "--config",
        type=Path,
        default=Path.home() / "fourcastnet" / "configs" / "phase0_nepal_box.yaml",
    )
    ap.add_argument("--dry-run", action="store_true", help="Only validate config + imports")
    args = ap.parse_args()

    report: dict = {
        "config": str(args.config),
        "dry_run": bool(args.dry_run),
        "ok": False,
    }

    try:
        if not args.config.is_file():
            report["error"] = f"missing config {args.config}"
            print(json.dumps(report, indent=2))
            return 1

        cfg = load_config(args.config)
        box = cfg.get("region", {}).get("lat_lon_box")
        locks = cfg.get("locks", {})
        report["box"] = box
        report["locks"] = locks
        report["splits"] = cfg.get("splits", {})

        blockers = []
        if not locks.get("geo_box_frozen"):
            blockers.append("geo_box_frozen is false")
        if not locks.get("years_frozen"):
            blockers.append("years_frozen is false")
        if box is None or any(v is None for v in box.values()):
            blockers.append("config region.lat_lon_box incomplete")
        # precip_in_v1 deliberately ignored — open policy, not a smoke blocker

        try:
            import torch

            report["torch"] = torch.__version__
            report["cuda"] = torch.cuda.is_available()
        except Exception as e:
            report["torch_error"] = str(e)
            blockers.append("torch not importable")

        try:
            import earth2studio  # noqa: F401
            from earth2studio.models.px.fcn3 import FCN3  # noqa: F401
            from makani.models.model_package import load_model_package  # noqa: F401

            report["earth2studio"] = True
            report["makani"] = True
        except Exception as e:
            report["earth2studio"] = False
            report["earth2studio_error"] = str(e)
            blockers.append(f"earth2studio/makani import failed: {e}")

        ckpt = Path(cfg.get("model", {}).get("checkpoint", "")).expanduser()
        report["checkpoint_exists"] = ckpt.is_file() if str(ckpt) else False
        if not report["checkpoint_exists"]:
            blockers.append(f"checkpoint missing: {ckpt}")

        if blockers:
            report["ok"] = False
            report["status"] = "blocked"
            report["blockers"] = blockers
            print(json.dumps(report, indent=2))
            print("BLOCKED — see blockers", file=sys.stderr)
            return 2

        if args.dry_run:
            report["ok"] = True
            report["status"] = "dry_run_ready"
            print(json.dumps(report, indent=2))
            return 0

        run_inference(cfg, report)
        report["ok"] = True
        report["status"] = "inference_smoke_ok"
        sidecar = Path(report["output_dir"]) / "smoke_report.json"
        sidecar.write_text(json.dumps(report, indent=2, default=str))
        report["sidecar"] = str(sidecar)
        print(json.dumps(report, indent=2, default=str))
        return 0

    except Exception as e:
        report["ok"] = False
        report["status"] = "failed"
        report["error"] = str(e)
        report["traceback"] = traceback.format_exc()
        report["peak_mem"] = _peak_mem_gb()
        print(json.dumps(report, indent=2, default=str))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
