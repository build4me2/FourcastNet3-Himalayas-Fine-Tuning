#!/usr/bin/env python3
"""Tier-0 Path B: global ERA5 IC → frozen FCN3 → Nepal crop at selected leads.

GATE_RECIPE §2 / G0_BASE_CALL next-eng #1.
Memory-safe: crop 21×37 each step; discard full global field. One GPU process.

Writes:
  runs/phase0/tier0/pairs/forecast/{ic_id}_{YYYYmmddTHHMM}_crop.npz
  runs/phase0/tier0/pairs/forecast_manifest.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from g0_base import (  # noqa: E402
    _peak_mem_gb,
    _squeeze_field,
    build_ic_tensor,
    load_config,
    load_fcn3,
    load_manifest,
    preview_indices,
    staged_ics,
    utc_now,
    verify_ic_file,
    wall_time_estimate,
    write_failure_artifacts,
)
from inference_smoke import crop_nepal  # noqa: E402


def expand(p: str | Path) -> Path:
    return Path(str(p)).expanduser().resolve()


def parse_ic_time_str(s: str) -> str:
    s = s.replace("Z", "")
    if "T" not in s:
        s = s + "T00:00"
    return s


def run_crop_rollout(
    cfg: dict,
    ics: list[dict],
    n_members: int,
    n_steps: int,
    seed0: int,
    leads_h: list[int],
    keep_vars: list[str],
    box: dict,
    out_dir: Path,
    report: dict,
) -> dict:
    import numpy as np
    import torch

    model, device = load_fcn3(cfg, report)
    pidx, var_names = preview_indices(model.variables, keep_vars)
    if not pidx:
        raise RuntimeError(f"none of {keep_vars} in model.variables")
    report["variables"] = var_names
    report["variable_indices"] = pidx
    report["leads_h"] = leads_h
    report["n_steps"] = n_steps
    report["ensemble_members"] = n_members
    report["storage"] = "nepal_crop_selected_leads"
    want = set(int(x) for x in leads_h)

    fc_dir = out_dir / "forecast"
    fc_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    records = []
    t_all0 = time.time()
    with torch.inference_mode():
        for ic in ics:
            path = Path(ic["path"]).expanduser()
            ic_time = parse_ic_time_str(ic["time"])
            ic_id = ic.get("id")
            print(
                json.dumps(
                    {
                        "ic_start": ic_id,
                        "time": ic_time,
                        "n_members": n_members,
                        "n_steps": n_steps,
                        "leads_h": leads_h,
                        "variables": var_names,
                    }
                ),
                flush=True,
            )
            info = verify_ic_file(ic)
            if not info["ok"]:
                raise RuntimeError(f"{ic_id} IC verify failed: {info.get('error')}")
            x0, coords0 = build_ic_tensor(path, ic_time, model, device)
            member_stacks = []
            lat_c = lon_c = None
            for m in range(n_members):
                seed = seed0 + m
                model.set_rng(seed=seed, reset=True)
                x = x0.clone()
                coords = OrderedDict(
                    (k, (v.copy() if hasattr(v, "copy") else v)) for k, v in coords0.items()
                )
                it = model.create_iterator(x, coords)
                kept = []
                t_m0 = time.time()
                for step in range(n_steps + 1):
                    out, c = next(it)
                    lead_h = 6 * step
                    if lead_h in want:
                        crop, lat_c, lon_c = crop_nepal(out, c, box)
                        field = _squeeze_field(crop)
                        # field (V, h, w) on device
                        sel = field[pidx].float().cpu().numpy()
                        if not np.isfinite(sel).all():
                            raise RuntimeError(
                                f"non-finite crop {ic_id} m={m} lead={lead_h}"
                            )
                        kept.append(sel)
                    del out
                    if step % 5 == 0 or lead_h in want:
                        print(
                            json.dumps(
                                {
                                    "progress": {
                                        "ic": ic_id,
                                        "member": m,
                                        "step": step,
                                        "lead_h": lead_h,
                                        "kept": lead_h in want,
                                        "peak_mem": _peak_mem_gb(),
                                    }
                                }
                            ),
                            flush=True,
                        )
                if len(kept) != len(leads_h):
                    raise RuntimeError(
                        f"{ic_id} m={m}: kept {len(kept)} leads, want {leads_h}"
                    )
                stacked = np.stack(kept, axis=0)  # (L, C, H, W)
                member_stacks.append(stacked)
                print(
                    json.dumps(
                        {
                            "member_done": {
                                "ic": ic_id,
                                "member": m,
                                "seed": seed,
                                "shape": list(stacked.shape),
                                "wall_s": round(time.time() - t_m0, 2),
                            }
                        }
                    ),
                    flush=True,
                )
                del it, x, coords, kept
            ens = np.stack(member_stacks, axis=0)  # (M, L, C, H, W)
            mean = ens.mean(axis=0).astype(np.float32)
            stem = f"{ic_id}_{ic_time.replace('-', '').replace(':', '')[:13]}_crop.npz"
            # ic_time like 2018-01-15T00:00 → 20180115T0000
            iso = ic_time.replace("-", "").replace(":", "")
            if iso.endswith("00") and "T" in iso:
                pass
            stem = f"{ic_id}_{iso[:13]}_crop.npz"
            fpath = fc_dir / stem
            np.savez_compressed(
                fpath,
                ens_mean=mean,
                members=ens.astype(np.float32),
                variables=np.array(var_names),
                leads_h=np.asarray(leads_h, dtype=np.int32),
                lat=np.asarray(lat_c, dtype=np.float64),
                lon=np.asarray(lon_c, dtype=np.float64),
                ic_id=np.array(ic_id),
                ic_time=np.array(ic["time"]),
                seeds=np.asarray([seed0 + m for m in range(n_members)], dtype=np.int32),
                ic_source=np.array("global_era5_staged"),
            )
            rec = {
                "ic_id": ic_id,
                "ic_time": ic["time"],
                "season": ic.get("season"),
                "region_class": ic.get("region_class"),
                "path": str(fpath),
                "ens_mean_shape": list(mean.shape),
                "members_shape": list(ens.shape),
                "finite": bool(np.isfinite(ens).all()),
                "crop_lat": [float(lat_c[0]), float(lat_c[-1]), int(lat_c.size)],
                "crop_lon": [float(lon_c[0]), float(lon_c[-1]), int(lon_c.size)],
            }
            records.append(rec)
            print(json.dumps({"ic_done": rec}, default=str), flush=True)
            del ens, mean, member_stacks, x0, coords0
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    report["ics"] = records
    report["rollout_s"] = round(time.time() - t_all0, 2)
    report["peak_mem"] = _peak_mem_gb()
    report["gpu_used"] = True
    report["fcn3_inference"] = True
    report["frozen"] = True
    report["train"] = False
    report["ok"] = all(r["finite"] for r in records) and len(records) == len(ics)
    report["status"] = "crop_rollout_ok" if report["ok"] else "crop_rollout_partial"
    report["n_ics"] = len(records)
    return report


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Tier-0 global-IC → FCN3 Nepal crop (GPU; Manisha science path)"
    )
    ap.add_argument(
        "--config",
        type=Path,
        default=Path.home() / "fourcastnet" / "configs" / "tier0_bias.yaml",
    )
    ap.add_argument(
        "--g0-config",
        type=Path,
        default=Path.home() / "fourcastnet" / "configs" / "g0_base.yaml",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path.home() / "fourcastnet" / "data" / "g0_ics" / "g0_ic_manifest.json",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-gpu", action="store_true")
    ap.add_argument("--members", type=int, default=None)
    ap.add_argument("--max-ics", type=int, default=None)
    ap.add_argument("--ids", default=None, help="Comma-separated IC ids to roll")
    ap.add_argument("--splits", default=None, help="Comma splits from manifest (train/val/test)")
    ap.add_argument(
        "--pairs-dir",
        type=Path,
        default=None,
        help="Override pairs output dir (forecast/ written here)",
    )
    args = ap.parse_args()

    tcfg = load_config(args.config)
    gcfg = load_config(args.g0_config)
    box = tcfg["region"]["lat_lon_box"]
    leads = list(tcfg.get("leads_h", [24, 72, 120]))
    variables = list(tcfg.get("variables", ["t2m", "u10m", "v10m"]))
    if "tcwv" not in variables:
        variables = variables + ["tcwv"]
    n_steps = max(leads) // 6
    n_members = int(args.members or tcfg.get("rollout", {}).get("ensemble_members", 2))
    seed0 = int(tcfg.get("rollout", {}).get("seed0", 333))
    out_dir = expand(
        args.pairs_dir
        or tcfg.get("data", {}).get("pairs_dir", "~/fourcastnet/runs/phase0/tier0/pairs")
    )
    man = load_manifest(expand(args.manifest))
    ics = staged_ics(man)
    if args.splits:
        want_splits = {s.strip() for s in args.splits.split(",") if s.strip()}
        ics = [ic for ic in ics if ic.get("split") in want_splits]
    if args.ids:
        want = {x.strip() for x in args.ids.split(",") if x.strip()}
        ics = [ic for ic in ics if ic.get("id") in want]
    if args.max_ics:
        ics = ics[: int(args.max_ics)]

    report: dict[str, Any] = {
        "schema": "tier0_crop_rollout/v1",
        "created_utc": utc_now(),
        "script": "code/phase0/tier0_crop_rollout.py",
        "recipe": "GATE_RECIPE_TIER0_G0.md §2 + G0_BASE_CALL.md",
        "path": "B_crop_from_global_ic",
        "target_label": "ERA5_interim (paired separately)",
        "claim_level": "interim_era5",
        "box": box,
        "leads_h": leads,
        "variables": variables,
        "n_ics": len(ics),
        "ensemble_members": n_members,
        "steps": n_steps,
        "seed0": seed0,
        "ic_source": "data/g0_ics (global ERA5; not Random; not Nepal crop)",
        "nepal_crop_as_ic": False,
        "random_ic": False,
        "gpu_used": False,
        "ok": False,
        "wall_estimate": wall_time_estimate(
            n_ics=len(ics),
            n_members=n_members,
            n_steps=n_steps,
            s_per_step=5.0,
            load_s=30.0,
        ),
    }

    if args.dry_run:
        report["ok"] = True
        report["status"] = "dry_run"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "forecast_dry_run.json").write_text(json.dumps(report, indent=2, default=str))
        print(json.dumps(report, indent=2, default=str))
        return 0

    if not args.allow_gpu:
        report["status"] = "gpu_not_allowed"
        report["error"] = "need --allow-gpu (Manisha science-path greenlight)"
        print(json.dumps(report, indent=2))
        return 3

    tb_path = Path.home() / "fourcastnet" / "logs" / "tier0_crop_rollout.traceback.txt"
    man_path = out_dir / "forecast_manifest.json"
    try:
        run_crop_rollout(
            gcfg,
            ics,
            n_members,
            n_steps,
            seed0,
            leads,
            variables,
            box,
            out_dir,
            report,
        )
        report["created_utc_end"] = utc_now()
        out_dir.mkdir(parents=True, exist_ok=True)
        man_path.write_text(json.dumps(report, indent=2, default=str))
        report["forecast_manifest"] = str(man_path)
        print(json.dumps(report, indent=2, default=str), flush=True)
        return 0 if report.get("ok") else 1
    except Exception as e:
        write_failure_artifacts(report, man_path, tb_path, e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
