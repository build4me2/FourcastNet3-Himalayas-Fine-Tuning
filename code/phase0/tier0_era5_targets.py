#!/usr/bin/env python3
"""Fetch ERA5_interim Nepal-box targets at G0 IC + leads (CPU / ARCO).

Does NOT use CDS — leaves regional backfill PID 611595 alone.
Source: ARCO ERA5 final zarr (same as G0 ICs).

Writes:
  runs/phase0/tier0/pairs/targets/{ic_id}_{YYYYmmddTHHMM}_era5.npz
  runs/phase0/tier0/pairs/targets_manifest.json

Optional local CDS cross-check when a monthly raw file covers the time
(currently only 2020-01 / ic03 among G0 ICs).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from arco_direct_fetch import fetch_crop_array  # noqa: E402

CDS_NAME = {"t2m": "t2m", "u10m": "u10", "v10m": "v10", "tcwv": "tcwv"}


def load_config(path: Path) -> dict:
    if yaml is None:
        raise RuntimeError("PyYAML required")
    return yaml.safe_load(path.read_text())


def expand(p: str | Path) -> Path:
    return Path(str(p)).expanduser().resolve()


def parse_ic_time(s: str) -> datetime:
    s = s.replace("Z", "")
    if "T" not in s:
        s = s + "T00:00:00"
    if len(s) == 16:
        s = s + ":00"
    return datetime.fromisoformat(s)


def try_local_cds_crop(
    raw_dir: Path,
    dt: datetime,
    channels: list[str],
    lat_ref,
    lon_ref,
) -> dict | None:
    """If era5_single_YYYY-MM.nc exists, extract the matching 6-h frame."""
    import numpy as np
    import netCDF4 as nc

    fn = raw_dir / f"era5_single_{dt.year:04d}-{dt.month:02d}.nc"
    if not fn.is_file():
        return None
    ds = nc.Dataset(fn)
    try:
        if "valid_time" in ds.variables:
            vt = np.asarray(ds.variables["valid_time"][:], dtype=np.int64)
            units = getattr(ds.variables["valid_time"], "units", "seconds since 1970-01-01")
        elif "time" in ds.variables:
            vt = np.asarray(ds.variables["time"][:], dtype=np.int64)
            units = getattr(ds.variables["time"], "units", "")
        else:
            return None
        target = int(dt.replace(tzinfo=timezone.utc).timestamp())
        if "seconds since 1970" in units:
            times = vt
        else:
            # fallback: treat as hours since 1900
            epoch = datetime(1900, 1, 1, tzinfo=timezone.utc)
            times = np.array(
                [int((epoch + timedelta(hours=int(h))).timestamp()) for h in vt],
                dtype=np.int64,
            )
        hits = np.where(times == target)[0]
        if hits.size == 0:
            return None
        ti = int(hits[0])
        lat = np.asarray(ds.variables["latitude"][:], dtype=np.float64)
        lon = np.asarray(ds.variables["longitude"][:], dtype=np.float64)
        fields = {}
        for ch in channels:
            cds = CDS_NAME.get(ch, ch)
            if cds not in ds.variables:
                return None
            fields[ch] = np.asarray(ds.variables[cds][ti], dtype=np.float32)
        return {
            "ok": True,
            "path": str(fn),
            "time_index": ti,
            "lat": lat,
            "lon": lon,
            "fields": fields,
        }
    finally:
        ds.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Tier-0 ERA5_interim crop targets (ARCO, CPU)")
    ap.add_argument(
        "--config",
        type=Path,
        default=Path.home() / "fourcastnet" / "configs" / "tier0_bias.yaml",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path.home() / "fourcastnet" / "data" / "g0_ics" / "g0_ic_manifest.json",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--ids", default=None, help="Comma-separated IC ids")
    ap.add_argument("--splits", default=None, help="Comma splits train/val/test")
    ap.add_argument("--pairs-dir", type=Path, default=None, help="Override pairs dir")
    args = ap.parse_args()

    cfg = load_config(args.config)
    box = cfg["region"]["lat_lon_box"]
    leads = list(cfg.get("leads_h", [24, 72, 120]))
    variables = list(cfg.get("variables", ["t2m", "u10m", "v10m"]))
    if "tcwv" not in variables:
        variables = variables + ["tcwv"]
    data = cfg.get("data", {})
    out_dir = expand(args.pairs_dir or data.get("pairs_dir", "~/fourcastnet/runs/phase0/tier0/pairs"))
    tgt_dir = out_dir / "targets"
    raw_dir = expand(data.get("era5_raw_dir", "~/fourcastnet/data/era5/raw"))

    man = json.loads(expand(args.manifest).read_text())
    ics = [ic for ic in man.get("ics", []) if ic.get("status") == "staged"]
    if args.splits:
        want_splits = {s.strip() for s in args.splits.split(",") if s.strip()}
        ics = [ic for ic in ics if ic.get("split") in want_splits]
    if args.ids:
        want = {x.strip() for x in args.ids.split(",") if x.strip()}
        ics = [ic for ic in ics if ic.get("id") in want]
    report: dict[str, Any] = {
        "script": "code/phase0/tier0_era5_targets.py",
        "target": "ERA5_interim",
        "source": "arco_zarr_direct_crop",
        "cds_used": False,
        "box": box,
        "leads_h": leads,
        "variables": variables,
        "n_ics": len(ics),
        "utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": bool(args.dry_run),
    }
    needed = []
    for ic in ics:
        ic_dt = parse_ic_time(ic["time"])
        for lh in leads:
            needed.append(
                {
                    "ic_id": ic.get("id"),
                    "ic_time": ic["time"],
                    "lead_h": lh,
                    "valid_time": (ic_dt + timedelta(hours=int(lh))).isoformat(),
                }
            )
    report["n_frames"] = len(needed)
    if args.dry_run:
        report["ok"] = True
        report["status"] = "dry_run"
        report["sample"] = needed[:3]
        print(json.dumps(report, indent=2))
        return 0

    import numpy as np

    tgt_dir.mkdir(parents=True, exist_ok=True)
    records = []
    t0 = datetime.now(timezone.utc)
    for ic in ics:
        ic_id = ic.get("id")
        ic_dt = parse_ic_time(ic["time"])
        frames = []
        metas = []
        cds_notes = []
        for lh in leads:
            vt = ic_dt + timedelta(hours=int(lh))
            print(json.dumps({"fetch": ic_id, "lead_h": lh, "valid": vt.isoformat()}), flush=True)
            arr, meta = fetch_crop_array(
                variables,
                vt,
                lat_south=float(box["lat_south"]),
                lat_north=float(box["lat_north"]),
                lon_west=float(box["lon_west"]),
                lon_east=float(box["lon_east"]),
                verbose=True,
            )
            frames.append(arr)
            metas.append(meta)
            lat = np.asarray(meta["lat"], dtype=np.float64)
            lon = np.asarray(meta["lon"], dtype=np.float64)
            local = try_local_cds_crop(raw_dir, vt, variables, lat, lon)
            if local and local.get("ok"):
                diffs = {}
                for i, ch in enumerate(variables):
                    a = arr[i]
                    b = local["fields"][ch]
                    if a.shape != b.shape:
                        diffs[ch] = {"shape_mismatch": [list(a.shape), list(b.shape)]}
                    else:
                        d = np.abs(a.astype(np.float64) - b.astype(np.float64))
                        diffs[ch] = {
                            "max_abs": float(d.max()),
                            "mean_abs": float(d.mean()),
                        }
                cds_notes.append({"lead_h": lh, "path": local["path"], "diff": diffs})
        stack = np.stack(frames, axis=0)  # (L, C, H, W)
        stem = f"{ic_id}_{ic_dt.strftime('%Y%m%dT%H%M')}_era5.npz"
        path = tgt_dir / stem
        np.savez_compressed(
            path,
            data=stack,
            variables=np.array(variables),
            leads_h=np.asarray(leads, dtype=np.int32),
            lat=lat,
            lon=lon,
            ic_id=np.array(ic_id),
            ic_time=np.array(ic["time"]),
            target=np.array("ERA5_interim"),
            source=np.array("arco_zarr_direct_crop"),
        )
        rec = {
            "ic_id": ic_id,
            "ic_time": ic["time"],
            "path": str(path),
            "shape": list(stack.shape),
            "finite": bool(np.isfinite(stack).all()),
            "cds_overlap": cds_notes,
        }
        records.append(rec)
        print(json.dumps({"wrote": rec}, default=str), flush=True)

    man_out = {
        **report,
        "ok": True,
        "status": "targets_ok",
        "ics": records,
        "wall_s": round((datetime.now(timezone.utc) - t0).total_seconds(), 2),
        "claim_level": "interim_era5",
        "year_split_frozen": True,
        "provisional_years": True,
        "year_split_note": (
            "PROVISIONAL year split (Manisha not hard-locked): "
            "train 2018-2021 / val 2022 / test 2023-2024."
        ),
    }
    man_path = out_dir / "targets_manifest.json"
    man_path.write_text(json.dumps(man_out, indent=2, default=str))
    print(json.dumps({"ok": True, "manifest": str(man_path), "n": len(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
