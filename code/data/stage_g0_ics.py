#!/usr/bin/env python3
"""Stage global ERA5 ICs for FCN3 G0 (Leonard GATE_RECIPE_TIER0_G0.md §3 step 1).

Fetches full-globe 0.25° (721×1440) fields for the 72 FCN3 channels in HF /
earth2studio channel order via ARCO ERA5 (preferred; no CDS competition with
regional backfill). Writes:

  data/g0_ics/ic_YYYYMMDDTHHMM_global.npy   # float32, shape (72, 721, 1440)
  data/g0_ics/g0_ic_manifest.json

Hard rules:
  - Never use Nepal regional crop as IC
  - Never write Random IC as a gate artifact
  - Do not touch ERA5 regional backfill processes
  - No GPU / no FCN3 rollout in this script
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
except ImportError as e:
    raise SystemExit("PyYAML required") from e


NEPAL_BOX = {"lat_south": 26.0, "lat_north": 31.0, "lon_west": 80.0, "lon_east": 89.0}


def load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def fcn3_channels() -> list[str]:
    """Canonical 72-channel order (matches models/fourcastnet3/config.json channel_names
    and earth2studio.models.px.fcn3.VARIABLES). Prefer local config, fall back to package.
    """
    local = Path("~/fourcastnet/models/fourcastnet3/config.json").expanduser()
    if local.is_file():
        cfg = json.loads(local.read_text())
        names = list(cfg.get("channel_names") or [])
        if len(names) == 72:
            return names
    from earth2studio.models.px.fcn3 import VARIABLES

    return list(VARIABLES)


def time_tag(dt: datetime) -> str:
    return dt.strftime("%Y%m%dT%H%M")


def parse_time(s: str) -> datetime:
    # Always treat as UTC naive wall-clock (ERA5 convention)
    s = s.strip().replace("Z", "")
    if "T" in s:
        return datetime.fromisoformat(s)
    return datetime.fromisoformat(s + "T00:00:00")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def manifest_path(out_dir: Path) -> Path:
    return out_dir / "g0_ic_manifest.json"


def load_manifest(out_dir: Path) -> dict:
    p = manifest_path(out_dir)
    if p.is_file():
        return json.loads(p.read_text())
    return {
        "schema": "g0_ic_manifest/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "updated_utc": None,
        "spec": {
            "grid": "0.25deg_global",
            "shape": [72, 721, 1440],
            "dtype": "float32",
            "layout": "channel_lat_lon",
            "init_hour_utc": 0,
            "channel_order_source": "fourcastnet3/config.json:channel_names (== earth2studio FCN3.VARIABLES)",
            "forbidden": [
                "Nepal regional crop as IC",
                "Random / synthetic IC as gate artifact",
            ],
            "nepal_box_locked": NEPAL_BOX,
        },
        "source_default": None,
        "channels": [],
        "ics": [],
        "counts": {},
        "notes": [],
    }


def save_manifest(out_dir: Path, man: dict) -> None:
    man["updated_utc"] = datetime.now(timezone.utc).isoformat()
    # recompute counts
    ics = man.get("ics") or []
    seasons = {}
    classes = {}
    staged = 0
    for e in ics:
        if e.get("status") == "staged":
            staged += 1
            seasons[e.get("season", "?")] = seasons.get(e.get("season", "?"), 0) + 1
            classes[e.get("region_class", "?")] = classes.get(e.get("region_class", "?"), 0) + 1
    man["counts"] = {
        "listed": len(ics),
        "staged": staged,
        "by_season_staged": seasons,
        "by_region_class_staged": classes,
    }
    p = manifest_path(out_dir)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(man, indent=2, sort_keys=False) + "\n")
    tmp.replace(p)


def upsert_ic(man: dict, entry: dict) -> None:
    ics = man.setdefault("ics", [])
    for i, e in enumerate(ics):
        if e.get("id") == entry["id"] or e.get("time") == entry["time"]:
            ics[i] = entry
            return
    ics.append(entry)


def validate_array(arr, channels: list[str]) -> dict:
    import numpy as np

    info = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "finite": bool(np.isfinite(arr).all()),
        "nan_count": int(np.isnan(arr).sum()),
        "inf_count": int(np.isinf(arr).sum()),
    }
    if arr.shape != (len(channels), 721, 1440):
        raise RuntimeError(f"bad shape {arr.shape}; expected ({len(channels)}, 721, 1440)")
    if arr.dtype != np.float32:
        raise RuntimeError(f"bad dtype {arr.dtype}; expected float32")
    if not info["finite"]:
        raise RuntimeError(f"non-finite values: nan={info['nan_count']} inf={info['inf_count']}")
    # sanity: t2m roughly 180–330 K; msl ~5e4–1.1e5
    try:
        t2m = arr[channels.index("t2m")]
        msl = arr[channels.index("msl")]
        info["t2m_min"] = float(t2m.min())
        info["t2m_max"] = float(t2m.max())
        info["msl_min"] = float(msl.min())
        info["msl_max"] = float(msl.max())
        if not (180.0 < info["t2m_min"] < 330.0 and 180.0 < info["t2m_max"] < 340.0):
            raise RuntimeError(f"t2m out of physical range: {info['t2m_min']}..{info['t2m_max']}")
        if not (45000.0 < info["msl_min"] < 110000.0):
            raise RuntimeError(f"msl out of physical range: {info['msl_min']}..{info['msl_max']}")
    except ValueError:
        pass
    return info


def fetch_arco(dt: datetime, channels: list[str], verbose: bool = True):
    """Fetch one global timestep from ARCO ERA5 zarr; return float32 (C,H,W).

    Prefer direct zarr reads with retries (pressure vars are ~150MB/chunk).
    Fall back to earth2studio.data.ARCO if direct import fails.
    """
    import numpy as np

    # Prefer sibling helper (code/phase0/arco_direct_fetch.py)
    try:
        from arco_direct_fetch import fetch_timestep_array

        return fetch_timestep_array(channels, dt, verbose=verbose)
    except ImportError:
        pass
    # path-based import when cwd is repo root
    import importlib.util

    helper = Path(__file__).resolve().parent / "arco_direct_fetch.py"
    if helper.is_file():
        spec = importlib.util.spec_from_file_location("arco_direct_fetch", helper)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        return mod.fetch_timestep_array(channels, dt, verbose=verbose)

    # last resort: earth2studio ARCO
    from earth2studio.data import ARCO

    ds = ARCO(cache=True, verbose=verbose, async_timeout=900)
    t0 = time.time()
    da = ds(dt, channels)
    elapsed = time.time() - t0
    var_names = [str(v) for v in da["variable"].values]
    if var_names != channels:
        idx = [var_names.index(c) for c in channels]
        da = da.isel(variable=idx)
    arr = np.asarray(da.isel(time=0).values, dtype=np.float32)
    if arr.shape != (len(channels), 721, 1440):
        raise RuntimeError(f"ARCO returned shape {arr.shape}")
    meta = {
        "source": "earth2studio.data.ARCO",
        "source_url": "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
        "product": "ERA5_final_ARCO",
        "fetch_s": round(elapsed, 2),
        "lat0": float(da["lat"].values[0]),
        "latN": float(da["lat"].values[-1]),
        "lon0": float(da["lon"].values[0]),
        "lonN": float(da["lon"].values[-1]),
    }
    return arr, meta


def stage_one(
    ic: dict,
    channels: list[str],
    out_dir: Path,
    man: dict,
    source: str,
    force: bool,
    verbose: bool,
) -> dict:
    import numpy as np

    dt = parse_time(ic["time"])
    if dt.hour != 0 or dt.minute != 0:
        print(f"WARN: init hour is {dt.hour:02d}{dt.minute:02d}, recipe prefers 00 UTC", flush=True)
    tag = time_tag(dt)
    out_name = f"ic_{tag}_global.npy"
    out_path = out_dir / out_name

    entry = {
        "id": ic["id"],
        "time": dt.strftime("%Y-%m-%dT%H:%M:%S"),
        "time_tag": tag,
        "season": ic.get("season"),
        "region_class": ic.get("region_class"),
        "region_label": ic.get("region_label"),
        "rationale": ic.get("rationale"),
        "path": str(out_path),
        "relative_path": out_name,
        "status": "pending",
    }

    if out_path.is_file() and not force:
        arr = np.load(out_path, mmap_mode="r")
        try:
            vinfo = validate_array(np.asarray(arr), channels)
            # Preserve prior source/meta if present
            prior = None
            for e in man.get("ics") or []:
                if e.get("id") == ic["id"] or e.get("time_tag") == tag:
                    prior = e
                    break
            entry.update(
                {
                    "status": "staged",
                    "bytes": out_path.stat().st_size,
                    "sha256": sha256_file(out_path),
                    "validation": vinfo,
                    "skipped": True,
                    "note": "existing file validated; use --force to re-fetch",
                    "is_global": True,
                    "is_nepal_crop": False,
                    "is_random": False,
                }
            )
            if prior and prior.get("source"):
                entry["source"] = prior["source"]
            elif not entry.get("source"):
                entry["source"] = {
                    "source": "arco_zarr_direct",
                    "source_url": "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
                    "product": "ERA5_final_ARCO",
                }
            upsert_ic(man, entry)
            save_manifest(out_dir, man)
            print(f"[skip] {ic['id']} {tag} already staged ({out_path.stat().st_size} bytes)", flush=True)
            return entry
        except Exception as e:
            print(f"[re-fetch] {ic['id']} existing failed validation: {e}", flush=True)

    print(f"[fetch] {ic['id']} {tag} via {source} …", flush=True)
    t0 = time.time()
    if source == "arco":
        arr, meta = fetch_arco(dt, channels, verbose=verbose)
    else:
        raise SystemExit(f"unsupported source: {source} (CDS deliberately not default; avoid competing with backfill)")

    vinfo = validate_array(arr, channels)
    # atomic write (file handle avoids np.save appending an extra .npy)
    tmp = out_path.with_name(out_path.name + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, arr)
    tmp.replace(out_path)

    entry.update(
        {
            "status": "staged",
            "bytes": out_path.stat().st_size,
            "sha256": sha256_file(out_path),
            "validation": vinfo,
            "source": meta,
            "wall_s": round(time.time() - t0, 2),
            "skipped": False,
            "is_global": True,
            "is_nepal_crop": False,
            "is_random": False,
        }
    )
    upsert_ic(man, entry)
    save_manifest(out_dir, man)
    print(
        f"[ok] {ic['id']} {tag} → {out_name} "
        f"{entry['bytes']} bytes sha256={entry['sha256'][:12]}… "
        f"wall={entry['wall_s']}s t2m=[{vinfo.get('t2m_min'):.1f},{vinfo.get('t2m_max'):.1f}]",
        flush=True,
    )
    return entry


def check_quota_rules(man: dict) -> list[str]:
    """Return warnings if staged set does not yet meet Leonard N≥8 mix."""
    warns = []
    staged = [e for e in man.get("ics", []) if e.get("status") == "staged"]
    if len(staged) < 8:
        warns.append(f"staged {len(staged)} < 8")
    seasons = {}
    classes = {}
    for e in staged:
        seasons[e.get("season")] = seasons.get(e.get("season"), 0) + 1
        classes[e.get("region_class")] = classes.get(e.get("region_class"), 0) + 1
    if seasons.get("DJF", 0) < 2:
        warns.append(f"DJF staged {seasons.get('DJF', 0)} < 2")
    if seasons.get("JJA", 0) < 2:
        warns.append(f"JJA staged {seasons.get('JJA', 0)} < 2")
    if classes.get("non_asia", 0) < 4:
        warns.append(f"non_asia staged {classes.get('non_asia', 0)} < 4")
    if classes.get("asia_ex_nepal", 0) < 2:
        warns.append(f"asia_ex_nepal staged {classes.get('asia_ex_nepal', 0)} < 2")
    if classes.get("monsoon_adjacent", 0) < 2:
        warns.append(f"monsoon_adjacent staged {classes.get('monsoon_adjacent', 0)} < 2")
    return warns


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--config",
        default=str(Path("~/fourcastnet/configs/g0_ics.yaml").expanduser()),
        help="YAML listing ≥8 IC dates + region rationale",
    )
    ap.add_argument("--out-dir", default=None, help="Override out_dir from config")
    ap.add_argument("--source", default=None, choices=["arco"], help="Data source (default: config/arco)")
    ap.add_argument("--smoke", action="store_true", help="Stage first 2 ICs only (ic01, ic02)")
    ap.add_argument("--ids", default=None, help="Comma-separated IC ids to stage")
    ap.add_argument("--all", action="store_true", help="Stage all ICs in config")
    ap.add_argument("--force", action="store_true", help="Re-fetch even if file exists")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="Print plan + write empty manifest scaffold")
    args = ap.parse_args(argv)

    cfg_path = Path(args.config).expanduser()
    if not cfg_path.is_file():
        print(f"config missing: {cfg_path}", file=sys.stderr)
        return 2
    cfg = load_cfg(cfg_path)
    out_dir = Path(args.out_dir or cfg.get("out_dir") or "~/fourcastnet/data/g0_ics").expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    source = args.source or cfg.get("source") or "arco"

    channels = fcn3_channels()
    if len(channels) != 72:
        print(f"ERROR: expected 72 channels, got {len(channels)}", file=sys.stderr)
        return 2

    ics = list(cfg.get("ics") or [])
    if args.smoke:
        selected = ics[:2]
    elif args.ids:
        want = {x.strip() for x in args.ids.split(",") if x.strip()}
        selected = [x for x in ics if x["id"] in want]
        missing = want - {x["id"] for x in selected}
        if missing:
            print(f"unknown ids: {sorted(missing)}", file=sys.stderr)
            return 2
    elif args.all:
        selected = ics
    else:
        # default: all (callers should pass --smoke or --all explicitly in automation)
        selected = ics

    man = load_manifest(out_dir)
    man["source_default"] = source
    man["channels"] = channels
    man["config_path"] = str(cfg_path)
    man.setdefault("notes", [])
    note = (
        "Staged via ARCO ERA5 (Google cloud zarr) to avoid competing with CDS "
        "regional backfill PID; product is ERA5 final ARCO curation."
    )
    if note not in man["notes"]:
        man["notes"].append(note)

    print(
        f"out_dir={out_dir} source={source} n_selected={len(selected)} "
        f"channels={len(channels)} dry_run={args.dry_run}",
        flush=True,
    )
    for ic in selected:
        print(
            f"  plan {ic['id']} {ic['time']} {ic.get('season')} "
            f"{ic.get('region_class')}/{ic.get('region_label')}",
            flush=True,
        )

    if args.dry_run:
        # ensure every listed IC appears as planned
        for ic in ics:
            dt = parse_time(ic["time"])
            upsert_ic(
                man,
                {
                    "id": ic["id"],
                    "time": dt.strftime("%Y-%m-%dT%H:%M:%S"),
                    "time_tag": time_tag(dt),
                    "season": ic.get("season"),
                    "region_class": ic.get("region_class"),
                    "region_label": ic.get("region_label"),
                    "rationale": ic.get("rationale"),
                    "status": "planned",
                    "relative_path": f"ic_{time_tag(dt)}_global.npy",
                },
            )
        save_manifest(out_dir, man)
        print(json.dumps({"ok": True, "dry_run": True, "manifest": str(manifest_path(out_dir))}, indent=2))
        return 0

    failures = []
    for ic in selected:
        try:
            stage_one(
                ic,
                channels=channels,
                out_dir=out_dir,
                man=man,
                source=source,
                force=args.force,
                verbose=not args.quiet,
            )
        except Exception as e:
            failures.append({"id": ic["id"], "error": str(e), "traceback": traceback.format_exc()})
            upsert_ic(
                man,
                {
                    "id": ic["id"],
                    "time": ic["time"],
                    "season": ic.get("season"),
                    "region_class": ic.get("region_class"),
                    "status": "failed",
                    "error": str(e),
                },
            )
            save_manifest(out_dir, man)
            print(f"[FAIL] {ic['id']}: {e}", flush=True)
            traceback.print_exc()

    warns = check_quota_rules(man)
    summary = {
        "ok": len(failures) == 0,
        "staged": man["counts"].get("staged", 0),
        "listed": man["counts"].get("listed", 0),
        "manifest": str(manifest_path(out_dir)),
        "failures": failures,
        "quota_warnings": warns,
        "gpu_used": False,
        "nepal_crop_used_as_ic": False,
        "random_ic": False,
    }
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
