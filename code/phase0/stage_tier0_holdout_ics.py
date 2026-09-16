#!/usr/bin/env python3
"""Stage Tier-0 holdout global ERA5 ICs with train/val/test split manifest.

- Train ICs: hardlink/reuse existing data/g0_ics/*.npy (2018–2021)
- Val/test: fetch via ARCO (same path as stage_g0_ics.py)
- Writes data/tier0_ics/tier0_ic_manifest.json with split labels
- provisional_years=false; year_split_frozen=true (HARD — YEAR_HARD_LOCK.md)

Never uses Nepal crop as IC. Leaves CDS ERA5 regional PID alone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from stage_g0_ics import (  # noqa: E402
    fcn3_channels,
    fetch_arco,
    parse_time,
    time_tag,
    validate_array,
)


def expand(p: str | Path) -> Path:
    return Path(str(p)).expanduser().resolve()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def hardlink_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "exists"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        default=str(Path("~/fourcastnet/configs/tier0_holdout_ics.yaml").expanduser()),
    )
    ap.add_argument("--splits", default="train,val,test", help="Comma splits to stage")
    ap.add_argument("--ids", default=None, help="Comma-separated IC ids to stage (incremental)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if yaml is None:
        raise SystemExit("PyYAML required")
    cfg = yaml.safe_load(Path(args.config).expanduser().read_text())
    out_dir = expand(cfg.get("out_dir", "~/fourcastnet/data/tier0_ics"))
    out_dir.mkdir(parents=True, exist_ok=True)
    want_splits = {s.strip() for s in args.splits.split(",") if s.strip()}
    channels = fcn3_channels()
    g0_man = Path("~/fourcastnet/data/g0_ics/g0_ic_manifest.json").expanduser()
    if g0_man.is_file():
        channels = json.loads(g0_man.read_text()).get("channels") or channels

    ics_cfg = [ic for ic in cfg.get("ics", []) if ic.get("split") in want_splits]
    if args.ids:
        want_ids = {x.strip() for x in args.ids.split(",") if x.strip()}
        ics_cfg = [ic for ic in ics_cfg if ic.get("id") in want_ids]
        missing = want_ids - {ic.get("id") for ic in ics_cfg}
        if missing:
            raise SystemExit(f"unknown/filtered ids: {sorted(missing)}")
    man_path = out_dir / "tier0_ic_manifest.json"
    if man_path.is_file():
        man = json.loads(man_path.read_text())
        man["channels"] = channels or man.get("channels")
        man["provisional_years"] = bool(cfg.get("provisional_years", False))
        man["year_split_frozen"] = bool(cfg.get("year_split_frozen", True))
        man["year_split"] = cfg.get("year_split") or man.get("year_split")
        man["year_split_note"] = (
            "HARD year split (Manisha locked 2026-09-13; docs/research/YEAR_HARD_LOCK.md). "
            "train 2018-2021 / val 2022 / test 2023-2024. "
            "provisional_years=false; year_split_frozen=true; claim_level stays interim_era5; g1_claimable=false."
        )
        man.setdefault("ics", [])
        man["config_path"] = str(expand(args.config))
        if cfg.get("holdout_expand") is not None:
            man["holdout_expand"] = cfg.get("holdout_expand")
    else:
        man = {
            "schema": "tier0_ic_manifest/v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config_path": str(expand(args.config)),
            "provisional_years": bool(cfg.get("provisional_years", False)),
            "year_split_frozen": bool(cfg.get("year_split_frozen", True)),
            "year_split_note": (
                "HARD year split (Manisha locked 2026-09-13; docs/research/YEAR_HARD_LOCK.md). "
                "train 2018-2021 / val 2022 / test 2023-2024. "
                "provisional_years=false; claim_level stays interim_era5; g1_claimable=false."
            ),
            "year_split": cfg.get("year_split"),
            "spec": {
                "grid": "0.25deg_global",
                "shape": [72, 721, 1440],
                "dtype": "float32",
                "layout": "channel_lat_lon",
                "init_hour_utc": 0,
                "forbidden": ["Nepal regional crop as IC", "Random / synthetic IC as gate artifact"],
            },
            "source_default": "arco",
            "channels": channels,
            "ics": [],
            "counts": {},
        }

    if args.dry_run:
        print(json.dumps({
            "dry_run": True,
            "out_dir": str(out_dir),
            "n": len(ics_cfg),
            "by_split": {s: sum(1 for ic in ics_cfg if ic.get("split") == s) for s in sorted(want_splits)},
            "ids": [ic["id"] for ic in ics_cfg],
        }, indent=2))
        return 0

    import numpy as np

    for ic in ics_cfg:
        dt = parse_time(ic["time"])
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
            "split": ic.get("split"),
            "year": dt.year,
            "path": str(out_path),
            "relative_path": out_name,
            "is_global": True,
            "is_nepal_crop": False,
            "is_random": False,
        }
        reuse = ic.get("reuse_from")
        t0 = time.time()
        if reuse:
            src = expand(reuse)
            if not src.is_file():
                raise FileNotFoundError(f"reuse_from missing: {src}")
            mode = hardlink_or_copy(src, out_path) if (args.force or not out_path.exists()) else "exists"
            if out_path.exists() and not args.force and mode == "exists":
                pass
            elif args.force and out_path.exists() and mode == "exists":
                out_path.unlink()
                mode = hardlink_or_copy(src, out_path)
            arr = np.load(out_path, mmap_mode="r")
            info = validate_array(np.asarray(arr[:]), channels)
            entry.update({
                "status": "staged",
                "bytes": out_path.stat().st_size,
                "sha256": sha256_file(out_path),
                "validation": info,
                "source": {"source": "reuse_g0_ics", "reuse_from": str(src), "link_mode": mode},
                "wall_s": round(time.time() - t0, 2),
                "skipped": mode == "exists",
            })
        else:
            if out_path.exists() and not args.force:
                arr = np.load(out_path, mmap_mode="r")
                info = validate_array(np.asarray(arr[:]), channels)
                entry.update({
                    "status": "staged",
                    "bytes": out_path.stat().st_size,
                    "sha256": sha256_file(out_path),
                    "validation": info,
                    "skipped": True,
                    "note": "existing file validated; use --force to re-fetch",
                })
            else:
                if not args.quiet:
                    print(json.dumps({"fetch": ic["id"], "time": entry["time"], "split": ic.get("split")}), flush=True)
                arr, meta = fetch_arco(dt, channels, verbose=not args.quiet)
                info = validate_array(arr, channels)
                np.save(out_path, arr)
                entry.update({
                    "status": "staged",
                    "bytes": out_path.stat().st_size,
                    "sha256": sha256_file(out_path),
                    "validation": info,
                    "source": meta,
                    "wall_s": round(time.time() - t0, 2),
                    "skipped": False,
                })
        replaced = False
        for i, e in enumerate(man["ics"]):
            if e.get("id") == entry["id"] or e.get("time_tag") == entry.get("time_tag"):
                man["ics"][i] = entry
                replaced = True
                break
        if not replaced:
            man["ics"].append(entry)
        if not args.quiet:
            print(json.dumps({"staged": {k: entry[k] for k in ("id", "split", "time_tag", "status", "bytes", "skipped") if k in entry}}, default=str), flush=True)

    # counts
    staged = [e for e in man["ics"] if e.get("status") == "staged"]
    by_split = {}
    for e in staged:
        by_split[e.get("split", "?")] = by_split.get(e.get("split", "?"), 0) + 1
    man["counts"] = {
        "listed": len(man["ics"]),
        "staged": len(staged),
        "by_split": by_split,
        "by_season_staged": {},
        "by_region_class_staged": {},
    }
    for e in staged:
        man["counts"]["by_season_staged"][e.get("season", "?")] = (
            man["counts"]["by_season_staged"].get(e.get("season", "?"), 0) + 1
        )
        man["counts"]["by_region_class_staged"][e.get("region_class", "?")] = (
            man["counts"]["by_region_class_staged"].get(e.get("region_class", "?"), 0) + 1
        )
    if cfg.get("holdout_expand") is not None:
        man["holdout_expand"] = cfg.get("holdout_expand")
    man["updated_utc"] = datetime.now(timezone.utc).isoformat()
    man_path = out_dir / "tier0_ic_manifest.json"
    man_path.write_text(json.dumps(man, indent=2, default=str))
    print(json.dumps({"ok": True, "manifest": str(man_path), "counts": man["counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
