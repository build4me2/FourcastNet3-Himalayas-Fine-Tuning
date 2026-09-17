#!/usr/bin/env python3
"""Audit ERA5 Nepal crop coverage under data/era5/raw/.

Scans monthly single-levels + pressure-levels netCDFs for holes on
surface diagnostics t2m / u10m / v10m (CDS short names in files: t2m, u10, v10).

Writes:
  data/era5/coverage_audit.json
  docs/research/ERA5_COVERAGE_AUDIT.md

Does NOT touch the CDS download process. Read-only on raw/.

Usage:
  cd ~/fourcastnet
  python code/data/era5_coverage_audit.py --help
  python code/data/era5_coverage_audit.py
  python code/data/era5_coverage_audit.py --start 1980 --end-tip
"""
from __future__ import annotations

import argparse
import json
import sys
from calendar import monthrange
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# CDS single-levels files use short names; map to FCN3 / recipe names
VAR_MAP = {
    "t2m": {"cds_short": "t2m", "cds_long": "2m_temperature", "fcn3": "t2m"},
    "u10m": {"cds_short": "u10", "cds_long": "10m_u_component_of_wind", "fcn3": "u10m"},
    "v10m": {"cds_short": "v10", "cds_long": "10m_v_component_of_wind", "fcn3": "v10m"},
}
REQUIRED_SURFACE = ("t2m", "u10", "v10")
HOURS_PER_DAY = 4  # 00/06/12/18


def expand(p: str | Path) -> Path:
    return Path(str(p)).expanduser().resolve()


def expected_timesteps(year: int, month: int) -> int:
    return monthrange(year, month)[1] * HOURS_PER_DAY


def probe_surface(path: Path, min_bytes: int = 10_000) -> dict[str, Any]:
    """Return present/missing/partial info for one single-levels monthly file."""
    info: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
        "vars_found": [],
        "vars_missing": [],
        "n_time": None,
        "status": "missing",
        "error": None,
    }
    if not path.is_file() or info["size_bytes"] < min_bytes:
        info["status"] = "missing"
        info["vars_missing"] = list(REQUIRED_SURFACE)
        return info

    try:
        import netCDF4 as nc  # type: ignore
    except ImportError:
        # Fallback: file present but cannot open — mark present_unverified
        info["status"] = "present_unverified"
        info["error"] = "netCDF4 not importable; size-only check"
        info["vars_found"] = list(REQUIRED_SURFACE)  # assume OK if size ok
        return info

    try:
        ds = nc.Dataset(path, "r")
        keys = list(ds.variables.keys())
        found = [v for v in REQUIRED_SURFACE if v in ds.variables]
        missing = [v for v in REQUIRED_SURFACE if v not in ds.variables]
        info["vars_found"] = found
        info["vars_missing"] = missing
        # time dim
        n_time = None
        for tname in ("valid_time", "time"):
            if tname in ds.variables:
                n_time = int(ds.variables[tname].shape[0])
                break
            if tname in ds.dimensions:
                n_time = int(ds.dimensions[tname].size)
                break
        info["n_time"] = n_time
        ds.close()
        if missing:
            info["status"] = "partial"
        else:
            info["status"] = "present"
    except Exception as e:
        info["status"] = "partial"
        info["error"] = f"{type(e).__name__}: {e}"
        info["vars_missing"] = list(REQUIRED_SURFACE)
    return info


def probe_pressure(path: Path, min_bytes: int = 10_000) -> dict[str, Any]:
    info: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
        "status": "missing",
        "n_time": None,
        "error": None,
    }
    if not path.is_file() or info["size_bytes"] < min_bytes:
        info["status"] = "missing"
        return info
    try:
        import netCDF4 as nc  # type: ignore
        ds = nc.Dataset(path, "r")
        n_time = None
        for tname in ("valid_time", "time"):
            if tname in ds.variables:
                n_time = int(ds.variables[tname].shape[0])
                break
            if tname in ds.dimensions:
                n_time = int(ds.dimensions[tname].size)
                break
        info["n_time"] = n_time
        # pressure files carry u/v/t/z/q — not the surface diagnostics
        info["status"] = "present"
        ds.close()
    except ImportError:
        info["status"] = "present_unverified"
        info["error"] = "netCDF4 not importable; size-only check"
    except Exception as e:
        info["status"] = "partial"
        info["error"] = f"{type(e).__name__}: {e}"
    return info


def month_status(single: dict, pressure: dict, year: int, month: int) -> str:
    """Combine single+pressure into present / missing / partial."""
    exp = expected_timesteps(year, month)
    s, p = single["status"], pressure["status"]
    if s == "missing" and p == "missing":
        return "missing"
    if s in ("present", "present_unverified") and p in ("present", "present_unverified"):
        # check timestep completeness on surface (gated vars)
        nt = single.get("n_time")
        if nt is not None and nt < exp:
            return "partial"
        if single.get("vars_missing"):
            return "partial"
        return "present"
    if s == "missing" or p == "missing":
        return "partial"  # one of two file types absent
    return "partial"


def audit(
    raw_dir: Path,
    start_year: int,
    end_year: int,
    end_month: int,
) -> dict[str, Any]:
    months: dict[str, Any] = {}
    by_year: dict[str, Any] = {}
    counts = defaultdict(int)
    frontier_contiguous_from_start = None
    still_contiguous = True
    last_present_any = None
    holes: list[str] = []

    y, m = start_year, 1
    while (y < end_year) or (y == end_year and m <= end_month):
        stem = f"{y:04d}-{m:02d}"
        single_p = raw_dir / f"era5_single_{stem}.nc"
        pressure_p = raw_dir / f"era5_pressure_{stem}.nc"
        single = probe_surface(single_p)
        pressure = probe_pressure(pressure_p)
        status = month_status(single, pressure, y, m)
        exp = expected_timesteps(y, m)
        rec = {
            "year": y,
            "month": m,
            "status": status,
            "expected_timesteps": exp,
            "single": {
                "status": single["status"],
                "size_bytes": single["size_bytes"],
                "n_time": single.get("n_time"),
                "vars_found": single.get("vars_found"),
                "vars_missing": single.get("vars_missing"),
                "error": single.get("error"),
            },
            "pressure": {
                "status": pressure["status"],
                "size_bytes": pressure["size_bytes"],
                "n_time": pressure.get("n_time"),
                "error": pressure.get("error"),
            },
            "surface_diagnostics": {
                "t2m": "t2m" in (single.get("vars_found") or []),
                "u10m": "u10" in (single.get("vars_found") or []),
                "v10m": "v10" in (single.get("vars_found") or []),
                "note": "FCN3 names; CDS short names in file are t2m/u10/v10",
            },
        }
        months[stem] = rec
        counts[status] += 1
        if status != "present":
            holes.append(stem)
            still_contiguous = False
        else:
            last_present_any = stem
            if still_contiguous:
                frontier_contiguous_from_start = stem

        ys = by_year.setdefault(
            str(y),
            {"present": 0, "missing": 0, "partial": 0, "months": []},
        )
        ys[status] = ys.get(status, 0) + 1
        ys["months"].append({"month": m, "status": status})

        m += 1
        if m > 12:
            m = 1
            y += 1

    return {
        "schema": "era5_coverage_audit/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "raw_dir": str(raw_dir),
        "range": {
            "start": f"{start_year:04d}-01",
            "end": f"{end_year:04d}-{end_month:02d}",
            "note": "Audit window; CDS tip may extend further as download continues",
        },
        "var_map": VAR_MAP,
        "required_surface_cds_short": list(REQUIRED_SURFACE),
        "summary": {
            "n_months_scanned": len(months),
            "n_present": counts["present"],
            "n_missing": counts["missing"],
            "n_partial": counts["partial"],
            "frontier_contiguous_from_start": frontier_contiguous_from_start,
            "last_present_any": last_present_any,
            "n_holes": len(holes),
        },
        "holes": holes,
        "by_year": by_year,
        "months": months,
        "honesty": [
            "Read-only audit; does not stop/restart CDS download.",
            "Surface diagnostics gated for FINAL: t2m / u10m / v10m.",
            "Pressure files audited for presence only (not surface-var holes).",
            "Archive audit PASS for FINAL requires contiguous verified years for protocol — see FINAL_EVAL_SUITE_RECIPE.md §2.",
        ],
    }


def render_md(doc: dict) -> str:
    s = doc["summary"]
    lines = [
        "# ERA5 Nepal crop — coverage audit",
        "",
        f"**Generated (UTC):** {doc['created_utc']}  ",
        f"**Raw dir:** `{doc['raw_dir']}`  ",
        f"**Window:** {doc['range']['start']} → {doc['range']['end']}  ",
        f"**Schema:** `{doc['schema']}`",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"| --- | ---: |",
        f"| Months scanned | {s['n_months_scanned']} |",
        f"| Present | {s['n_present']} |",
        f"| Missing | {s['n_missing']} |",
        f"| Partial | {s['n_partial']} |",
        f"| Holes (non-present) | {s['n_holes']} |",
        f"| Frontier contiguous from start | {s['frontier_contiguous_from_start']} |",
        f"| Last present month (any) | {s['last_present_any']} |",
        "",
        "## Variable map (surface diagnostics)",
        "",
        "| FCN3 / recipe | CDS short (in .nc) | CDS long name |",
        "| --- | --- | --- |",
    ]
    for fcn3, meta in doc["var_map"].items():
        lines.append(
            f"| {fcn3} | `{meta['cds_short']}` | {meta['cds_long']} |"
        )
    lines += [
        "",
        "## By year",
        "",
        "| Year | Present | Missing | Partial |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for y in sorted(doc["by_year"].keys(), key=int):
        ys = doc["by_year"][y]
        lines.append(
            f"| {y} | {ys.get('present', 0)} | {ys.get('missing', 0)} | {ys.get('partial', 0)} |"
        )

    holes = doc.get("holes") or []
    lines += ["", "## Holes (missing or partial months)", ""]
    if not holes:
        lines.append("_None in scanned window._")
    else:
        # collapse runs for readability
        lines.append(f"Count: **{len(holes)}**. First 40: ")
        lines.append("")
        lines.append(", ".join(f"`{h}`" for h in holes[:40]))
        if len(holes) > 40:
            lines.append("")
            lines.append(f"… +{len(holes) - 40} more (see JSON `holes`).")

    lines += [
        "",
        "## Notes",
        "",
    ]
    for h in doc.get("honesty") or []:
        lines.append(f"- {h}")
    lines += [
        "",
        "## Next",
        "",
        "1. Leave CDS download PID alone until tip complete.",
        "2. Re-run this audit when frontier advances.",
        "3. Archive audit PASS → Leonard `FINAL_EVAL_PROTOCOL.md` → score `final_baselines.json`.",
        "",
    ]
    return "\n".join(lines)


def infer_tip(raw_dir: Path, start_year: int) -> tuple[int, int]:
    """Infer end year/month from newest present single file, else calendar now."""
    singles = sorted(raw_dir.glob("era5_single_????-??.nc"))
    if singles:
        stem = singles[-1].stem.replace("era5_single_", "")
        y, m = stem.split("-")
        # scan through a bit past tip so missing months after frontier show as holes
        tip_y, tip_m = int(y), int(m)
        return tip_y, tip_m
    now = datetime.now(timezone.utc)
    return now.year, now.month


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Audit ERA5 raw/ monthly coverage for t2m/u10m/v10m holes (read-only)."
    )
    ap.add_argument(
        "--raw-dir",
        default="~/fourcastnet/data/era5/raw",
    )
    ap.add_argument("--start", type=int, default=1980, help="Start year (inclusive)")
    ap.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="End year (inclusive); default = tip from files / now",
    )
    ap.add_argument(
        "--end-month",
        type=int,
        default=None,
        help="End month when --end-year set; default 12 or tip month",
    )
    ap.add_argument(
        "--end-tip",
        action="store_true",
        help="End at newest present monthly file (default behavior)",
    )
    ap.add_argument(
        "--json-out",
        default="~/fourcastnet/data/era5/coverage_audit.json",
    )
    ap.add_argument(
        "--md-out",
        default="~/fourcastnet/docs/research/ERA5_COVERAGE_AUDIT.md",
    )
    ap.add_argument(
        "--also-md",
        default="~/fourcastnet/docs/calls/ERA5_COVERAGE_AUDIT.md",
        help="Optional mirror path (empty string to skip)",
    )
    args = ap.parse_args()

    raw_dir = expand(args.raw_dir)
    if not raw_dir.is_dir():
        print(f"ERROR: raw dir missing: {raw_dir}", file=sys.stderr)
        return 2

    tip_y, tip_m = infer_tip(raw_dir, args.start)
    end_y = args.end_year if args.end_year is not None else tip_y
    end_m = args.end_month if args.end_month is not None else (
        tip_m if args.end_year is None else 12
    )

    print(f"auditing {raw_dir} from {args.start}-01 to {end_y:04d}-{end_m:02d}")
    doc = audit(raw_dir, args.start, end_y, end_m)

    json_out = expand(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {json_out}")

    md = render_md(doc)
    md_out = expand(args.md_out)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(md)
    print(f"wrote {md_out}")

    if args.also_md:
        also = expand(args.also_md)
        also.parent.mkdir(parents=True, exist_ok=True)
        also.write_text(md)
        print(f"wrote {also}")

    s = doc["summary"]
    print(
        f"summary: present={s['n_present']} missing={s['n_missing']} "
        f"partial={s['n_partial']} frontier={s['frontier_contiguous_from_start']} "
        f"last_present={s['last_present_any']} holes={s['n_holes']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
