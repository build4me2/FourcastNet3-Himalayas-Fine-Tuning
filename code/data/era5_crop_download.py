#!/usr/bin/env python3
"""
ERA5 regional crop downloader for FCN3 Nepal/HKH (Howard / Spark).

- Monthly CDS chunks (single-levels + pressure-levels)
- Area crop: 31/80/26/89 (N/W/S/E)
- 6-hourly 00/06/12/18
- Resume-friendly: skip existing complete monthly files
- Retry/backoff on CDS queue / transient errors

Never prints CDS credentials.
"""
from __future__ import annotations

import argparse
import calendar
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

try:
    import cdsapi
except ImportError as e:
    print("cdsapi not installed in this env:", e, file=sys.stderr)
    sys.exit(1)


LOG = logging.getLogger("era5_crop")


def setup_logging(log_path: Optional[Path], verbose: bool = True) -> None:
    LOG.handlers.clear()
    LOG.setLevel(logging.DEBUG if verbose else logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    LOG.addHandler(sh)
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        LOG.addHandler(fh)


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def cds_area(cfg: dict) -> list:
    a = cfg["area"]
    return [a["north"], a["west"], a["south"], a["east"]]


def month_days(year: int, month: int) -> list[str]:
    n = calendar.monthrange(year, month)[1]
    return [f"{d:02d}" for d in range(1, n + 1)]


def out_paths(cfg: dict, year: int, month: int) -> tuple[Path, Path]:
    raw = Path(cfg["paths"]["raw_dir"])
    monthly = Path(cfg["paths"]["monthly_dir"])
    stem = f"{year:04d}-{month:02d}"
    return (
        raw / f"era5_single_{stem}.nc",
        raw / f"era5_pressure_{stem}.nc",
    )


def file_ok(path: Path, min_bytes: int = 10_000) -> bool:
    return path.is_file() and path.stat().st_size >= min_bytes


def load_state(path: Path) -> dict:
    if path.is_file():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {"completed": {}, "failed": {}}
    return {"completed": {}, "failed": {}}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(path)


def classify_retryable(exc: BaseException) -> bool:
    msg = str(exc).lower()
    needles = (
        "queue",
        "timeout",
        "timed out",
        "temporarily",
        "429",
        "502",
        "503",
        "504",
        "connection",
        "reset",
        "busy",
        "try again",
        "rate",
        "limit",
        "unavailable",
        "gateway",
        "internal server",
    )
    return any(n in msg for n in needles)


def retrieve_with_retry(
    client: Any,
    dataset: str,
    request: dict,
    target: Path,
    max_retries: int,
    base_sec: float,
    max_sec: float,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    last_err: Optional[BaseException] = None
    for attempt in range(1, max_retries + 1):
        try:
            if partial.exists():
                partial.unlink()
            LOG.info(
                "CDS retrieve %s -> %s (attempt %d/%d)",
                dataset,
                target.name,
                attempt,
                max_retries,
            )
            client.retrieve(dataset, request, str(partial))
            if not partial.is_file() or partial.stat().st_size < 1000:
                raise RuntimeError(f"download too small or missing: {partial}")
            partial.replace(target)
            LOG.info(
                "OK %s size=%.2f MB",
                target.name,
                target.stat().st_size / 1e6,
            )
            return
        except Exception as e:
            last_err = e
            LOG.warning("retrieve failed attempt %d: %s", attempt, e)
            if partial.exists():
                try:
                    partial.unlink()
                except OSError:
                    pass
            if attempt >= max_retries or not classify_retryable(e):
                break
            sleep_s = min(max_sec, base_sec * (2 ** (attempt - 1)))
            # jitter
            sleep_s = sleep_s * (0.8 + 0.4 * (os.getpid() % 100) / 100.0)
            LOG.info("backoff %.0fs before retry", sleep_s)
            time.sleep(sleep_s)
    assert last_err is not None
    raise last_err


def build_single_request(cfg: dict, year: int, month: int) -> dict:
    vars_ = [v["name_cds"] for v in cfg["surface_variables"]]
    return {
        "product_type": cfg["cds"]["product_type"],
        "variable": vars_,
        "year": f"{year:04d}",
        "month": f"{month:02d}",
        "day": month_days(year, month),
        "time": cfg["time"]["hours"],
        "area": cds_area(cfg),
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def build_pressure_request(cfg: dict, year: int, month: int) -> dict:
    vars_ = [v["name_cds"] for v in cfg["pressure_variables"]]
    return {
        "product_type": cfg["cds"]["product_type"],
        "variable": vars_,
        "pressure_level": list(cfg["pressure_levels_hpa"]),
        "year": f"{year:04d}",
        "month": f"{month:02d}",
        "day": month_days(year, month),
        "time": cfg["time"]["hours"],
        "area": cds_area(cfg),
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def iter_months(start_year: int, end_year: int, end_month: int = 12) -> Iterable[tuple[int, int]]:
    for y in range(start_year, end_year + 1):
        last_m = end_month if y == end_year else 12
        for m in range(1, last_m + 1):
            yield y, m


def resolve_end(cfg: dict, force_end_year: Optional[int], force_end_month: Optional[int]) -> tuple[int, int]:
    """Return (end_year, end_month) inclusive. Null end_year => current UTC year/month."""
    if force_end_year is not None:
        ey = force_end_year
        em = force_end_month or 12
        return ey, em
    ey = cfg["time"].get("end_year")
    if ey is None:
        now = datetime.now(timezone.utc)
        # ERA5T tip lags ~5–10 days; still request through current month — CDS will
        # fail individual months if unavailable; we mark and continue.
        return now.year, now.month
    return int(ey), int(force_end_month or 12)


def download_month(
    client: Any,
    cfg: dict,
    year: int,
    month: int,
    skip_existing: bool = True,
) -> dict:
    single_path, pressure_path = out_paths(cfg, year, month)
    cds = cfg["cds"]
    key = f"{year:04d}-{month:02d}"
    result = {"month": key, "single": None, "pressure": None, "skipped": False}

    need_s = not (skip_existing and file_ok(single_path))
    need_p = not (skip_existing and file_ok(pressure_path))
    if not need_s and not need_p:
        LOG.info("skip existing %s", key)
        result["skipped"] = True
        result["single"] = str(single_path)
        result["pressure"] = str(pressure_path)
        return result

    if need_s:
        retrieve_with_retry(
            client,
            cds["dataset_single"],
            build_single_request(cfg, year, month),
            single_path,
            int(cds.get("max_retries", 8)),
            float(cds.get("retry_base_sec", 60)),
            float(cds.get("retry_max_sec", 1800)),
        )
    else:
        LOG.info("skip existing single %s", single_path.name)
    result["single"] = str(single_path)

    if need_p:
        retrieve_with_retry(
            client,
            cds["dataset_pressure"],
            build_pressure_request(cfg, year, month),
            pressure_path,
            int(cds.get("max_retries", 8)),
            float(cds.get("retry_base_sec", 60)),
            float(cds.get("retry_max_sec", 1800)),
        )
    else:
        LOG.info("skip existing pressure %s", pressure_path.name)
    result["pressure"] = str(pressure_path)
    return result


def verify_files(paths: list[Path]) -> list[dict]:
    out = []
    for p in paths:
        info = {"path": str(p), "exists": p.is_file(), "size_bytes": 0, "vars": [], "ok": False}
        if p.is_file():
            info["size_bytes"] = p.stat().st_size
            try:
                import xarray as xr

                ds = xr.open_dataset(p)
                info["vars"] = list(ds.data_vars)
                info["dims"] = {k: int(v) for k, v in ds.sizes.items()}
                ds.close()
                info["ok"] = True
            except Exception as e:
                info["error"] = str(e)
        out.append(info)
    return out


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="ERA5 FCN3 regional crop downloader")
    ap.add_argument(
        "--config",
        default=os.path.expanduser("~/fourcastnet/configs/era5_nepal_crop.yaml"),
    )
    ap.add_argument("--smoke", action="store_true", help="Download smoke month only")
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument("--month", type=int, default=None)
    ap.add_argument("--start-year", type=int, default=None)
    ap.add_argument("--end-year", type=int, default=None)
    ap.add_argument("--end-month", type=int, default=None)
    ap.add_argument("--no-skip", action="store_true")
    ap.add_argument("--log", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(Path(args.config).expanduser())
    for d in (
        cfg["paths"]["raw_dir"],
        cfg["paths"]["monthly_dir"],
        cfg["paths"]["log_dir"],
    ):
        Path(d).mkdir(parents=True, exist_ok=True)

    log_path = Path(args.log) if args.log else Path(cfg["paths"]["log_dir"]) / "era5_pull.log"
    setup_logging(log_path)
    LOG.info("config=%s", args.config)
    LOG.info("area N/W/S/E=%s", cds_area(cfg))
    LOG.info("raw_dir=%s", cfg["paths"]["raw_dir"])

    # cdsapi reads ~/.cdsapirc; never log key
    client = cdsapi.Client(quiet=True, progress=True)

    state_path = Path(cfg["paths"]["state_file"])
    state = load_state(state_path)

    if args.smoke:
        y = args.year or cfg.get("smoke", {}).get("year", 2020)
        m = args.month or cfg.get("smoke", {}).get("month", 1)
        months = [(y, m)]
        LOG.info("SMOKE mode: %04d-%02d", y, m)
    elif args.year and args.month:
        months = [(args.year, args.month)]
    else:
        start = args.start_year or int(cfg["time"]["start_year"])
        ey, em = resolve_end(cfg, args.end_year, args.end_month)
        months = list(iter_months(start, ey, em))
        LOG.info("FULL pull: %d months from %d-01 through %04d-%02d", len(months), start, ey, em)

    failures = []
    for y, m in months:
        key = f"{y:04d}-{m:02d}"
        try:
            res = download_month(client, cfg, y, m, skip_existing=not args.no_skip)
            state.setdefault("completed", {})[key] = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "single": res["single"],
                "pressure": res["pressure"],
                "skipped": res["skipped"],
            }
            state.get("failed", {}).pop(key, None)
            save_state(state_path, state)
            if args.smoke or (args.year and args.month):
                sp, pp = out_paths(cfg, y, m)
                ver = verify_files([sp, pp])
                for v in ver:
                    LOG.info(
                        "verify %s exists=%s size=%.2fMB ok=%s vars=%s dims=%s",
                        v["path"],
                        v["exists"],
                        v["size_bytes"] / 1e6,
                        v.get("ok"),
                        v.get("vars"),
                        v.get("dims"),
                    )
                # write smoke summary json
                smoke_json = Path(cfg["paths"]["raw_dir"]) / f"smoke_{key}.json"
                smoke_json.write_text(json.dumps(ver, indent=2))
                LOG.info("smoke summary -> %s", smoke_json)
        except Exception as e:
            LOG.error("FAILED %s: %s", key, e)
            LOG.debug(traceback.format_exc())
            state.setdefault("failed", {})[key] = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "error": str(e)[:500],
            }
            save_state(state_path, state)
            failures.append(key)
            if args.smoke:
                return 1
            # continue full backfill
            continue

    if failures:
        LOG.warning("completed with %d failures: %s", len(failures), failures[:20])
        return 2
    LOG.info("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
