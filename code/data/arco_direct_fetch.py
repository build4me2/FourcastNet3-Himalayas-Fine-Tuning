"""Direct ARCO ERA5 zarr fetch helpers (robust retries; no CDS)."""
from __future__ import annotations

import time
from datetime import datetime
from typing import Iterable

import numpy as np

ARCO_ZARR = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
# FCN3 pressure levels (hPa)
FCN3_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

# Map FCN3 channel -> (zarr_name, level_or_None)
SURFACE_MAP = {
    "u10m": ("10m_u_component_of_wind", None),
    "v10m": ("10m_v_component_of_wind", None),
    "u100m": ("100m_u_component_of_wind", None),
    "v100m": ("100m_v_component_of_wind", None),
    "t2m": ("2m_temperature", None),
    "msl": ("mean_sea_level_pressure", None),
    "tcwv": ("total_column_water_vapour", None),
}
PRESSURE_PREFIX = {
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "z": "geopotential",
    "t": "temperature",
    "q": "specific_humidity",
}


def channel_to_spec(ch: str) -> tuple[str, int | None]:
    if ch in SURFACE_MAP:
        return SURFACE_MAP[ch]
    for pref, zname in PRESSURE_PREFIX.items():
        if ch.startswith(pref) and ch[len(pref) :].isdigit():
            return zname, int(ch[len(pref) :])
    raise KeyError(f"unknown FCN3 channel: {ch}")


def hours_since_1900(dt: datetime) -> int:
    epoch = datetime(1900, 1, 1)
    return int((dt - epoch).total_seconds() // 3600)


def open_arco_group():
    import fsspec
    import zarr

    mapper = fsspec.get_mapper(ARCO_ZARR, token="anon")
    return zarr.open_group(mapper, mode="r")


_TIME_CACHE: dict[str, np.ndarray] = {}


def find_time_index(group, dt: datetime) -> int:
    """Locate time index for dt (UTC). Uses hours-since-1900 + searchsorted."""
    target = hours_since_1900(dt)
    # time array is large (~10MB ints); cache once per process
    if ARCO_ZARR not in _TIME_CACHE:
        _TIME_CACHE[ARCO_ZARR] = np.asarray(group["time"][:], dtype=np.int64)
    t = _TIME_CACHE[ARCO_ZARR]
    idx = int(np.searchsorted(t, target))
    if idx >= len(t) or int(t[idx]) != target:
        # try nearby ±1h in case of encoding quirks
        for d in (0, -1, 1, -2, 2):
            j = idx + d
            if 0 <= j < len(t) and int(t[j]) == target:
                return j
        raise KeyError(f"time {dt.isoformat()} (hours={target}) not in ARCO time axis near idx={idx}")
    return idx


def _read_with_retries(reader, n_retries: int = 8, base_sleep: float = 2.0):
    last = None
    for attempt in range(n_retries):
        try:
            return reader()
        except Exception as e:
            last = e
            sleep = base_sleep * (2 ** min(attempt, 5)) + (0.1 * attempt)
            print(f"  retry {attempt+1}/{n_retries} after {type(e).__name__}: {e} (sleep {sleep:.1f}s)", flush=True)
            time.sleep(sleep)
    raise RuntimeError(f"failed after {n_retries} retries: {last}")


def fetch_timestep_array(channels: list[str], dt: datetime, verbose: bool = True) -> tuple[np.ndarray, dict]:
    """Return float32 (C, 721, 1440) for channels at dt from ARCO zarr."""
    t0 = time.time()
    g = open_arco_group()
    t_idx = find_time_index(g, dt)
    levels = np.asarray(g["level"][:], dtype=np.int32)
    level_to_i = {int(lev): i for i, lev in enumerate(levels.tolist())}

    # Group channels by zarr array to fetch each chunk once
    needed_arrays: dict[str, list[tuple[int, int | None]]] = {}
    # maps zarr_name -> list of (channel_index, level_or_None)
    for ci, ch in enumerate(channels):
        zname, lev = channel_to_spec(ch)
        needed_arrays.setdefault(zname, []).append((ci, lev))

    out = np.empty((len(channels), 721, 1440), dtype=np.float32)
    meta_reads = []

    for zname, specs in needed_arrays.items():
        arr = g[zname]
        if verbose:
            print(f"  reading {zname} time_idx={t_idx} shape={arr.shape} chunks={arr.chunks}", flush=True)

        if arr.ndim == 3:
            # (time, lat, lon)
            def reader(a=arr, ti=t_idx):
                return np.asarray(a[ti], dtype=np.float32)

            slab = _read_with_retries(reader)
            if slab.shape != (721, 1440):
                raise RuntimeError(f"{zname} unexpected slab {slab.shape}")
            for ci, lev in specs:
                assert lev is None
                out[ci] = slab
            meta_reads.append({"array": zname, "bytes": int(slab.nbytes), "ndim": 3})
        elif arr.ndim == 4:
            # (time, level, lat, lon) — chunk is all 37 levels; read once
            def reader(a=arr, ti=t_idx):
                return np.asarray(a[ti], dtype=np.float32)

            slab = _read_with_retries(reader)
            if slab.shape[1:] != (721, 1440):
                raise RuntimeError(f"{zname} unexpected slab {slab.shape}")
            for ci, lev in specs:
                assert lev is not None
                if lev not in level_to_i:
                    raise KeyError(f"level {lev} missing in ARCO for {zname}")
                out[ci] = slab[level_to_i[lev]]
            meta_reads.append({"array": zname, "bytes": int(slab.nbytes), "ndim": 4})
        else:
            raise RuntimeError(f"{zname} ndim={arr.ndim}")

    meta = {
        "source": "arco_zarr_direct",
        "source_url": ARCO_ZARR,
        "product": "ERA5_final_ARCO",
        "earth2studio": "lexicon-compatible identity units; direct zarr for retry robustness",
        "time_index": t_idx,
        "fetch_s": round(time.time() - t0, 2),
        "reads": meta_reads,
        "lat0": 90.0,
        "latN": -90.0,
        "lon0": 0.0,
        "lonN": 359.75,
    }
    return out, meta
