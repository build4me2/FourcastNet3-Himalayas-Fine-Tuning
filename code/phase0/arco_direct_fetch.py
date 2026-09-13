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



def grid_lat_lon():
    """FCN3 / ARCO 0.25° grid (N→S lat, 0→359.75 lon)."""
    lat = np.linspace(90.0, -90.0, 721)
    lon = np.linspace(0.0, 360.0, 1440, endpoint=False)
    return lat, lon


def crop_slices(lat, lon, lat_south: float, lat_north: float, lon_west: float, lon_east: float):
    lat_idx = np.where((lat >= lat_south) & (lat <= lat_north))[0]
    lon_idx = np.where((lon >= lon_west) & (lon <= lon_east))[0]
    if lat_idx.size == 0 or lon_idx.size == 0:
        raise RuntimeError(
            f"empty crop lat=[{lat_south},{lat_north}] lon=[{lon_west},{lon_east}] "
            f"grid lat=[{lat.min()},{lat.max()}] lon=[{lon.min()},{lon.max()}]"
        )
    lat_sl = slice(int(lat_idx[0]), int(lat_idx[-1]) + 1)
    lon_sl = slice(int(lon_idx[0]), int(lon_idx[-1]) + 1)
    return lat_sl, lon_sl, lat[lat_sl], lon[lon_sl]


def fetch_crop_array(
    channels: list[str],
    dt: datetime,
    lat_south: float = 26.0,
    lat_north: float = 31.0,
    lon_west: float = 80.0,
    lon_east: float = 89.0,
    verbose: bool = True,
) -> tuple[np.ndarray, dict]:
    """ERA5 crop (C, nlat, nlon) at dt from ARCO. Surface preferred; pressure OK.

    Reads only the lat/lon window (tiny vs full 721×1440). Used for Tier-0
    targets and as the hook for G0 verifying-ERA5 *regional* scores.
    Global verifying fields for CRPS/SSR/PSD still need fetch_timestep_array.
    """
    t0 = time.time()
    g = open_arco_group()
    t_idx = find_time_index(g, dt)
    lat_g, lon_g = grid_lat_lon()
    # Prefer ARCO coordinate arrays if present
    try:
        if "latitude" in g:
            lat_g = np.asarray(g["latitude"][:], dtype=np.float64)
        elif "lat" in g:
            lat_g = np.asarray(g["lat"][:], dtype=np.float64)
        if "longitude" in g:
            lon_g = np.asarray(g["longitude"][:], dtype=np.float64)
        elif "lon" in g:
            lon_g = np.asarray(g["lon"][:], dtype=np.float64)
    except Exception:
        pass
    lat_sl, lon_sl, lat_c, lon_c = crop_slices(
        lat_g, lon_g, lat_south, lat_north, lon_west, lon_east
    )
    levels = None
    level_to_i = {}
    if any(channel_to_spec(ch)[1] is not None for ch in channels):
        levels = np.asarray(g["level"][:], dtype=np.int32)
        level_to_i = {int(lev): i for i, lev in enumerate(levels.tolist())}

    h, w = int(lat_c.size), int(lon_c.size)
    out = np.empty((len(channels), h, w), dtype=np.float32)
    meta_reads = []
    for ci, ch in enumerate(channels):
        zname, lev = channel_to_spec(ch)
        arr = g[zname]
        if verbose:
            print(f"  crop {ch}->{zname} lev={lev} t={dt.isoformat()} idx={t_idx}", flush=True)
        if arr.ndim == 3:
            def reader(a=arr, ti=t_idx, ys=lat_sl, xs=lon_sl):
                return np.asarray(a[ti, ys, xs], dtype=np.float32)
            slab = _read_with_retries(reader)
        elif arr.ndim == 4:
            if lev is None:
                raise RuntimeError(f"{zname} is 4D but channel {ch} has no level")
            if lev not in level_to_i:
                raise KeyError(f"level {lev} missing in ARCO for {zname}")
            li = level_to_i[lev]
            def reader(a=arr, ti=t_idx, li=li, ys=lat_sl, xs=lon_sl):
                return np.asarray(a[ti, li, ys, xs], dtype=np.float32)
            slab = _read_with_retries(reader)
        else:
            raise RuntimeError(f"{zname} ndim={arr.ndim}")
        if slab.shape != (h, w):
            raise RuntimeError(f"{zname} crop shape {slab.shape} != {(h, w)}")
        out[ci] = slab
        meta_reads.append({"channel": ch, "array": zname, "bytes": int(slab.nbytes)})
    meta = {
        "source": "arco_zarr_direct_crop",
        "source_url": ARCO_ZARR,
        "product": "ERA5_final_ARCO",
        "time_index": t_idx,
        "fetch_s": round(time.time() - t0, 2),
        "reads": meta_reads,
        "lat": lat_c.tolist(),
        "lon": lon_c.tolist(),
        "shape": list(out.shape),
        "box": {
            "lat_south": lat_south,
            "lat_north": lat_north,
            "lon_west": lon_west,
            "lon_east": lon_east,
        },
    }
    return out, meta
