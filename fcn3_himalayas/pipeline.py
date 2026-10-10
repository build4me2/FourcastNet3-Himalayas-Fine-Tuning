"""Inference pipeline, mirroring the FINAL eval path exactly:

  code/phase0/g0_base.py           : FCN3 via Earth2Studio, IC tensor (1,1,1,72,721,1440)
  code/phase0/tier0_crop_rollout.py: model.set_rng(seed=333, reset=True); create_iterator; 6 h/step
  code/phase0/inference_smoke.py   : crop_nepal (box 26-31N, 80-89E, inclusive -> 21x37)
  code/tier_a/v1_3_joint/dataset.py: x = [t2m,u10m,v10m (physical units), elev_norm, lead_norm]
  code/tier_a/v1_3_joint/train.py  : corrected = raw + adapter(x)
"""
from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from importlib import resources
from pathlib import Path

import numpy as np

ADAPTER_REPO = "build4me2/fcn3-himalayas-final-residual-v0"
ADAPTER_FILE = "best_residual.pt"
ADAPTER_MD5 = "586ab17b843757bb84e66e2e3af8dc01"
BOX = {"lat_south": 26.0, "lat_north": 31.0, "lon_west": 80.0, "lon_east": 89.0}
VARS = ["t2m", "u10m", "v10m"]
SEED = 333  # seed used for every FINAL-eval FCN3 rollout
TRAINED_LEADS = (24, 72, 120)


def log(msg: str) -> None:
    print(f"[fcn3-himalayas] {msg}", flush=True)


def pick_device(device: str):
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("ERROR: --device cuda requested but CUDA is not available")
    return torch.device(device)


def load_elevation() -> dict:
    with resources.files("fcn3_himalayas").joinpath("data/box_elevation.npz").open("rb") as f:
        z = np.load(f)
        elev = z["elevation_m"].astype(np.float32)
        lat, lon = z["lat"], z["lon"]
    elev_norm = (elev - float(np.nanmean(elev))) / (float(np.nanstd(elev)) + 1e-6)
    return {"elev_m": elev, "elev_norm": elev_norm.astype(np.float32), "lat": lat, "lon": lon}


def load_adapter(device, path: str | None = None):
    import torch

    from .adapter_model import ElevCondResidualUNet

    if path is None:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(ADAPTER_REPO, ADAPTER_FILE)
    md5 = hashlib.md5(Path(path).read_bytes()).hexdigest()
    if md5 != ADAPTER_MD5:
        log(f"WARNING: adapter md5 {md5} != published {ADAPTER_MD5}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    m = ElevCondResidualUNet(in_channels=5, out_channels=3, base=16, depth=3, cond_dim=2, dropout=0.20)
    m.load_state_dict(state)
    return m.to(device).eval(), md5


def load_fcn3(device, package: str | None = None):
    try:
        from earth2studio.models.auto import Package
        from earth2studio.models.px.fcn3 import FCN3
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "ERROR: Earth2Studio FCN3 not importable (" + str(e) + ").\n"
            "Install with: pip install 'earth2studio[fcn3]' and makani "
            "(see README Quickstart)."
        )
    pkg = Package(package, cache=False) if package else FCN3.load_default_package()
    model = FCN3.load_model(pkg).to(device)
    model.eval()
    return model


def initial_condition(model, init: np.datetime64, device, ic_npy: str | None = None, ic_source: str = "arco"):
    import torch

    variables = list(model.variables)
    if ic_npy:
        arr = np.load(ic_npy)
        src = f"npy:{ic_npy}"
    elif ic_source == "arco":
        from datetime import datetime

        from .arco import ARCO_ZARR, fetch_timestep_array

        log("fetching initial condition from ARCO ERA5 (public Google Cloud Zarr, no account; ~5-15 min)...")
        arr, _ = fetch_timestep_array(variables, datetime.fromisoformat(str(init)), verbose=False)
        src = f"ARCO ERA5 {ARCO_ZARR} (direct zarr)"
    elif ic_source == "e2s-arco":
        from earth2studio.data import ARCO
        from earth2studio.data.utils import fetch_data

        log("fetching initial condition via earth2studio.data.ARCO (no account)...")
        x, _ = fetch_data(ARCO(cache=True, verbose=False), time=np.array([init], dtype="datetime64[ns]"),
                          variable=np.array(variables), device="cpu")
        arr = x.numpy().reshape(len(variables), 721, 1440)
        src = "earth2studio.data.ARCO (gs://gcp-public-data-arco-era5)"
    else:
        raise SystemExit(f"ERROR: unknown --ic-source {ic_source}")
    if arr.shape != (len(variables), 721, 1440):
        raise SystemExit(f"ERROR: IC shape {arr.shape} != ({len(variables)}, 721, 1440)")
    if not np.isfinite(arr).all():
        raise SystemExit("ERROR: initial condition contains non-finite values")
    x = torch.from_numpy(np.array(arr, dtype=np.float32)).to(device).view(1, 1, 1, *arr.shape)
    coords = OrderedDict(
        batch=np.array([0]),
        time=np.array([init]),
        lead_time=np.array([np.timedelta64(0, "h")]),
        variable=np.array(variables),
        lat=np.linspace(90.0, -90.0, 721),
        lon=np.linspace(0, 360, 1440, endpoint=False),
    )
    return x, coords, src


def crop_box(x, coords):
    lats, lons = np.asarray(coords["lat"]), np.asarray(coords["lon"])
    li = np.where((lats >= BOX["lat_south"]) & (lats <= BOX["lat_north"]))[0]
    lo = np.where((lons >= BOX["lon_west"]) & (lons <= BOX["lon_east"]))[0]
    return (x[..., li[0]: li[-1] + 1, lo[0]: lo[-1] + 1],
            lats[li[0]: li[-1] + 1], lons[lo[0]: lo[-1] + 1])


def run_forecast(init: str, lead: int, out: str, device: str = "auto", every: int = 24,
                 ic_npy: str | None = None, fcn3_package: str | None = None,
                 adapter_path: str | None = None, seed: int = SEED, ic_source: str = "arco") -> dict:
    import torch
    import xarray as xr

    if lead < 24 or lead > 120 or lead % 6:
        raise SystemExit("ERROR: --lead must be a multiple of 6 between 24 and 120 h "
                         "(the adapter was trained on +24/+72/+120 h)")
    if every % 6 or every <= 0:
        raise SystemExit("ERROR: --every must be a positive multiple of 6")
    t_init = np.datetime64(init.replace("Z", ""), "h")
    if int(t_init.astype(object).hour) % 6:
        raise SystemExit("ERROR: --init hour must be 00, 06, 12 or 18 UTC")
    leads = [h for h in range(24, lead + 1, every)]
    if lead not in leads:
        leads.append(lead)

    dev = pick_device(device)
    log(f"device={dev}")
    t0 = time.time()
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    elev = load_elevation()
    # The adapter is tiny (~0.17M params): run it on CPU in fp32 for bit-reproducible output.
    adapter, md5 = load_adapter("cpu", adapter_path)
    # Fetch the IC *before* loading FCN3: ARCO/gcsfs async I/O was observed to hang on Spark
    # when started after the FCN3 load in the same process.
    from earth2studio.models.px.fcn3 import VARIABLES as FCN3_VARIABLES

    class _V:
        variables = list(FCN3_VARIABLES)

    x0, coords, src = initial_condition(_V, t_init, dev, ic_npy, ic_source)
    t_ic = time.time() - t0
    log("loading FourCastNet3 (Earth2Studio, hf://nvidia/fourcastnet3)...")
    fcn3 = load_fcn3(dev, fcn3_package)
    if list(fcn3.variables) != list(FCN3_VARIABLES):
        raise SystemExit("ERROR: FCN3 variable order mismatch")
    t_load = time.time() - t0 - t_ic
    vidx = [list(fcn3.variables).index(v) for v in VARS]

    raw = {}
    lat_c = lon_c = None
    t1 = time.time()
    with torch.inference_mode():
        fcn3.set_rng(seed=seed, reset=True)
        it = fcn3.create_iterator(x0, coords)
        for step in range(lead // 6 + 1):
            o, c = next(it)
            h = 6 * step
            if h in leads:
                crop, lat_c, lon_c = crop_box(o, c)
                f = crop.detach().float().cpu().numpy().reshape(-1, len(lat_c), len(lon_c))
                raw[h] = f[vidx]
                log(f"FCN3 +{h:3d} h done")
        del it, o
    t_fcn3 = time.time() - t1
    if not (np.allclose(lat_c, elev["lat"]) and np.allclose(lon_c, elev["lon"])):
        raise SystemExit("ERROR: FCN3 crop grid does not match adapter elevation grid")

    rawa = np.stack([raw[h] for h in leads]).astype(np.float32)  # (L,3,H,W)
    xs = []
    for i, h in enumerate(leads):
        lead_map = np.full_like(elev["elev_norm"], np.float32((h - 24.0) / 96.0))
        xs.append(np.concatenate([rawa[i], elev["elev_norm"][None], lead_map[None]], 0))
    with torch.no_grad():
        res = adapter(torch.from_numpy(np.stack(xs))).float().numpy()
    corr = rawa + res

    peak = torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else None
    data = {}
    units = {"t2m": "K", "u10m": "m s-1", "v10m": "m s-1"}
    for k, v in enumerate(VARS):
        data[f"{v}_raw"] = (("lead_time", "lat", "lon"), rawa[:, k], {"units": units[v], "long_name": f"FCN3 raw {v}"})
        data[f"{v}_corrected"] = (("lead_time", "lat", "lon"), corr[:, k], {"units": units[v], "long_name": f"FCN3 + Himalayas adapter {v}"})
    data["elevation"] = (("lat", "lon"), elev["elev_m"], {"units": "m"})
    ds = xr.Dataset(
        data,
        coords={"lead_time": ("lead_time", np.array(leads, dtype=np.int32), {"units": "hours"}),
                "lat": lat_c.astype(np.float64), "lon": lon_c.astype(np.float64),
                "valid_time": ("lead_time", np.array([t_init + np.timedelta64(h, "h") for h in leads], dtype="datetime64[ns]"))},
        attrs={"init_time": str(t_init) + ":00Z", "ic_source": src, "fcn3": "nvidia/fourcastnet3 via Earth2Studio",
               "adapter": f"hf://{ADAPTER_REPO}/{ADAPTER_FILE} md5={md5}", "fcn3_seed": seed,
               "box": "26-31N, 80-89E (0.25 deg, 21x37)",
               "note": "Adapter trained/evaluated on +24/+72/+120 h; other leads are interpolated in lead_norm.",
               "fcn3_himalayas_version": __import__("fcn3_himalayas").__version__},
    )
    ds.to_netcdf(out)
    stats = {"out": out, "device": str(dev), "load_s": round(t_load, 1), "ic_s": round(t_ic, 1),
             "fcn3_s": round(t_fcn3, 1), "total_s": round(time.time() - t0, 1),
             "peak_gpu_mem_gib": None if peak is None else round(peak, 2), "leads_h": leads}
    log(f"wrote {out}  {stats}")
    return stats
