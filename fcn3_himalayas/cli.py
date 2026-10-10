from __future__ import annotations

import argparse
import json


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="fcn3-himalayas",
                                 description="FourCastNet3 + Himalayas residual adapter (box 26-31N, 80-89E)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("forecast", help="run FCN3 from an ERA5 init time and apply the adapter")
    f.add_argument("--init", required=True, help="init time UTC, e.g. 2024-07-01T00 (hour 00/06/12/18)")
    f.add_argument("--lead", type=int, default=120, help="max lead in hours, 24..120 (default 120)")
    f.add_argument("--every", type=int, default=24, help="output interval in hours (default 24)")
    f.add_argument("--out", default="fcn3_himalayas_forecast.nc", help="output NetCDF path")
    f.add_argument("--device", default="auto", help="auto | cuda | cpu")
    f.add_argument("--ic-source", default="arco", choices=["arco", "e2s-arco"],
                   help="ERA5 IC source: arco (direct public zarr, default) | e2s-arco (earth2studio.data.ARCO)")
    f.add_argument("--ic-npy", help="advanced: local global IC .npy (72,721,1440) instead of ARCO")
    f.add_argument("--fcn3-package", help="advanced: local FCN3 package dir instead of hf://nvidia/fourcastnet3")
    f.add_argument("--adapter", help="advanced: local adapter .pt instead of the Hugging Face file")
    a = ap.parse_args(argv)
    from .pipeline import run_forecast

    stats = run_forecast(a.init, a.lead, a.out, device=a.device, every=a.every, ic_npy=a.ic_npy,
                         fcn3_package=a.fcn3_package, adapter_path=a.adapter, ic_source=a.ic_source)
    print(json.dumps(stats))
    return 0
