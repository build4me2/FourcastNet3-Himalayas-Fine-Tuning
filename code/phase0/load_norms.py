#!/usr/bin/env python3
"""Phase 0 — load FCN3 norm / aux artifacts from models/fourcastnet3.

Does not require full Earth2Studio. Verifies global_means/stds/mins/maxs shapes
and presence of orography + land_mask.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


DEFAULT_ROOT = Path.home() / "fourcastnet" / "models" / "fourcastnet3"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = p.parse_args()
    root: Path = args.root
    needed = [
        "global_means.npy",
        "global_stds.npy",
        "mins.npy",
        "maxs.npy",
        "orography.nc",
        "land_mask.nc",
        "config.json",
        "training_checkpoints/best_ckpt_mp0.tar",
    ]
    report = {"root": str(root), "files": {}}
    missing = []
    for name in needed:
        path = root / name
        ok = path.is_file()
        entry = {"path": str(path), "exists": ok}
        if ok:
            entry["size_bytes"] = path.stat().st_size
        else:
            missing.append(name)
        report["files"][name] = entry

    try:
        import numpy as np

        for key in ("global_means.npy", "global_stds.npy", "mins.npy", "maxs.npy"):
            path = root / key
            if path.is_file():
                arr = np.load(path)
                report["files"][key]["shape"] = list(arr.shape)
                report["files"][key]["dtype"] = str(arr.dtype)
    except Exception as e:
        report["numpy_load_error"] = str(e)

    cfg = root / "config.json"
    if cfg.is_file():
        try:
            data = json.loads(cfg.read_text())
            report["config_keys_sample"] = sorted(list(data.keys()))[:40]
            for k in ("in_channels", "out_channels", "dhours", "nettype"):
                if k in data:
                    report[k] = data[k]
        except Exception as e:
            report["config_error"] = str(e)

    print(json.dumps(report, indent=2))
    if missing:
        print(f"MISSING: {missing}", file=sys.stderr)
        return 1
    print("OK: FCN3 package artifacts present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
