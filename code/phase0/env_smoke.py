#!/usr/bin/env python3
"""Phase 0 — environment / hardware smoke (no training).

Prints torch/CUDA availability, optional earth2studio / torch_harmonics,
and peak allocated mem if a tiny CUDA tensor is created.

Exit 0 if torch imports; exit 2 if torch missing; exit 3 if CUDA expected but absent.
Measured claims only — write results into docs/PROGRESS.md by hand or via runner.
"""
from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone


def main() -> int:
    report: dict = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
    }
    try:
        import torch
    except Exception as e:
        report["torch"] = {"ok": False, "error": str(e)}
        print(json.dumps(report, indent=2))
        print("FAIL: torch not importable (env rebuild mid-flight?)", file=sys.stderr)
        return 2

    report["torch"] = {
        "ok": True,
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        report["torch"]["device0"] = torch.cuda.get_device_name(0)
        try:
            x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
            y = x @ x
            torch.cuda.synchronize()
            report["torch"]["bf16_matmul_ok"] = True
            report["torch"]["mem_allocated_mb"] = round(torch.cuda.memory_allocated() / 1e6, 2)
            del x, y
        except Exception as e:
            report["torch"]["bf16_matmul_ok"] = False
            report["torch"]["bf16_matmul_error"] = str(e)

    for mod in ("earth2studio", "torch_harmonics", "physicsnemo", "makani"):
        try:
            m = __import__(mod)
            report[mod] = {"ok": True, "version": getattr(m, "__version__", "unknown")}
        except Exception as e:
            report[mod] = {"ok": False, "error": str(e)}

    print(json.dumps(report, indent=2))
    if not report["torch"]["cuda_available"]:
        print("WARN: CUDA not available", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
