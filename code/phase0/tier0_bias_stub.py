#!/usr/bin/env python3
"""Back-compat wrapper — real implementation is tier0_bias.py (CPU-only)."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

target = Path(__file__).resolve().parent / "tier0_bias.py"
sys.argv[0] = str(target)
runpy.run_path(str(target), run_name="__main__")
