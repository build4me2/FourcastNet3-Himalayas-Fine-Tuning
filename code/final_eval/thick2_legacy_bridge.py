#!/usr/bin/env python3
"""Thick-2 legacy bridge scorer stub (FINAL_EVAL_SUITE_RECIPE.md §1).

Re-score a FINAL (or candidate) residual ckpt on the interim thick-2 12/16/16
protocol for the continuity table vs living headlines:
  t2m 1.770 / 1.775 / 1.982 · WV 0.69560 / 0.73877

Bridge is **report + honesty only** — not a second promote path.
CPU scaffold: --help / --dry-run work without GPU; real score reuses
score_living_wind_baseline helpers when invoked with a ckpt + thick-2 config.

Does NOT write .pt files. Does NOT overwrite living weights.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

LIVING_CONTINUITY = {
    "source": "docs/research/TIER_A_V1_3_JOINT_CALL.md + claim_16ic",
    "val_t2m_pooled": 1.770077,
    "test_t2m_pooled_16ic": 1.774864,
    "val_t2m_plus_120h": 1.982401,
    "val_wind_vector_lead_mean": 0.69560168,
    "test_wind_vector_lead_mean": 0.73876564,
    "protocol": "thick2_12_16_16",
    "holdout_expand": "v2",
    "bars_set": "thick2",
}


def expand(p: str | Path) -> Path:
    return Path(str(p)).expanduser().resolve()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Thick-2 legacy bridge stub: continuity re-score on interim 12/16/16. "
            "Report-only. No train."
        )
    )
    ap.add_argument(
        "--config",
        default="~/fourcastnet/configs/tier_a_v1_3_joint.yaml",
        help="Thick-2 / living joint config (interim protocol)",
    )
    ap.add_argument(
        "--ckpt",
        default=None,
        help="Residual ckpt to re-score (default: living v1_3_joint — continuity self-check)",
    )
    ap.add_argument(
        "--out",
        default="~/fourcastnet/runs/phase0/final_eval/thick2_legacy_bridge.json",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--write-stub",
        action="store_true",
        help="Write stub JSON with continuity reference floats (not a new score)",
    )
    args = ap.parse_args()

    cfg_path = expand(args.config)
    ckpt = expand(
        args.ckpt
        or "~/fourcastnet/runs/phase0/tier_a/v1_3_joint/best_residual.pt"
    )
    out = expand(args.out)

    print("=== thick2_legacy_bridge stub ===")
    print(f"config: {cfg_path} present={cfg_path.is_file()}")
    print(f"ckpt:   {ckpt} present={ckpt.is_file()}")
    print(f"out:    {out}")
    print("continuity reference (living interim; do not invent new bars):")
    for k, v in LIVING_CONTINUITY.items():
        print(f"  {k}: {v}")

    if args.dry_run:
        print("dry-run OK — real bridge score wires to score_living_wind_baseline.accumulate")
        print("NOTE: bridge is report+honesty only; not a G1/FINAL promote path")
        return 0

    if args.write_stub or True:
        # Default: write stub documenting continuity table slots
        doc = {
            "schema": "thick2_legacy_bridge/v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "STUB",
            "scored": False,
            "protocol": "thick2_12_16_16",
            "role": "continuity_table_only",
            "not_promote_path": True,
            "ckpt": str(ckpt),
            "ckpt_present": ckpt.is_file(),
            "config": str(cfg_path),
            "living_continuity_reference": LIVING_CONTINUITY,
            "scores": {
                "val": {"t2m_pooled": None, "t2m_plus_120h": None, "wind_vector_lead_mean": None},
                "test": {"t2m_pooled": None, "wind_vector_lead_mean": None},
            },
            "note": (
                "Stub only. When a FINAL ckpt exists, re-score on thick-2 interim "
                "ICs via score_living_wind_baseline helpers and fill scores.*. "
                "Compare to living_continuity_reference for honesty table."
            ),
            "wind_vector_definition": (
                "sqrt(mean((u_err^2+v_err^2)/2)) grid-pooled then lead-mean "
                "over {24,72,120}"
            ),
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"wrote stub {out}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
