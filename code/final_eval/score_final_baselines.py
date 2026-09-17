#!/usr/bin/env python3
"""CPU scaffold: score three baselines on FINAL protocol → final_baselines.json.

Baselines (FINAL_EVAL_SUITE_RECIPE.md §8):
  1. raw FCN3 (no residual)
  2. Tier-0 adapter
  3. living runs/phase0/tier_a/v1_3_joint/ residual

Wind-vector LOCKED (same as score_living_wind_baseline.py):
  per-lead: sqrt( mean_over_valid_grid( (u_err^2 + v_err^2) / 2 ) )
  headline: lead-mean over gate leads {24,72,120}

Constraints:
  - No GPU train, no .pt writes, do not touch living weights
  - Do NOT invent floats — real scores only when PROTOCOL ICs exist
  - Missing ICs → exit cleanly explaining PROTOCOL required

Usage:
  cd ~/fourcastnet
  python code/final_eval/score_final_baselines.py --help
  python code/final_eval/score_final_baselines.py --dry-run
  python code/final_eval/score_final_baselines.py --write-schema
  # Real scoring (blocked until FINAL_EVAL_PROTOCOL.md + filled IC lists):
  python code/final_eval/score_final_baselines.py --config configs/final_eval_baselines.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

try:
    import yaml
except ImportError:
    yaml = None

SCHEMA_VERSION = "final_baselines/v1"
BASELINES = ("raw_fcn3", "tier0", "living_v1_3_joint")
GATE_LEADS_DEFAULT = (24, 72, 120)
REPORT_LEADS_DEFAULT = (24, 48, 72, 96, 120)


def expand(p: str | Path) -> Path:
    return Path(str(p)).expanduser().resolve()


def load_cfg(path: Path) -> dict:
    if yaml is None:
        raise SystemExit("PyYAML required: pip install pyyaml")
    return yaml.safe_load(path.read_text())


def md5_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wind_vector_definition(cfg: dict) -> dict:
    wv = cfg.get("wind_vector") or {}
    leads = (cfg.get("leads") or {}).get("gate_h") or list(GATE_LEADS_DEFAULT)
    return {
        "formula": wv.get(
            "formula", "sqrt(mean((u_err^2 + v_err^2)/2))"
        ),
        "pool": wv.get("pool", "grid_then_lead_mean"),
        "channels": list(wv.get("channels") or ["u10m", "v10m"]),
        "valid_mask": wv.get("valid_mask", "elev_bin_id >= 0"),
        "gate_leads_h": list(leads),
        "headline": f"lead_mean over gate leads {list(leads)}",
        "locked_with": "code/tier_a/v1_3_joint/score_living_wind_baseline.py",
    }


def empty_metric_block(leads: list[int]) -> dict:
    """Placeholder metric slots — null floats until real score."""
    per_lead = {
        str(lh): {
            "n": None,
            "t2m_rmse": None,
            "u10m_rmse": None,
            "v10m_rmse": None,
            "wind_vector_rmse": None,
        }
        for lh in leads
    }
    return {
        "per_lead": per_lead,
        "grid_pooled_gate_leads": {
            "n": None,
            "t2m_rmse": None,
            "u10m_rmse": None,
            "v10m_rmse": None,
            "wind_vector_rmse": None,
        },
        "lead_mean": {
            "t2m_rmse": None,
            "u10m_rmse": None,
            "v10m_rmse": None,
            "wind_vector_rmse": None,
            "leads_h": list(leads),
            "definition": (
                "mean over leads of per-lead grid-pooled RMSE; "
                "wind_vector per-lead = sqrt(mean((u_err^2+v_err^2)/2))"
            ),
        },
        "t2m_val_plus_120h": None,
        "status": "PENDING_PROTOCOL_ICS",
    }


def build_schema_placeholder(cfg: dict) -> dict:
    """Write-only schema/placeholder JSON — no invented skill floats."""
    gate = list((cfg.get("leads") or {}).get("gate_h") or GATE_LEADS_DEFAULT)
    report = list((cfg.get("leads") or {}).get("report_h") or REPORT_LEADS_DEFAULT)
    ys = cfg.get("year_split") or {}
    ics = cfg.get("ics") or {}
    region = cfg.get("region") or {}
    baselines_cfg = cfg.get("baselines") or {}
    living = baselines_cfg.get("living_v1_3_joint") or {}
    living_ckpt = expand(living.get("ckpt") or "~/fourcastnet/runs/phase0/tier_a/v1_3_joint/best_residual.pt")

    split_template = {
        b: empty_metric_block(gate) for b in BASELINES
    }
    # report leads sidecar (P1)
    report_template = {
        b: {
            "per_lead": {
                str(lh): {
                    "n": None,
                    "t2m_rmse": None,
                    "u10m_rmse": None,
                    "v10m_rmse": None,
                    "wind_vector_rmse": None,
                }
                for lh in report
            },
            "status": "PENDING_PROTOCOL_ICS",
        }
        for b in BASELINES
    }

    return {
        "schema": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "SCHEMA_PLACEHOLDER",
        "scored": False,
        "recipe": cfg.get("recipe", "docs/research/FINAL_EVAL_SUITE_RECIPE.md"),
        "call": cfg.get("call", "docs/research/FINAL_EVAL_SUITE_CALL.md"),
        "claim_level": cfg.get("claim_level", "pending_final_protocol"),
        "g1_claimable": False,
        "box": {
            "lat_south": region.get("lat_south", 26.0),
            "lat_north": region.get("lat_north", 31.0),
            "lon_west": region.get("lon_west", 80.0),
            "lon_east": region.get("lon_east", 89.0),
        },
        "wind_vector_definition": wind_vector_definition(cfg),
        "leads": {"gate_h": gate, "report_h": report},
        "variables_out": list(cfg.get("variables_out") or ["t2m", "u10m", "v10m"]),
        "year_split": {
            "status": ys.get("status", "TODO_PENDING_ARCHIVE_AUDIT"),
            "train": list(ys.get("train") or []),
            "val": list(ys.get("val") or []),
            "test": list(ys.get("test") or []),
            "interim_legacy_bridge": ys.get("interim_legacy_bridge"),
        },
        "ics": {
            "status": ics.get("status", "TODO_PENDING_FINAL_EVAL_PROTOCOL"),
            "n_train": len(ics.get("train_ids") or []),
            "n_val": len(ics.get("val_ids") or []),
            "n_test": len(ics.get("test_ids") or []),
            "train_ids_sha256": None,
            "val_ids_sha256": None,
            "test_ids_sha256": None,
            "floors": ics.get("floors"),
            "note": "IC lists empty until FINAL_EVAL_PROTOCOL.md; do not invent ICs.",
        },
        "baselines": {
            "raw_fcn3": {
                "role": "absolute_floor",
                "residual": False,
                "ckpt": None,
            },
            "tier0": {
                "role": "lightweight_adapter_reference",
                "adapter_maps": str(
                    (baselines_cfg.get("tier0") or {}).get(
                        "adapter_maps",
                        "~/fourcastnet/runs/phase0/tier0/tier0_bias_maps.nc",
                    )
                ),
                "residual": False,
            },
            "living_v1_3_joint": {
                "role": "continuity_promote_bar",
                "residual": True,
                "ckpt": str(living_ckpt),
                "ckpt_md5": md5_file(living_ckpt),
                "ckpt_present": living_ckpt.is_file(),
                "note": "read-only; do not overwrite living weights",
            },
        },
        "splits": {
            "val": split_template,
            "test": {b: empty_metric_block(gate) for b in BASELINES},
        },
        "report_leads": {
            "val": report_template,
            "test": {
                b: {
                    "per_lead": {
                        str(lh): {
                            "n": None,
                            "t2m_rmse": None,
                            "u10m_rmse": None,
                            "v10m_rmse": None,
                            "wind_vector_rmse": None,
                        }
                        for lh in report
                    },
                    "status": "PENDING_PROTOCOL_ICS",
                }
                for b in BASELINES
            },
        },
        "legacy_bridge": {
            "enabled": bool((cfg.get("legacy_bridge") or {}).get("enabled", True)),
            "protocol": (cfg.get("legacy_bridge") or {}).get(
                "protocol", "thick2_12_16_16"
            ),
            "status": "STUB",
            "scorer": "code/final_eval/thick2_legacy_bridge.py",
            "note": (
                "Re-score FINAL ckpt on thick-2 12/16/16 interim protocol for "
                "continuity vs living 1.770/1.775/1.982 and WV 0.69560/0.73877. "
                "Report + honesty only — not a second promote path."
            ),
            "scores": None,
        },
        "pass_fail_schema": {
            "note": "Numeric bars TBD from this JSON after real score; Leonard FINAL_EVAL_BARS_CALL.md",
            "A_vs_raw": [
                "val t2m pooled strictly < measured raw val t2m",
                "test t2m pooled strictly < measured raw test t2m",
                "val +120h t2m <= measured raw val +120h",
                "val & test wind-vector strictly < measured raw WV",
                "n_eligible_saves >= 1; reload=composite_eligible",
            ],
            "B_vs_tier0": [
                "val t2m strictly < measured Tier-0 val t2m",
                "test t2m strictly < measured Tier-0 test t2m",
            ],
            "C_vs_living_on_FINAL": [
                "val t2m strictly < living re-score val t2m",
                "test t2m strictly < living re-score test t2m",
                "val & test wind-vector strictly < living re-score WV",
            ],
            "final_g1_candidate_pass": "A ∧ B ∧ C (floats not frozen yet)",
        },
        "honesty": list(
            cfg.get("honesty")
            or [
                "No FINAL train until archive audit + protocol + baselines + bars call + Manisha.",
                "Do not invent beat-this K bars.",
                "Living v1_3_joint remains interim_era5.",
            ]
        ),
        "non_claims": [
            "Not global WeatherBench 2 / FCN3 leaderboard SOTA",
            "Not operational NWP replacement over Nepal",
            "Not km-scale CorrDiff / RCM downscaling",
            "Not precip skill",
            "Not IMDAA / station-verified",
            "Not calibrated probabilistic ensembles",
            "Interim thick-2 / v1_3_joint remain interim_era5 — do not relabel as FINAL",
        ],
    }


def validate_schema(doc: dict) -> list[str]:
    """Return list of validation errors (empty = OK)."""
    errs: list[str] = []
    if doc.get("schema") != SCHEMA_VERSION:
        errs.append(f"schema must be {SCHEMA_VERSION}, got {doc.get('schema')}")
    if "splits" not in doc:
        errs.append("missing splits")
    else:
        for split in ("val", "test"):
            if split not in doc["splits"]:
                errs.append(f"missing splits.{split}")
                continue
            for b in BASELINES:
                if b not in doc["splits"][split]:
                    errs.append(f"missing splits.{split}.{b}")
    if doc.get("g1_claimable") is True and not doc.get("scored"):
        errs.append("g1_claimable cannot be true on unscored placeholder")
    wv = doc.get("wind_vector_definition") or {}
    if "sqrt" not in str(wv.get("formula", "")):
        errs.append("wind_vector_definition.formula must include locked sqrt form")
    box = doc.get("box") or {}
    if box.get("lat_south") != 26.0 or box.get("lat_north") != 31.0:
        errs.append("box lat must be 26–31N")
    if box.get("lon_west") != 80.0 or box.get("lon_east") != 89.0:
        errs.append("box lon must be 80–89E")
    return errs


def protocol_ready(cfg: dict) -> tuple[bool, str]:
    """True iff year slots + IC lists are non-empty (FINAL_EVAL_PROTOCOL landed)."""
    ys = cfg.get("year_split") or {}
    ics = cfg.get("ics") or {}
    reasons = []
    if not (ys.get("train") and ys.get("val") and ys.get("test")):
        reasons.append("year_split train/val/test empty (TODO_PENDING_ARCHIVE_AUDIT)")
    if not (ics.get("train_ids") and ics.get("val_ids") and ics.get("test_ids")):
        reasons.append("ics train/val/test_ids empty (TODO_PENDING_FINAL_EVAL_PROTOCOL)")
    protocol = expand(
        (cfg.get("paths") or {}).get(
            "protocol_doc", "~/fourcastnet/docs/research/FINAL_EVAL_PROTOCOL.md"
        )
    )
    if not protocol.is_file():
        reasons.append(f"missing {protocol} — Leonard writes after archive audit")
    if reasons:
        return False, "; ".join(reasons)
    return True, "protocol years + ICs present"


def dry_run(cfg: dict, cfg_path: Path) -> int:
    ready, reason = protocol_ready(cfg)
    living = (cfg.get("baselines") or {}).get("living_v1_3_joint") or {}
    ckpt = expand(living.get("ckpt") or "~/fourcastnet/runs/phase0/tier_a/v1_3_joint/best_residual.pt")
    out = expand(
        (cfg.get("paths") or {}).get(
            "out_json", "~/fourcastnet/runs/phase0/final_eval/final_baselines.json"
        )
    )
    print("=== score_final_baselines dry-run (CPU, no train) ===")
    print(f"config:     {cfg_path}")
    print(f"out:        {out}")
    print(f"box:        26–31N / 80–89E")
    print(f"gate leads: {(cfg.get('leads') or {}).get('gate_h')}")
    print(f"report:     {(cfg.get('leads') or {}).get('report_h')}")
    print(f"living ckpt present: {ckpt.is_file()}  md5={md5_file(ckpt)}")
    print(f"protocol ready: {ready}")
    print(f"  reason: {reason}")
    doc = build_schema_placeholder(cfg)
    verrs = validate_schema(doc)
    print(f"schema validation: {'OK' if not verrs else verrs}")
    if not ready:
        print(
            "\nPROTOCOL required before real scoring.\n"
            "Next: archive audit PASS → FINAL_EVAL_PROTOCOL.md (years + IC hashes) "
            "→ re-run without --dry-run.\n"
            "Until then: use --write-schema to refresh placeholder JSON only."
        )
        return 0
    print("Protocol looks filled — real scoring path would run (not invoked in dry-run).")
    return 0


def write_schema(cfg: dict, out_path: Path) -> int:
    doc = build_schema_placeholder(cfg)
    verrs = validate_schema(doc)
    if verrs:
        print("SCHEMA VALIDATION FAILED:", verrs, file=sys.stderr)
        return 2
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote schema placeholder {out_path} (scored=false; no invented floats)")
    return 0


def score_real(cfg: dict, out_path: Path) -> int:
    """Real scoring entry — blocked until protocol ICs exist.

    When ready, this should reuse helpers from:
      code/tier_a/v1_3_joint/score_living_wind_baseline.py
      (accumulate / summarize wind-vector lead-mean)
    and Tier-0 adapter maps under runs/phase0/tier0/.
    """
    ready, reason = protocol_ready(cfg)
    if not ready:
        print(
            "ERROR: PROTOCOL required — cannot score final_baselines.json yet.\n"
            f"  {reason}\n"
            "  See docs/research/FINAL_EVAL_SUITE_RECIPE.md §2, §12.\n"
            "  Use --write-schema for placeholder only; do not invent floats.",
            file=sys.stderr,
        )
        # Still refresh placeholder so path exists
        write_schema(cfg, out_path)
        return 1

    # Scaffold only: real IC loaders / FCN3 crop rollouts not wired until protocol lands.
    print(
        "PROTOCOL years/ICs detected, but full FINAL IC→crop→score path is not "
        "wired in this scaffold yet. Writing schema placeholder; eng will extend "
        "accumulate/summarize from score_living_wind_baseline.py once IC staging exists.",
        file=sys.stderr,
    )
    return write_schema(cfg, out_path)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "CPU scaffold for FINAL baselines (raw FCN3 / Tier-0 / living v1_3_joint). "
            "No train. No invented floats. Requires FINAL_EVAL_PROTOCOL for real scores."
        )
    )
    ap.add_argument(
        "--config",
        default="~/fourcastnet/configs/final_eval_baselines.yaml",
        help="YAML config (IC lists placeholder until protocol)",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Override output JSON (default: paths.out_json in config)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config/paths/schema; exit 0; no score invent",
    )
    ap.add_argument(
        "--write-schema",
        action="store_true",
        help="Write/refresh placeholder final_baselines.json (null metrics)",
    )
    ap.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate existing --out JSON schema and exit",
    )
    ap.add_argument(
        "--device",
        default="cpu",
        choices=["cpu"],
        help="CPU only for this scaffold (no GPU train)",
    )
    args = ap.parse_args()

    cfg_path = expand(args.config)
    if not cfg_path.is_file():
        print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
        return 2
    cfg = load_cfg(cfg_path)
    out_path = expand(
        args.out
        or (cfg.get("paths") or {}).get(
            "out_json", "~/fourcastnet/runs/phase0/final_eval/final_baselines.json"
        )
    )

    if args.validate_only:
        if not out_path.is_file():
            print(f"ERROR: missing {out_path}", file=sys.stderr)
            return 2
        doc = json.loads(out_path.read_text())
        verrs = validate_schema(doc)
        if verrs:
            print("FAIL:", verrs)
            return 2
        print(f"OK schema {doc.get('schema')} scored={doc.get('scored')} @ {out_path}")
        return 0

    if args.dry_run:
        return dry_run(cfg, cfg_path)

    if args.write_schema:
        return write_schema(cfg, out_path)

    return score_real(cfg, out_path)


if __name__ == "__main__":
    raise SystemExit(main())
