#!/usr/bin/env python3
"""Tier-0 holdout: train-only elevation-binned bias fit + val/test score.

Provisional years (Manisha not hard-locked):
  train 2018–2021 / val 2022 / test 2023–2024
  provisional_years=true; year_split_frozen=true (provisional label)
  claim_level=interim_era5; g1_claimable=false

Reuses fit_binned_linear / load helpers from tier0_bias.py.
Writes runs/phase0/tier0_holdout/ metrics + maps.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from tier0_bias import (  # noqa: E402
    Box,
    build_elevation_bins,
    load_orography_crop,
    expand,
    fit_binned_linear,
    load_config,
    load_real_pairs,
    check_paths,
    save_bias_maps_nc,
    subset_lead,
    write_metrics_csv,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_split_map(manifest_path: Path) -> dict[str, str]:
    man = json.loads(manifest_path.read_text())
    return {ic["id"]: ic.get("split", "train") for ic in man.get("ics", []) if ic.get("status") == "staged"}


def filter_pairs_by_ids(pair: dict, want_ids: set[str]) -> dict:
    """Subset a load_real_pairs result to selected IC ids (axis 0)."""
    import numpy as np

    if not pair.get("ok"):
        return pair
    ics = pair["ics"]
    keep = [i for i, e in enumerate(ics) if e["ic_id"] in want_ids]
    if not keep:
        return {"ok": False, "error": f"no ICs in pair match {sorted(want_ids)}"}
    fields = {}
    for v, d in pair["fields"].items():
        fields[v] = {
            "pred": np.asarray(d["pred"])[keep],
            "true": np.asarray(d["true"])[keep],
        }
    out = dict(pair)
    out["fields"] = fields
    out["ics"] = [ics[i] for i in keep]
    out["n_ics"] = len(keep)
    return out


def apply_fit_score(fit: dict, pred, true, elev_m, bins: dict) -> dict:
    """Apply a previously fitted per-bin linear/const bias; report RMSE metrics."""
    import numpy as np

    residual = true - pred  # not used for fit; for bias reporting
    bin_id = bins["bin_id"]
    # broadcast elev/bin over leading dims
    lead_shape = pred.shape[:-2]
    elev_b = np.broadcast_to(elev_m, pred.shape)
    bid_b = np.broadcast_to(bin_id, pred.shape)

    corr_const = pred.copy().astype(np.float64)
    corr_lin = pred.copy().astype(np.float64)
    a_const_map = {e["bin"]: e.get("a_const") for e in fit["per_bin"]}
    a_lin_map = {e["bin"]: e.get("a_lin") for e in fit["per_bin"]}
    b_lin_map = {e["bin"]: e.get("b_lin") for e in fit["per_bin"]}

    for b, a_c in a_const_map.items():
        if a_c is None:
            continue
        m = bid_b == b
        corr_const = np.where(m, corr_const + float(a_c), corr_const)
        a_l = a_lin_map.get(b)
        b_l = b_lin_map.get(b)
        if a_l is not None and b_l is not None:
            corr_lin = np.where(m, corr_lin + float(a_l) + float(b_l) * elev_b, corr_lin)

    land = elev_m  # elevation already land-aware via bins; use bin_id >= 0
    valid = bid_b >= 0
    def rmse(a, b, mask):
        d = (a - b)[mask]
        if d.size == 0:
            return None
        return float(np.sqrt(np.mean(d.astype(np.float64) ** 2)))

    def bias(a, b, mask):
        d = (b - a)[mask]  # truth - pred convention matching fit
        if d.size == 0:
            return None
        return float(np.mean(d.astype(np.float64)))

    # For corrected: truth - corr
    per_bin = []
    for e in fit["per_bin"]:
        b = e["bin"]
        m = (bid_b == b)
        per_bin.append({
            "bin": b,
            "label": e["label"],
            "n": int(m.sum()),
            "n_grid": e.get("n_grid"),
            "a_const_applied": a_const_map.get(b),
            "a_lin_applied": a_lin_map.get(b),
            "b_lin_applied": b_lin_map.get(b),
            "rmse_raw": rmse(pred, true, m),
            "rmse_after_const": rmse(corr_const, true, m),
            "rmse_after_lin": rmse(corr_lin, true, m),
            "mean_bias_raw": bias(pred, true, m),
        })

    return {
        "variable": fit["variable"],
        "model": "apply_train_elev_binned_linear",
        "lead_h": fit.get("lead_h"),
        "scope": fit.get("scope"),
        "rmse_raw_valid": rmse(pred, true, valid),
        "rmse_after_const_valid": rmse(corr_const, true, valid),
        "rmse_after_lin_valid": rmse(corr_lin, true, valid),
        "n_samples_valid": int(valid.sum()),
        "mean_bias_raw_valid": bias(pred, true, valid),
        "per_bin": per_bin,
    }


def fit_all(pair: dict, oro: dict, bins: dict, leads: list[int]) -> list[dict]:
    fits = []
    for lh in leads:
        for vname, flds in subset_lead(pair["fields"], pair["leads_h"], lh).items():
            fit = fit_binned_linear(flds["pred"], flds["true"], oro["elev_m"], bins, vname)
            fit["lead_h"] = lh
            fit["scope"] = f"lead_{lh}h"
            fits.append(fit)
    for vname, flds in pair["fields"].items():
        fit = fit_binned_linear(flds["pred"], flds["true"], oro["elev_m"], bins, vname)
        fit["lead_h"] = "pooled"
        fit["scope"] = "pooled_all_leads"
        fits.append(fit)
    return fits


def score_with_train_fits(pair: dict, oro: dict, bins: dict, train_fits: list[dict], leads: list[int]) -> list[dict]:
    # Index train fits by (variable, scope)
    by_key = {(f["variable"], f["scope"]): f for f in train_fits}
    scores = []
    for lh in leads:
        sub = subset_lead(pair["fields"], pair["leads_h"], lh)
        for vname, flds in sub.items():
            key = (vname, f"lead_{lh}h")
            if key not in by_key:
                continue
            sc = apply_fit_score(by_key[key], flds["pred"], flds["true"], oro["elev_m"], bins)
            sc["scope"] = f"lead_{lh}h"
            sc["lead_h"] = lh
            sc["applied_from"] = "train_fit"
            scores.append(sc)
    for vname, flds in pair["fields"].items():
        key = (vname, "pooled_all_leads")
        if key not in by_key:
            continue
        sc = apply_fit_score(by_key[key], flds["pred"], flds["true"], oro["elev_m"], bins)
        sc["scope"] = "pooled_all_leads"
        sc["lead_h"] = "pooled"
        sc["applied_from"] = "train_fit"
        scores.append(sc)
    return scores


def write_split_csv(path: Path, rows: list[dict], split: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "split", "variable", "scope", "lead_h", "bin", "label", "n",
        "rmse_raw", "rmse_after_const", "rmse_after_lin", "mean_bias_raw",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for sc in rows:
            for e in sc.get("per_bin", []):
                w.writerow({
                    "split": split,
                    "variable": sc["variable"],
                    "scope": sc.get("scope"),
                    "lead_h": sc.get("lead_h"),
                    "bin": e.get("bin"),
                    "label": e.get("label"),
                    "n": e.get("n"),
                    "rmse_raw": e.get("rmse_raw"),
                    "rmse_after_const": e.get("rmse_after_const"),
                    "rmse_after_lin": e.get("rmse_after_lin"),
                    "mean_bias_raw": e.get("mean_bias_raw"),
                })


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path.home() / "fourcastnet" / "configs" / "tier0_bias.yaml")
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path.home() / "fourcastnet" / "data" / "tier0_ics" / "tier0_ic_manifest.json",
    )
    ap.add_argument(
        "--pairs-dir",
        type=Path,
        default=Path.home() / "fourcastnet" / "runs" / "phase0" / "tier0_holdout" / "pairs",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "fourcastnet" / "runs" / "phase0" / "tier0_holdout",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(expand(args.config))
    locks = cfg.get("locks", {})
    box = Box.from_cfg(cfg)
    paths = check_paths(cfg)
    # override output
    out_dir = expand(args.output_dir)
    pairs_dir = expand(args.pairs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_map = load_split_map(expand(args.manifest))
    train_ids = {i for i, s in split_map.items() if s == "train"}
    val_ids = {i for i, s in split_map.items() if s == "val"}
    test_ids = {i for i, s in split_map.items() if s == "test"}

    report: dict[str, Any] = {
        "schema": "tier0_holdout/v1",
        "created_utc": utc_now(),
        "script": "code/phase0/tier0_holdout.py",
        "recipe": "GATE_RECIPE_TIER0_G0.md §2",
        "claim_level": "interim_era5",
        "target": "ERA5_interim",
        "g1_claimable": False,
        "provisional_years": True,
        "year_split_frozen": True,
        "year_split_note": (
            "PROVISIONAL (Manisha has not hard-locked). "
            "train 2018–2021 / val 2022 / test 2023–2024. provisional_years=true."
        ),
        "year_split": {"train": "2018-2021", "val": "2022", "test": "2023-2024"},
        "n_ids": {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)},
        "train_ids": sorted(train_ids),
        "val_ids": sorted(val_ids),
        "test_ids": sorted(test_ids),
        "pairs_dir": str(pairs_dir),
        "output_dir": str(out_dir),
        "box": {
            "lon_west": box.lon_west,
            "lon_east": box.lon_east,
            "lat_south": box.lat_south,
            "lat_north": box.lat_north,
        },
    }

    if args.dry_run:
        report["status"] = "dry_run"
        report["ok"] = True
        (out_dir / "tier0_holdout_dry_run.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return 0

    oro = load_orography_crop(paths["orography"], paths["land_mask"], box)
    edges = list(cfg.get("elevation_bins_m", [0, 1500, 3000, 4500, 9000]))
    bins = build_elevation_bins(oro["elev_m"], edges, land_mask=oro.get("land_mask"))
    variables = list(cfg.get("variables", ["t2m", "u10m", "v10m", "tcwv"]))
    leads = list(cfg.get("leads_h", [24, 72, 120]))

    all_pair = load_real_pairs(pairs_dir, variables, leads, oro["elev_m"].shape)
    if not all_pair.get("ok"):
        report["status"] = "blocked_pairs"
        report["error"] = all_pair.get("error")
        (out_dir / "tier0_holdout_metrics.json").write_text(json.dumps(report, indent=2, default=str))
        print(json.dumps(report, indent=2, default=str))
        return 2

    train_pair = filter_pairs_by_ids(all_pair, train_ids)
    val_pair = filter_pairs_by_ids(all_pair, val_ids) if val_ids else {"ok": False, "error": "no val"}
    test_pair = filter_pairs_by_ids(all_pair, test_ids) if test_ids else {"ok": False, "error": "no test"}
    if not train_pair.get("ok"):
        report["status"] = "blocked_train"
        report["error"] = train_pair.get("error")
        (out_dir / "tier0_holdout_metrics.json").write_text(json.dumps(report, indent=2, default=str))
        print(json.dumps(report, indent=2, default=str))
        return 2

    train_fits = fit_all(train_pair, oro, bins, leads)
    report["train"] = {
        "n_ics": train_pair["n_ics"],
        "ics": train_pair["ics"],
        "fits": train_fits,
    }

    holdout_scores = {}
    for name, pair in (("val", val_pair), ("test", test_pair), ("train_self", train_pair)):
        if not pair.get("ok"):
            holdout_scores[name] = {"ok": False, "error": pair.get("error")}
            continue
        scores = score_with_train_fits(pair, oro, bins, train_fits, leads)
        holdout_scores[name] = {
            "ok": True,
            "n_ics": pair["n_ics"],
            "ics": [e["ic_id"] for e in pair["ics"]],
            "scores": scores,
        }
        write_split_csv(out_dir / f"tier0_holdout_metrics_{name}.csv", scores, name)

    # Headline beat-this from holdout (prefer test pooled t2m lin RMSE; else val)
    def headline_t2m(split_name: str):
        block = holdout_scores.get(split_name) or {}
        if not block.get("ok"):
            return None
        for sc in block["scores"]:
            if sc["variable"] == "t2m" and sc.get("scope") == "pooled_all_leads":
                return {
                    "split": split_name,
                    "rmse_raw": sc.get("rmse_raw_valid"),
                    "rmse_after_lin": sc.get("rmse_after_lin_valid"),
                    "rmse_after_const": sc.get("rmse_after_const_valid"),
                    "n": sc.get("n_samples_valid"),
                }
        return None

    beat = {
        "previous_all_ics_pooled_t2m_lin_rmse": 1.978,
        "train_self": headline_t2m("train_self"),
        "val": headline_t2m("val"),
        "test": headline_t2m("test"),
        "note": (
            "Previous 1.978 K was fit+score on all 8 train-era ICs (optimistic). "
            "Holdout beat-this should use val/test after train-only fit."
        ),
    }
    # Prefer test lin RMSE as new bar if available; else val
    for key in ("test", "val"):
        h = beat.get(key)
        if h and h.get("rmse_after_lin") is not None:
            beat["recommended_beat_this_t2m_rmse_lin"] = h["rmse_after_lin"]
            beat["recommended_beat_this_source"] = key
            break

    # Save train bias maps (train-only fit)
    bias_maps = out_dir / "tier0_bias_maps_train_only.nc"
    save_bias_maps_nc(bias_maps, oro, bins, train_fits)

    # Also write train metrics csv in same style
    write_metrics_csv(out_dir / "tier0_metrics_train_fit.csv", train_fits)

    report["holdout_scores"] = holdout_scores
    report["beat_this"] = beat
    report["artifacts"] = {
        "bias_maps_train_only": str(bias_maps),
        "metrics_json": str(out_dir / "tier0_holdout_metrics.json"),
        "metrics_train_csv": str(out_dir / "tier0_metrics_train_fit.csv"),
        "metrics_val_csv": str(out_dir / "tier0_holdout_metrics_val.csv"),
        "metrics_test_csv": str(out_dir / "tier0_holdout_metrics_test.csv"),
        "metrics_train_self_csv": str(out_dir / "tier0_holdout_metrics_train_self.csv"),
    }
    report["elevation_bins_m"] = edges
    report["bin_labels"] = bins["labels"]
    report["n_per_bin"] = [int(x) for x in bins["n_per_bin"]]
    report["leads_h"] = leads
    report["variables"] = variables
    report["not_skill"] = False
    report["status"] = "holdout_ok"
    report["ok"] = True
    report["disclaimer"] = (
        "Train-only elevation-binned linear bias on global-ERA5-IC → FCN3 Nepal crop "
        "vs ARCO ERA5. Val/test scored with frozen train coefficients. "
        "target=ERA5_interim; provisional_years=true; g1_claimable=false."
    )

    # JSON-serialize: strip non-serializable if any
    out_json = out_dir / "tier0_holdout_metrics.json"
    out_json.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({
        "ok": True,
        "status": "holdout_ok",
        "beat_this": beat,
        "artifacts": report["artifacts"],
        "n_ids": report["n_ids"],
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
