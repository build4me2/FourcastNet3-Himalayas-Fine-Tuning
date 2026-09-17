#!/usr/bin/env python3
"""CPU/GPU: raw FCN3 + living residual wind baseline for v1.3 joint.

Wind-vector RMSE definition (Leonard TIER_A_V1_3_JOINT_RECIPE.md section 5):
  per-lead: sqrt( mean_over_valid_grid( (u_err^2 + v_err^2) / 2 ) )
  headline: mean of per-lead values over leads {24,72,120}  (= lead-mean)

Also reports grid-pooled-all-leads and component u10m/v10m for honesty.
Does NOT overwrite living residual weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dataset import build_loaders, expand  # noqa: E402
from model import build_model  # noqa: E402


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


@torch.no_grad()
def accumulate(model, loader, device, variables_out, apply_residual: bool):
    leads = defaultdict(lambda: {
        "t2m_sq": 0.0, "u_sq": 0.0, "v_sq": 0.0, "wvec_sum": 0.0, "n": 0
    })
    idx = {v: i for i, v in enumerate(variables_out)}
    assert "t2m" in idx and "u10m" in idx and "v10m" in idx
    if model is not None:
        model.eval()
    for batch in loader:
        x = batch["x"].to(device)
        pred_out = batch["pred_out"].to(device)
        truth = batch["truth"].to(device)
        bin_id = batch["bin_id"].to(device)
        lead_h = batch["lead_h"]
        if apply_residual:
            assert model is not None
            corr = pred_out + model(x)
        else:
            corr = pred_out
        B = x.shape[0]
        for b in range(B):
            lh = int(lead_h[b].item() if hasattr(lead_h[b], "item") else lead_h[b])
            m = bin_id[b] >= 0
            n = int(m.sum().item())
            if n == 0:
                continue
            t_e = (corr[b, idx["t2m"]] - truth[b, idx["t2m"]]).float()[m]
            u_e = (corr[b, idx["u10m"]] - truth[b, idx["u10m"]]).float()[m]
            v_e = (corr[b, idx["v10m"]] - truth[b, idx["v10m"]]).float()[m]
            a = leads[lh]
            a["t2m_sq"] += float((t_e ** 2).sum().item())
            a["u_sq"] += float((u_e ** 2).sum().item())
            a["v_sq"] += float((v_e ** 2).sum().item())
            a["wvec_sum"] += float((((u_e ** 2) + (v_e ** 2)) / 2.0).sum().item())
            a["n"] += n
    return dict(leads)


def summarize(leads_acc: dict, want_leads=(24, 72, 120)) -> dict:
    per_lead = {}
    for lh in want_leads:
        a = leads_acc.get(lh)
        if not a or a["n"] <= 0:
            continue
        n = a["n"]
        per_lead[str(lh)] = {
            "n": n,
            "t2m_rmse": float(np.sqrt(a["t2m_sq"] / n)),
            "u10m_rmse": float(np.sqrt(a["u_sq"] / n)),
            "v10m_rmse": float(np.sqrt(a["v_sq"] / n)),
            "wind_vector_rmse": float(np.sqrt(a["wvec_sum"] / n)),
        }
    tot = {"t2m_sq": 0.0, "u_sq": 0.0, "v_sq": 0.0, "wvec_sum": 0.0, "n": 0}
    for lh in want_leads:
        a = leads_acc.get(lh)
        if not a:
            continue
        for k in tot:
            tot[k] += a[k]
    n = tot["n"]
    pooled = None
    if n > 0:
        pooled = {
            "n": n,
            "t2m_rmse": float(np.sqrt(tot["t2m_sq"] / n)),
            "u10m_rmse": float(np.sqrt(tot["u_sq"] / n)),
            "v10m_rmse": float(np.sqrt(tot["v_sq"] / n)),
            "wind_vector_rmse": float(np.sqrt(tot["wvec_sum"] / n)),
        }

    def lead_mean(key):
        vals = [per_lead[str(lh)][key] for lh in want_leads if str(lh) in per_lead]
        return float(sum(vals) / len(vals)) if vals else None

    return {
        "per_lead": per_lead,
        "grid_pooled_all_leads": pooled,
        "lead_mean": {
            "t2m_rmse": lead_mean("t2m_rmse"),
            "u10m_rmse": lead_mean("u10m_rmse"),
            "v10m_rmse": lead_mean("v10m_rmse"),
            "wind_vector_rmse": lead_mean("wind_vector_rmse"),
            "leads_h": list(want_leads),
            "definition": (
                "mean over leads of per-lead grid-pooled RMSE; "
                "wind_vector per-lead = sqrt(mean((u_err^2+v_err^2)/2))"
            ),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="~/fourcastnet/configs/tier_a_v1_3_joint.yaml")
    ap.add_argument(
        "--living-ckpt",
        default="~/fourcastnet/runs/phase0/tier_a/v1_2b_thick2_train/best_residual.pt",
    )
    ap.add_argument(
        "--out",
        default="~/fourcastnet/runs/phase0/tier_a/v1_3_joint/living_wind_baseline.json",
    )
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    cfg_path = expand(args.config)
    if not cfg_path.is_file():
        cfg_path = expand("~/fourcastnet/configs/tier_a_v1_2b_thick2_train.yaml")
    ckpt_path = expand(args.living_ckpt)
    out_path = expand(args.out)
    md5_before = md5(ckpt_path)

    cfg = load_cfg(cfg_path)
    ys = cfg["year_split"]
    splits = {
        "train": list(ys["train_ids"]),
        "val": list(ys["val_ids"]),
        "test": list(ys["test_ids"]),
    }
    device = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )
    loaders = build_loaders(cfg, splits, use_members_train=False)
    vout = list(cfg["variables_out"])

    model = build_model(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    model.load_state_dict(state)

    report = {
        "schema": "living_wind_baseline/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "recipe": "docs/research/TIER_A_V1_3_JOINT_RECIPE.md",
        "living_ckpt": str(ckpt_path),
        "living_ckpt_md5": md5_before,
        "config_used": str(cfg_path),
        "device": str(device),
        "holdout_expand": cfg.get("holdout_expand", "v2"),
        "bars_set": cfg.get("bars_set", "thick2"),
        "year_split": {
            "train": ys.get("train"),
            "val": ys.get("val"),
            "test": ys.get("test"),
            "n_train": len(splits["train"]),
            "n_val": len(splits["val"]),
            "n_test": len(splits["test"]),
        },
        "wind_vector_definition": {
            "formula": "sqrt(mean_over_valid_grid((u_err^2 + v_err^2)/2))",
            "headline": "lead_mean over leads [24,72,120]",
            "also_reported": "grid_pooled_all_leads (single pool across leads)",
            "channels": ["u10m", "v10m"],
            "valid_mask": "elev_bin_id >= 0",
        },
        "promote_gate_uses": "lead_mean.wind_vector_rmse",
        "splits": {},
        "published_living_t2m": {
            "source": "runs/phase0/tier_a/v1_2b_thick2_train/tier_a_v1_2b_thick2_train_results.json",
            "val_pooled": 1.759883,
            "test_pooled": 1.779660,
            "val_120h": 1.962487,
            "note": "canonical living bars; baseline recompute should match val/test pooled t2m",
        },
        "honesty": (
            "Living residual trained with channel_weights [2.0,0.5,0.5]; "
            "eval historically t2m-only. This baseline scores raw FCN3 and living "
            "residual on u/v/wind-vector without retraining. No overwrite of living weights."
        ),
    }

    for split in ("val", "test"):
        raw_acc = accumulate(None, loaders[split], device, vout, apply_residual=False)
        liv_acc = accumulate(model, loaders[split], device, vout, apply_residual=True)
        report["splits"][split] = {
            "raw_fcn3": summarize(raw_acc),
            "living_residual": summarize(liv_acc),
        }
        print(
            f"{split} wind_vector lead_mean raw="
            f"{report['splits'][split]['raw_fcn3']['lead_mean']['wind_vector_rmse']:.6f} "
            f"living="
            f"{report['splits'][split]['living_residual']['lead_mean']['wind_vector_rmse']:.6f}",
            flush=True,
        )
        print(
            f"{split} t2m pooled living="
            f"{report['splits'][split]['living_residual']['grid_pooled_all_leads']['t2m_rmse']:.6f}",
            flush=True,
        )

    report["vs_living_wind_bars"] = {
        "val_wind_vector_lead_mean_strict_lt": report["splits"]["val"]["living_residual"]["lead_mean"]["wind_vector_rmse"],
        "test_wind_vector_lead_mean_strict_lt": report["splits"]["test"]["living_residual"]["lead_mean"]["wind_vector_rmse"],
        "val_raw_wind_vector_lead_mean": report["splits"]["val"]["raw_fcn3"]["lead_mean"]["wind_vector_rmse"],
        "test_raw_wind_vector_lead_mean": report["splits"]["test"]["raw_fcn3"]["lead_mean"]["wind_vector_rmse"],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    md5_after = md5(ckpt_path)
    if md5_before != md5_after:
        print("ERROR: living ckpt MD5 changed", file=sys.stderr)
        return 2
    print(f"wrote {out_path}; living md5 unchanged {md5_before}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
