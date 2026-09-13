#!/usr/bin/env python3
"""Tier-A v1.1 train/eval — smaller/regularized elev-conditioned residual UNet.

Leonard TIER_A_V1_CALL.md: v1 FAIL (test overfit). Iterate v1.1. NO diffusion.
FCN3 frozen. Writes ONLY under runs/phase0/tier_a/v1_1/.
Does NOT overwrite v0/, v0_1/, v1/, g0_verifying, tier0_holdout.
Does NOT touch ERA5 PID 611595 / FCN3 weights.

Usage (from ~/fourcastnet):
  ~/fcn3-venv/bin/python code/tier_a/v1_1/train.py --config configs/tier_a_v1_1.yaml --dry-run
  ~/fcn3-venv/bin/python code/tier_a/v1_1/train.py --config configs/tier_a_v1_1.yaml --train --epochs 400
  ~/fcn3-venv/bin/python code/tier_a/v1_1/train.py --config configs/tier_a_v1_1_expand.yaml --eval-only \
      --pairs-dir runs/phase0/tier0_holdout_expand/pairs \
      --out-dir runs/phase0/tier_a/v1_1_expand \
      --ckpt runs/phase0/tier_a/v1_1/best_residual.pt
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # ~/fourcastnet  (code/tier_a/v1_1/)
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    import yaml
except ImportError:
    yaml = None

from dataset import build_loaders, expand, load_split_ids  # noqa: E402
from model import build_model  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config(path: Path) -> dict:
    if yaml is None:
        raise RuntimeError("PyYAML required")
    return yaml.safe_load(path.read_text())


def artifact_stem(cfg: dict) -> str:
    return str(cfg.get("artifact_stem") or "tier_a_v1_1")


def loss_mode(cfg: dict) -> str:
    return str(cfg.get("train", {}).get("loss", "mse_residual_elev_weighted_lead_weighted"))


def lead_weights(cfg: dict) -> dict[int, float]:
    """Per-lead loss weights. w_120 MUST be ≥ 1 (Leonard v1.1 / v1-reg freeze)."""
    t = cfg.get("train", {})
    raw = t.get("lead_weights") or {}
    w24 = float(raw.get(24, raw.get("24", 1.0)))
    w72 = float(raw.get(72, raw.get("72", 1.0)))
    w120 = float(raw.get(120, raw.get("120", t.get("w_120", 1.0))))
    if w120 < 1.0 - 1e-12:
        raise RuntimeError(f"w_120 must be >= 1 (got {w120}); Leonard TIER_A_V1_CALL v1.1")
    return {24: w24, 72: w72, 120: w120}


# Thin-set bars (historical only — v0/v0.1/v1/v1.1 thin calls).
THIN_BEAT_THIS = {
    "val_t2m_pooled_rmse_lin_strict_lt": 1.948661,
    "val_headline_lt": 1.949,
    "test_t2m_pooled_rmse_lin_strict_lt": 1.850338,
    "val_120h_t2m_rmse_raw_le": 2.219970831929039,
    "val_120h_rule": (
        "val +120h t2m RMSE <= raw (2.219971 K); hard gate for v1.1; "
        "v0/v0.1 grandfathered; v1 FAIL frozen"
    ),
    "source": "runs/phase0/tier0_holdout/tier0_holdout_metrics.json",
    "call": "docs/research/TIER_A_V1_CALL.md",
    "note": "thin historical only; living expand bars in HOLDOUT_EXPAND_CALL.md",
    "frozen_exact": {
        "val_rmse_after_lin": 1.9486610005712885,
        "test_rmse_after_lin": 1.8503382009036373,
        "val_120h_t2m_rmse_raw": 2.219970831929039,
    },
}

# Expand bars FROZEN (Leonard docs/research/HOLDOUT_EXPAND_CALL.md) — supersede thin for Tier-A.
EXPAND_BEAT_THIS = {
    "val_t2m_pooled_rmse_lin_strict_lt": 1.987014,
    "val_headline_lt": 1.987,
    "test_t2m_pooled_rmse_lin_strict_lt": 1.897298,
    "val_120h_t2m_rmse_raw_le": 2.131515989123864,
    "val_120h_rule": (
        "val +120h t2m RMSE <= expand-val raw (2.131516 K); hard gate for v1.x; "
        "thin 2.220 historical only"
    ),
    "source": "runs/phase0/tier0_holdout_expand/tier0_holdout_expand_metrics.json",
    "call": "docs/research/HOLDOUT_EXPAND_CALL.md",
    "note": (
        "primary=val strict < 1.987014; secondary=test strict < 1.897298; "
        "+120h <= expand raw 2.131516; thin bars historical only; val-only win = FAIL"
    ),
    "frozen_exact": {
        "val_rmse_after_lin": 1.987014136483935,
        "test_rmse_after_lin": 1.8972984228975271,
        "val_120h_t2m_rmse_raw": 2.131515989123864,
    },
}


def uses_expand_bars(cfg: dict) -> bool:
    """True when holdout_expand=v1 or config/variant/paths indicate expand."""
    he = cfg.get("holdout_expand")
    if he is True or str(he or "").strip().lower() in {"v1", "true", "1", "yes"}:
        return True
    variant = str(cfg.get("variant") or "").lower()
    stem = str(cfg.get("artifact_stem") or "").lower()
    out = str(cfg.get("output_dir") or "").lower()
    if "expand" in variant or "expand" in stem or "v1_1_expand" in out:
        return True
    data = cfg.get("data") or {}
    for key in ("pairs_dir", "holdout_metrics"):
        p = str(data.get(key) or "").lower()
        if "holdout_expand" in p or "tier0_holdout_expand" in p:
            return True
    return False


def thin_beat_this_echo() -> dict:
    g0 = {
        "baseline": "runs/phase0/g0/g0_verifying_results.json",
        "rule": "≤~5% CRPS degrade midlat @ +15 d; PSD/SSR floors",
        "tier_a_status": "not_applicable_fcn3_frozen",
    }
    out = dict(THIN_BEAT_THIS)
    out["g0_adapter"] = g0
    out["frozen_exact"] = dict(THIN_BEAT_THIS["frozen_exact"])
    return out


def raw_val_120h_ref(cfg: dict) -> float:
    """Exact raw val +120h t2m RMSE. Do not beat a softer number."""
    bt = cfg.get("beat_this", {})
    t = cfg.get("train", {})
    default = (
        EXPAND_BEAT_THIS["val_120h_t2m_rmse_raw_le"]
        if uses_expand_bars(cfg)
        else THIN_BEAT_THIS["val_120h_t2m_rmse_raw_le"]
    )
    return float(
        t.get("raw_val_120h_t2m_rmse")
        or bt.get("val_120h_t2m_rmse_raw_le")
        or default
    )


def beat_this_echo(cfg: dict) -> dict:
    bt = cfg.get("beat_this", {})
    g0 = bt.get(
        "g0_adapter",
        {
            "baseline": "runs/phase0/g0/g0_verifying_results.json",
            "rule": "≤~5% CRPS degrade midlat @ +15 d; PSD/SSR floors",
            "tier_a_status": "not_applicable_fcn3_frozen",
        },
    )
    defaults = EXPAND_BEAT_THIS if uses_expand_bars(cfg) else THIN_BEAT_THIS
    return {
        "val_t2m_pooled_rmse_lin_strict_lt": float(
            bt.get("val_t2m_pooled_rmse_lin_strict_lt", defaults["val_t2m_pooled_rmse_lin_strict_lt"])
        ),
        "val_headline_lt": float(bt.get("val_headline_lt", defaults["val_headline_lt"])),
        "test_t2m_pooled_rmse_lin_strict_lt": float(
            bt.get("test_t2m_pooled_rmse_lin_strict_lt", defaults["test_t2m_pooled_rmse_lin_strict_lt"])
        ),
        "val_120h_t2m_rmse_raw_le": float(
            bt.get("val_120h_t2m_rmse_raw_le", raw_val_120h_ref(cfg))
        ),
        "val_120h_rule": bt.get("val_120h_rule", defaults["val_120h_rule"]),
        "source": bt.get("source", defaults["source"]),
        "call": bt.get("call", defaults["call"]),
        "note": bt.get("note", defaults["note"]),
        "g0_adapter": g0,
        "frozen_exact": dict(defaults["frozen_exact"]),
        "bars_set": "expand" if uses_expand_bars(cfg) else "thin_historical",
    }


def elev_weights(bin_id, boost: float = 1.5):
    import torch

    w = torch.ones_like(bin_id, dtype=torch.float32)
    w = torch.where(bin_id >= 2, w * boost, w)
    w = torch.where(bin_id >= 3, w * boost, w)
    return w


def _channel_weights(pred_res, cfg) -> "torch.Tensor":
    import torch

    cw = cfg.get("train", {}).get("channel_weights") or [2.0, 0.5, 0.5]
    c = pred_res.shape[1]
    vals = [float(cw[i]) if i < len(cw) else 1.0 for i in range(c)]
    return torch.tensor(vals, device=pred_res.device, dtype=pred_res.dtype).view(1, c, 1, 1)


def _elev_weighted_mse(pred_res, true_res, bin_id, boost: float, cfg: dict | None = None):
    import torch

    w = elev_weights(bin_id, boost)
    err = (pred_res - true_res) ** 2
    if cfg is not None:
        err = err * _channel_weights(pred_res, cfg)
    w3 = w.unsqueeze(1)
    return (err * w3).sum() / (w3.expand_as(err).sum().clamp_min(1.0))


def residual_loss(
    pred_res,
    true_res,
    bin_id,
    boost: float,
    lead_h=None,
    weights: dict[int, float] | None = None,
    leads_h: list[int] | None = None,
    cfg: dict | None = None,
):
    """Elev-weighted residual MSE, then weighted mean across leads (w_120 ≥ 1)."""
    import torch

    if lead_h is None:
        return _elev_weighted_mse(pred_res, true_res, bin_id, boost, cfg)
    if not torch.is_tensor(lead_h):
        lead_h = torch.as_tensor(lead_h, device=pred_res.device)
    else:
        lead_h = lead_h.to(pred_res.device)

    ref = list(leads_h) if leads_h else sorted({int(x) for x in lead_h.detach().cpu().tolist()})
    wmap = weights or {24: 1.0, 72: 1.0, 120: 1.0}
    acc = pred_res.new_zeros(())
    wsum = 0.0
    for lh in ref:
        mask = lead_h == int(lh)
        if int(mask.sum().item()) == 0:
            continue
        wl = float(wmap.get(int(lh), 1.0))
        acc = acc + wl * _elev_weighted_mse(pred_res[mask], true_res[mask], bin_id[mask], boost, cfg)
        wsum += wl
    if wsum <= 0:
        return _elev_weighted_mse(pred_res, true_res, bin_id, boost, cfg)
    return acc / wsum


def mean_per_lead_rmse(metrics: dict) -> float | None:
    pl = metrics.get("per_lead") or {}
    vals = [float(v["rmse_tier_a"]) for v in pl.values() if v.get("rmse_tier_a") is not None]
    if not vals:
        return metrics.get("rmse_tier_a")
    return float(sum(vals) / len(vals))


def eval_split(model, loader, device, variables_out: list[str]) -> dict:
    import numpy as np
    import torch

    model.eval()
    sq_raw = 0.0
    sq_corr = 0.0
    n = 0
    per_lead: dict[int, dict[str, float]] = {}
    per_bin: dict[int, dict[str, float]] = {}

    for batch in loader:
        x = batch["x"].to(device)
        pred_out = batch["pred_out"].to(device)
        truth = batch["truth"].to(device)
        bin_id = batch["bin_id"].to(device)
        leads = batch["lead_h"]
        with torch.no_grad():
            res = model(x)
        corr = pred_out + res
        for b in range(x.shape[0]):
            lh = int(leads[b].item() if hasattr(leads[b], "item") else leads[b])
            raw_e = (pred_out[b, 0] - truth[b, 0]).float()
            cor_e = (corr[b, 0] - truth[b, 0]).float()
            m = bin_id[b] >= 0
            sq_raw += float((raw_e[m] ** 2).sum().item())
            sq_corr += float((cor_e[m] ** 2).sum().item())
            n += int(m.sum().item())

            if lh not in per_lead:
                per_lead[lh] = {"sq_raw": 0.0, "sq_corr": 0.0, "n": 0}
            per_lead[lh]["sq_raw"] += float((raw_e[m] ** 2).sum().item())
            per_lead[lh]["sq_corr"] += float((cor_e[m] ** 2).sum().item())
            per_lead[lh]["n"] += int(m.sum().item())

            for bb in range(4):
                mb = bin_id[b] == bb
                if int(mb.sum().item()) == 0:
                    continue
                if bb not in per_bin:
                    per_bin[bb] = {"sq_raw": 0.0, "sq_corr": 0.0, "n": 0}
                per_bin[bb]["sq_raw"] += float((raw_e[mb] ** 2).sum().item())
                per_bin[bb]["sq_corr"] += float((cor_e[mb] ** 2).sum().item())
                per_bin[bb]["n"] += int(mb.sum().item())

    def rmse(sq, nn):
        return float(np.sqrt(sq / nn)) if nn > 0 else None

    out = {
        "variable": "t2m",
        "n": n,
        "rmse_raw": rmse(sq_raw, n),
        "rmse_tier_a": rmse(sq_corr, n),
        "per_lead": {
            str(k): {
                "rmse_raw": rmse(v["sq_raw"], v["n"]),
                "rmse_tier_a": rmse(v["sq_corr"], v["n"]),
                "n": v["n"],
            }
            for k, v in sorted(per_lead.items())
        },
        "per_elev_bin": {
            str(k): {
                "rmse_raw": rmse(v["sq_raw"], v["n"]),
                "rmse_tier_a": rmse(v["sq_corr"], v["n"]),
                "n": v["n"],
            }
            for k, v in sorted(per_bin.items())
        },
    }
    out["rmse_tier_a_lead_mean"] = mean_per_lead_rmse(out)
    return out


def val_120h_rmse(metrics: dict) -> float | None:
    pl = (metrics.get("per_lead") or {}).get("120") or {}
    return pl.get("rmse_tier_a")


def constraint_ok(val_m: dict, raw_120: float) -> bool:
    v120 = val_120h_rmse(val_m)
    return v120 is not None and float(v120) <= float(raw_120) + 1e-9


def gate_pass_v1(
    val_rmse: float | None,
    test_rmse: float | None,
    val_120: float | None,
    raw_120: float,
    bars: dict,
) -> dict:
    vbar = bars["val_t2m_pooled_rmse_lin_strict_lt"]
    tbar = bars["test_t2m_pooled_rmse_lin_strict_lt"]
    val_ok = val_rmse is not None and val_rmse < vbar
    test_ok = test_rmse is not None and test_rmse < tbar
    lead120_ok = val_120 is not None and float(val_120) <= float(raw_120) + 1e-9
    return {
        "val_pass": bool(val_ok),
        "test_pass": bool(test_ok),
        "val_120h_le_raw": bool(lead120_ok),
        "val_120h_rule": bars.get("val_120h_rule"),
        "raw_val_120h_t2m_rmse": float(raw_120),
        "val_120h_t2m_rmse": val_120,
        "val_120h_delta_vs_raw": (
            None if val_120 is None else float(val_120) - float(raw_120)
        ),
        "tier_a_interim_pass": bool(val_ok and test_ok and lead120_ok),
        "val_only_win_is_fail": True,
        "plus120h_regress_is_fail": True,
        "note": (
            f"v1.1 INTERIM PASS requires (1) val pooled < {vbar} AND "
            f"(2) test pooled < {tbar} AND (3) val +120h t2m RMSE <= raw "
            f"({raw_120:.6f} K). claim_level remains interim_era5; "
            "g1_claimable=false. Pooled-only win with +120h regress = FAIL. "
            "v1/ is frozen FAIL (do not overwrite). "
            "Expand bars: docs/research/HOLDOUT_EXPAND_CALL.md."
        ),
    }


def file_md5(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def freeze_snapshot() -> dict:
    paths = {
        "tier_a_v0_results": expand("~/fourcastnet/runs/phase0/tier_a/v0/tier_a_v0_results.json"),
        "tier_a_v0_1_results": expand("~/fourcastnet/runs/phase0/tier_a/v0_1/tier_a_v0_1_results.json"),
        "tier_a_v1_results": expand("~/fourcastnet/runs/phase0/tier_a/v1/tier_a_v1_results.json"),
        "g0_verifying": expand("~/fourcastnet/runs/phase0/g0/g0_verifying_results.json"),
        "tier0_holdout": expand("~/fourcastnet/runs/phase0/tier0_holdout/tier0_holdout_metrics.json"),
    }
    return {k: {"path": str(p), "md5": file_md5(p), "mtime": (p.stat().st_mtime if p.exists() else None)} for k, p in paths.items()}


def assert_no_overwrite(out_dir: Path, cfg: dict) -> None:
    out_r = out_dir.resolve()
    forbidden = [
        expand("~/fourcastnet/runs/phase0/g0/g0_verifying_results.json"),
        expand("~/fourcastnet/runs/phase0/tier0_holdout/tier0_holdout_metrics.json"),
        expand("~/fourcastnet/runs/phase0/tier0_holdout"),
        expand("~/fourcastnet/runs/phase0/tier_a/v0"),
        expand("~/fourcastnet/runs/phase0/tier_a/v0_1"),
        expand("~/fourcastnet/runs/phase0/tier_a/v1"),
    ]
    for p in forbidden:
        pr = p.resolve()
        if out_r == pr or (pr.exists() and (out_r == pr or str(out_r).startswith(str(pr) + "/"))):
            raise RuntimeError(f"refusing to write into frozen path {p}")
    thin = expand("~/fourcastnet/runs/phase0/tier_a/v1_1").resolve()
    expand_root = expand("~/fourcastnet/runs/phase0/tier_a/v1_1_expand").resolve()
    allowed_roots = [thin, expand_root]
    ok = any(out_r == a or str(out_r).startswith(str(a) + "/") for a in allowed_roots)
    if not ok:
        raise RuntimeError(
            f"v1.1 must write under runs/phase0/tier_a/v1_1/ or v1_1_expand/ (got {out_r})"
        )
    # Expand zero-shot must not overwrite the frozen thin v1_1 train directory.
    if cfg.get("variant") == "v1_1_expand_zero_shot":
        if not (out_r == expand_root or str(out_r).startswith(str(expand_root) + "/")):
            raise RuntimeError(
                f"expand zero-shot must write under v1_1_expand/ (got {out_r}); thin v1_1/ is frozen"
            )
    for p in [
        expand("~/fourcastnet/runs/phase0/g0/g0_verifying_results.json"),
        expand("~/fourcastnet/runs/phase0/tier0_holdout/tier0_holdout_metrics.json"),
        expand("~/fourcastnet/runs/phase0/tier_a/v1/tier_a_v1_results.json"),
    ]:
        if not p.exists():
            raise RuntimeError(f"forbidden artifact missing (do not recreate casually): {p}")


def train_loop(model, loaders, cfg, device, epochs: int, out_dir: Path) -> dict:
    import torch

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    boost = float(cfg["train"].get("elev_weight_boost", 1.5))
    wmap = lead_weights(cfg)
    leads_ref = [int(x) for x in cfg.get("leads_h", [24, 72, 120])]
    raw_120 = raw_val_120h_ref(cfg)
    patience = int(cfg["train"].get("early_stop_patience", 60))
    select_by = str(cfg["train"].get("ckpt_select", "composite"))
    hist = []
    best_eligible_score = float("inf")
    best_eligible_pooled = None
    best_eligible_ep = None
    best_uncon_120 = float("inf")
    best_path = out_dir / "best_residual.pt"
    uncon_path = out_dir / "best_unconstrained_120h.pt"
    last_improve_ep = 0
    n_eligible_saves = 0

    def _score(val_m: dict) -> float | None:
        pooled = val_m.get("rmse_tier_a")
        lead_mean = val_m.get("rmse_tier_a_lead_mean")
        if select_by == "pooled":
            return pooled
        return lead_mean if lead_mean is not None else pooled

    # epoch 0 = identity (zero-init head)
    val0 = eval_split(model, loaders["val"], device, cfg["variables_out"])
    s0 = _score(val0)
    ok0 = constraint_ok(val0, raw_120)
    hist.append(
        {
            "epoch": 0,
            "train_loss": None,
            "val_t2m_rmse": val0["rmse_tier_a"],
            "val_t2m_rmse_lead_mean": val0.get("rmse_tier_a_lead_mean"),
            "val_120h": val_120h_rmse(val0),
            "constraint_ok": ok0,
            "select_score": s0,
            "saved_eligible": False,
        }
    )
    print(
        f"[ep 000/{epochs}] identity val_pooled={val0['rmse_tier_a']:.5f} "
        f"val_lead_mean={val0.get('rmse_tier_a_lead_mean'):.5f} "
        f"val_120h={val_120h_rmse(val0):.5f} constraint_ok={ok0}",
        flush=True,
    )

    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        nbat = 0
        for batch in loaders["train"]:
            x = batch["x"].to(device)
            y = batch["y_residual"].to(device)
            bid = batch["bin_id"].to(device)
            lead_h = batch["lead_h"]
            if not torch.is_tensor(lead_h):
                lead_h = torch.as_tensor(lead_h, device=device)
            else:
                lead_h = lead_h.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(x)
            loss = residual_loss(
                pred,
                y,
                bid,
                boost,
                lead_h=lead_h,
                weights=wmap,
                leads_h=leads_ref,
                cfg=cfg,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item())
            nbat += 1
        tr_loss = total / max(nbat, 1)
        val_m = eval_split(model, loaders["val"], device, cfg["variables_out"])
        pooled = val_m["rmse_tier_a"]
        lead_mean = val_m.get("rmse_tier_a_lead_mean")
        v120 = val_120h_rmse(val_m)
        score = _score(val_m)
        ok = constraint_ok(val_m, raw_120)
        saved = False
        # unconstrained tracker: lowest val +120h (diagnostics only)
        if v120 is not None and v120 < best_uncon_120:
            best_uncon_120 = v120
            torch.save(
                {
                    "epoch": ep,
                    "model": model.state_dict(),
                    "val_t2m_rmse": pooled,
                    "val_t2m_rmse_lead_mean": lead_mean,
                    "val_120h": v120,
                    "constraint_ok": ok,
                    "kind": "unconstrained_best_120h",
                },
                uncon_path,
            )
        # composite: REJECT if +120h constraint fails; else save by lead-mean/pooled
        if ok and score is not None and score < best_eligible_score:
            best_eligible_score = score
            best_eligible_pooled = pooled
            best_eligible_ep = ep
            last_improve_ep = ep
            n_eligible_saves += 1
            saved = True
            torch.save(
                {
                    "epoch": ep,
                    "model": model.state_dict(),
                    "val_t2m_rmse": pooled,
                    "val_t2m_rmse_lead_mean": lead_mean,
                    "val_120h": v120,
                    "select_score": score,
                    "select_by": select_by,
                    "constraint_ok": True,
                    "raw_val_120h": raw_120,
                    "w_120": wmap[120],
                    "lead_weights": wmap,
                    "cfg_schema": cfg.get("schema"),
                    "loss": loss_mode(cfg),
                    "kind": "composite_eligible",
                },
                best_path,
            )
        hist.append(
            {
                "epoch": ep,
                "train_loss": tr_loss,
                "val_t2m_rmse": pooled,
                "val_t2m_rmse_lead_mean": lead_mean,
                "val_120h": v120,
                "constraint_ok": ok,
                "select_score": score,
                "saved_eligible": saved,
            }
        )
        if ep == 1 or ep % 5 == 0 or ep == epochs or saved:
            print(
                f"[ep {ep:03d}/{epochs}] train_loss={tr_loss:.5f} "
                f"val_pooled={pooled:.5f} val_lead_mean={lead_mean:.5f} "
                f"val_120h={v120 if v120 is not None else float('nan'):.5f} "
                f"constraint_ok={ok} saved={saved} "
                f"best_elig={best_eligible_score if best_eligible_score < float('inf') else float('nan'):.5f}",
                flush=True,
            )
        if ep - last_improve_ep >= patience:
            print(
                f"[early-stop] no eligible composite improve for {patience} ep "
                f"(last={last_improve_ep}, now={ep})",
                flush=True,
            )
            break

    used_kind = None
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        used_kind = "composite_eligible"
    elif uncon_path.exists():
        ckpt = torch.load(uncon_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        used_kind = "fallback_unconstrained_120h"
        print("WARN: no +120h-eligible ckpt; reloaded unconstrained best +120h (gate will FAIL)", flush=True)
    return {
        "history": hist,
        "best_val_t2m_rmse": best_eligible_pooled,
        "best_select_score": (best_eligible_score if best_eligible_score < float("inf") else None),
        "best_eligible_epoch": best_eligible_ep,
        "n_eligible_saves": n_eligible_saves,
        "select_by": select_by,
        "lead_weights": wmap,
        "w_120": wmap[120],
        "raw_val_120h": raw_120,
        "early_stop_patience": patience,
        "stopped_epoch": hist[-1]["epoch"] if hist else 0,
        "reload_kind": used_kind,
        "best_ckpt": str(best_path) if best_path.exists() else None,
        "unconstrained_120h_ckpt": str(uncon_path) if uncon_path.exists() else None,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def maybe_load_compare(name: str, path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        prev = json.loads(path.read_text())
    except Exception:
        return None
    m = prev.get("metrics", {})
    out = {"path": str(path), "variant": prev.get("variant"), "gates": prev.get("gates")}
    for split in ("val", "test"):
        sm = m.get(split) or {}
        out[split] = {
            "rmse_tier_a": sm.get("rmse_tier_a"),
            "per_lead": {
                k: {"rmse_tier_a": (v or {}).get("rmse_tier_a"), "rmse_raw": (v or {}).get("rmse_raw")}
                for k, v in (sm.get("per_lead") or {}).items()
            },
        }
    return out


def lead120_delta(m: dict) -> dict:
    pl = (m.get("per_lead") or {}).get("120") or {}
    raw = pl.get("rmse_raw")
    corr = pl.get("rmse_tier_a")
    return {
        "rmse_raw": raw,
        "rmse_tier_a": corr,
        "delta_vs_raw": (None if raw is None or corr is None else float(corr) - float(raw)),
        "le_raw": (None if raw is None or corr is None else bool(float(corr) <= float(raw) + 1e-9)),
    }



def _tier0_summary(hm: dict) -> dict:
    """Pull headline Tier-0 RMSE bars from metrics JSON if present."""
    out: dict[str, Any] = {}
    for split in ("val", "test", "train"):
        sm = hm.get(split) or (hm.get("metrics") or {}).get(split) or {}
        if not isinstance(sm, dict):
            continue
        for key in (
            "rmse_after_lin",
            "t2m_pooled_rmse_lin",
            "rmse_t2m_lin",
            "pooled_t2m_rmse",
        ):
            if key in sm:
                out[f"{split}_{key}"] = sm[key]
        if "per_lead" in sm:
            out[f"{split}_per_lead"] = sm["per_lead"]
        if "t2m" in sm:
            out[f"{split}_t2m"] = sm["t2m"]
    for k in (
        "val_rmse_after_lin",
        "test_rmse_after_lin",
        "val_t2m_pooled_rmse_lin",
        "test_t2m_pooled_rmse_lin",
    ):
        if k in hm:
            out[k] = hm[k]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Tier-A v1.1 residual train/eval (no diffusion)")
    ap.add_argument("--config", default="configs/tier_a_v1_1.yaml")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="short CPU smoke train")
    ap.add_argument("--train", action="store_true", help="full train (train-split only)")
    ap.add_argument(
        "--eval-only",
        action="store_true",
        help="zero-shot eval: load frozen ckpt, no optimize (holdout expand)",
    )
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--allow-gpu", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument(
        "--pairs-dir",
        default=None,
        help="Override cfg data.pairs_dir (expand pairs)",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Override cfg output_dir (e.g. runs/phase0/tier_a/v1_1_expand)",
    )
    ap.add_argument(
        "--ckpt",
        default=None,
        help="Frozen residual ckpt for --eval-only (default: v1_1/best_residual.pt)",
    )
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (ROOT / cfg_path).resolve()
    cfg = load_config(cfg_path)
    if args.pairs_dir:
        cfg.setdefault("data", {})["pairs_dir"] = args.pairs_dir
    if args.out_dir:
        cfg["output_dir"] = args.out_dir
    bars = beat_this_echo(cfg)
    stem = artifact_stem(cfg)
    out_dir = expand(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    assert_no_overwrite(out_dir, cfg)
    freeze_before = freeze_snapshot()
    wmap = lead_weights(cfg)
    raw_120 = raw_val_120h_ref(cfg)

    g0p = expand(cfg["data"]["g0_verifying"])
    holdp = expand(cfg["data"]["holdout_metrics"])
    if not g0p.is_file() or not holdp.is_file():
        print(f"WARN: missing baseline artifact g0={g0p.exists()} holdout={holdp.exists()}")

    splits_cfg = cfg["year_split"]
    man = expand(cfg["data"]["ic_manifest"])
    if man.is_file():
        splits = load_split_ids(man)
    else:
        splits = {
            "train": list(splits_cfg["train_ids"]),
            "val": list(splits_cfg["val_ids"]),
            "test": list(splits_cfg["test_ids"]),
        }

    report: dict[str, Any] = {
        "schema": cfg.get("schema", "tier_a_v1_1_results/v1"),
        "created_utc": utc_now(),
        "script": "code/tier_a/v1_1/train.py",
        "config": str(cfg_path),
        "method": cfg.get("method"),
        "tier": "A",
        "variant": cfg.get("variant"),
        "cut": "v1.1-reg",
        "diffusion": False,
        "fcn3_weights": "FROZEN",
        "claim_level": cfg["locks"]["claim_level"],
        "target": cfg["locks"]["target"],
        "g1_claimable": False,
        "provisional_years": True,
        "year_split_frozen": True,
        "year_split": {
            "train": splits_cfg["train"],
            "val": splits_cfg["val"],
            "test": splits_cfg["test"],
        },
        "train_ids": splits["train"],
        "val_ids": splits["val"],
        "test_ids": splits["test"],
        "box": cfg["region"]["lat_lon_box"],
        "beat_this": bars,
        "loss": {
            "name": loss_mode(cfg),
            "lead_balance": True,
            "w_120": wmap[120],
            "lead_weights": {str(k): v for k, v in wmap.items()},
            "description": (
                "Multi-lead elev-weighted residual MSE with explicit per-lead "
                f"weights w_24={wmap[24]}, w_72={wmap[72]}, w_120={wmap[120]} "
                "(w_120 >= 1 required). Missing leads in a batch skipped / "
                "renormalized. Composite ckpt: REJECT if val +120h t2m RMSE > raw; "
                "else save best lead-mean (pooled tracked)."
            ),
            "leads_h": list(cfg.get("leads_h", [24, 72, 120])),
            "elev_weight_boost": float(cfg["train"].get("elev_weight_boost", 1.5)),
            "channel_weights": list(cfg["train"].get("channel_weights") or [2.0, 0.5, 0.5]),
            "ckpt_select": str(cfg["train"].get("ckpt_select", "composite")),
            "composite": {
                "primary": "lead_mean",
                "constraint": "val_120h_t2m_rmse <= raw",
                "raw_val_120h_t2m_rmse": raw_120,
                "reject_if_constraint_fails": True,
            },
        },
        "model_spec": cfg.get("model"),
        "overwrite_forbidden_ok": True,
        "era5_pid_611595": "untouched",
        "freeze_before": freeze_before,
        "honesty": cfg.get("notes", {}).get(
            "honesty",
            (
                "Tier-A v1.1 smaller/regularized residual on frozen FCN3 crop → ERA5_interim. "
                "Not v1-diff / not CorrDiff; not FCN3 weight FT; not MSE-only dynamics "
                "claim; not G1/IMDAA."
            ),
        ),
    }

    if args.dry_run:
        pairs = expand(cfg["data"]["pairs_dir"])
        elev = expand(cfg["data"]["elevation_mask"])
        report.update(
            {
                "mode": "dry_run",
                "status": "dry_run_ok" if pairs.is_dir() and elev.is_file() else "dry_run_blocked",
                "paths": {
                    "pairs_dir": str(pairs),
                    "elevation_mask": str(elev),
                    "output_dir": str(out_dir),
                    "g0_verifying": str(g0p),
                    "holdout_metrics": str(holdp),
                },
                "n_ids": {k: len(v) for k, v in splits.items()},
                "w_120": wmap[120],
                "freeze_after": freeze_snapshot(),
            }
        )
        out_json = out_dir / f"{stem}_dry_run.json"
        out_json.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {"ok": report["status"] == "dry_run_ok", "wrote": str(out_json), "beat_this": bars, "w_120": wmap[120]},
                indent=2,
            )
        )
        return 0 if report["status"] == "dry_run_ok" else 2

    import random

    import numpy as np
    import torch

    seed = int(args.seed if args.seed is not None else cfg["train"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if args.eval_only:
        epochs = 0
        want_gpu = bool(args.allow_gpu)
        mode = "eval_only"
    elif args.smoke:
        epochs = int(args.epochs or cfg["train"]["epochs_smoke"])
        want_gpu = False
        mode = "smoke"
    elif args.train:
        epochs = int(args.epochs or cfg["train"]["epochs_full"])
        want_gpu = bool(args.allow_gpu)
        mode = "train"
    else:
        print("Specify --dry-run, --smoke, --train, or --eval-only", file=sys.stderr)
        return 2

    if want_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    t0 = time.time()
    loaders = build_loaders(
        cfg, splits, use_members_train=(mode != "eval_only")
    )
    model = build_model(cfg).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(
        f"Tier-A {cfg.get('variant', '?')} cut=v1.1-reg diffusion=NO "
        f"mode={mode} device={device} epochs={epochs} "
        f"loss={loss_mode(cfg)} w_120={wmap[120]} lead_weights={wmap} "
        f"n_train={loaders['n_train']} n_val={loaders['n_val']} n_test={loaders['n_test']} "
        f"params={nparams}",
        flush=True,
    )

    ckpt_path = None
    train_info: dict[str, Any]
    if mode == "eval_only":
        ckpt_path = expand(
            args.ckpt
            or "~/fourcastnet/runs/phase0/tier_a/v1_1/best_residual.pt"
        )
        if not ckpt_path.is_file():
            print(f"ERROR: frozen ckpt missing: {ckpt_path}", file=sys.stderr)
            return 2
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt:
            state = ckpt["model"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
        model.load_state_dict(state)
        model.eval()
        train_info = {
            "best_val_t2m_rmse": None,
            "best_select_score": None,
            "best_eligible_epoch": None,
            "n_eligible_saves": 0,
            "select_by": "frozen_ckpt_zero_shot",
            "lead_weights": {str(k): v for k, v in wmap.items()},
            "w_120": wmap[120],
            "reload_kind": "eval_only_frozen",
            "stopped_epoch": None,
            "early_stop_patience": None,
            "best_ckpt": str(ckpt_path),
            "history": [],
        }
        print(f"eval-only loaded frozen ckpt: {ckpt_path}", flush=True)
    else:
        train_info = train_loop(model, loaders, cfg, device, epochs, out_dir)

    val_m = eval_split(model, loaders["val"], device, cfg["variables_out"])
    test_m = eval_split(model, loaders["test"], device, cfg["variables_out"])
    train_m = eval_split(model, loaders["train"], device, cfg["variables_out"])
    gates = gate_pass_v1(
        val_m["rmse_tier_a"],
        test_m["rmse_tier_a"],
        val_120h_rmse(val_m),
        raw_120,
        bars,
    )
    elapsed = time.time() - t0
    freeze_after = freeze_snapshot()
    freeze_ok = all(
        freeze_before[k]["md5"] == freeze_after[k]["md5"] for k in freeze_before
    )

    expand_tier0 = None
    expand_holdp = expand(
        cfg.get("data", {}).get(
            "holdout_metrics",
            "~/fourcastnet/runs/phase0/tier0_holdout_expand/tier0_holdout_expand_metrics.json",
        )
    )
    if expand_holdp.is_file():
        try:
            expand_tier0 = {
                "path": str(expand_holdp),
                "summary": _tier0_summary(json.loads(expand_holdp.read_text())),
            }
        except Exception as e:
            expand_tier0 = {"path": str(expand_holdp), "error": str(e)}

    status = (
        "eval_only_ok"
        if mode == "eval_only"
        else ("smoke_ok" if mode == "smoke" else "train_ok")
    )
    report.update(
        {
            "mode": mode,
            "status": status,
            "zero_shot": bool(mode == "eval_only"),
            "holdout_expand": ("v1" if (mode == "eval_only" or uses_expand_bars(cfg)) else None),
            "provisional_years": True,
            "ckpt": str(ckpt_path) if mode == "eval_only" else train_info.get("best_ckpt"),
            "device": str(device),
            "epochs": epochs,
            "seed": seed,
            "n_params": nparams,
            "n_samples": {
                "train": loaders["n_train"],
                "val": loaders["n_val"],
                "test": loaders["n_test"],
            },
            "train_info": {
                "best_val_t2m_rmse": train_info["best_val_t2m_rmse"],
                "best_select_score": train_info.get("best_select_score"),
                "best_eligible_epoch": train_info.get("best_eligible_epoch"),
                "n_eligible_saves": train_info.get("n_eligible_saves"),
                "select_by": train_info.get("select_by"),
                "lead_weights": train_info.get("lead_weights"),
                "w_120": train_info.get("w_120"),
                "reload_kind": train_info.get("reload_kind"),
                "stopped_epoch": train_info.get("stopped_epoch"),
                "early_stop_patience": train_info.get("early_stop_patience"),
                "best_ckpt": train_info["best_ckpt"],
                "history_tail": train_info["history"][-8:],
            },
            "metrics": {
                "train_self": train_m,
                "val": val_m,
                "test": test_m,
            },
            "lead_120h": {
                "val": lead120_delta(val_m),
                "test": lead120_delta(test_m),
                "raw_val_reference": raw_120,
                "rule": "val +120h t2m RMSE <= raw (hard); v0/v0.1 grandfathered",
            },
            "compare_v0": maybe_load_compare("v0", expand("~/fourcastnet/runs/phase0/tier_a/v0/tier_a_v0_results.json")),
            "compare_v0_1": maybe_load_compare("v0_1", expand("~/fourcastnet/runs/phase0/tier_a/v0_1/tier_a_v0_1_results.json")),
            "compare_v1": maybe_load_compare("v1", expand("~/fourcastnet/runs/phase0/tier_a/v1/tier_a_v1_results.json")),
            "gates": gates,
            "thin_beat_this_bars": thin_beat_this_echo(),
            "expand_tier0_bars": expand_tier0,
            "elapsed_s": elapsed,
            "freeze_after": freeze_after,
            "freeze_md5_unchanged": freeze_ok,
            "artifacts": {
                "results_json": str(out_dir / f"{stem}_results.json"),
                "best_ckpt": train_info["best_ckpt"],
                "metrics_csv": str(out_dir / f"{stem}_metrics.csv"),
            },
            "not_skill_claims": [
                "not G1 / not IMDAA",
                "not FCN3 weight FT",
                "not v1-diff / not CorrDiff",
                "provisional_years=true",
                "claim_level=interim_era5",
                "g1_claimable=false",
            ],
        }
    )

    out_json = out_dir / f"{stem}_results.json"
    out_json.write_text(json.dumps(report, indent=2) + "\n")

    rows = []
    for split, m in [("train_self", train_m), ("val", val_m), ("test", test_m)]:
        rows.append(
            {
                "split": split,
                "variable": "t2m",
                "scope": "pooled",
                "rmse_raw": m["rmse_raw"],
                "rmse_tier_a": m["rmse_tier_a"],
                "n": m["n"],
                "beat_val_bar": bars["val_t2m_pooled_rmse_lin_strict_lt"],
                "beat_test_bar": bars["test_t2m_pooled_rmse_lin_strict_lt"],
                "raw_val_120h_bar": raw_120,
            }
        )
        rows.append(
            {
                "split": split,
                "variable": "t2m",
                "scope": "lead_mean",
                "rmse_raw": None,
                "rmse_tier_a": m.get("rmse_tier_a_lead_mean"),
                "n": m["n"],
                "beat_val_bar": bars["val_t2m_pooled_rmse_lin_strict_lt"],
                "beat_test_bar": bars["test_t2m_pooled_rmse_lin_strict_lt"],
                "raw_val_120h_bar": raw_120,
            }
        )
        for lh, d in m["per_lead"].items():
            rows.append(
                {
                    "split": split,
                    "variable": "t2m",
                    "scope": f"lead_{lh}h",
                    "rmse_raw": d["rmse_raw"],
                    "rmse_tier_a": d["rmse_tier_a"],
                    "n": d["n"],
                    "beat_val_bar": bars["val_t2m_pooled_rmse_lin_strict_lt"],
                    "beat_test_bar": bars["test_t2m_pooled_rmse_lin_strict_lt"],
                    "raw_val_120h_bar": raw_120,
                }
            )
    write_csv(out_dir / f"{stem}_metrics.csv", rows)

    summary = {
        "ok": True,
        "mode": mode,
        "variant": cfg.get("variant"),
        "cut": "v1.1-reg",
        "diffusion": False,
        "wrote": str(out_json),
        "val_t2m_rmse_tier_a": val_m["rmse_tier_a"],
        "test_t2m_rmse_tier_a": test_m["rmse_tier_a"],
        "val_lead_mean": val_m.get("rmse_tier_a_lead_mean"),
        "val_120h": (val_m.get("per_lead") or {}).get("120"),
        "test_120h": (test_m.get("per_lead") or {}).get("120"),
        "w_120": wmap[120],
        "gates": gates,
        "beat_this": bars,
        "n_params": nparams,
        "elapsed_s": elapsed,
        "device": str(device),
        "freeze_md5_unchanged": freeze_ok,
    }
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
