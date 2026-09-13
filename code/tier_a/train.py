#!/usr/bin/env python3
"""Tier-A residual train/eval — frozen FCN3 crop + tiny residual UNet.

RRCA-FD Plan A. Supports v0 (elev-weighted pooled) and v0.1 (lead-balanced /
equal-weight multi-lead loss). Echoes frozen beat-this bars in every result JSON.
Does NOT overwrite g0_verifying_results.json, tier0_holdout, or frozen tier_a/v0.
Does NOT touch ERA5 PID 611595 / FCN3 weights.

Usage (from ~/fourcastnet):
  ~/fcn3-venv/bin/python code/tier_a/train.py --config configs/tier_a_v0.yaml --dry-run
  ~/fcn3-venv/bin/python code/tier_a/train.py --config configs/tier_a_v0_1.yaml --train --epochs 300
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # ~/fourcastnet
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
    """Filename stem for results/metrics (variant-aware; default tier_a_v0)."""
    return str(cfg.get("artifact_stem") or "tier_a_v0")


def loss_mode(cfg: dict) -> str:
    return str(cfg.get("train", {}).get("loss", "mse_residual_elev_weighted"))


def lead_balance_enabled(cfg: dict) -> bool:
    """True for v0.1 equal-weight multi-lead loss."""
    t = cfg.get("train", {})
    if "lead_balance" in t:
        return bool(t["lead_balance"])
    return loss_mode(cfg) in {
        "mse_residual_elev_weighted_lead_balanced",
        "mse_lead_balanced",
        "multi_lead_equal",
    }


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
    # keep legacy key if present for v0 compatibility
    if "tier_a_v0_status" in g0 and "tier_a_status" not in g0:
        g0 = dict(g0)
        g0["tier_a_status"] = g0["tier_a_v0_status"]
    return {
        "val_t2m_pooled_rmse_lin_strict_lt": float(bt.get("val_t2m_pooled_rmse_lin_strict_lt", 1.948661)),
        "val_headline_lt": float(bt.get("val_headline_lt", 1.949)),
        "test_t2m_pooled_rmse_lin_strict_lt": float(bt.get("test_t2m_pooled_rmse_lin_strict_lt", 1.850338)),
        "source": bt.get("source", "runs/phase0/tier0_holdout/tier0_holdout_metrics.json"),
        "call": bt.get("call", "docs/research/TIER0_HOLDOUT_CALL.md"),
        "note": bt.get("note", "val-only win = FAIL"),
        "g0_adapter": g0,
        "frozen_exact": {
            "val_rmse_after_lin": 1.9486610005712885,
            "test_rmse_after_lin": 1.8503382009036373,
        },
    }


def elev_weights(bin_id, boost: float = 1.5):
    import torch

    # bins 0..3; boost higher elevation
    w = torch.ones_like(bin_id, dtype=torch.float32)
    w = torch.where(bin_id >= 2, w * boost, w)
    w = torch.where(bin_id >= 3, w * boost, w)
    return w


def _elev_weighted_mse(pred_res, true_res, bin_id, boost: float):
    """Elev-weighted MSE over a (sub)batch — channels broadcast."""
    import torch

    w = elev_weights(bin_id, boost)  # (B,H,W)
    err = (pred_res - true_res) ** 2
    w3 = w.unsqueeze(1)
    return (err * w3).sum() / (w3.expand_as(err).sum().clamp_min(1.0))


def residual_loss(
    pred_res,
    true_res,
    bin_id,
    boost: float,
    lead_h=None,
    lead_balance: bool = False,
    leads_h: list[int] | None = None,
):
    """Elev-weighted residual MSE; optionally equal-weight across leads.

    lead_balance=True (v0.1): compute elev-weighted MSE *per lead present in the
    batch*, then take the unweighted mean of those lead losses so +24/+72/+120h
    each contribute equally even if a batch is short-lead heavy. Missing leads
    in a given batch are skipped (renormalize over present leads).
    """
    import torch

    if not lead_balance or lead_h is None:
        return _elev_weighted_mse(pred_res, true_res, bin_id, boost)

    if not torch.is_tensor(lead_h):
        lead_h = torch.as_tensor(lead_h, device=pred_res.device)
    else:
        lead_h = lead_h.to(pred_res.device)

    ref = list(leads_h) if leads_h else sorted({int(x) for x in lead_h.detach().cpu().tolist()})
    losses = []
    for lh in ref:
        mask = lead_h == int(lh)
        if int(mask.sum().item()) == 0:
            continue
        losses.append(_elev_weighted_mse(pred_res[mask], true_res[mask], bin_id[mask], boost))
    if not losses:
        return _elev_weighted_mse(pred_res, true_res, bin_id, boost)
    return torch.stack(losses).mean()


def mean_per_lead_rmse(metrics: dict) -> float | None:
    """Lead-balanced selection score = mean of per-lead t2m RMSE."""
    pl = metrics.get("per_lead") or {}
    vals = [float(v["rmse_tier_a"]) for v in pl.values() if v.get("rmse_tier_a") is not None]
    if not vals:
        return metrics.get("rmse_tier_a")
    return float(sum(vals) / len(vals))


def eval_split(model, loader, device, variables_out: list[str]) -> dict:
    import numpy as np
    import torch

    model.eval()
    # accumulate corrected vs truth for t2m (index 0)
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
        # pooled over all vars for logging; headline uses t2m=ch0
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


def gate_pass(val_rmse: float | None, test_rmse: float | None, bars: dict) -> dict:
    vbar = bars["val_t2m_pooled_rmse_lin_strict_lt"]
    tbar = bars["test_t2m_pooled_rmse_lin_strict_lt"]
    val_ok = val_rmse is not None and val_rmse < vbar
    test_ok = test_rmse is not None and test_rmse < tbar
    return {
        "val_pass": bool(val_ok),
        "test_pass": bool(test_ok),
        "tier_a_interim_pass": bool(val_ok and test_ok),
        "val_only_win_is_fail": True,
        "note": (
            "interim PASS requires val AND test strict beat-this; "
            "claim_level remains interim_era5; g1_claimable=false"
        ),
    }


def assert_no_overwrite(out_dir: Path, cfg: dict) -> None:
    forbidden = [
        expand("~/fourcastnet/runs/phase0/g0/g0_verifying_results.json"),
        expand("~/fourcastnet/runs/phase0/tier0_holdout/tier0_holdout_metrics.json"),
    ]
    # freeze v0 when writing elsewhere (v0.1 must not overwrite v0)
    freeze_v0 = bool(cfg.get("locks", {}).get("freeze_tier_a_v0", False))
    v0_dir = expand("~/fourcastnet/runs/phase0/tier_a/v0")
    if freeze_v0 and out_dir.resolve() == v0_dir.resolve():
        raise RuntimeError("refusing to write into frozen runs/phase0/tier_a/v0/")
    for p in forbidden:
        if out_dir.resolve() == p.resolve() or str(p.resolve()).startswith(str(out_dir.resolve())):
            raise RuntimeError(f"refusing to write into forbidden path {p}")
    for p in forbidden:
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
    do_lb = lead_balance_enabled(cfg)
    leads_ref = [int(x) for x in cfg.get("leads_h", [24, 72, 120])]
    # selection: lead-mean when balancing; else pooled val RMSE (v0 protocol)
    select_by = str(cfg["train"].get("ckpt_select", "lead_mean" if do_lb else "pooled"))
    hist = []
    best_score = float("inf")
    best_val_pooled = float("inf")
    best_path = out_dir / "best_residual.pt"

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
                lead_balance=do_lb,
                leads_h=leads_ref,
            )
            loss.backward()
            opt.step()
            total += float(loss.item())
            nbat += 1
        tr_loss = total / max(nbat, 1)
        val_m = eval_split(model, loaders["val"], device, cfg["variables_out"])
        pooled = val_m["rmse_tier_a"]
        lead_mean = val_m.get("rmse_tier_a_lead_mean")
        score = lead_mean if select_by == "lead_mean" else pooled
        hist.append(
            {
                "epoch": ep,
                "train_loss": tr_loss,
                "val_t2m_rmse": pooled,
                "val_t2m_rmse_lead_mean": lead_mean,
                "select_score": score,
            }
        )
        if score is not None and score < best_score:
            best_score = score
            best_val_pooled = pooled if pooled is not None else best_val_pooled
            torch.save(
                {
                    "epoch": ep,
                    "model": model.state_dict(),
                    "val_t2m_rmse": pooled,
                    "val_t2m_rmse_lead_mean": lead_mean,
                    "select_score": best_score,
                    "select_by": select_by,
                    "cfg_schema": cfg.get("schema"),
                    "loss": loss_mode(cfg),
                    "lead_balance": do_lb,
                },
                best_path,
            )
        if ep == 1 or ep % 5 == 0 or ep == epochs:
            pl120 = (val_m.get("per_lead") or {}).get("120", {}).get("rmse_tier_a")
            print(
                f"[ep {ep:03d}/{epochs}] train_loss={tr_loss:.5f} "
                f"val_pooled={pooled:.5f} val_lead_mean={lead_mean:.5f} "
                f"val_120h={pl120 if pl120 is not None else float('nan'):.5f} "
                f"best_sel={best_score:.5f} ({select_by})",
                flush=True,
            )

    # reload best
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
    return {
        "history": hist,
        "best_val_t2m_rmse": best_val_pooled,
        "best_select_score": best_score,
        "select_by": select_by,
        "lead_balance": do_lb,
        "best_ckpt": str(best_path),
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


def maybe_load_v0_compare() -> dict | None:
    p = expand("~/fourcastnet/runs/phase0/tier_a/v0/tier_a_v0_results.json")
    if not p.is_file():
        return None
    try:
        v0 = json.loads(p.read_text())
    except Exception:
        return None
    m = v0.get("metrics", {})
    out = {"path": str(p), "variant": v0.get("variant"), "gates": v0.get("gates")}
    for split in ("val", "test"):
        sm = m.get(split) or {}
        out[split] = {
            "rmse_tier_a": sm.get("rmse_tier_a"),
            "per_lead": {
                k: {"rmse_tier_a": (v or {}).get("rmse_tier_a")}
                for k, v in (sm.get("per_lead") or {}).items()
            },
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Tier-A residual train/eval (v0 / v0.1)")
    ap.add_argument("--config", default="configs/tier_a_v0.yaml")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="short CPU smoke train")
    ap.add_argument("--train", action="store_true", help="longer train")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--allow-gpu", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (ROOT / cfg_path).resolve()
    cfg = load_config(cfg_path)
    bars = beat_this_echo(cfg)
    stem = artifact_stem(cfg)
    out_dir = expand(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    assert_no_overwrite(out_dir, cfg)

    # verify forbidden artifacts untouched policy (existence check only)
    g0p = expand(cfg["data"]["g0_verifying"])
    holdp = expand(cfg["data"]["holdout_metrics"])
    if not g0p.is_file() or not holdp.is_file():
        print(f"WARN: missing baseline artifact g0={g0p.exists()} holdout={holdp.exists()}")

    splits_cfg = cfg["year_split"]
    # prefer manifest splits; fall back to config ids
    man = expand(cfg["data"]["ic_manifest"])
    if man.is_file():
        splits = load_split_ids(man)
    else:
        splits = {
            "train": list(splits_cfg["train_ids"]),
            "val": list(splits_cfg["val_ids"]),
            "test": list(splits_cfg["test_ids"]),
        }

    do_lb = lead_balance_enabled(cfg)
    report: dict[str, Any] = {
        "schema": cfg.get("schema", "tier_a_v0_results/v1"),
        "created_utc": utc_now(),
        "script": "code/tier_a/train.py",
        "config": str(cfg_path),
        "method": cfg.get("method"),
        "tier": "A",
        "variant": cfg.get("variant"),
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
            "lead_balance": do_lb,
            "description": (
                "Equal-weight multi-lead: elev-weighted MSE computed per lead "
                "(+24/+72/+120h), then unweighted mean across leads present in "
                "the batch. Prevents short-lead-heavy pooled optimization from "
                "regressing +120h (Leonard TIER_A_V0_CALL WARN)."
                if do_lb
                else "Elev-weighted pooled residual MSE (v0)."
            ),
            "leads_h": list(cfg.get("leads_h", [24, 72, 120])),
            "elev_weight_boost": float(cfg["train"].get("elev_weight_boost", 1.5)),
            "ckpt_select": str(
                cfg["train"].get("ckpt_select", "lead_mean" if do_lb else "pooled")
            ),
        },
        "overwrite_forbidden_ok": True,
        "era5_pid_611595": "untouched",
        "honesty": cfg.get("notes", {}).get(
            "honesty",
            (
                "Tier-A diagnostic residual on frozen FCN3 crop → ERA5_interim. "
                "Not CorrDiff yet; not FCN3 weight FT; not MSE-only dynamics claim; "
                "not G1/IMDAA."
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
            }
        )
        out_json = out_dir / f"{stem}_dry_run.json"
        out_json.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {"ok": report["status"] == "dry_run_ok", "wrote": str(out_json), "beat_this": bars},
                indent=2,
            )
        )
        return 0 if report["status"] == "dry_run_ok" else 2

    import torch
    import random
    import numpy as np

    seed = int(args.seed if args.seed is not None else cfg["train"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if args.smoke:
        epochs = int(args.epochs or cfg["train"]["epochs_smoke"])
        want_gpu = False
        mode = "smoke"
    elif args.train:
        epochs = int(args.epochs or cfg["train"]["epochs_full"])
        want_gpu = bool(args.allow_gpu)
        mode = "train"
    else:
        print("Specify --dry-run, --smoke, or --train", file=sys.stderr)
        return 2

    if want_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    t0 = time.time()
    loaders = build_loaders(cfg, splits, use_members_train=True)
    model = build_model(cfg).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(
        f"Tier-A {cfg.get('variant', '?')} mode={mode} device={device} epochs={epochs} "
        f"loss={loss_mode(cfg)} lead_balance={do_lb} "
        f"n_train={loaders['n_train']} n_val={loaders['n_val']} n_test={loaders['n_test']} "
        f"params={nparams}",
        flush=True,
    )

    train_info = train_loop(model, loaders, cfg, device, epochs, out_dir)
    val_m = eval_split(model, loaders["val"], device, cfg["variables_out"])
    test_m = eval_split(model, loaders["test"], device, cfg["variables_out"])
    train_m = eval_split(model, loaders["train"], device, cfg["variables_out"])
    gates = gate_pass(val_m["rmse_tier_a"], test_m["rmse_tier_a"], bars)

    # +120h non-regression vs raw (informational; not a hard gate)
    def lead120_delta(m: dict) -> dict:
        pl = (m.get("per_lead") or {}).get("120") or {}
        raw = pl.get("rmse_raw")
        corr = pl.get("rmse_tier_a")
        return {
            "rmse_raw": raw,
            "rmse_tier_a": corr,
            "delta_vs_raw": (None if raw is None or corr is None else float(corr) - float(raw)),
            "improves_vs_raw": (
                None if raw is None or corr is None else bool(corr < raw)
            ),
        }

    elapsed = time.time() - t0
    v0_cmp = maybe_load_v0_compare() if do_lb else None
    report.update(
        {
            "mode": mode,
            "status": "smoke_ok" if mode == "smoke" else "train_ok",
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
                "select_by": train_info.get("select_by"),
                "lead_balance": train_info.get("lead_balance"),
                "best_ckpt": train_info["best_ckpt"],
                "history_tail": train_info["history"][-5:],
            },
            "metrics": {
                "train_self": train_m,
                "val": val_m,
                "test": test_m,
            },
            "lead_120h": {
                "val": lead120_delta(val_m),
                "test": lead120_delta(test_m),
            },
            "compare_v0": v0_cmp,
            "gates": gates,
            "elapsed_s": elapsed,
            "artifacts": {
                "results_json": str(out_dir / f"{stem}_results.json"),
                "best_ckpt": train_info["best_ckpt"],
                "metrics_csv": str(out_dir / f"{stem}_metrics.csv"),
            },
            "not_skill_claims": [
                "not G1 / not IMDAA",
                "not FCN3 weight FT",
                "not CorrDiff (tiny UNet residual only)",
                "provisional_years=true",
                "not lead-uniform unless lead_balance documented + measured",
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
                }
            )
    write_csv(out_dir / f"{stem}_metrics.csv", rows)

    summary = {
        "ok": True,
        "mode": mode,
        "variant": cfg.get("variant"),
        "wrote": str(out_json),
        "val_t2m_rmse_tier_a": val_m["rmse_tier_a"],
        "test_t2m_rmse_tier_a": test_m["rmse_tier_a"],
        "val_lead_mean": val_m.get("rmse_tier_a_lead_mean"),
        "val_120h": (val_m.get("per_lead") or {}).get("120"),
        "test_120h": (test_m.get("per_lead") or {}).get("120"),
        "gates": gates,
        "beat_this": bars,
        "loss": report["loss"],
        "elapsed_s": elapsed,
        "device": str(device),
    }
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
