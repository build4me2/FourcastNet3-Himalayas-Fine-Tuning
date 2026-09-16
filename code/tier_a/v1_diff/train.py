#!/usr/bin/env python3
"""Tier-A v1-diff CorrDiff-lite — Leonard TIER_A_V1_DIFF_RECIPE.md.

Custom 2D EDM residual diffuser. FCN3 FROZEN. Living residual FROZEN as mean
path / conditioner. Target: leftover = ERA5 − (FCN3 + frozen residual mean).
Decode: mean path + ens-mean K=4. Writes ONLY under runs/phase0/tier_a/v1_diff/.
Do NOT overwrite v1_2b_thick2_train/ or CDS PID 611595.

Usage (from ~/fourcastnet):
  ~/fcn3-venv/bin/python code/tier_a/v1_diff/train.py \\
      --config configs/tier_a_v1_diff.yaml --dry-run
  PYTHONUNBUFFERED=1 ~/fcn3-venv/bin/python -u code/tier_a/v1_diff/train.py \\
      --config configs/tier_a_v1_diff.yaml --train --allow-gpu --epochs 200
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
ROOT = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    import yaml
except ImportError:
    yaml = None

from dataset import (  # noqa: E402
    build_loaders,
    expand,
    load_frozen_residual,
    load_split_ids,
)
from model import (  # noqa: E402
    build_model,
    count_params,
    edm_loss_weight,
    ens_mean_leftover,
    one_step_mean_leftover,
    sample_sigma,
)

import torch  # noqa: E402  — needed for @torch.no_grad on eval_split


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config(path: Path) -> dict:
    if yaml is None:
        raise RuntimeError("PyYAML required")
    return yaml.safe_load(path.read_text())


def artifact_stem(cfg: dict) -> str:
    return str(cfg.get("artifact_stem") or "tier_a_v1_diff")


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
        "living_best_residual": expand(
            "~/fourcastnet/runs/phase0/tier_a/v1_2b_thick2_train/best_residual.pt"
        ),
        "living_results": expand(
            "~/fourcastnet/runs/phase0/tier_a/v1_2b_thick2_train/tier_a_v1_2b_thick2_train_results.json"
        ),
        "v1_2b_best": expand("~/fourcastnet/runs/phase0/tier_a/v1_2b/best_residual.pt"),
        "v1_2b_thick2_results": expand(
            "~/fourcastnet/runs/phase0/tier_a/v1_2b_thick2/tier_a_v1_2b_thick2_results.json"
        ),
        "g0_verifying": expand("~/fourcastnet/runs/phase0/g0/g0_verifying_results.json"),
        "tier0_holdout_expand_v2": expand(
            "~/fourcastnet/runs/phase0/tier0_holdout_expand_v2/tier0_holdout_expand_v2_metrics.json"
        ),
    }
    return {
        k: {
            "path": str(p),
            "md5": file_md5(p),
            "mtime": (p.stat().st_mtime if p.exists() else None),
        }
        for k, p in paths.items()
    }


def assert_no_overwrite(out_dir: Path) -> None:
    out_r = out_dir.resolve()
    forbidden = [
        expand("~/fourcastnet/runs/phase0/g0"),
        expand("~/fourcastnet/runs/phase0/tier0_holdout"),
        expand("~/fourcastnet/runs/phase0/tier0_holdout_expand"),
        expand("~/fourcastnet/runs/phase0/tier0_holdout_expand_v2"),
        expand("~/fourcastnet/runs/phase0/tier_a/v0"),
        expand("~/fourcastnet/runs/phase0/tier_a/v0_1"),
        expand("~/fourcastnet/runs/phase0/tier_a/v1"),
        expand("~/fourcastnet/runs/phase0/tier_a/v1_1"),
        expand("~/fourcastnet/runs/phase0/tier_a/v1_1_expand"),
        expand("~/fourcastnet/runs/phase0/tier_a/v1_2"),
        expand("~/fourcastnet/runs/phase0/tier_a/v1_2b"),
        expand("~/fourcastnet/runs/phase0/tier_a/v1_2b_thick2"),
        expand("~/fourcastnet/runs/phase0/tier_a/v1_2b_thick2_train"),
    ]
    for p in forbidden:
        pr = p.resolve()
        if out_r == pr or str(out_r).startswith(str(pr) + "/"):
            raise RuntimeError(f"refusing to write into frozen path {p}")
    allowed = expand("~/fourcastnet/runs/phase0/tier_a/v1_diff").resolve()
    if not (out_r == allowed or str(out_r).startswith(str(allowed) + "/")):
        raise RuntimeError(f"v1-diff must write under runs/phase0/tier_a/v1_diff/ (got {out_r})")


def cds_alive(pid: int = 611595) -> dict:
    import os

    alive = False
    cmd = None
    try:
        os.kill(pid, 0)
        alive = True
        cmd_path = Path(f"/proc/{pid}/cmdline")
        if cmd_path.exists():
            cmd = cmd_path.read_bytes().replace(b"\x00", b" ").decode(errors="replace").strip()
    except OSError:
        alive = False
    return {"pid": pid, "alive": alive, "cmd": cmd, "policy": "DO_NOT_KILL"}


def beat_this_echo(cfg: dict) -> dict:
    bt = cfg.get("beat_this", {})
    living = bt.get("living_residual") or {}
    return {
        "val_t2m_pooled_rmse_lin_strict_lt": float(
            bt.get("val_t2m_pooled_rmse_lin_strict_lt", 1.988588)
        ),
        "test_t2m_pooled_rmse_lin_strict_lt": float(
            bt.get("test_t2m_pooled_rmse_lin_strict_lt", 1.918383)
        ),
        "val_120h_t2m_rmse_raw_le": float(
            bt.get("val_120h_t2m_rmse_raw_le", 2.178841818082155)
        ),
        "bars_set": str(bt.get("bars_set") or cfg.get("bars_set") or "thick2"),
        "living_residual": {
            "path": living.get("path", "runs/phase0/tier_a/v1_2b_thick2_train/"),
            "ckpt": living.get(
                "ckpt",
                "runs/phase0/tier_a/v1_2b_thick2_train/best_residual.pt",
            ),
            "best_val_pooled": float(living.get("best_val_pooled", 1.759883)),
            "best_test_pooled": float(living.get("best_test_pooled", 1.7797)),
            "vs_living_val_strict_lt": float(living.get("vs_living_val_strict_lt", 1.759883)),
            "vs_living_test_le": float(living.get("vs_living_test_le", 1.7897)),
            "prefer_test_strict_lt": float(living.get("prefer_test_strict_lt", 1.7797)),
        },
        "recipe": bt.get("recipe", "docs/research/TIER_A_V1_DIFF_RECIPE.md"),
        "note": bt.get("note"),
    }


def channel_weights_tensor(n_ch: int, cfg: dict, device, dtype):
    import torch

    cw = cfg.get("train", {}).get("channel_weights") or [2.0, 0.5, 0.5]
    vals = [float(cw[i]) if i < len(cw) else 1.0 for i in range(n_ch)]
    return torch.tensor(vals, device=device, dtype=dtype).view(1, n_ch, 1, 1)


def elev_weights(bin_id, boost: float = 1.5):
    import torch

    w = torch.ones_like(bin_id, dtype=torch.float32)
    w = torch.where(bin_id >= 2, w * boost, w)
    w = torch.where(bin_id >= 3, w * boost, w)
    return w


def edm_train_step(model, batch, cfg, device, amp: bool = False):
    import torch

    cond = batch["cond"].to(device)
    leftover = batch["leftover"].to(device)
    bin_id = batch["bin_id"].to(device)
    B = leftover.shape[0]
    mcfg = cfg.get("model", {})
    sigma = sample_sigma(
        B,
        device,
        leftover.dtype,
        sigma_min=float(mcfg.get("sigma_min", 0.002)),
        sigma_max=float(mcfg.get("sigma_max", 80.0)),
        rho=float(mcfg.get("rho", 7.0)),
    )
    noise = torch.randn_like(leftover)
    x_noisy = leftover + sigma[:, None, None, None] * noise

    use_bf16 = bool(amp and device.type == "cuda")
    from contextlib import nullcontext
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()
    with ctx:
        denoised = model(x_noisy, sigma, cond)
        w = edm_loss_weight(sigma, float(mcfg.get("sigma_data", 1.0)))
        err = (denoised.float() - leftover.float()) ** 2
        err = err * channel_weights_tensor(err.shape[1], cfg, device, err.dtype)
        ew = elev_weights(bin_id, float(cfg["train"].get("elev_weight_boost", 1.5))).to(device)
        # lead weight
        lead_h = batch["lead_h"]
        if not torch.is_tensor(lead_h):
            lead_h = torch.as_tensor(lead_h, device=device)
        else:
            lead_h = lead_h.to(device)
        lw_map = cfg.get("train", {}).get("lead_weights") or {24: 1.0, 72: 1.0, 120: 2.0}
        lw = torch.ones(B, device=device, dtype=err.dtype)
        for lh, wv in lw_map.items():
            lw = torch.where(lead_h == int(lh), lw * float(wv), lw)
        per = (err * ew.unsqueeze(1)).mean(dim=(1, 2, 3))
        loss = (w * lw * per).mean()
    return loss


@torch.no_grad()
def eval_split(model, loader, cfg, device) -> dict:
    import numpy as np
    import torch

    model.eval()
    K = int(cfg["train"].get("decode_k", 4))
    steps = int(cfg["train"].get("decode_steps", 18))
    mcfg = cfg.get("model", {})
    sigma_min = float(mcfg.get("sigma_min", 0.002))
    sigma_max = float(mcfg.get("sigma_max", 80.0))
    rho = float(mcfg.get("rho", 7.0))

    sq_raw = 0.0  # FCN3 only
    sq_living = 0.0  # FCN3 + frozen residual
    sq_diff = 0.0  # + diffusion mean
    n = 0
    per_lead: dict[int, dict[str, float]] = {}
    per_bin: dict[int, dict[str, float]] = {}

    for batch in loader:
        cond = batch["cond"].to(device)
        pred_out = batch["pred_out"].to(device)
        mean_path = batch["mean_path"].to(device)
        truth = batch["truth"].to(device)
        bin_id = batch["bin_id"].to(device)
        leads = batch["lead_h"]

        decode_mode = str(cfg["train"].get("decode_mode", "ens_mean"))
        if decode_mode in {"one_step", "zero_noise", "one_step_mean"}:
            leftover_hat = one_step_mean_leftover(
                model, cond, sigma=float(sigma_min)
            )
        else:
            leftover_hat = ens_mean_leftover(
                model,
                cond,
                K=K,
                steps=steps,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                rho=rho,
            )
        corr = mean_path + leftover_hat

        for b in range(cond.shape[0]):
            lh = int(leads[b].item() if hasattr(leads[b], "item") else leads[b])
            raw_e = (pred_out[b, 0] - truth[b, 0]).float()
            liv_e = (mean_path[b, 0] - truth[b, 0]).float()
            dif_e = (corr[b, 0] - truth[b, 0]).float()
            m = bin_id[b] >= 0
            sq_raw += float((raw_e[m] ** 2).sum().item())
            sq_living += float((liv_e[m] ** 2).sum().item())
            sq_diff += float((dif_e[m] ** 2).sum().item())
            n += int(m.sum().item())

            if lh not in per_lead:
                per_lead[lh] = {"sq_raw": 0.0, "sq_living": 0.0, "sq_diff": 0.0, "n": 0}
            per_lead[lh]["sq_raw"] += float((raw_e[m] ** 2).sum().item())
            per_lead[lh]["sq_living"] += float((liv_e[m] ** 2).sum().item())
            per_lead[lh]["sq_diff"] += float((dif_e[m] ** 2).sum().item())
            per_lead[lh]["n"] += int(m.sum().item())

            for bb in range(4):
                mb = bin_id[b] == bb
                if int(mb.sum().item()) == 0:
                    continue
                if bb not in per_bin:
                    per_bin[bb] = {"sq_raw": 0.0, "sq_living": 0.0, "sq_diff": 0.0, "n": 0}
                per_bin[bb]["sq_raw"] += float((raw_e[mb] ** 2).sum().item())
                per_bin[bb]["sq_living"] += float((liv_e[mb] ** 2).sum().item())
                per_bin[bb]["sq_diff"] += float((dif_e[mb] ** 2).sum().item())
                per_bin[bb]["n"] += int(mb.sum().item())

    def rmse(sq, nn):
        return float(np.sqrt(sq / nn)) if nn > 0 else None

    out = {
        "variable": "t2m",
        "n": n,
        "decode_k": K,
        "decode_steps": steps,
        "rmse_raw": rmse(sq_raw, n),
        "rmse_living_residual": rmse(sq_living, n),
        "rmse_tier_a": rmse(sq_diff, n),  # FCN3 + residual + diffusion mean
        "per_lead": {
            str(k): {
                "rmse_raw": rmse(v["sq_raw"], v["n"]),
                "rmse_living_residual": rmse(v["sq_living"], v["n"]),
                "rmse_tier_a": rmse(v["sq_diff"], v["n"]),
                "n": v["n"],
            }
            for k, v in sorted(per_lead.items())
        },
        "per_elev_bin": {
            str(k): {
                "rmse_raw": rmse(v["sq_raw"], v["n"]),
                "rmse_living_residual": rmse(v["sq_living"], v["n"]),
                "rmse_tier_a": rmse(v["sq_diff"], v["n"]),
                "n": v["n"],
            }
            for k, v in sorted(per_bin.items())
        },
    }
    pl_vals = [
        float(v["rmse_tier_a"])
        for v in out["per_lead"].values()
        if v.get("rmse_tier_a") is not None
    ]
    out["rmse_tier_a_lead_mean"] = (
        float(sum(pl_vals) / len(pl_vals)) if pl_vals else out["rmse_tier_a"]
    )
    return out


def val_120h_rmse(metrics: dict) -> float | None:
    pl = (metrics.get("per_lead") or {}).get("120") or {}
    return pl.get("rmse_tier_a")


def gate_pass(val_m, test_m, bars: dict, n_eligible: int, reload_kind: str | None) -> dict:
    vbar = bars["val_t2m_pooled_rmse_lin_strict_lt"]
    tbar = bars["test_t2m_pooled_rmse_lin_strict_lt"]
    raw_120 = bars["val_120h_t2m_rmse_raw_le"]
    living = bars["living_residual"]
    val_rmse = val_m.get("rmse_tier_a") if val_m else None
    test_rmse = test_m.get("rmse_tier_a") if test_m else None
    val_120 = val_120h_rmse(val_m) if val_m else None

    val_ok = val_rmse is not None and val_rmse < vbar
    test_ok = test_rmse is not None and test_rmse < tbar
    lead120_ok = val_120 is not None and float(val_120) <= float(raw_120) + 1e-9

    vs_val = val_rmse is not None and val_rmse < float(living["vs_living_val_strict_lt"])
    vs_test = test_rmse is not None and test_rmse <= float(living["vs_living_test_le"]) + 1e-12
    vs_test_prefer = test_rmse is not None and test_rmse < float(living["prefer_test_strict_lt"])
    improves = bool(vs_val and vs_test)

    eligible_ok = int(n_eligible or 0) >= 1 and reload_kind == "composite_eligible"
    if reload_kind == "fallback_unconstrained_120h":
        eligible_ok = False

    thick2_pass = bool(val_ok and test_ok and lead120_ok and eligible_ok)
    interim = bool(thick2_pass and improves)

    return {
        "val_pass": bool(val_ok),
        "test_pass": bool(test_ok),
        "val_120h_le_raw": bool(lead120_ok),
        "n_eligible_saves": int(n_eligible or 0),
        "eligible_ckpt_ok": bool(eligible_ok),
        "reload_kind": reload_kind,
        "thick2_bars_pass": thick2_pass,
        "vs_living_val_strict_lt": bool(vs_val),
        "vs_living_test_le": bool(vs_test),
        "vs_living_test_prefer_strict": bool(vs_test_prefer),
        "diff_improves_residual": improves,
        "tier_a_interim_pass": interim,
        "val_t2m_rmse": val_rmse,
        "test_t2m_rmse": test_rmse,
        "val_120h_t2m_rmse": val_120,
        "living_val_bar": float(living["vs_living_val_strict_lt"]),
        "living_test_bar_le": float(living["vs_living_test_le"]),
        "note": (
            "If thick2 bars pass but fail vs living => FAIL as product upgrade; "
            "diff_improves_residual=false; living residual stays canonical."
        ),
    }


def train_loop(model, loaders, cfg, device, epochs: int, out_dir: Path) -> dict:
    import torch

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    accum = max(int(cfg["train"].get("grad_accum", 8)), 1)
    use_bf16 = bool(cfg["train"].get("bf16", True)) and device.type == "cuda"
    raw_120 = float(cfg["train"].get("raw_val_120h_t2m_rmse", 2.178841818082155))
    patience = int(cfg["train"].get("early_stop_patience", 40))
    hist = []
    best_eligible_score = float("inf")
    best_eligible_pooled = None
    best_eligible_ep = None
    best_uncon_120 = float("inf")
    best_path = out_dir / "best_diffusion.pt"
    uncon_path = out_dir / "best_unconstrained_120h.pt"
    last_improve_ep = 0
    n_eligible_saves = 0

    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        nbat = 0
        opt.zero_grad(set_to_none=True)
        for bi, batch in enumerate(loaders["train"]):
            loss = edm_train_step(model, batch, cfg, device, amp=use_bf16)
            (loss / accum).backward()
            if (bi + 1) % accum == 0 or (bi + 1) == len(loaders["train"]):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
            total += float(loss.item())
            nbat += 1
        tr_loss = total / max(nbat, 1)

        # eval every epoch (dataset small); K may be reduced mid-train for speed
        cfg_eval = dict(cfg)
        cfg_eval["train"] = dict(cfg["train"])
        # during train use K=1 for speed; final reload eval uses full K
        cfg_eval["train"]["decode_k"] = 1
        cfg_eval["train"]["decode_steps"] = min(10, int(cfg["train"].get("decode_steps", 18)))
        cfg_eval["train"]["decode_mode"] = "one_step"  # stable selection; final uses ens_mean K
        val_m = eval_split(model, loaders["val"], cfg_eval, device)
        pooled = val_m["rmse_tier_a"]
        lead_mean = val_m.get("rmse_tier_a_lead_mean")
        v120 = val_120h_rmse(val_m)
        score = lead_mean if lead_mean is not None else pooled
        ok = v120 is not None and float(v120) <= raw_120 + 1e-9
        saved = False

        if v120 is not None and v120 < best_uncon_120:
            best_uncon_120 = v120
            torch.save(
                {
                    "epoch": ep,
                    "model": model.state_dict(),
                    "val_t2m_rmse": pooled,
                    "val_120h": v120,
                    "kind": "unconstrained_best_120h",
                },
                uncon_path,
            )
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
                    "kind": "composite_eligible",
                    "arch": "ResidualEDM2D",
                    "diffusion_backend": "edm",
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
                f"constraint_ok={ok} saved={saved}",
                flush=True,
            )
        if ep - last_improve_ep >= patience:
            print(f"[early-stop] no eligible improve for {patience} ep", flush=True)
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
        print("WARN: no +120h-eligible ckpt; gate will FAIL vs thick-2", flush=True)

    return {
        "history": hist,
        "best_val_t2m_rmse": best_eligible_pooled,
        "best_select_score": (
            best_eligible_score if best_eligible_score < float("inf") else None
        ),
        "best_eligible_epoch": best_eligible_ep,
        "n_eligible_saves": n_eligible_saves,
        "reload_kind": used_kind,
        "stopped_epoch": hist[-1]["epoch"] if hist else 0,
        "early_stop_patience": patience,
        "best_ckpt": str(best_path) if best_path.exists() else None,
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


def main() -> int:
    # late import so dry-run path-check can work without torch if needed
    ap = argparse.ArgumentParser(description="Tier-A v1-diff CorrDiff-lite train/eval")
    ap.add_argument("--config", default="configs/tier_a_v1_diff.yaml")
    ap.add_argument("--dry-run", action="store_true", help="1 forward step; EXIT 0")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--train", action="store_true")
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
    assert_no_overwrite(out_dir)
    freeze_before = freeze_snapshot()
    cds = cds_alive(611595)

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
        "schema": cfg.get("schema", "tier_a_v1_diff_results/v1"),
        "created_utc": utc_now(),
        "script": "code/tier_a/v1_diff/train.py",
        "config": str(cfg_path),
        "method": cfg.get("method"),
        "tier": "A",
        "variant": cfg.get("variant"),
        "cut": cfg.get("cut"),
        "diffusion": True,
        "diffusion_kind": cfg.get("diffusion_kind", "edm_custom_2d"),
        "arch_choice": {
            "backend": "edm",
            "implementation": "custom_2d_ResidualEDM2D",
            "physicsnemo": False,
            "reason": "PhysicsNeMo stock UNet is 3D; Nepal 21x37 uses custom 2D EDM",
        },
        "fcn3_weights": "FROZEN",
        "living_residual_weights": "FROZEN",
        "living_residual_ckpt": str(expand(cfg["data"]["living_residual_ckpt"])),
        "target_mode": cfg.get("train", {}).get(
            "target_mode", "leftover_vs_fcn3_plus_residual"
        ),
        "claim_level": cfg["locks"]["claim_level"],
        "g1_claimable": False,
        "provisional_years": False,
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
        "model_spec": cfg.get("model"),
        "decode": {
            "formula": "FCN3 + frozen_residual_mean + diffusion_ens_mean",
            "K": int(cfg["train"].get("decode_k", 4)),
            "steps": int(cfg["train"].get("decode_steps", 18)),
        },
        "overwrite_forbidden_ok": True,
        "era5_pid_611595": cds,
        "freeze_before": freeze_before,
        "honesty": (cfg.get("notes") or {}).get("honesty"),
    }

    import random

    import numpy as np
    import torch

    seed = int(args.seed if args.seed is not None else cfg["train"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if args.dry_run:
        device = torch.device("cpu")
        frozen, fpath = load_frozen_residual(cfg, device="cpu")
        loaders = build_loaders(
            cfg, splits, frozen_model=frozen, use_members_train=False, device="cpu"
        )
        model = build_model(cfg).to(device)
        nparams = count_params(model)
        batch = next(iter(loaders["train"]))
        loss = edm_train_step(model, batch, cfg, device, amp=False)
        # one decode step (K=1, few steps) to prove sampler path
        cfg_d = dict(cfg)
        cfg_d["train"] = dict(cfg["train"])
        cfg_d["train"]["decode_k"] = 1
        cfg_d["train"]["decode_steps"] = 2
        with torch.no_grad():
            leftover_hat = ens_mean_leftover(
                model,
                batch["cond"],
                K=1,
                steps=2,
                sigma_min=float(cfg["model"].get("sigma_min", 0.002)),
                sigma_max=float(cfg["model"].get("sigma_max", 80.0)),
                rho=float(cfg["model"].get("rho", 7.0)),
            )
        assert leftover_hat.shape == batch["leftover"].shape
        freeze_after = freeze_snapshot()
        freeze_ok = all(
            freeze_before[k]["md5"] == freeze_after[k]["md5"] for k in freeze_before
        )
        report.update(
            {
                "mode": "dry_run",
                "status": "dry_run_ok",
                "device": "cpu",
                "n_params": nparams,
                "n_samples": {
                    "train": loaders["n_train"],
                    "val": loaders["n_val"],
                    "test": loaders["n_test"],
                },
                "n_ids": {k: len(v) for k, v in splits.items()},
                "forward_loss": float(loss.item()),
                "leftover_hat_shape": list(leftover_hat.shape),
                "living_residual_loaded": fpath,
                "freeze_after": freeze_after,
                "freeze_md5_unchanged": freeze_ok,
                "paths": {
                    "pairs_dir": str(expand(cfg["data"]["pairs_dir"])),
                    "output_dir": str(out_dir),
                    "living_residual_ckpt": fpath,
                },
            }
        )
        out_json = out_dir / f"{stem}_dry_run.json"
        out_json.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "ok": True,
                    "status": "dry_run_ok",
                    "wrote": str(out_json),
                    "n_params": nparams,
                    "forward_loss": float(loss.item()),
                    "arch": "ResidualEDM2D/edm",
                    "beat_this": bars,
                    "freeze_md5_unchanged": freeze_ok,
                    "cds": cds,
                },
                indent=2,
            ),
            flush=True,
        )
        return 0

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
    frozen, fpath = load_frozen_residual(cfg, device=str(device))
    loaders = build_loaders(
        cfg,
        splits,
        frozen_model=frozen,
        use_members_train=False,
        device=str(device),
    )
    model = build_model(cfg).to(device)
    nparams = count_params(model)
    print(
        f"Tier-A v1_diff arch=ResidualEDM2D/edm mode={mode} device={device} "
        f"epochs={epochs} params={nparams} n_train={loaders['n_train']} "
        f"n_val={loaders['n_val']} n_test={loaders['n_test']} "
        f"living={fpath}",
        flush=True,
    )
    if nparams > 2_500_000:
        print(f"WARN: params={nparams} exceed ~2M soft cap; consider shrink", flush=True)

    train_info = train_loop(model, loaders, cfg, device, epochs, out_dir)

    # final eval with full K
    val_m = eval_split(model, loaders["val"], cfg, device)
    test_m = eval_split(model, loaders["test"], cfg, device)
    gates = gate_pass(
        val_m,
        test_m,
        bars,
        n_eligible=int(train_info.get("n_eligible_saves") or 0),
        reload_kind=train_info.get("reload_kind"),
    )
    elapsed = time.time() - t0
    freeze_after = freeze_snapshot()
    freeze_ok = all(
        freeze_before[k]["md5"] == freeze_after[k]["md5"] for k in freeze_before
    )
    cds_after = cds_alive(611595)

    report.update(
        {
            "mode": mode,
            "status": "train_ok" if mode == "train" else "smoke_ok",
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
                "reload_kind": train_info.get("reload_kind"),
                "stopped_epoch": train_info.get("stopped_epoch"),
                "early_stop_patience": train_info.get("early_stop_patience"),
                "best_ckpt": train_info.get("best_ckpt"),
                "history_tail": train_info["history"][-8:],
            },
            "metrics": {"val": val_m, "test": test_m},
            "lead_120h": {
                "val": {
                    "rmse_tier_a": val_120h_rmse(val_m),
                    "rmse_raw": (val_m.get("per_lead") or {}).get("120", {}).get("rmse_raw"),
                    "le_raw": (
                        val_120h_rmse(val_m) is not None
                        and float(val_120h_rmse(val_m))
                        <= float(bars["val_120h_t2m_rmse_raw_le"]) + 1e-9
                    ),
                },
                "test": {
                    "rmse_tier_a": (test_m.get("per_lead") or {}).get("120", {}).get(
                        "rmse_tier_a"
                    ),
                },
                "raw_val_reference": bars["val_120h_t2m_rmse_raw_le"],
            },
            "gates": gates,
            "diff_improves_residual": gates["diff_improves_residual"],
            "elapsed_s": elapsed,
            "freeze_after": freeze_after,
            "freeze_md5_unchanged": freeze_ok,
            "era5_pid_611595_after": cds_after,
            "artifacts": {
                "results_json": str(out_dir / f"{stem}_results.json"),
                "best_ckpt": train_info.get("best_ckpt"),
                "metrics_csv": str(out_dir / f"{stem}_metrics.csv"),
            },
            "not_skill_claims": [
                "not G1 / not IMDAA",
                "not FCN3 weight FT",
                "not NVIDIA CorrDiff parity",
                "claim_level=interim_era5",
                "living residual stays canonical if diff_improves_residual=false",
            ],
        }
    )

    out_json = out_dir / f"{stem}_results.json"
    out_json.write_text(json.dumps(report, indent=2) + "\n")

    rows = []
    for split, m in [("val", val_m), ("test", test_m)]:
        rows.append(
            {
                "split": split,
                "scope": "pooled",
                "rmse_raw": m["rmse_raw"],
                "rmse_living_residual": m["rmse_living_residual"],
                "rmse_tier_a": m["rmse_tier_a"],
                "n": m["n"],
            }
        )
        for lh, d in m["per_lead"].items():
            rows.append(
                {
                    "split": split,
                    "scope": f"lead_{lh}h",
                    "rmse_raw": d["rmse_raw"],
                    "rmse_living_residual": d["rmse_living_residual"],
                    "rmse_tier_a": d["rmse_tier_a"],
                    "n": d["n"],
                }
            )
    write_csv(out_dir / f"{stem}_metrics.csv", rows)

    summary = {
        "ok": True,
        "mode": mode,
        "variant": "v1_diff",
        "arch": "ResidualEDM2D/edm",
        "n_params": nparams,
        "wrote": str(out_json),
        "val_t2m_rmse": val_m["rmse_tier_a"],
        "test_t2m_rmse": test_m["rmse_tier_a"],
        "val_120h": val_120h_rmse(val_m),
        "gates": gates,
        "diff_improves_residual": gates["diff_improves_residual"],
        "freeze_md5_unchanged": freeze_ok,
        "cds": cds_after,
        "elapsed_s": elapsed,
        "device": str(device),
    }
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
