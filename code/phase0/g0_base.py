#!/usr/bin/env python3
"""G0 base integrity probe — Path A (GATE_RECIPE_TIER0_G0.md §1).

Protocol
--------
  Global ERA5 IC (manifest) → frozen FCN3 global rollout → global metrics
  (CRPS / SSR / PSD stubs or real finite-field checks).

Memory (Spark 128GB UMA)
------------------------
  Do NOT retain full (S, V, H, W) global tensors across steps/members.
  Per step: extract preview_variables only → CPU; discard full field.
  Ensemble kept as (M, S, Vp, H, W) float32 ≈ 1.5 GB/member × M
  (Vp=6 preview vars), not ~18 GB/member × M for all 72 channels.
  Default M=4 is memory-safe with preview-only; M=2 was considered for
  Spark safety if full-field storage were ever reintroduced.

Modes (GPU gated)
-----------------
  --dry-run   Default safe. Validate config + manifest + staged ICs + imports;
              print wall-time estimate; write plan JSON. NO GPU / NO FCN3 load.
  --smoke     1 IC × few steps. Requires --allow-gpu. Optional tiny check only.
  --full      All staged ICs (≥ min_staged_for_full) × +15 d (60 steps) × M members.
              Requires --allow-gpu. Writes g0_base_results.json.

Reuse
-----
  Model load pattern mirrors code/phase0/inference_smoke.py
  (Package + FCN3.load_model from local package_root).

Usage
-----
  cd ~/fourcastnet
  PYTHONUNBUFFERED=1 ~/fcn3-venv/bin/python code/phase0/g0_base.py --dry-run
  # ONLY after Manisha GPU greenlight:
  # PYTHONUNBUFFERED=1 ~/fcn3-venv/bin/python code/phase0/g0_base.py --full --allow-gpu
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None

EXPECTED_SHAPE = (72, 721, 1440)
DEFAULT_CONFIG = Path.home() / "fourcastnet" / "configs" / "g0_base.yaml"


def load_config(path: Path) -> dict:
    if yaml is None:
        raise RuntimeError("PyYAML required; pip install pyyaml")
    return yaml.safe_load(path.read_text())


def expand(p: str | Path) -> Path:
    return Path(str(p)).expanduser().resolve()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _peak_mem_gb() -> dict:
    out: dict[str, Any] = {
        "rss_gb": None,
        "cuda_alloc_gb": None,
        "cuda_reserved_gb": None,
    }
    try:
        import resource

        out["rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 3)
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            out["cuda_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
            out["cuda_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 3)
    except Exception:
        pass
    return out


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def staged_ics(manifest: dict) -> list[dict]:
    return [ic for ic in manifest.get("ics", []) if ic.get("status") == "staged"]


def verify_ic_file(ic: dict, expect_channels: list[str] | None = None) -> dict:
    """Validate one staged IC on disk. CPU / numpy only."""
    import numpy as np

    rel = ic.get("relative_path") or Path(ic.get("path", "")).name
    path = Path(ic.get("path", "")).expanduser()
    info: dict[str, Any] = {
        "id": ic.get("id"),
        "time": ic.get("time"),
        "path": str(path),
        "relative_path": rel,
        "ok": False,
    }
    if not path.is_file():
        info["error"] = "missing_file"
        return info
    # mmap to avoid 286 MiB × N resident during dry-run of many ICs
    arr = np.load(path, mmap_mode="r")
    info["shape"] = list(arr.shape)
    info["dtype"] = str(arr.dtype)
    info["bytes"] = int(path.stat().st_size)
    if tuple(arr.shape) != EXPECTED_SHAPE:
        info["error"] = f"bad_shape want={EXPECTED_SHAPE} got={tuple(arr.shape)}"
        return info
    if arr.dtype != np.float32:
        info["error"] = f"bad_dtype want=float32 got={arr.dtype}"
        return info
    # Sample finite check (full scan is expensive; trust staging validation + spot check)
    sample = np.asarray(arr[4, ::40, ::40])  # t2m-ish channel index 4
    info["sample_finite"] = bool(np.isfinite(sample).all())
    if not info["sample_finite"]:
        info["error"] = "non_finite_sample"
        return info
    # Honesty flags from manifest
    if ic.get("is_nepal_crop") is True or ic.get("is_random") is True:
        info["error"] = "forbidden_ic_type"
        return info
    if ic.get("is_global") is False:
        info["error"] = "not_global"
        return info
    info["ok"] = True
    info["season"] = ic.get("season")
    info["region_class"] = ic.get("region_class")
    info["region_label"] = ic.get("region_label")
    info["sha256"] = ic.get("sha256")
    return info


def wall_time_estimate(
    n_ics: int,
    n_members: int,
    n_steps: int,
    s_per_step: float,
    load_s: float,
) -> dict:
    """Estimate from measured smoke (~5 s/step disco; load ~30 s)."""
    roll_s = n_ics * n_members * n_steps * s_per_step
    total_s = load_s + roll_s
    return {
        "n_ics": n_ics,
        "n_members": n_members,
        "n_steps": n_steps,
        "s_per_step_assumed": s_per_step,
        "load_s_assumed": load_s,
        "rollout_s_est": round(roll_s, 1),
        "total_s_est": round(total_s, 1),
        "total_h_est": round(total_s / 3600.0, 2),
        "basis": (
            "inference_smoke 2026-09-11: disco CUDA True; "
            "~5 s/step; 4×16 rollout 335.22 s; peak CUDA alloc ~53 GB; load 28.47 s"
        ),
        "formula": "load_s + n_ics * n_members * n_steps * s_per_step",
        "note": (
            "Sequential members (batch=1) for Spark 128GB UMA safety — "
            "same as inference_smoke. Preview-only CPU storage (~1.5 GB/member). "
            "Does not include verification ERA5 fetch."
        ),
    }


def preview_indices(variables: Any, preview_names: list[str]) -> tuple[list[int], list[str]]:
    """Map preview variable names → channel indices present in model.variables."""
    import numpy as np

    vars_arr = np.asarray(variables)
    idx: list[int] = []
    names_hit: list[str] = []
    for n in preview_names:
        hits = np.where(vars_arr == n)[0]
        if hits.size:
            idx.append(int(hits[0]))
            names_hit.append(n)
    return idx, names_hit


def metrics_stubs_global(
    ens: Any,
    preview_names: list[str],
    lead_hours: list[int],
) -> dict:
    """Path-A metric stubs on preview-only ensemble [M, S, Vp, H, W].

    Real CRPS/SSR need verifying analysis at each lead — not wired here.
    Finite-field + spread diagnostics are real; CRPS/SSR/PSD marked stub.
    """
    import numpy as np

    ens = np.asarray(ens)
    out: dict[str, Any] = {
        "ensemble_shape": list(ens.shape),
        "preview_variables": list(preview_names),
        "storage": "preview_only",
        "lead_hours": lead_hours,
        "stub": True,
    }
    finite = bool(np.isfinite(ens).all()) if ens.size else True
    out["finite_fields"] = {
        "all_finite": finite,
        "nan_count": int(np.isnan(ens).sum()) if ens.size else 0,
        "inf_count": int(np.isinf(ens).sum()) if ens.size else 0,
        "pass": finite,
        "scope": "preview_variables_only",
    }
    spread = {}
    if ens.ndim == 5 and ens.shape[0] >= 1 and ens.shape[1] >= 1:
        last = ens[:, -1]  # [M, Vp, H, W]
        if last.shape[0] >= 2:
            std = last.std(axis=0)  # [Vp, H, W]
            for i, name in enumerate(preview_names):
                if i < std.shape[0]:
                    spread[name] = {
                        "global_mean_std": float(std[i].mean()),
                        "global_max_std": float(std[i].max()),
                    }
        else:
            spread["note"] = "need >=2 members for spread"
    out["ssr_stub"] = {
        "status": "stub",
        "reason": "SSR needs verifying analysis + ensemble mean RMSE; spread-only logged",
        "spread_at_final_lead": spread,
        "pass_provisional": None,
    }
    out["crps_stub"] = {
        "status": "stub",
        "reason": "Spatial CRPS @ +15 d needs verifying ERA5 (or ERA5T) at lead; not fetched in scaffolding",
        "pass_provisional": None,
    }
    out["psd_stub"] = {
        "status": "stub",
        "reason": "Angular/zonal PSD vs ERA5 @ +15 d not computed in scaffolding",
        "pass_provisional": None,
    }
    out["hmm_identity"] = {
        "status": "assumed_ok_via_earth2studio_fcn3_iterator",
        "note": "1 forward / member / step via FCN3.create_iterator (not iterative diffusion substitute)",
    }
    out["gate_ready"] = False
    out["gate_note"] = (
        "Finite fields real (preview vars); CRPS/SSR/PSD stubs — "
        "wire verifying analysis before G0 PASS claim"
    )
    return out


def build_ic_tensor(path: Path, ic_time: str, model, device):
    """Load staged global npy → FCN3 input tensor + coords (path A)."""
    import numpy as np
    import torch

    arr = np.load(path)  # (72, 721, 1440) float32
    if tuple(arr.shape) != EXPECTED_SHAPE:
        raise RuntimeError(f"IC shape {arr.shape} != {EXPECTED_SHAPE}")
    if not np.isfinite(arr).all():
        raise RuntimeError(f"IC has non-finite values: {path}")
    # Earth2Studio FCN3 input: (batch, time, lead_time, variable, lat, lon)
    x = torch.from_numpy(np.array(arr, copy=True)).to(device=device, dtype=torch.float32)
    x = x.view(1, 1, 1, *EXPECTED_SHAPE)
    t = np.datetime64(ic_time.replace("Z", ""))
    if "T" not in str(t):
        t = np.datetime64(str(t) + "T00:00")
    coords = OrderedDict(
        {
            "batch": np.array([0]),
            "time": np.array([t]),
            "lead_time": np.array([np.timedelta64(0, "h")]),
            "variable": np.array(model.variables),
            "lat": np.linspace(90.0, -90.0, 721),
            "lon": np.linspace(0, 360, 1440, endpoint=False),
        }
    )
    return x, coords


def load_fcn3(cfg: dict, report: dict):
    """Mirror inference_smoke.py Package + FCN3.load_model path."""
    import torch
    from earth2studio.models.auto import Package
    from earth2studio.models.px.fcn3 import FCN3

    pkg_root = expand(cfg["model"]["package_root"])
    ckpt = expand(cfg["model"]["checkpoint"])
    report["package_root"] = str(pkg_root)
    report["checkpoint"] = str(ckpt)
    if not ckpt.is_file():
        raise FileNotFoundError(f"checkpoint missing: {ckpt}")
    if not (pkg_root / "config.json").is_file():
        raise FileNotFoundError(f"package config.json missing under {pkg_root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report["device"] = str(device)
    report["cuda_available"] = torch.cuda.is_available()
    if device.type != "cuda":
        raise RuntimeError("G0 GPU path requires CUDA; cpu-only not supported for FCN3 G0")

    t0 = time.time()
    package = Package(str(pkg_root), cache=False)
    model = FCN3.load_model(package)
    model = model.to(device)
    model.eval()
    report["load_s"] = round(time.time() - t0, 2)
    report["n_params"] = int(sum(p.numel() for p in model.parameters()))
    report["n_variables"] = int(len(model.variables))
    report["torch"] = torch.__version__
    return model, device


def _squeeze_field(out) -> Any:
    """Reduce FCN3 iterator output to (V, H, W) tensor."""
    t = out.detach()
    while t.ndim > 4 and t.shape[0] == 1:
        t = t.squeeze(0)
    while t.ndim > 4 and t.shape[0] == 1:
        t = t.squeeze(0)
    if t.ndim == 4 and t.shape[0] == 1:
        t = t[0]
    if t.ndim != 3:
        raise RuntimeError(f"unexpected out ndim={t.ndim} shape={tuple(t.shape)}")
    return t


def run_path_a(
    cfg: dict,
    ics: list[dict],
    n_members: int,
    n_steps: int,
    seed0: int,
    out_dir: Path,
    report: dict,
    save_global: bool = False,
    progress_every: int = 5,
    partial_dir: Path | None = None,
) -> dict:
    """Global IC → global rollout → metrics stubs (Path A).

    Memory-safe: keep only preview_variables on CPU; discard full fields each step.
    """
    import numpy as np
    import torch

    model, device = load_fcn3(cfg, report)
    preview_cfg = list(cfg.get("g0_base", {}).get("preview_variables", ["t2m", "z500"]))
    pidx, preview_names = preview_indices(model.variables, preview_cfg)
    if not pidx:
        raise RuntimeError(f"none of preview_variables {preview_cfg} found in model.variables")
    report["preview_variables"] = preview_names
    report["preview_indices"] = pidx
    report["storage_mode"] = "preview_only"
    # ~bytes estimate for one member preview stack
    bytes_per_member = (n_steps + 1) * len(pidx) * 721 * 1440 * 4
    report["preview_mem_est_gb_per_member"] = round(bytes_per_member / 1e9, 3)
    report["preview_mem_est_gb_ensemble"] = round(bytes_per_member * n_members / 1e9, 3)

    lead_hours = [6 * s for s in range(n_steps + 1)]
    ic_results = []
    if partial_dir is None:
        partial_dir = out_dir / "partial"
    partial_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t_all0 = time.time()
    with torch.inference_mode():
        for ic in ics:
            path = Path(ic["path"]).expanduser()
            ic_time = ic["time"]
            ic_id = ic.get("id")
            print(
                json.dumps(
                    {
                        "ic_start": ic_id,
                        "time": ic_time,
                        "path": str(path),
                        "n_members": n_members,
                        "n_steps": n_steps,
                        "preview_variables": preview_names,
                        "storage": "preview_only",
                    }
                ),
                flush=True,
            )
            x0, coords0 = build_ic_tensor(path, ic_time, model, device)
            members_meta = []
            # Preview-only stacks: list of (S, Vp, H, W) float32 numpy
            member_previews: list[Any] = []
            running_nan = 0
            running_inf = 0
            all_finite_full_stream = True

            for m in range(n_members):
                seed = seed0 + m
                model.set_rng(seed=seed, reset=True)
                x = x0.clone()
                coords = OrderedDict(
                    (k, (v.copy() if hasattr(v, "copy") else v)) for k, v in coords0.items()
                )
                t_m0 = time.time()
                it = model.create_iterator(x, coords)
                preview_frames: list[Any] = []
                member_finite = True
                for step in range(n_steps + 1):
                    t_step0 = time.time()
                    out, c = next(it)
                    field = _squeeze_field(out)  # (V, H, W) on device
                    # Streaming finite check on full field (cheap-ish vs storing)
                    # Sample every progress_every steps for full-field; always check preview
                    preview_t = field[pidx].float().cpu()  # (Vp, H, W)
                    del field, out
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                    finite_preview = bool(torch.isfinite(preview_t).all().item())
                    if not finite_preview:
                        member_finite = False
                        all_finite_full_stream = False
                        running_nan += int(torch.isnan(preview_t).sum().item())
                        running_inf += int(torch.isinf(preview_t).sum().item())

                    preview_frames.append(preview_t.numpy())
                    del preview_t

                    step_s = round(time.time() - t_step0, 2)
                    if step % progress_every == 0 or step == n_steps:
                        print(
                            json.dumps(
                                {
                                    "progress": {
                                        "ic": ic_id,
                                        "member": m,
                                        "step": step,
                                        "lead_h": 6 * step,
                                        "step_s": step_s,
                                        "finite_preview": finite_preview,
                                        "peak_mem": _peak_mem_gb(),
                                    }
                                }
                            ),
                            flush=True,
                        )

                stacked = np.stack(preview_frames, axis=0)  # (S, Vp, H, W)
                member_previews.append(stacked)
                m_info = {
                    "member": m,
                    "seed": seed,
                    "preview_shape": list(stacked.shape),
                    "wall_s": round(time.time() - t_m0, 2),
                    "finite_preview": member_finite and bool(np.isfinite(stacked).all()),
                    "storage": "preview_only",
                }
                members_meta.append(m_info)
                print(
                    json.dumps({"member_done": {**m_info, "ic": ic_id}}),
                    flush=True,
                )
                del preview_frames, stacked, x, it, coords
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            ens = np.stack(member_previews, axis=0)  # (M, S, Vp, H, W)
            metrics = metrics_stubs_global(ens, preview_names, lead_hours)
            # Attach streaming counters
            metrics["finite_fields"]["stream_nan_preview"] = running_nan
            metrics["finite_fields"]["stream_inf_preview"] = running_inf
            metrics["finite_fields"]["stream_all_finite_preview"] = all_finite_full_stream
            ic_rec = {
                "id": ic_id,
                "time": ic_time,
                "path": str(path),
                "season": ic.get("season"),
                "region_class": ic.get("region_class"),
                "region_label": ic.get("region_label"),
                "members": members_meta,
                "metrics": metrics,
                "path_a": True,
                "storage": "preview_only",
            }
            if save_global:
                npz_path = out_dir / f"{ic_id}_ensemble_preview.npz"
                np.savez_compressed(
                    npz_path,
                    preview=ens,
                    preview_variables=np.array(preview_names),
                    lead_hours=np.array(lead_hours),
                    seeds=np.array([seed0 + m for m in range(n_members)]),
                    ic_time=ic_time,
                    ic_id=ic_id,
                )
                ic_rec["preview_npz"] = str(npz_path)

            # Per-IC checkpoint (partial)
            partial_path = partial_dir / f"{ic_id}.json"
            partial_path.write_text(json.dumps(ic_rec, indent=2, default=str))
            ic_rec["partial_json"] = str(partial_path)
            print(
                json.dumps(
                    {
                        "ic_done": ic_id,
                        "partial_json": str(partial_path),
                        "finite": metrics["finite_fields"]["all_finite"],
                        "peak_mem": _peak_mem_gb(),
                    }
                ),
                flush=True,
            )

            ic_results.append(ic_rec)
            del ens, member_previews, x0, coords0
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    report["rollout_s"] = round(time.time() - t_all0, 2)
    report["peak_mem"] = _peak_mem_gb()
    report["ic_results"] = ic_results
    report["path"] = "A"
    report["frozen"] = True
    report["train"] = False
    all_finite = all(
        r["metrics"]["finite_fields"]["all_finite"] for r in ic_results
    ) if ic_results else False
    report["aggregate"] = {
        "n_ics": len(ic_results),
        "all_finite": all_finite,
        "crps_ssr_psd": "stub",
        "storage": "preview_only",
        "g0_pass_claimable": False,
        "g0_pass_note": (
            "Finite OK (preview) does not equal G0 PASS — need real CRPS/SSR/PSD vs verifying ERA5"
            if all_finite
            else "Non-finite preview fields → G0 FAIL per GATE_RECIPE §1.4"
        ),
    }
    return report


def dry_run(cfg: dict, manifest: dict, report: dict) -> int:
    """Validate scaffolding; no GPU / no model load."""
    blockers: list[str] = []
    ics_cfg = cfg.get("ics", {})
    g0 = cfg.get("g0_base", {})

    staged = staged_ics(manifest)
    report["manifest_counts"] = manifest.get("counts")
    report["n_listed"] = len(manifest.get("ics", []))
    report["n_staged"] = len(staged)
    report["min_staged_for_full"] = int(ics_cfg.get("min_staged_for_full", 8))

    verified = []
    for ic in staged:
        info = verify_ic_file(ic)
        verified.append(info)
        if not info["ok"]:
            blockers.append(f"{ic.get('id')}: {info.get('error')}")
    report["ics_verified"] = verified

    for ic in manifest.get("ics", []):
        if ic.get("is_nepal_crop") or ic.get("is_random"):
            blockers.append(f"{ic.get('id')}: forbidden nepal/random IC in manifest")

    ckpt = expand(cfg["model"]["checkpoint"])
    pkg = expand(cfg["model"]["package_root"])
    report["checkpoint_exists"] = ckpt.is_file()
    report["package_config_exists"] = (pkg / "config.json").is_file()
    if not report["checkpoint_exists"]:
        blockers.append(f"checkpoint missing: {ckpt}")
    if not report["package_config_exists"]:
        blockers.append(f"package config.json missing: {pkg}")

    try:
        import torch

        report["torch"] = torch.__version__
        report["cuda_available"] = torch.cuda.is_available()
    except Exception as e:
        blockers.append(f"torch import failed: {e}")
        report["torch_error"] = str(e)

    try:
        import earth2studio  # noqa: F401
        from earth2studio.models.px.fcn3 import FCN3  # noqa: F401
        from makani.models.model_package import load_model_package  # noqa: F401

        report["earth2studio"] = True
        report["makani"] = True
    except Exception as e:
        blockers.append(f"earth2studio/makani import failed: {e}")
        report["earth2studio"] = False
        report["earth2studio_error"] = str(e)

    report["wall_estimate_full"] = wall_time_estimate(
        n_ics=int(ics_cfg.get("min_staged_for_full", 8)),
        n_members=int(g0.get("ensemble_members", 4)),
        n_steps=int(g0.get("steps", 60)),
        s_per_step=float(g0.get("wall_estimate_s_per_step", 5.0)),
        load_s=float(g0.get("wall_estimate_load_s", 30.0)),
    )
    smoke = cfg.get("smoke", {})
    report["wall_estimate_smoke"] = wall_time_estimate(
        n_ics=int(smoke.get("max_ics", 1)),
        n_members=int(smoke.get("ensemble_members", 1)),
        n_steps=int(smoke.get("steps", 2)),
        s_per_step=float(g0.get("wall_estimate_s_per_step", 5.0)),
        load_s=float(g0.get("wall_estimate_load_s", 30.0)),
    )
    # Document memory math
    n_prev = len(g0.get("preview_variables", ["t2m", "z500"]))
    n_steps_full = int(g0.get("steps", 60))
    n_mem = int(g0.get("ensemble_members", 4))
    preview_gb = (n_steps_full + 1) * n_prev * 721 * 1440 * 4 * n_mem / 1e9
    full_gb = (n_steps_full + 1) * 72 * 721 * 1440 * 4 * n_mem / 1e9
    report["memory_notes"] = {
        "preview_only_ensemble_gb_est": round(preview_gb, 2),
        "full_field_ensemble_gb_est_DO_NOT_USE": round(full_gb, 2),
        "model_cuda_gb_est": "~53–69",
        "spark_uma_gb": 128,
        "ensemble_members_default": n_mem,
        "note": (
            "Full-field storage (~18 GB/member) caused OOM on first full run. "
            "Preview-only (~1.5 GB/member) keeps M=4 memory-safe. "
            "Consider M=2 only if full fields are reintroduced."
        ),
    }

    report["gpu_used"] = False
    report["fcn3_loaded"] = False
    report["path"] = "A"
    report["locks"] = cfg.get("locks", {})

    if len(staged) < 1:
        blockers.append("no staged ICs in manifest — run stage_g0_ics.py first")

    out_dir = expand(g0.get("output_dir", "~/fourcastnet/runs/phase0/g0"))
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = out_dir / "g0_base_dry_run.json"

    if blockers:
        report["ok"] = False
        report["status"] = "blocked"
        report["blockers"] = blockers
        plan_path.write_text(json.dumps(report, indent=2, default=str))
        report["plan_json"] = str(plan_path)
        print(json.dumps(report, indent=2, default=str))
        print("BLOCKED — see blockers", file=sys.stderr)
        return 2

    report["ok"] = True
    report["status"] = "dry_run_ready"
    report["full_run"] = False
    report["ready_for_full"] = len(staged) >= int(ics_cfg.get("min_staged_for_full", 8))
    report["ready_note"] = (
        "staged>=8 and --allow-gpu required for --full"
        if report["ready_for_full"]
        else f"only {len(staged)}/8 staged — wait for IC backfill before --full"
    )
    plan_path.write_text(json.dumps(report, indent=2, default=str))
    report["plan_json"] = str(plan_path)
    print(json.dumps(report, indent=2, default=str))
    return 0


def write_failure_artifacts(
    report: dict,
    results_path: Path | None,
    traceback_path: Path,
    exc: BaseException,
) -> None:
    """Flush traceback + JSON error for OOM / GPU failures."""
    tb = traceback.format_exc()
    report["ok"] = False
    report["status"] = "failed"
    report["error"] = str(exc)
    report["error_type"] = type(exc).__name__
    report["traceback"] = tb
    report["peak_mem"] = _peak_mem_gb()
    report["failed_utc"] = utc_now()
    try:
        traceback_path.parent.mkdir(parents=True, exist_ok=True)
        traceback_path.write_text(tb)
        report["traceback_path"] = str(traceback_path)
    except Exception as e2:
        report["traceback_write_error"] = str(e2)
    if results_path is not None:
        try:
            results_path.parent.mkdir(parents=True, exist_ok=True)
            results_path.write_text(json.dumps(report, indent=2, default=str))
            report["results_json"] = str(results_path)
        except Exception as e2:
            report["results_write_error"] = str(e2)
    print(json.dumps(report, indent=2, default=str), flush=True)
    print(tb, file=sys.stderr, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="G0 base Path A runner (global IC → global rollout → metrics)"
    )
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Override manifest path (default from config)",
    )
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate only (default if no --smoke/--full)",
    )
    mode.add_argument(
        "--smoke",
        action="store_true",
        help="1 IC × few steps; requires --allow-gpu",
    )
    mode.add_argument(
        "--full",
        action="store_true",
        help="All ICs × +15 d; requires --allow-gpu",
    )
    ap.add_argument(
        "--allow-gpu",
        action="store_true",
        help="Required for --smoke / --full (Manisha GPU greenlight)",
    )
    ap.add_argument(
        "--save-preview-npz",
        action="store_true",
        help="On GPU runs, save preview-var ensemble npz per IC",
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=5,
        help="Print progress JSON every N steps (default 5; step 0 and final always)",
    )
    args = ap.parse_args()

    if not args.smoke and not args.full:
        args.dry_run = True

    report: dict[str, Any] = {
        "schema": "g0_base_results/v1",
        "created_utc": utc_now(),
        "config": str(args.config),
        "mode": "dry_run" if args.dry_run else ("smoke" if args.smoke else "full"),
        "allow_gpu": bool(args.allow_gpu),
        "ok": False,
        "path": "A",
        "recipe": "docs/research/GATE_RECIPE_TIER0_G0.md §1 path A",
        "storage_mode": "preview_only",
    }

    results_path: Path | None = None
    traceback_path = (
        Path.home() / "fourcastnet" / "logs" / "g0_base_full.traceback.txt"
        if args.full
        else Path.home() / "fourcastnet" / "logs" / "g0_base_smoke.traceback.txt"
    )

    try:
        if not args.config.is_file():
            report["error"] = f"missing config {args.config}"
            print(json.dumps(report, indent=2), flush=True)
            return 1
        cfg = load_config(args.config)
        man_path = expand(
            args.manifest
            if args.manifest
            else cfg.get("ics", {}).get(
                "manifest", "~/fourcastnet/data/g0_ics/g0_ic_manifest.json"
            )
        )
        report["manifest"] = str(man_path)
        if not man_path.is_file():
            report["error"] = f"missing manifest {man_path}"
            print(json.dumps(report, indent=2), flush=True)
            return 1
        manifest = load_manifest(man_path)

        if args.dry_run:
            return dry_run(cfg, manifest, report)

        # GPU modes
        if not args.allow_gpu:
            report["ok"] = False
            report["status"] = "gpu_not_allowed"
            report["error"] = (
                "--smoke/--full require --allow-gpu (Manisha GPU greenlight not set)"
            )
            print(json.dumps(report, indent=2), flush=True)
            print("REFUSED — pass --allow-gpu only after Manisha greenlight", file=sys.stderr)
            return 3

        g0 = cfg.get("g0_base", {})
        out_dir = expand(g0.get("output_dir", "~/fourcastnet/runs/phase0/g0"))
        out_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = Path.home() / "fourcastnet" / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        staged = staged_ics(manifest)
        if not staged:
            report["error"] = "no staged ICs"
            print(json.dumps(report, indent=2), flush=True)
            return 1

        for ic in staged:
            info = verify_ic_file(ic)
            if not info["ok"]:
                report["error"] = f"IC verify failed: {info}"
                print(json.dumps(report, indent=2, default=str), flush=True)
                return 1

        seed0 = int(g0.get("seed0", 333))
        if args.smoke:
            smoke = cfg.get("smoke", {})
            n_members = int(smoke.get("ensemble_members", 1))
            n_steps = int(smoke.get("steps", 2))
            ics = staged[: int(smoke.get("max_ics", 1))]
            results_path = out_dir / "g0_base_smoke.json"
            traceback_path = logs_dir / "g0_base_smoke.traceback.txt"
            report["status"] = "smoke_running"
        else:
            min_n = int(cfg.get("ics", {}).get("min_staged_for_full", 8))
            if len(staged) < min_n:
                report["ok"] = False
                report["status"] = "insufficient_ics"
                report["error"] = (
                    f"full G0 needs ≥{min_n} staged ICs; have {len(staged)}. "
                    "Wait for backfill (do not kill PID staging)."
                )
                report["n_staged"] = len(staged)
                print(json.dumps(report, indent=2, default=str), flush=True)
                return 4
            n_members = int(g0.get("ensemble_members", 4))
            n_steps = int(g0.get("steps", 60))
            ics = staged
            results_path = expand(
                g0.get(
                    "results_json",
                    "~/fourcastnet/runs/phase0/g0/g0_base_results.json",
                )
            )
            traceback_path = logs_dir / "g0_base_full.traceback.txt"
            report["status"] = "full_running"

        report["n_ics"] = len(ics)
        report["ensemble_members"] = n_members
        report["steps"] = n_steps
        report["wall_estimate"] = wall_time_estimate(
            len(ics),
            n_members,
            n_steps,
            float(g0.get("wall_estimate_s_per_step", 5.0)),
            float(g0.get("wall_estimate_load_s", 30.0)),
        )
        report["gpu_used"] = True
        report["output_dir"] = str(out_dir)
        report["results_json"] = str(results_path)

        print(
            json.dumps(
                {
                    "gpu_run_start": report["mode"],
                    "n_ics": len(ics),
                    "n_members": n_members,
                    "n_steps": n_steps,
                    "storage": "preview_only",
                    "progress_every": args.progress_every,
                }
            ),
            flush=True,
        )

        try:
            run_path_a(
                cfg,
                ics,
                n_members,
                n_steps,
                seed0,
                out_dir,
                report,
                save_global=bool(args.save_preview_npz),
                progress_every=max(1, int(args.progress_every)),
                partial_dir=out_dir / "partial",
            )
        except Exception as e:
            write_failure_artifacts(report, results_path, traceback_path, e)
            return 1

        report["ok"] = True
        report["status"] = "smoke_ok" if args.smoke else "full_ok"
        report["full_run"] = bool(args.full)
        report["results_json"] = str(results_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps(report, indent=2, default=str))
        print(json.dumps(report, indent=2, default=str), flush=True)
        return 0

    except Exception as e:
        write_failure_artifacts(report, results_path, traceback_path, e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
