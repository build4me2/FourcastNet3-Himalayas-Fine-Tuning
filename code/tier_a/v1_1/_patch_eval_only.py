#!/usr/bin/env python3
"""Patch code/tier_a/v1_1/train.py for --eval-only expand zero-shot."""
from pathlib import Path
import py_compile

p = Path("code/tier_a/v1_1/train.py")
bak = Path("code/tier_a/v1_1/train.py.bak_pre_eval_only")
if not bak.exists():
    bak.write_text(p.read_text())
t = p.read_text()

old_doc = """Usage (from ~/fourcastnet):
  ~/fcn3-venv/bin/python code/tier_a/v1_1/train.py --config configs/tier_a_v1_1.yaml --dry-run
  ~/fcn3-venv/bin/python code/tier_a/v1_1/train.py --config configs/tier_a_v1_1.yaml --train --epochs 400
"""
new_doc = """Usage (from ~/fourcastnet):
  ~/fcn3-venv/bin/python code/tier_a/v1_1/train.py --config configs/tier_a_v1_1.yaml --dry-run
  ~/fcn3-venv/bin/python code/tier_a/v1_1/train.py --config configs/tier_a_v1_1.yaml --train --epochs 400
  ~/fcn3-venv/bin/python code/tier_a/v1_1/train.py --config configs/tier_a_v1_1_expand.yaml --eval-only \\
      --pairs-dir runs/phase0/tier0_holdout_expand/pairs \\
      --out-dir runs/phase0/tier_a/v1_1_expand \\
      --ckpt runs/phase0/tier_a/v1_1/best_residual.pt
"""
if old_doc not in t:
    raise SystemExit("docstring usage block missing")
t = t.replace(old_doc, new_doc, 1)

old_assert = '''def assert_no_overwrite(out_dir: Path, cfg: dict) -> None:
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
    allowed = expand("~/fourcastnet/runs/phase0/tier_a/v1_1").resolve()
    if out_r != allowed and not str(out_r).startswith(str(allowed) + "/"):
        raise RuntimeError(f"v1.1 must write under runs/phase0/tier_a/v1_1/ (got {out_r})")
    for p in [
        expand("~/fourcastnet/runs/phase0/g0/g0_verifying_results.json"),
        expand("~/fourcastnet/runs/phase0/tier0_holdout/tier0_holdout_metrics.json"),
        expand("~/fourcastnet/runs/phase0/tier_a/v1/tier_a_v1_results.json"),
    ]:
        if not p.exists():
            raise RuntimeError(f"forbidden artifact missing (do not recreate casually): {p}")
'''

new_assert = '''def assert_no_overwrite(out_dir: Path, cfg: dict) -> None:
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
'''

if old_assert not in t:
    raise SystemExit("assert_no_overwrite block missing")
t = t.replace(old_assert, new_assert, 1)

old_args = '''    ap.add_argument("--config", default="configs/tier_a_v1_1.yaml")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="short CPU smoke train")
    ap.add_argument("--train", action="store_true", help="full train (train-split only)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--allow-gpu", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
'''
new_args = '''    ap.add_argument("--config", default="configs/tier_a_v1_1.yaml")
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
'''
if old_args not in t:
    raise SystemExit("argparse block missing")
t = t.replace(old_args, new_args, 1)

old_cfg_load = '''    cfg = load_config(cfg_path)
    bars = beat_this_echo(cfg)
    stem = artifact_stem(cfg)
    out_dir = expand(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    assert_no_overwrite(out_dir, cfg)
'''
new_cfg_load = '''    cfg = load_config(cfg_path)
    if args.pairs_dir:
        cfg.setdefault("data", {})["pairs_dir"] = args.pairs_dir
    if args.out_dir:
        cfg["output_dir"] = args.out_dir
    bars = beat_this_echo(cfg)
    stem = artifact_stem(cfg)
    out_dir = expand(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    assert_no_overwrite(out_dir, cfg)
'''
if old_cfg_load not in t:
    raise SystemExit("cfg load block missing")
t = t.replace(old_cfg_load, new_cfg_load, 1)

old_mode = '''    if args.smoke:
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
        f"Tier-A {cfg.get('variant', '?')} cut=v1.1-reg diffusion=NO "
        f"mode={mode} device={device} epochs={epochs} "
        f"loss={loss_mode(cfg)} w_120={wmap[120]} lead_weights={wmap} "
        f"n_train={loaders['n_train']} n_val={loaders['n_val']} n_test={loaders['n_test']} "
        f"params={nparams}",
        flush=True,
    )

    train_info = train_loop(model, loaders, cfg, device, epochs, out_dir)
    val_m = eval_split(model, loaders["val"], device, cfg["variables_out"])
    test_m = eval_split(model, loaders["test"], device, cfg["variables_out"])
    train_m = eval_split(model, loaders["train"], device, cfg["variables_out"])
'''

new_mode = '''    if args.eval_only:
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
'''
if old_mode not in t:
    raise SystemExit("mode selection block missing")
t = t.replace(old_mode, new_mode, 1)

old_report_update = '''    report.update(
        {
            "mode": mode,
            "status": "smoke_ok" if mode == "smoke" else "train_ok",
            "device": str(device),
'''
new_report_update = '''    expand_tier0 = None
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
            "holdout_expand": "v1" if mode == "eval_only" else None,
            "provisional_years": True,
            "ckpt": str(ckpt_path) if mode == "eval_only" else train_info.get("best_ckpt"),
            "device": str(device),
'''
if old_report_update not in t:
    raise SystemExit("report.update status block missing")
t = t.replace(old_report_update, new_report_update, 1)

old_gates = '''            "gates": gates,
            "elapsed_s": elapsed,
'''
new_gates = '''            "gates": gates,
            "thin_beat_this_bars": bars,
            "expand_tier0_bars": expand_tier0,
            "elapsed_s": elapsed,
'''
if old_gates not in t:
    raise SystemExit("gates block missing")
t = t.replace(old_gates, new_gates, 1)

helper = '''
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


'''
marker = "def main() -> int:"
if "def _tier0_summary" not in t:
    if marker not in t:
        raise SystemExit("main marker missing")
    t = t.replace(marker, helper + marker, 1)

# Fix summary status / mode print for eval_only
old_summary_mode = '''        "mode": mode,
        "variant": cfg.get("variant"),
        "cut": "v1.1-reg",
        "diffusion": False,
        "wrote": str(out_json),
'''
# leave as-is; mode already correct

p.write_text(t)
py_compile.compile(str(p), doraise=True)
print("OK patched", p, "bytes", p.stat().st_size)
print("eval_only mentions", t.count("eval_only"))
print("zero_shot mentions", t.count("zero_shot"))
