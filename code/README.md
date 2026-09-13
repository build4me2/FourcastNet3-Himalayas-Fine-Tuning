# Code — FCN3 RRCA-FD (new work only)

No Idea 1/2 lake/GLOF code lives here.

## Layout

| Path | Role |
|------|------|
| `phase0/` | Bring-up: env smoke, norm load, frozen FCN3 inference smoke, Tier-0 CPU bias scaffold |
| (later) `tier_a/` | CorrDiff/StormCast-lite regional diagnostic |
| (later) `eval/` | G0–G3 gate scripts |

## Run (after env ready)

```bash
source ~/fourcastnet/venv/bin/activate   # or ~/fcn3-venv
cd ~/fourcastnet
python code/phase0/env_smoke.py
python code/phase0/load_norms.py
# After box/years lock + ERA5 crop:
# python code/phase0/inference_smoke.py --config configs/phase0_nepal_box.yaml
```

Coordinate GPU long runs with Raj.


## Tier-0 (CPU only)

```bash
source ~/fcn3-venv/bin/activate
cd ~/fourcastnet
python code/phase0/tier0_bias.py --help
python code/phase0/tier0_bias.py --dry-run
python code/phase0/tier0_bias.py --smoke   # synthetic residual; NOT skill
```

Skill / G0 needs global ERA5 ICs (see canonical `docs/04-gates/GATE_RECIPE_TIER0_G0.md`). Do not run GPU jobs until Manisha greenlights.

## G0 global IC staging

```bash
source ~/fcn3-venv/bin/activate
cd ~/fourcastnet
python code/phase0/stage_g0_ics.py --dry-run
python code/phase0/stage_g0_ics.py --smoke          # 2 ICs
python code/phase0/stage_g0_ics.py --all            # all 8 (skips existing)
# or: bash code/phase0/run_g0_ics_backfill.sh
```

Outputs: `data/g0_ics/ic_YYYYMMDDTHHMM_global.npy` + `g0_ic_manifest.json`.
Uses ARCO ERA5 zarr (not CDS). Does not run FCN3 / GPU.

## G0 base Path A (GPU gated)

```bash
source ~/fcn3-venv/bin/activate
cd ~/fourcastnet
python code/phase0/g0_base.py --dry-run          # default safe; no GPU
# ONLY after Manisha GPU greenlight AND staged>=8 ICs:
# python code/phase0/g0_base.py --full --allow-gpu
# optional tiny check:
# python code/phase0/g0_base.py --smoke --allow-gpu
```

Outputs: `runs/phase0/g0/g0_base_dry_run.json` (plan); full → `runs/phase0/g0/g0_base_results.json`.
CRPS/SSR/PSD are stubs until verifying ERA5 at leads is wired; finite-field check is real.

## Tier-0 real (global-IC → crop)

```bash
# CPU targets (ARCO; does not use CDS / PID 611595)
~/fcn3-venv/bin/python code/phase0/tier0_era5_targets.py

# GPU crop from staged global ICs (one process)
PYTHONUNBUFFERED=1 ~/fcn3-venv/bin/python -u code/phase0/tier0_crop_rollout.py --allow-gpu

# Fit elevation-binned linear; writes runs/phase0/tier0/
~/fcn3-venv/bin/python code/phase0/tier0_bias.py --real

# G0 verifying-ERA5 plan (CRPS hook; do not claim PASS)
~/fcn3-venv/bin/python code/phase0/g0_verifying_era5.py --dry-run
```
