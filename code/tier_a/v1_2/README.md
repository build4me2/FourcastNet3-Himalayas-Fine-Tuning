# Tier-A v1.2 — expand-pairs scaffold from v1.1 (NO diffusion)

**Status:** **SCAFFOLDED** — CPU dry-run only. **Manisha has NOT confirmed train.** Leonard: CPU scaffold only.
**Parent:** same arch as v1.1 (`base=16`, `dropout=0.20`, `w_120=1.5`, composite ckpt).
**Data:** `holdout_expand:v1` pairs (32 ICs). Beat-this = **expand bars** (not thin).

## Gates (expand — Leonard HOLDOUT_EXPAND_CALL.md)

| Gate | Bar |
|------|-----|
| val t2m pooled RMSE | strictly **< 1.987014 K** |
| test t2m pooled RMSE | strictly **< 1.897298 K** |
| **hard** val +120 h t2m RMSE | **≤ expand-val raw 2.131516 K** |
| Splits | train ic01–08; val ic09–12+ic17–24; test expand 12 (ic13–16+ic25–32) |
| Labels | `provisional_years=true`; `claim_level=interim_era5`; `g1_claimable=false` |

## vs v1.1

| Item | v1.1 thin / zero-shot expand | **v1.2** |
|------|------------------------------|----------|
| pairs_dir | thin or expand (eval-only) | **expand pairs** (train scaffold) |
| beat_this | thin historically; expand for zero-shot | **expand bars** |
| out_dir | `v1_1/` / `v1_1_expand/` | **`v1_2/` only** |
| arch | base16 / drop0.20 / w_120=1.5 | **same** |

## Do not overwrite

`runs/phase0/tier_a/v0/`, `v0_1/`, `v1/`, **`v1_1/`**, **`v1_1_expand/`**, `tier0_holdout/`, `tier0_holdout_expand/`, G0 verifying. Leave ERA5 PID **611595** alone.

## Quick start (Spark) — dry-run NOW; train ONLY after Manisha confirms

```bash
cd ~/fourcastnet
~/fcn3-venv/bin/python code/tier_a/v1_2/train.py --config configs/tier_a_v1_2.yaml --dry-run
# AFTER Manisha confirms (NO --allow-gpu unless she says so):
# PYTHONUNBUFFERED=1 ~/fcn3-venv/bin/python -u code/tier_a/v1_2/train.py \
#   --config configs/tier_a_v1_2.yaml --train --epochs 400
```

Outputs (when trained): `runs/phase0/tier_a/v1_2/{tier_a_v1_2_results.json,tier_a_v1_2_metrics.csv,best_residual.pt}`.
