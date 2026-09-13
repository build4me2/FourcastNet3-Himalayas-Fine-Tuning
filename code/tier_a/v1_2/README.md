# Tier-A v1.2 — expand-pairs train from v1.1 (NO diffusion)

**Status:** **TRAINED — INTERIM FAIL** (frozen FAIL reference). See [`docs/calls/TIER_A_V1_2_CALL.md`](../../docs/calls/TIER_A_V1_2_CALL.md).
**Parent arch:** same as v1.1 (`base=16`, `dropout=0.20`, `w_120=1.5`, composite ckpt).
**Data:** `holdout_expand:v1` pairs (32 ICs). Beat-this = **expand bars** (not thin).
**Living successor:** **v1.2b** (INTERIM PASS) — [`docs/calls/TIER_A_V1_2B_CALL.md`](../../docs/calls/TIER_A_V1_2B_CALL.md).

## Gates (expand — Leonard HOLDOUT_EXPAND_CALL.md)

| Gate | Bar | v1.2 result |
|------|-----|-------------|
| val t2m pooled RMSE | strictly **< 1.987014 K** | PASS (~1.905) |
| test t2m pooled RMSE | strictly **< 1.897298 K** | PASS (~1.887) |
| **hard** val +120 h t2m RMSE | **≤ expand-val raw 2.131516 K** | **FAIL** (~2.132; Δ ≈ +5e-4 K) |
| Eligible composite ckpt | preferred | **0** saves → `fallback_unconstrained_120h` ⇒ auto FAIL |
| Labels | `provisional_years=true`; `claim_level=interim_era5`; `g1_claimable=false` | |

## vs v1.1 / v1.2b

| Item | v1.1 thin / zero-shot expand | **v1.2** | **v1.2b** |
|------|------------------------------|----------|-----------|
| pairs_dir | thin or expand (eval-only) | **expand pairs** (trained) | expand pairs |
| `w_120` / patience | 1.5 / — | 1.5 / 50 | **2.0 / 80** |
| INTERIM PASS | ZS expand YES | **NO** | **YES** |
| out_dir | `v1_1/` / `v1_1_expand/` | **`v1_2/` only** | `v1_2b/` |

## Do not overwrite

`runs/phase0/tier_a/v0/`, `v0_1/`, `v1/`, **`v1_1/`**, **`v1_1_expand/`**, **`v1_2/`** (this FAIL freeze), `tier0_holdout/`, `tier0_holdout_expand/`, G0 verifying. Leave ERA5 PID **611595** alone.

## Quick start (Spark) — reference / reproduce only

```bash
cd ~/fourcastnet
~/fcn3-venv/bin/python code/tier_a/v1_2/train.py --config configs/tier_a_v1_2.yaml --dry-run
# Reproduce train (do not overwrite frozen v1_2 artifacts unless Manisha asks):
# PYTHONUNBUFFERED=1 ~/fcn3-venv/bin/python -u code/tier_a/v1_2/train.py \
#   --config configs/tier_a_v1_2.yaml --train --allow-gpu --epochs 400
```

Outputs: `runs/phase0/tier_a/v1_2/{tier_a_v1_2_results.json,tier_a_v1_2_metrics.csv,best_residual.pt}` (Spark-local weights not in git).
