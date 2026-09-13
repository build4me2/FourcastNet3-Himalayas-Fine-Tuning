# Tier-A v1.1 — smaller/regularized elev-conditioned residual (NO diffusion)

**Call:** Leonard `TIER_A_V1_CALL.md` — v1-reg **FAIL** (test overfit). Iterate **v1.1**. No `v1-diff`.
**Method:** RRCA-FD Plan A. FCN3 weights **frozen**. Nepal crop 21×37 → ERA5 interim.

## Gates (all required for INTERIM PASS) — same as v1

| Gate | Bar |
|------|-----|
| val 2022 t2m pooled RMSE | strictly **< 1.948661 K** |
| test 2023–24 t2m pooled RMSE | strictly **< 1.850338 K** |
| **hard** val +120 h t2m RMSE | **≤ raw 2.219971 K** (v0/v0.1 grandfathered; v1 FAIL frozen) |
| Labels | `provisional_years=true`; `claim_level=interim_era5`; `g1_claimable=false` |
| G0 adapter | N/A while FCN3 frozen |

Pooled-only win with +120 h regress = **FAIL v1.1**.

## vs v1-reg (frozen FAIL)

| Knob | v1 | **v1.1** |
|------|---:|----------|
| `base_channels` | 32 | **16** (Leonard 16 or 24; smaller/regularized) |
| dropout | 0.10 | **0.20** |
| `w_120` | 3.0 | **1.5** (still ≥1) |
| weight_decay | 3e-4 | **1e-3** |
| patience | 80 | **50** |
| composite +120 h reject | yes | **yes** |

## Do not overwrite

`runs/phase0/tier_a/v0/`, `v0_1/`, **`v1/`**, `runs/phase0/g0/g0_verifying_results.json`, `runs/phase0/tier0_holdout/`. Leave ERA5 PID **611595** alone.

## Quick start (Spark)

```bash
cd ~/fourcastnet
~/fcn3-venv/bin/python code/tier_a/v1_1/train.py --config configs/tier_a_v1_1.yaml --dry-run
PYTHONUNBUFFERED=1 ~/fcn3-venv/bin/python -u code/tier_a/v1_1/train.py \
  --config configs/tier_a_v1_1.yaml --train --epochs 400
```

Outputs: `runs/phase0/tier_a/v1_1/{tier_a_v1_1_results.json,tier_a_v1_1_metrics.csv,best_residual.pt}`.
