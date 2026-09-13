# Tier-A v1-reg — elev-conditioned residual (NO diffusion)

**Call:** Leonard `TIER_A_V1_SKETCH_CALL.md` — implement **v1-reg first**. No `v1-diff`.
**Method:** RRCA-FD Plan A. FCN3 weights **frozen**. Nepal crop 21×37 → ERA5 interim.

## Gates (all required for INTERIM PASS)

| Gate | Bar |
|------|-----|
| val 2022 t2m pooled RMSE | strictly **< 1.948661 K** |
| test 2023–24 t2m pooled RMSE | strictly **< 1.850338 K** |
| **NEW hard** val +120 h t2m RMSE | **≤ raw 2.219971 K** (no regress vs raw; v0/v0.1 grandfathered) |
| Labels | `provisional_years=true`; `claim_level=interim_era5`; `g1_claimable=false` |
| G0 adapter | N/A while FCN3 frozen |

Pooled-only win with +120 h regress = **FAIL v1-reg**.

## Loss / ckpt

- Multi-lead elev-weighted residual MSE with **`w_120 = 3.0` (≥ 1)**; `w_24 = w_72 = 1.0`.
- Composite ckpt: **REJECT** any epoch with val +120 h t2m RMSE > raw; among eligible, save best lead-mean (pooled tracked).
- Early-stop patience 80 on eligible composite.

## Do not overwrite

`runs/phase0/tier_a/v0/`, `runs/phase0/tier_a/v0_1/`, `runs/phase0/g0/g0_verifying_results.json`, `runs/phase0/tier0_holdout/`. Leave ERA5 PID **611595** alone.

## Quick start (Spark)

```bash
cd ~/fourcastnet
~/fcn3-venv/bin/python code/tier_a/v1/train.py --config configs/tier_a_v1.yaml --dry-run
~/fcn3-venv/bin/python code/tier_a/v1/train.py --config configs/tier_a_v1.yaml --smoke
PYTHONUNBUFFERED=1 ~/fcn3-venv/bin/python -u code/tier_a/v1/train.py \
  --config configs/tier_a_v1.yaml --train --epochs 400
```

Outputs: `runs/phase0/tier_a/v1/{tier_a_v1_results.json,tier_a_v1_metrics.csv,best_residual.pt}`.
