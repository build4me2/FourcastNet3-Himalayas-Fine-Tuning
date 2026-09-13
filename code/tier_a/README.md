# Tier-A — frozen FCN3 + residual UNet (v0 / v0.1 / v1-reg / v1.1)

**Method:** RRCA-FD Plan A (`FINETUNE_METHOD_DESIGN.md`). FCN3 weights **frozen**.
**Calls:** Leonard `TIER0_HOLDOUT_CALL.md` (GO) · `TIER_A_V0_CALL.md` (v0 INTERIM PASS) · `TIER_A_V1_SKETCH_CALL.md` (v1-reg first; no diffusion) · `TIER_A_V1_CALL.md` (v1 FAIL; v1.1 iterate).

## Beat-this (FROZEN — echoed in result JSON)

| Gate | Bar |
|------|-----|
| val 2022 t2m pooled RMSE | strictly **< 1.948661 K** (headline < 1.949) |
| test 2023–24 | strictly **< 1.850338 K** (val-only = FAIL) |
| G0 adapter | N/A while FCN3 frozen |
| Labels | `provisional_years=true`; `claim_level=interim_era5`; `g1_claimable=false` |

**Do not overwrite:** `runs/phase0/g0/g0_verifying_results.json`, `runs/phase0/tier0_holdout/`, **`runs/phase0/tier_a/v0/`** (frozen after Leonard PASS).

## Variants

| ID | Config | Loss | Outputs |
|----|--------|------|---------|
| **v0** | `configs/tier_a_v0.yaml` | elev-weighted pooled MSE | `runs/phase0/tier_a/v0/` |
| **v0.1** | `configs/tier_a_v0_1.yaml` | **equal-weight multi-lead** elev-weighted MSE (+24/+72/+120h); ckpt by lead-mean val RMSE | `runs/phase0/tier_a/v0_1/` |
| **v1-reg** | `configs/tier_a_v1.yaml` | **w_120=3.0** multi-lead; composite reject val +120h>raw; ElevCondResidualUNet ~665k | `runs/phase0/tier_a/v1/` (**FAIL freeze**) |
| **v1.1** | `configs/tier_a_v1_1.yaml` | **w_120=1.5**; dropout 0.20; base 16; composite +120h reject; ~168k | `runs/phase0/tier_a/v1_1/` |

### v0.1 lead-balanced loss (documented)

For each batch: compute elev-weighted residual MSE **separately per lead** present among `{24,72,120}`, then take the **unweighted mean** of those lead losses. Missing leads in a batch are skipped (renormalize). Early-stop / best-ckpt uses `mean(per-lead val t2m RMSE)` instead of pooled RMSE so +120h cannot be silently sacrificed for short-lead pooled wins (Leonard WARN: val +120h 2.220→2.382 under v0).

## Quick start (Spark)

```bash
cd ~/fourcastnet
# v0.1 dry-run
~/fcn3-venv/bin/python code/tier_a/train.py --config configs/tier_a_v0_1.yaml --dry-run
# v0.1 smoke (CPU)
~/fcn3-venv/bin/python code/tier_a/train.py --config configs/tier_a_v0_1.yaml --smoke
# v0.1 full train (leave ERA5 PID 611595 alone; GPU optional)
~/fcn3-venv/bin/python code/tier_a/train.py --config configs/tier_a_v0_1.yaml --train --epochs 300
```

## Outputs (v0.1)

`runs/phase0/tier_a/v0_1/` — `best_residual.pt`, `tier_a_v0_1_results.json` (bars echoed), `tier_a_v0_1_metrics.csv`.

## v1-reg (2026-09-11)

See `code/tier_a/v1/README.md` and `docs/calls/TIER_A_V1_NOTE.md`. **Do not overwrite** `v0/` or `v0_1/`. No `v1-diff`.

## v1.1 (2026-09-11)

See `code/tier_a/v1_1/README.md` and `docs/calls/TIER_A_V1_1_NOTE.md`.
**Do not overwrite** `v0/`, `v0_1/`, or **`v1/`**. No `v1-diff`.

```bash
cd ~/fourcastnet
PYTHONUNBUFFERED=1 ~/fcn3-venv/bin/python -u code/tier_a/v1_1/train.py \
  --config configs/tier_a_v1_1.yaml --train --epochs 400
```
