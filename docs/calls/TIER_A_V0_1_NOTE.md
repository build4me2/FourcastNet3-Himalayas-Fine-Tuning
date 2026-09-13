# Tier-A v0.1 note — lead-balanced residual (2026-09-11 PT)

**Parent call:** `docs/calls/TIER_A_V0_CALL.md` (v0 INTERIM PASS YES — FREEZE v0).  
**Method:** RRCA-FD Plan A — same TinyElevResidualUNet (~35k), same IC splits/bars, **FCN3 frozen**.  
**Change:** equal-weight multi-lead elev-weighted MSE + early-stop on lead-mean val RMSE.

## Motivation (Leonard WARN)

v0 pooled PASS was short-lead heavy: val **+120 h** raw 2.220 → Tier-A **2.382** (regress).  
v0.1 objective: do not let +24/+72 dominate the loss / ckpt selection.

## Loss (documented)

For each train batch:

1. Compute elev-weighted residual MSE **separately** for each lead in `{24, 72, 120}` present in the batch.
2. Take the **unweighted mean** of those per-lead losses (equal weight; missing leads skipped / renormalized).
3. Best ckpt selected by `mean(per-lead val t2m RMSE)` (`ckpt_select=lead_mean`), not pooled RMSE.

Config: `configs/tier_a_v0_1.yaml` · `loss=mse_residual_elev_weighted_lead_balanced` · `lead_balance=true`.

## Artifacts (Spark `~/fourcastnet`)

| Item | Path |
|------|------|
| Results JSON | `runs/phase0/tier_a/v0_1/tier_a_v0_1_results.json` |
| Metrics CSV | `runs/phase0/tier_a/v0_1/tier_a_v0_1_metrics.csv` |
| Best ckpt | `runs/phase0/tier_a/v0_1/best_residual.pt` |
| Dry-run | `runs/phase0/tier_a/v0_1/tier_a_v0_1_dry_run.json` |
| Log | `logs/tier_a_v0_1_train300.nohup.out` |
| Code | `code/tier_a/train.py` (shared; v0-compatible) |
| **Frozen v0** | `runs/phase0/tier_a/v0/` — **not overwritten** |

## Beat-this (unchanged; echoed in JSON)

- val t2m pooled RMSE strictly **< 1.948661**
- test strictly **< 1.850338**
- `provisional_years=true`; `g1_claimable=false`; G0 adapter N/A (FCN3 frozen)

## Measured (CPU, 300 ep, ~156 s; best by lead_mean)

| Split | raw | v0 | **v0.1** | bar |
|-------|----:|---:|---------:|----:|
| val pooled | 1.949 | 1.892 | **1.889** | <1.948661 ✓ |
| test pooled | 1.893 | 1.733 | **1.786** | <1.850338 ✓ |

### Per-lead val (headline WARN)

| Lead | raw | v0 | **v0.1** | v0.1 vs raw | v0.1 vs v0 |
|------|----:|---:|---------:|------------:|----------:|
| +24 h | 1.691 | 1.561 | **1.554** | improve | slight improve |
| +72 h | 1.901 | 1.620 | **1.658** | improve | slightly worse than v0 |
| **+120 h** | **2.220** | **2.382** | **2.354** | **still regress** | better than v0 by ~0.027 |

### Per-lead test

| Lead | raw | v0 | **v0.1** |
|------|----:|---:|---------:|
| +24 h | 1.784 | 1.571 | 1.624 |
| +72 h | 2.041 | 1.822 | 1.889 |
| +120 h | 1.844 | 1.794 | 1.832 (slight improve vs raw) |

**Mechanical gates:** `val_pass=true`, `test_pass=true`, `tier_a_interim_pass=true` (claim_level still interim_era5).

## Honesty / non-claims

- +120 h val **still regresses vs raw** under lead-balanced loss — WARN only partially mitigated vs v0.
- Small-N holdout (4+4). ERA5_interim only. Not G1 / IMDAA / CorrDiff / FCN3 FT.
- Not lead-uniform skill. `provisional_years=true`.
- ERA5 PID 611595, G0 verifying, tier0_holdout, and **tier_a/v0** left untouched.

## Ping Leonard

Paths for call: `runs/phase0/tier_a/v0_1/tier_a_v0_1_results.json` (+ metrics CSV / ckpt).  
Compare to frozen v0: `runs/phase0/tier_a/v0/tier_a_v0_results.json`.  
Question: accept mechanical interim PASS with residual +120 h WARN, or iterate (elev-band aux / stronger long-lead weight / RRCA-FD v1)?
