# Tier-A v1.1 note — smaller/regularized residual (2026-09-11 PT)

**Parent call:** `docs/calls/TIER_A_V1_CALL.md` (v1-reg **FAIL** on test; iterate v1.1; **NO diffusion**).  
**Method:** RRCA-FD Plan A — smaller/regularized elev/lead-FiLM residual UNet on **frozen FCN3** crop → ERA5 interim.  
**Not this cut:** v1-diff / CorrDiff; FCN3 weight FT; G1/IMDAA.

## Motivation

v1-reg (base 32 / dropout 0.10 / w_120=3.0 / ~665k) passed val + **hard +120 h** but **overfit** 4 val ICs: test pooled **1.950** (worse than raw 1.893; miss bar 1.850338). Leonard: freeze `v1/` as FAIL; iterate smaller/regularized under `v1_1/`.

## Model / loss (documented)

| Item | Spec |
|------|------|
| Model | `ElevCondResidualUNet` 3-level, **base=16**, **dropout=0.20**, FiLM(lead, mean elev) |
| Params | **167,843** (v1 665,411; v0/v0.1 TinyElevResidualUNet ~35k) |
| PhysicsNeMo | stock UNet is **3D** — not used on 21×37 crop |
| Loss | multi-lead elev-weighted residual MSE; **w_24=1, w_72=1, w_120=1.5 (≥1)** |
| Channel weights | t2m=2.0, u10m=0.5, v10m=0.5 (gates are t2m) |
| weight_decay | **1e-3** (v1 was 3e-4; Leonard ≥1e-3) |
| Ckpt | **composite**: REJECT epoch if val +120 h > raw; else save best lead-mean |
| Early-stop | patience **50** on eligible composite (v1 was 80) |
| Train split | ic01–ic08 only; val ic09–12 / test ic13–16 |
| Diffusion | **false** |

Config: `configs/tier_a_v1_1.yaml`.

## Artifacts (Spark `~/fourcastnet`)

| Item | Path |
|------|------|
| **Results JSON** | `runs/phase0/tier_a/v1_1/tier_a_v1_1_results.json` |
| Metrics CSV | `runs/phase0/tier_a/v1_1/tier_a_v1_1_metrics.csv` |
| Best ckpt (composite) | `runs/phase0/tier_a/v1_1/best_residual.pt` |
| Unconstrained +120h ckpt | `runs/phase0/tier_a/v1_1/best_unconstrained_120h.pt` |
| Dry-run | `runs/phase0/tier_a/v1_1/tier_a_v1_1_dry_run.json` |
| Log | `logs/tier_a_v1_1_train400.nohup.out` |
| Code | `code/tier_a/v1_1/{model,dataset,train}.py` |
| **Frozen v1 FAIL** | `runs/phase0/tier_a/v1/` — **not overwritten** |
| **Frozen v0 / v0.1** | `runs/phase0/tier_a/v0/` · `v0_1/` — **not overwritten** |

## Beat-this (echoed in JSON)

- val t2m pooled RMSE strictly **< 1.948661**
- test strictly **< 1.850338**
- **hard:** val +120 h t2m RMSE **≤ 2.219970831929039**
- `provisional_years=true`; `g1_claimable=false`; G0 adapter N/A (FCN3 frozen)

## Measured (CPU, early-stop ep 283 / best eligible ep 233, 14 eligible saves, ~252 s)

| Split | raw | v0 | v0.1 | v1-reg | **v1.1** | Bar | Gate |
|-------|----:|---:|-----:|-------:|---------:|----:|------|
| val pooled | 1.949 | 1.892 | 1.889 | 1.796 | **1.716** | <1.948661 | **PASS** |
| test pooled | 1.893 | 1.733 | 1.786 | 1.950 | **1.812** | <1.850338 | **PASS** |
| val +120 h | 2.220 | 2.382 | 2.354 | 2.018 | **2.015** | ≤2.219971 | **PASS** (−0.205 vs raw) |

### Per-lead val

| Lead | raw | v0 | v0.1 | v1-reg | **v1.1** |
|------|----:|---:|-----:|-------:|---------:|
| +24 h | 1.691 | 1.561 | 1.554 | 1.494 | **1.475** |
| +72 h | 1.901 | 1.620 | 1.658 | 1.837 | **1.612** |
| **+120 h** | **2.220** | 2.382 | 2.354 | 2.018 | **2.015** |

### Per-lead test

| Lead | raw | v0 | v0.1 | v1-reg | **v1.1** |
|------|----:|---:|-----:|-------:|---------:|
| +24 h | 1.784 | 1.571 | 1.624 | 1.706 | **1.586** |
| +72 h | 2.041 | 1.822 | 1.889 | 2.133 | **1.828** |
| +120 h | 1.844 | 1.794 | 1.832 | 1.987 | **2.000** (still > raw) |

## Gates (mechanical)

| Gate | Result |
|------|--------|
| val_pass | **true** |
| test_pass | **true** |
| val_120h_le_raw | **true** |
| **tier_a_interim_pass** | **true** |

claim_level remains `interim_era5`. First cut that meets **all three** v1-reg gates (v0/v0.1 grandfathered on +120 h; v1 FAIL on test).

## Honesty

- Smaller/regularized head **fixed the v1 test miss** (1.950 → 1.812) while keeping +120 h (2.015 ≤ 2.220).
- Test pooled **1.812 still worse than v0 1.733 / v0.1 1.786** (v0 remains best test-pooled on the shelf) but under bar.
- Test **+120 h still regresses vs raw** (2.000 vs 1.844) — not a gate; documented.
- Small-N (8/4/4). Not G1/IMDAA. Not CorrDiff. Not FCN3 weight FT. Not v1-diff.
- Frozen md5 unchanged: v0 `bdbde3ce…`, v0.1 `65f1ec78…`, v1 `4ea36cde…`, g0_verifying `806263fd…`, tier0_holdout `f85b7e80…`. ERA5 PID 611595 untouched.

## Suggested Leonard call

(a) Accept v1.1 **INTERIM PASS** (all three gates; claim_level still interim_era5).  
(b) Freeze `v1_1/` as the first +120 h-compliant PASS.  
(c) Do **not** start diffusion — this is a pass, not a plateau; pathway still says more ICs / year hard-lock before Phase 3.  
(d) Optional next: more val/test ICs (N=4 is binding) or leave-one-IC-out early-stop proxy — still never tune on test.
