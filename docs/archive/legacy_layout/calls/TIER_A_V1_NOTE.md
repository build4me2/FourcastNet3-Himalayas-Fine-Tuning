# Tier-A v1-reg note — elev-conditioned residual (2026-09-11 PT)

**Parent call:** `docs/calls/TIER_A_V1_SKETCH_CALL.md` (implement **v1-reg first; NO diffusion**).  
**Method:** RRCA-FD Plan A — larger elev/lead-FiLM residual UNet on **frozen FCN3** crop → ERA5 interim.  
**Not this cut:** v1-diff / CorrDiff; FCN3 weight FT; G1/IMDAA.

## Motivation

v0 / v0.1 mechanical pooled PASS was short-lead heavy. Leonard WARN: val **+120 h** raw 2.220 → v0 2.382 / v0.1 2.354 (regress).  
v1-reg freeze adds a **hard** gate: val +120 h t2m RMSE **≤ raw (2.219971 K)**; v0/v0.1 grandfathered. Capacity goes to a larger residual head, not FCN3 FT.

## Model / loss (documented)

| Item | Spec |
|------|------|
| Model | `ElevCondResidualUNet` 3-level, base=32, dropout=0.10, FiLM(lead, mean elev) |
| Params | **665,411** (v0/v0.1 TinyElevResidualUNet ~35k) |
| PhysicsNeMo | stock UNet is **3D** — not used on 21×37 crop |
| Loss | multi-lead elev-weighted residual MSE; **w_24=1, w_72=1, w_120=3.0 (≥1)** |
| Channel weights | t2m=2.0, u10m=0.5, v10m=0.5 (gates are t2m) |
| Ckpt | **composite**: REJECT epoch if val +120 h > raw; else save best lead-mean |
| Early-stop | patience 80 on eligible composite |
| Train split | ic01–ic08 only; val ic09–12 / test ic13–16 |
| Diffusion | **false** |

Config: `configs/tier_a_v1.yaml`.

## Artifacts (Spark `~/fourcastnet`)

| Item | Path |
|------|------|
| **Results JSON** | `runs/phase0/tier_a/v1/tier_a_v1_results.json` |
| Metrics CSV | `runs/phase0/tier_a/v1/tier_a_v1_metrics.csv` |
| Best ckpt (composite) | `runs/phase0/tier_a/v1/best_residual.pt` |
| Unconstrained +120h ckpt | `runs/phase0/tier_a/v1/best_unconstrained_120h.pt` |
| Dry-run | `runs/phase0/tier_a/v1/tier_a_v1_dry_run.json` |
| Log | `logs/tier_a_v1_train400.nohup.out` |
| Code | `code/tier_a/v1/{model,dataset,train}.py` |
| **Frozen v0** | `runs/phase0/tier_a/v0/` — **not overwritten** |
| **Frozen v0.1** | `runs/phase0/tier_a/v0_1/` — **not overwritten** |

## Beat-this (echoed in JSON)

- val t2m pooled RMSE strictly **< 1.948661**
- test strictly **< 1.850338**
- **NEW hard:** val +120 h t2m RMSE **≤ 2.219970831929039**
- `provisional_years=true`; `g1_claimable=false`; G0 adapter N/A (FCN3 frozen)

## Measured (CPU, early-stop ep 180 / best eligible ep 100, ~268 s)

| Split | raw | v0 | v0.1 | **v1-reg** | Bar | Gate |
|-------|----:|---:|-----:|----------:|----:|------|
| val pooled | 1.949 | 1.892 | 1.889 | **1.796** | <1.948661 | **PASS** |
| test pooled | 1.893 | 1.733 | 1.786 | **1.950** | <1.850338 | **FAIL** (worse than raw) |
| val +120 h | 2.220 | 2.382 | 2.354 | **2.018** | ≤2.219971 | **PASS** (−0.202 vs raw) |

### Per-lead val

| Lead | raw | v0 | v0.1 | **v1-reg** |
|------|----:|---:|-----:|----------:|
| +24 h | 1.691 | 1.561 | 1.554 | **1.494** |
| +72 h | 1.901 | 1.620 | 1.658 | **1.837** |
| **+120 h** | **2.220** | 2.382 | 2.354 | **2.018** |

### Per-lead test

| Lead | raw | v0 | v0.1 | **v1-reg** |
|------|----:|---:|-----:|----------:|
| +24 h | 1.784 | 1.571 | 1.624 | **1.706** |
| +72 h | 2.041 | 1.822 | 1.889 | **2.133** |
| +120 h | 1.844 | 1.794 | 1.832 | **1.987** |

## Gates (mechanical)

| Gate | Result |
|------|--------|
| val_pass | **true** |
| test_pass | **false** |
| val_120h_le_raw | **true** |
| **tier_a_interim_pass** | **false** |

Pooled-only win is FAIL; here the miss is **test** (val-only + +120 h win). claim_level remains `interim_era5`.

## Honesty

- Composite + w_120=3.0 **fixed** the val +120 h WARN (2.220 → 2.018) but the larger head **overfit** 4 val ICs (best ep 100); test pooled 1.950 **regresses vs raw 1.893** and misses 1.850338.
- Small-N (8/4/4). Not G1/IMDAA. Not CorrDiff. Not FCN3 weight FT. Not v1-diff.
- Frozen md5 unchanged: v0 `bdbde3ce…`, v0.1 `65f1ec78…`, g0_verifying `806263fd…`, tier0_holdout `f85b7e80…`. ERA5 PID 611595 untouched.

## Suggested Leonard call

(a) Accept v1-reg **FAIL** (test miss; +120 h rule now operational).  
(b) Iterate v1-reg capacity/reg (smaller base, w_120 closer to 1, earlier eligible ckpt) **before** any `v1-diff`.  
(c) Do **not** start diffusion — test bar not met; pathway says debug (1)–(2) first.
