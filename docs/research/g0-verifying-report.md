# G0 verifying report — CRPS / SSR / PSD vs ERA5

**Date:** 2026-09-11 PT (completed ~5:55 PM PT)  
**Host:** Spark `chandmanisha00@100.121.160.49` (spark-61dd)  
**Priority:** Leonard #1 — wire real verifying metrics so `g0_pass_claimable` can be evaluated  

## Verdict

| Layer | Result |
| --- | --- |
| Implementation | **DONE** — streaming lead-only scorer in `code/phase0/g0_verifying_era5.py` |
| Smoke | **OK** — 1 IC × 2 mem × +120 h |
| Full | **OK** — 8/8 ICs × 4 mem × 60 steps; leads +120/+240/+360 h |
| Metrics | **REAL** (not stubs) — fair CRPS, SSR, zonal PSD vs ARCO ERA5 |
| `g0_pass_claimable` | **TRUE** (base GATE §1.5: finite + SSR/PSD sane + protocol complete) |
| Leonard ~5% CRPS rule | **N/A for base** — rule is adapter ≤~5% degrade **vs this baseline** |
| ERA5 regional PID 611595 | **Untouched** (alive through run) |

## Leonard-pingable paths

| Artifact | Path |
| --- | --- |
| Verifying results | `~/fourcastnet/runs/phase0/g0/g0_verifying_results.json` |
| Sidecar (claimable pointer) | `~/fourcastnet/runs/phase0/g0/g0_base_results_verifying_sidecar.json` |
| Stub plumbing (preserved) | `~/fourcastnet/runs/phase0/g0/g0_base_results.json` (`crps_ssr_psd=stub`) |
| Smoke copy | `~/fourcastnet/runs/phase0/g0/g0_verifying_smoke_results.json` |
| ERA5 verify frames | `~/fourcastnet/data/g0_verify/` (24 npz) |
| Partial per-IC | `~/fourcastnet/runs/phase0/g0/partial_verify/ic0{1–8}.json` |
| Full log | `~/fourcastnet/logs/g0_verifying_full.nohup.out` |
| Code | `~/fourcastnet/code/phase0/g0_verifying_era5.py` |

## Method (memory-safe)

1. Fetch ARCO ERA5 preview vars (`t2m,u10m,v10m,tcwv,z500,t850`) at score leads only.  
2. Re-roll frozen FCN3 with `preview_only` / **lead-only** keep (no 61×72×721×1440×M ≈ 73 GB).  
3. Score online after 4 members per IC:  
   - Fair ensemble CRPS (area-weighted; midlat 20–70°)  
   - SSR = spread / RMSE(ensemble mean)  
   - Zonal PSD synoptic ratio (k=4..20) ens-mean vs ERA5  

Peak during full: CUDA alloc **52.9 GB**, reserved **69.4 GB**, RSS **9.9 GB**. Rollout wall **9971 s** (~2.77 h) + ~38 min ERA5 fetch.

## Measured — 8-IC mean midlat

### +15 d (360 h) — primary gate lead

| Var | CRPS midlat | SSR midlat | PSD ratio | Sane |
| --- | ---: | ---: | ---: | --- |
| t2m | **1.277** | 0.666 | 0.712 | yes |
| u10m | 2.312 | 0.727 | 0.381 | yes* |
| v10m | 2.345 | 0.752 | 0.406 | yes* |
| tcwv | 2.983 | 0.777 | 0.451 | yes* |
| z500 | **440.1** | 0.732 | 0.438 | yes* |
| t850 | **1.971** | 0.787 | 0.515 | yes |

\*PSD winds/z500 ~0.38–0.44 is inside coded sane band [0.25, 4] but **borderline** toward spectral blur — flag for Leonard.

### +10 d (240 h)

| Var | CRPS | SSR | PSD |
| --- | ---: | ---: | ---: |
| t2m | 1.054 | 0.684 | 0.768 |
| z500 | 324.3 | 0.752 | 0.517 |
| t850 | 1.584 | 0.792 | 0.554 |

### +5 d (120 h)

| Var | CRPS | SSR | PSD |
| --- | ---: | ---: | ---: |
| t2m | 0.673 | 0.637 | 0.928 |
| z500 | 157.1 | 0.635 | 0.884 |
| t850 | 0.893 | 0.696 | 0.866 |

Smoke (ic01, M=2, +120 h) matched order-of-magnitude: t2m CRPS 0.692, SSR 0.536.

## Aggregate gate fields

```text
g0_pass_claimable: true
crps_ssr_psd: real
all_finite: true
ssr_sane_all: true
psd_sane_all: true
protocol_complete: true  (≥8 ICs, primary lead 360)
```

## Context notes (steering)

- Provisional years: train **2018–2021** / val **2022** / test **2023–2024**  
- Tier-0 interim beat-this: t2m RMSE **< 1.978 K** (holdout re-score required)  
- Stub `g0_base_results.json` remains the finite-plumbing reference; **skill baseline for adapters is `g0_verifying_results.json`**

## Honesty / blockers

- **No blockers for claimable base metrics path** — real CRPS/SSR/PSD on disk.  
- Adapter PASS still requires ≤~5% CRPS degrade vs these numbers.  
- PSD at +15 d on winds/z500 is borderline low energy — not a coded FAIL, but worth Leonard eyes before treating spectra as strong.  
- Units: z500 CRPS in geopotential (m²/s²) as in ARCO/FCN3, not geopotential height meters.

## Next

1. Ping Leonard with paths above + +15 d table.  
2. Use verifying JSON as frozen base for Tier-A/B adapter G0 compares.  
3. Holdout re-score Tier-0 against provisional year split when ready.
