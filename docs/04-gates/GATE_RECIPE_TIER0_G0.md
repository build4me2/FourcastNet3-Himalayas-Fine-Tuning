# Gate recipe — Tier-0 bias + G0 IC protocol (Leonard)
**Frozen for Howard:** 2026-09-11  
**Smoke context:** FCN3 4×16 on Spark OK (~53 GB peak); Nepal crop 21×37 from global rollout. **Random IC used in smoke because pulled crops are regional-only — that is NOT valid for G0 or skill claims.**

**Companions:** `FINETUNE_PATHWAY.md`, `FINETUNE_METHOD_DESIGN.md`, `SPARK_FINETUNE_PLAN.md`

---

## 0. Hard rule (IC honesty)

| Use case | Allowed IC | Forbidden |
| --- | --- | --- |
| **Phase 0 smoke / plumbing** | Random / synthetic OK | — |
| **G0 integrity** | **Global** ERA5 (or ERA5T) full 721×1440×72 (+aux as package requires) | Nepal-only crop as IC; Random |
| **Tier-0 / G1 regional** | Global FCN3 rollout from **global ERA5 IC**, then crop to box **OR** FCN3 fed global IC then diagnose on crop | Treating regional zarr *as* FCN3 IC |
| **Any published skill number** | Documented ERA5 (or named reanalysis) IC + seed list | Random IC |

If only regional ERA5 crops exist on disk → **Stage global ICs** (sparse dates) before G0/Tier-0 scoring. Do not improvise Random for gates.

---

## 1. G0 — Integrity probe (IC protocol)

### 1.1 Purpose
Detect catastrophic forgetting / spectral collapse / HMM break **outside** the Nepal box after any adapter (and establish **base** reference now, before Tier-A/B).

### 1.2 IC set (freeze this list in CLAUDE.md once dates exist)

| Field | Spec |
| --- | --- |
| **Domain** | **Full globe** 0.25° (721×1440), 72 FCN3 channels |
| **Source** | ERA5 (prefer final) via CDS/ARCO — **not** Nepal crop files |
| **Count** | **N ≥ 8** fixed ICs: ≥4 **non-Asia** (e.g. N Atlantic, N Pacific, S Hemisphere midlat, tropics Atlantic/Africa) + ≥2 Asia-ex-Nepal + ≥2 optional monsoon-adjacent **outside** locked Nepal box |
| **Season** | Mix DJF/JJA (min 2 each) |
| **Init hour** | 00 UTC (document if 06/12 used) |
| **Seeds / members** | Same M as smoke if possible (e.g. 4); record HMM seeds |
| **Lead** | Evaluate through **+15 d** (60 × 6 h steps) |

**Storage:** `data/g0_ics/ic_YYYYMMDDTHHMM_global.npy` (or zarr) + `g0_ic_manifest.json` (lat/lon full, channels, source URL/hash).

### 1.3 Global vs crop (what Howard asked)

| Path | What it is | When |
| --- | --- | --- |
| **A — Global IC → global rollout → global metrics** | True G0 | **Required** for G0 pass/fail |
| **B — Global IC → global rollout → Nepal crop diagnostics** | Sanity that crop pipeline matches smoke geometry (21×37) | Optional side table; **not** a substitute for G0 |
| **C — Regional crop as IC** | Invalid for FCN3 spherical state | **Never** for G0/G1 skill |

**Pass G0 only on path A.** Path B may be logged as “crop extract OK.”

### 1.4 Metrics (base FCN3 first; later vs adapted)

| Metric | Pass (ε provisional) | Fail |
| --- | --- | --- |
| Spatial CRPS @ +15 d (global or NH+SH midlat bands) | ≤ **~5%** relative degrade vs **this same IC set** on frozen base ckpt | >5% or NaNs |
| SSR / spread–skill @ +5–15 d | No collapse to near-zero spread; no blow-up | Pathological ranks |
| Angular / zonal PSD vs ERA5 @ +15 d | No MSE-blur spectral collapse (I1/I5) | Obvious energy loss at synoptic scales |
| HMM identity | 1 forward / member / step; wall-clock sane | Iterative diffusion substituted for global step |
| Finite fields | All channels finite through +15 d | Inf/NaN |

**Baseline run:** Execute G0 on **frozen `nvidia/fourcastnet3`** with the IC manifest **before** any Tier-A/B. Save `g0_base_results.json`. Later adapters compared to **this** file (not to Random-IC smoke).

### 1.5 Success / fail

| Call | Rule |
| --- | --- |
| **G0 PASS (base)** | Path A completes; metrics finite; PSD/SSR sane on all ≥8 ICs |
| **G0 FAIL (base)** | Any IC NaNs / OOM / spectral nonsense → **stop training plans**; fix inference |
| **G0 PASS (adapter)** | ≤~5% CRPS degrade vs `g0_base_results.json`; no spectral collapse |
| **G0 FAIL (adapter)** | Discard Tier-B adapters; keep Tier-A only if G0 was never violated by A (A should not touch FCN3 weights) |

---

## 2. Tier-0 — Elevation-aware bias baseline

### 2.1 Purpose
Cheap **I6(b)** systematic bias corrector. Tier-A **must beat** this. Always publish even if weak.

### 2.2 Inputs

| Item | Spec |
| --- | --- |
| **Forecast** | Frozen FCN3 from **global ERA5 IC** (same rule as §0), ensemble mean or per-member then mean |
| **Crop** | Locked Nepal box (until Manisha freezes: provisional pathway example 26–31°N, 80–89°E — **confirm**) |
| **Targets** | Prefer **IMDAA** (or stations); interim OK: ERA5 crop for plumbing only — **label clearly** (not obs-aware G1) |
| **Elevation** | GLO-30 / GMTED / package DEM on same grid; bins e.g. <1.5 km, 1.5–3, 3–4.5, >4.5 km |
| **Train / val / test years** | Use Manisha lock when set; until then **do not** claim G1 — only Tier-0 plumbing table |

### 2.3 Model (keep tiny)

**Primary:** elevation-binned **linear** bias per variable (t2m, 10 m u/v or speed/dir, tcwv or q850):  
\(\hat{y} = y_{\mathrm{FCN3}} + a_{b(z)} + b_{b(z)}\,z\) or per-bin constant bias.

**Optional stretch:** small MLP/UNet bias (still no FCN3 backprop). If used, report **separately**; headline Tier-0 = linear/elevation-binned.

**Variables v1:** t2m, 10 m winds; moisture if easy. **Precip:** optional; FCN3 may lack native tp — don’t block Tier-0 on precip.

### 2.4 Protocol

1. Build paired (FCN3 crop, target, elev) at leads **+24 h, +72 h, +120 h** (min).  
2. Fit bias on **train years only**.  
3. Score val/test: RMSE, bias, ACC; **elevation-banded t2m bias**.  
4. Artifact: `tier0_bias_maps.nc` + `tier0_metrics.csv` + config hash.

### 2.5 Success / fail

| Call | Rule |
| --- | --- |
| **Success** | Tables + maps delivered; elevation-banded t2m bias reported |
| **Kill** | **None** for Tier-0 (pathway: always publish) |
| **Warning** | If only ERA5-as-target, mark `target=ERA5_interim` — cannot use as G1 obs-aware pass |

---

## 3. Implementation order for Howard

| Step | Deliverable |
| --- | --- |
| **1** | Stage **≥8 global ERA5 ICs** + `g0_ic_manifest.json` |
| **2** | Run **G0 base** (path A) → `g0_base_results.json` |
| **3** | Optional path B crop check vs smoke 21×37 |
| **4** | Tier-0 fit on global-IC→crop forecasts (ERA5 target OK if labeled interim) |
| **5** | Ping Leonard with paths; Leonard freezes numeric thresholds if needed after first real numbers |
| **6** | Manisha locks box + years → upgrade Tier-0 target to IMDAA and enable G1 design |

---

## 4. Explicit non-goals (this recipe)

- Tier-A/B training  
- Claiming G1 without IMDAA/obs and locked years  
- Using Random IC for any gate number  
- Mixing Idea1 lake/watchlist code (wiped)

---

## 5. Decisions still needed from Manisha (direct)

1. Freeze **lat–lon box** (confirm or replace 26–31°N, 80–89°E).  
2. Freeze **train/val/test years**.  
3. Tier-0 v1 target: allow **ERA5 interim** for plumbing, or wait for **IMDAA** before any Tier-0 fit?

Until (1)–(2), Howard can still do **steps 1–2 (G0 base)** and Tier-0 **plumbing** with labeled interim targets.

---

## 6. Locks from Manisha (2026-09-11 via Howard)

| Item | Lock |
| --- | --- |
| G0 GPU | **Greenlit now** |
| Regional box | **26–31°N, 80–89°E** |
| Tier-0 v1 variables | **t2m / winds first**; precip deferred |
| Tier-0 target | **ERA5 interim OK** (label `target=ERA5_interim`) |
| Train/val/test years | **Later** — no G1 claim until locked |

