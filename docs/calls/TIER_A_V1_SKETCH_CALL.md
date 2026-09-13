# Tier-A v1 sketch call (Leonard) — 2026-09-11

**Parent:** `TIER_A_V0_1_CALL.md` · Pathway: `FINETUNE_METHOD_DESIGN.md` Phase 2→3 · `FINETUNE_PATHWAY.md`  
**Howard sketch:** staged v1-reg then optional v1-diff; 1× Spark; PhysicsNeMo; no diffusers yet; freeze v0/v0.1.

## Freeze

| Item | Decision |
| --- | --- |
| **First cut** | **v1-reg only** — implement now |
| **Diffusion in first v1 cut?** | **NO** — `v1-diff` only if v1-reg plateaus on bars / +120 h gate with capacity left |
| **FCN3 weight FT / Tier-B** | **Out of scope** for v1 |
| **Full NVIDIA CorrDiff CONUS / globe** | **Not recommended** (agree) |
| **Dirs** | `runs/phase0/tier_a/v1/` · `code/tier_a/v1/` · do not overwrite `v0/` or `v0_1/` |

Matches pathway Phase 2 (regression) before Phase 3 (diffusion).

## v1-reg spec (frozen)

| Field | Spec |
| --- | --- |
| **Role** | Larger elev-conditioned residual regression (UNet or PhysicsNeMo residual) on frozen FCN3 crop → ERA5 interim |
| **Conditioning** | FCN3 crop (+ elev / orography); FCN3 weights **FROZEN**; never backprop into FCN3 |
| **Data** | Reuse `tier0_holdout` pairs; same IC splits; lead-balanced sampling; expand ICs only after Manisha year hard-lock |
| **Loss** | Multi-lead elev-weighted residual MSE (or spatial CRPS if cheap); **explicit +120 h weight ≥ 1** (equal or higher — document `w_120`); carry v0.1 `lead_balance` discipline |
| **Ckpt select** | Prefer lead-mean val RMSE; must also track pooled |
| **Target label** | `claim_level=interim_era5`; `g1_claimable=false`; `provisional_years=true` |
| **Memory** | Fit 1× Spark 128 GB UMA; BF16/batch=1+accum OK; Nepal 21×37 is tiny — capacity goes to depth/channels, not patching yet |

### Gates (v1-reg INTERIM PASS — all required)

1. Val t2m pooled RMSE **&lt; 1.948661 K** (Tier-0 holdout bar).
2. Test t2m pooled RMSE **&lt; 1.850338 K**.
3. **NEW hard:** val **+120 h** t2m RMSE **≤ raw val +120 h** (no regress vs raw; v0/v0.1 grandfathered). Exact raw reference from holdout/raw table: **2.219971 K** (use JSON raw at call time; do not beat a softer number).
4. G0 adapter: **N/A** while FCN3 frozen; base verifying PASS remains the reference.
5. Echo bars + `provisional_years` in result JSON; leave G0 verifying / tier0_holdout / v0 / v0_1 untouched.

### FAIL / iterate
- Misses (1)–(2) → debug data/alignment/capacity before any diffusion.
- Passes pooled but fails (3) → **FAIL v1-reg** (not interim PASS); try higher `w_120` or architecture before `v1-diff`.
- OOM → shrink channels / use PhysicsNeMo lighter residual; do not touch FCN3.

## v1-diff (deferred — optional second)

Only if v1-reg **plateaus** (PASS on bars but skill ceiling clear) **and** GPU time available:
- CorrDiff-lite / residual diffusion on Nepal crop (patch diffusion / smaller window if needed).
- New subdir `runs/phase0/tier_a/v1_diff/`; same beat-this + +120 h rule on **ensemble-mean or deterministic decode** plus separate CRPS/spread table.
- Kill per pathway: unstable / no lift → keep regression; still valid Tier-A story.

## Explicit non-goals (v1)
- Tier-B / LoRA on FCN3  
- IMDAA until access + year hard-lock (then upgrade target; re-score)  
- Claiming CorrDiff parity with NVIDIA CONUS  
- Precip-first (t2m/winds still primary per Manisha Tier-0 lock)

## Next eng (Howard)

1. Implement **v1-reg** under `code/tier_a/v1/` + `runs/phase0/tier_a/v1/`.
2. Document `w_120` and model size in config YAML.
3. Smoke → full train; ping Leonard with `tier_a_v1_results.json` for PASS/FAIL.
4. Do **not** start `v1-diff` until Leonard calls v1-reg plateau or Manisha orders diffusion early.

## Answer to Howard’s question
**OK to implement v1-reg first. Do not put diffusion in the first v1 cut.**
