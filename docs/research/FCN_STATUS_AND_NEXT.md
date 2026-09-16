# FCN3 Phase 0 — status & next (Leonard) — 2026-09-14

**Manisha:** work on FourCastNet (Howard = FCN-only; TTA parked).

## Frozen state (do not overwrite)

| Layer | Status | Path / bar |
| --- | --- | --- |
| G0 verifying | **PASS claimable** | `g0_verifying_results.json`; adapters ≤+5% CRPS; PSD hard &lt;0.25 |
| Tier-0 thin | historical | val 1.948661 / test 1.850338 |
| Tier-0 expand-v1 | historical | val 1.987014 / test 1.897298 / +120h ≤ 2.131516 |
| Tier-0 **thick-2** | **living beat-this** | val **&lt; 1.988588** / test **&lt; 1.918383** / +120h ≤ **2.178842** |
| Tier-A v0 | best **test pooled** champ (1.733 thin) | `tier_a/v0/` |
| Tier-A v1.1 | first +120h-compliant PASS (thin) | `tier_a/v1_1/` |
| Tier-A v1.1 expand ZS | INTERIM PASS (expand-v1 hist.) | val 1.808 / test 1.826 / +120h 1.998 |
| Tier-A v1.2b | INTERIM PASS (expand train) → **historical** | `tier_a/v1_2b/` · still frozen |
| Tier-A v1.2b **thick-2 ZS** | INTERIM PASS → **historical** (ZS demoted) | val 1.829 / test 1.897 / +120h 2.015 · `tier_a/v1_2b_thick2/` |
| Tier-A v1.2b **thick-2 retrain** | **INTERIM PASS CONFIRM YES** · **living residual** | val **1.760** / test **1.780** / +120h **1.962** · `tier_a/v1_2b_thick2_train/` |
| Tier-A **v1-diff** | **NULL — do not promote** | `tier_a/v1_diff/` · keep as null artifact; no material lift vs living |
| Years | **HARD-LOCKED** 2018–21 / 2022 / 2023–24 | `provisional_years=false`; see YEAR_HARD_LOCK.md |
| Labels | `interim_era5`; `g1_claimable=false` | — |

Calls: `TIER_A_V1_DIFF_CALL.md` (NULL), `TIER_A_V1_2B_THICK2_TRAIN_CALL.md` (living), `HOLDOUT_THICK2_CALL.md` (bars), `HOLDOUT_EXPAND_CALL.md` (historical), `TIER_A_V1_2B_CALL.md`, `G0_VERIFYING_CALL.md`.

## Living residual (LOCKED)

| Role | Path |
| --- | --- |
| **Living (canonical)** | `runs/phase0/tier_a/v1_2b_thick2_train/` (`best_residual.pt`) |
| v1-diff (null artifact) | `runs/phase0/tier_a/v1_diff/` · **do not delete** · do not promote |
| Prior expand-train v1.2b | `v1_2b/` → historical (still frozen) |
| Thick-2 ZS | `v1_2b_thick2/` → historical (ZS demoted) |
| v1.2 FAIL | `v1_2/` → keep |

Do **not** overwrite living residual or v1_diff results.

## Recommended next (default)

**Idle pending Manisha.** Living residual = `v1_2b_thick2_train/`.  
**v1-diff = NULL** — stop leftover-target diffusion scaling; do not promote.  
Await a **different** Phase-3 fork if assigned (not more leftover-target scaling).  
Keep CDS **611595** alone. No GPU.

### Living bars (echo in future result JSON)
- val t2m pooled RMSE lin strictly **&lt; 1.988588**
- test t2m pooled RMSE lin strictly **&lt; 1.918383**
- val +120h t2m RMSE **≤ 2.178842** (thick-2 val raw)

### v1-diff null (headline)
- one_step 0-noise: val ~1.7598 / test ~1.7799 / +120h ~1.962 (≈ living; Δ ~1e-5)
- ens K=4: **worse** (val ~1.937 / test ~1.965)
- `diff_improves_residual=false`

## Alternatives (if Manisha overrides)

| Option | When |
| --- | --- |
| **Different Phase-3 fork** | Explicit assign — e.g. target ERA5−FCN3 without residual mean, winds/CRPS-first, or pause writeup — **not** more leftover-target scaling |
| **More ICs (test SON)** | Recipe-perfect seasons without reopening architecture |
| **Winds/elev-band deep dive** | Science writeup without new train |
| **Pause eng** | Status-only; Howard idle on FCN (current default) |

## Non-goals until ordered
- TTA harness (Howard not assigned)
- FCN3 weight FT / Tier-B
- IMDAA / G1 claims
- Scaling leftover-target diffusion further (STOP)
- Overwriting frozen artifacts (incl. living `v1_2b_thick2_train/`, null `v1_diff/`, `v1_2b/`, `v1_2b_thick2/`)

## Next eng (Howard)
1. Mirror Leonard `TIER_A_V1_DIFF_CALL.md` · status: living = `v1_2b_thick2_train/`; v1-diff = NULL — **done**.
2. No GPU; leave ERA5 CDS PID 611595 alone.
3. Idle pending Manisha (different Phase-3 fork only if assigned).
