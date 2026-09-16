# Status — what works / what failed (Stanford-plain)

**Date:** 2026-09-15 PT · **Eng:** Howard (FCN on Spark) · **Owner:** Manisha  
**Living residual (LOCKED):** `runs/phase0/tier_a/v1_2b_thick2_train/` — **do not overwrite.**  
**CDS:** PID **611595** still filling Nepal ERA5 crop — leave alone.

## What works

| Piece | One-line |
| --- | --- |
| **G0 verifying** | PASS / claimable — adapters ≤~5% CRPS; PSD hard floor held (`g0_verifying_results.json`). |
| **Thick-2 protocol** | Holdout **12 / 16 / 16** (ic33–ic44) staged on ARCO; beat-this frozen: val **&lt; 1.988588** / test **&lt; 1.918383** / +120h **≤ 2.178842**. |
| **Year hard-lock** | Train **2018–2021** / val **2022** / test **2023–2024**. `provisional_years=false`, `year_split_frozen=true`. |
| **Living residual** | Tier-A v1.2b thick-2 **retrain** — INTERIM PASS CONFIRM. Pooled t2m RMSE: **val 1.760** / **test 1.780** / **val +120h 1.962**. |

Exact floats (from results JSON): val **1.759883** / test **1.779660** / val+120h **1.962487**.

## What failed / null

| Piece | Verdict |
| --- | --- |
| **v1.2** | **FAIL** (frozen) — +120h gate miss; keep `tier_a/v1_2/` as historical FAIL. |
| **v1-diff** | **NULL** — leftover-target diffusion ≈ living residual (Δ ~1e-5); ensemble worse. **Do not promote.** Stop leftover-target diffusion scaling. |

## Living path (canonical)

1. Weights: `runs/phase0/tier_a/v1_2b_thick2_train/best_residual.pt`
2. Results: `tier_a_v1_2b_thick2_train_results.json`
3. Prior ZS thick-2 (`v1_2b_thick2/`) and expand-train v1.2b (`v1_2b/`) → **historical**, still frozen.

## Non-claims (honesty)

- Label stays **`interim_era5`** (ERA5 interim targets; not IMDAA).
- **`g1_claimable=false`** — no G1 / IMDAA / Canvas claim from these bars.
- Nepal CDS crop is **archive for later**, not IC source (ICs = global ARCO).

## Background

- CDS ERA5 Nepal crop download still running (PID 611595).
- No new architecture train ordered; idle / science writeup / optional test-SON IC only.

See also: `FCN_STATUS_AND_NEXT.md`, `TIER_A_V1_2B_THICK2_TRAIN_CALL.md`, `TIER_A_V1_DIFF_CALL.md`, `YEAR_HARD_LOCK.md`.
