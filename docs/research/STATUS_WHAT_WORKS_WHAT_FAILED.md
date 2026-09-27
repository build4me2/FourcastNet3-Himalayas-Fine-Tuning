# Status — what works / what failed (Stanford-plain)

**Date:** 2026-09-27 13:45 PT · **Eng:** Howard (FCN on Spark) · **Owner:** Manisha  
**Living residual (FINAL protocol / LOCKED):** `runs/phase0/final_eval/final_residual_v0/` — **do not overwrite.**  
**Prior interim living (historical frozen):** `runs/phase0/tier_a/v1_3_joint/` — **do not overwrite.**  
**Claim:** PASS A∧B∧C per `FINAL_RESIDUAL_V0_CALL.md` · **`g1_claimable=true`** (Leonard unlock 2026-09-27 · `G1_CLAIMABLE_UNLOCK_CALL.md`) — regional FINAL G1 candidate clearing A∧B∧C. Hard non-claims: no global SOTA/ops/CorrDiff/precip/stations; thick-2 bridge report-only; no HF.

## What works

| Piece | One-line |
| --- | --- |
| **G0 verifying** | PASS / claimable — adapters ≤~5% CRPS; PSD hard floor held (`g0_verifying_results.json`). |
| **FINAL protocol** | Locked ICs 320/64/64 · years 1980–2019 / 2020–2021 / 2022–2025 · gate leads 24/72/120. |
| **Living residual (FINAL)** | **`final_residual_v0`** — PASS A∧B∧C · PROMOTE. Pooled t2m RMSE: **val 1.8199** / **test 1.8564** / **val +120h 2.0104**. Wind-vector lead-mean: val **0.8004** / test **0.8250**. |
| **Interim v1.3 joint** | Preserved as **historical interim** (thick-2 16-IC): val **1.770** / test **1.775** / +120h **1.982** · WV **0.69560 / 0.73877**. Still frozen. |
| **Thick-2 protocol** | Holdout **12 / 16 / 16** (ic33–ic44) staged on ARCO — historical beat-this record (interim era). |
| **Year hard-lock (interim)** | Train **2018–2021** / val **2022** / test **2023–2024**. `provisional_years=false`, `year_split_frozen=true`. |

Exact FINAL floats (from call): val t2m **1.819863** / test **1.856390** / val+120h **2.010414** · WV **0.800354 / 0.825004**.

## What failed / null / WARN

| Piece | Verdict |
| --- | --- |
| **v1.2** | **FAIL** (frozen) — +120h gate miss; keep `tier_a/v1_2/` as historical FAIL. |
| **v1-diff** | **NULL** — leftover-target diffusion ≈ living residual (Δ ~1e-5); ensemble worse. **Do not promote.** Stop leftover-target diffusion scaling. |
| **Thick-2 bridge (FINAL)** | **WARN-missing** from `final_residual_v0_results.json` — report-only; **non-blocking** for A∧B∧C / promote. |

## Living path (canonical)

1. Weights: `runs/phase0/final_eval/final_residual_v0/best_residual.pt` (md5 `586ab17b843757bb84e66e2e3af8dc01`)
2. Results: `final_residual_v0_results.json` · metrics CSV · call `FINAL_RESIDUAL_V0_CALL.md`
3. Prior interim `v1_3_joint/` → **historical frozen** (md5 `c81a5a4c4f07ca0a52bb0bffee0045fa`). Prior thick-2 train / ZS / expand → **historical**, still frozen.

## Non-claims (honesty)

- Living FINAL: **`g1_claimable=true`** (unlock call). Allowed: regional FINAL G1 candidate clearing A∧B∧C. Interim thick-2 `v1_3_joint/` stays historical `interim_era5` (not relabeled G1).
- Hard non-claims: no global WB2 / FCN3 SOTA · not ops · not CorrDiff · no precip · no stations · no IMDAA/Canvas from interim bars · no HF.
- Do **not** invent new bars; thick-2 bridge remains report-only.
- Nepal CDS crop is **archive for later**, not IC source (ICs = global ARCO).

## Background

- Status echo 2026-09-25 evening PT: new living = `final_residual_v0/`; interim `v1_3_joint/` preserved.
- Idle on next model / iterate unless Manisha assigns. Optional later: thick-2 bridge score for continuity table.

See also: `LIVING_INDEX.md`, `FCN_STATUS_AND_NEXT.md`, `FINAL_RESIDUAL_V0_CALL.md`, `G1_CLAIMABLE_UNLOCK_CALL.md`, `TIER_A_V1_3_JOINT_CALL.md`, `TIER_A_V1_DIFF_CALL.md`, `YEAR_HARD_LOCK.md`, `ERA5_COVERAGE_AUDIT.md`.
