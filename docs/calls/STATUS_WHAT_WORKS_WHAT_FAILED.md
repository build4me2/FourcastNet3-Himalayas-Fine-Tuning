# Status — what works / what failed (Stanford-plain)

**Date:** 2026-09-21 PT · **Eng:** Howard (FCN on Spark) · **Owner:** Manisha  
**Living residual (LOCKED):** `runs/phase0/tier_a/v1_3_joint/` — **do not overwrite.**  
**CDS:** Nepal ERA5 CDS crop still filling — leave download alone (check `ps` / `logs/era5_pull.log` for live PID; do not hardcode).

## What works

| Piece | One-line |
| --- | --- |
| **G0 verifying** | PASS / claimable — adapters ≤~5% CRPS; PSD hard floor held (`g0_verifying_results.json`). |
| **Thick-2 protocol** | Holdout **12 / 16 / 16** (ic33–ic44) staged on ARCO; beat-this frozen: val **&lt; 1.988588** / test **&lt; 1.918383** / +120h **≤ 2.178842**. |
| **Year hard-lock** | Train **2018–2021** / val **2022** / test **2023–2024**. `provisional_years=false`, `year_split_frozen=true`. |
| **Living residual** | Tier-A **v1.3 joint** — INTERIM PASS CONFIRM · PROMOTE. Pooled t2m RMSE: **val 1.770** / **test 1.775** (16-IC) / **val +120h 1.982**. Wind-vector lead-mean: val **0.69560** / test **0.73877**. |

Exact floats (16-IC claim / wind_score): val **1.770** / test **1.775** / val+120h **1.982** · WV **0.69560 / 0.73877**.

## What failed / null

| Piece | Verdict |
| --- | --- |
| **v1.2** | **FAIL** (frozen) — +120h gate miss; keep `tier_a/v1_2/` as historical FAIL. |
| **v1-diff** | **NULL** — leftover-target diffusion ≈ living residual (Δ ~1e-5); ensemble worse. **Do not promote.** Stop leftover-target diffusion scaling. |

## Living path (canonical)

1. Weights: `runs/phase0/tier_a/v1_3_joint/best_residual.pt`
2. Results: `tier_a_v1_3_joint_results.json` · 16-IC claim: `tier_a_v1_3_joint_results_claim_16ic.json` · winds: `v1_3_joint_wind_score.json`
3. Prior living `v1_2b_thick2_train/` → **historical frozen** (do not overwrite). Prior ZS thick-2 (`v1_2b_thick2/`) and expand-train v1.2b (`v1_2b/`) → **historical**, still frozen.

## Non-claims (honesty)

- Label stays **`interim_era5`** (ERA5 interim targets; not IMDAA).
- **`g1_claimable=false`** — no G1 / IMDAA / Canvas claim from these bars.
- Nepal CDS crop is **archive for later**, not IC source (ICs = global ARCO).
- Interim thick-2 ≠ FINAL / G1 — see FINAL suite docs.

## Background

- Nepal ERA5 CDS crop still filling — leave download alone.
- No new architecture train ordered; idle / science writeup / doc hygiene while CDS fills. **No GPU / no train / no .pt writes** this pass.

See also: `LIVING_INDEX.md`, `FCN_STATUS_AND_NEXT.md`, `TIER_A_V1_3_JOINT_CALL.md`, `TIER_A_V1_DIFF_CALL.md`, `YEAR_HARD_LOCK.md`, `ERA5_COVERAGE_AUDIT.md`.
