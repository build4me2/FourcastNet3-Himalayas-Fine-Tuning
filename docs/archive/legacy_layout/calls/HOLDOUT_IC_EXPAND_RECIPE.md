# Holdout IC expand recipe (Leonard) — 2026-09-12

**Manisha unlock:** expand holdout ICs first · **no** year hard-lock yet · **no** diffusion.  
**Years (still provisional):** train 2018–2021 / val 2022 / test 2023–2024.  
**Thin set now:** 8 / 4 / 4 (manifest `data/tier0_ics/tier0_ic_manifest.json`).

## Goal

Thicken **val + test** so N=4 stops binding claims. Keep train at **8** (reuse). Re-score Tier-0 beat-this and Tier-A v1.1 **before** any new Tier-A architecture.

## Target counts

| Split | Years | Now | **Target** | Action |
| --- | --- | ---: | ---: | --- |
| train | 2018–2021 | 8 | **8** | keep (no change this round) |
| val | 2022 | 4 | **12** | add **8** new |
| test | 2023–2024 | 4 | **12** | add **8** new (≈4 / year) |
| **Total** | | 16 | **32** | |

All ICs: **global** ERA5 ARCO, 72×721×1440, init **00 UTC**. Nepal 21×37 crop ≠ IC.

## Balance rules (hard)

1. **Season (per split):** after expand, each of **DJF / MAM / JJA / SON** has **≥3** ICs in val and **≥3** in test (thin set lacked MAM/SON — fix that).
2. **Region class (per split, soft):** mix `non_asia` / `asia_ex_nepal` / `monsoon_adjacent`; no single class &gt;50% of the split.
3. **Cadence:** ≤1 IC per calendar month per split-year (avoid near-duplicates). Prefer mid-month dates (…-15) unless colliding with an existing IC.
4. **No Nepal-box IC:** monsoon_adjacent labels are regime tags only; state is still full globe.
5. **IDs:** keep ic01–ic16; new ICs continue **ic17+**. Do not renumber.
6. **Manifest:** `provisional_years=true`; append-only expand; write `holdout_expand: v1` in manifest metadata.

## Concrete new IC list (stage these)

### Val 2022 — add ic17–ic24

| id | time (00Z) | season | region_class | region_label |
| --- | --- | --- | --- | --- |
| ic17 | 2022-02-15 | DJF | non_asia | N_Atlantic |
| ic18 | 2022-03-15 | MAM | asia_ex_nepal | East_Asia |
| ic19 | 2022-04-15 | MAM | non_asia | SH_midlat |
| ic20 | 2022-05-15 | MAM | monsoon_adjacent | Bay_of_Bengal_premonsoon |
| ic21 | 2022-08-15 | JJA | asia_ex_nepal | West_Pacific_typhoon |
| ic22 | 2022-09-15 | SON | monsoon_adjacent | Bay_of_Bengal_retreat |
| ic23 | 2022-10-15 | SON | non_asia | N_Pacific |
| ic24 | 2022-11-15 | SON | asia_ex_nepal | East_Asia |

**Val season totals after:** DJF 3 (Jan/Feb/Dec) · MAM 3 · JJA 3 (Jun/Jul/Aug) · SON 3.

### Test 2023–2024 — add ic25–ic32

| id | time (00Z) | season | region_class | region_label |
| --- | --- | --- | --- | --- |
| ic25 | 2023-03-15 | MAM | non_asia | N_Atlantic |
| ic26 | 2023-05-15 | MAM | monsoon_adjacent | Bay_of_Bengal_premonsoon |
| ic27 | 2023-09-15 | SON | asia_ex_nepal | West_Pacific |
| ic28 | 2023-11-15 | SON | monsoon_adjacent | NE_India_Myanmar_fringe |
| ic29 | 2024-02-15 | DJF | non_asia | SH_midlat |
| ic30 | 2024-04-15 | MAM | asia_ex_nepal | East_Asia |
| ic31 | 2024-07-12 | JJA | non_asia | N_Pacific |
| ic32 | 2024-10-15 | SON | monsoon_adjacent | Bay_of_Bengal_retreat |

**Test season totals after:** DJF 3 · MAM 3 · JJA 3 · SON 3.

If an ARCO date is missing/corrupt: shift **±1 day** same month; document in manifest `time_adjusted_from`.

## Staging / eng order (Howard)

1. Append ic17–ic32 to `configs/tier0_holdout_ics.yaml` + stage global `.npy` under `data/tier0_ics/`.
2. Update `tier0_ic_manifest.json` (counts 8/12/12).
3. Build pairs for **new ICs only** (forecast crop rollout + ERA5 interim targets) into a new pairs root, e.g. `runs/phase0/tier0_holdout_expand/pairs/` (do not delete thin-set pairs).
4. **Tier-0 re-score (required):**
   - Re-fit elev-binned linear on **train ic01–ic08 only** (same as before).
   - Score **full val ic09–ic24** and **full test ic13+ic25–ic32** (all 12+12).
   - Write `runs/phase0/tier0_holdout_expand/tier0_holdout_expand_metrics.json`.
   - Keep thin-set `tier0_holdout/` frozen as historical.
5. **Tier-A v1.1 zero-shot re-score (required before any new Tier-A train):**
   - Load frozen `runs/phase0/tier_a/v1_1/best_residual.pt` — **do not retrain**.
   - Score thickened val/test with same three gates vs **new** Tier-0 expand bars (and report vs old bars for honesty).
   - Write `runs/phase0/tier_a/v1_1_expand/tier_a_v1_1_expand_results.json`.
6. Ping Leonard with both JSONs — Leonard freezes updated beat-this and confirms/revokes v1.1 INTERIM PASS on the thick set.

## Re-score policy (frozen)

| Work | Allowed now? |
| --- | --- |
| Stage + pair new ICs | YES |
| Tier-0 re-fit train / score expand | YES (required) |
| v1.1 zero-shot on expand | YES (required) |
| New Tier-A train (v1.2 / v1-diff / etc.) | **NO** until Leonard calls expand results |
| Year hard-lock | **NO** (Manisha deferred) |
| Diffusion | **NO** |

### Expected Leonard call after expand (preview)
- New primary beat-this = **expand val** pooled t2m lin RMSE (replace 1.948661).
- New secondary = **expand test** lin RMSE (replace 1.850338).
- v1.1: PASS only if it still beats **new** bars AND val +120 h ≤ expand-val raw +120 h; else revoke interim PASS and iterate.

## Non-goals this round
- More train ICs  
- IMDAA / G1  
- Diffusion / FCN3 FT  
- Changing provisional year split  
