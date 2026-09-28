# FCN next unlock (Leonard) — 2026-09-13

**Manisha override (via Howard):** before any more FCN models / **no diffusion** / **no new Tier-A train** — unlock is either thicken ICs beyond 8/12/12 **or** hard-lock years.

**Living residual:** v1.2b INTERIM PASS · v1.2 FAIL frozen · expand bars live · ERA5 Nepal CDS PID 611595 still running (~1995).

## Freeze — order

| Priority | Unlock | Decision |
| --- | --- | --- |
| **1st** | **(B) Hard-lock years** | **NOW** |
| **2nd** | **(A) Thicken ICs** | After B; before any new model / diffusion |
| Blocked until A+B done | New Tier-A architecture, v1-diff, Tier-B | — |

**Why B first:** split is already what expand + v1.2b used; hard-lock is zero-GPU claim hygiene. Thickening under a still-provisional label leaves the same honesty hole. (Override if you wanted A-only first.)

---

## B — Year hard-lock (LOCKED)

| Split | Years | ICs (current) |
| --- | --- | --- |
| **Train** | **2018–2021** | ic01–ic08 |
| **Val** | **2022** | ic09–ic12 + ic17–ic24 (12) |
| **Test** | **2023–2024** | ic13–ic16 + ic25–ic32 (12) |

**Claim labels after B:**
- `provisional_years=false`
- `year_split_frozen=true` (**hard** — Manisha locked)
- `claim_level` stays `interim_era5` until IMDAA/obs
- `g1_claimable=false` until IMDAA/obs

**Eng (Howard) — do now (CPU/docs only):**
1. Patch configs + manifests to echo hard-lock labels.
2. Write short `docs/research/YEAR_HARD_LOCK.md` on Spark pointing at this call.
3. **Do not** retrain Tier-A solely for the label flip.
4. Optional: zero-shot re-tag v1.2b JSON with new labels (no weight change).
5. Mirror this file to `~/fourcastnet/docs/research/FCN_NEXT_UNLOCK.md` on Spark.

**Bars:** unchanged expand beat-this (val < 1.987014 / test < 1.897298 / +120h ≤ 2.131516) until unlock A re-scores.

---

## A — Thicken ICs (LOCKED recipe; run after B)

| Split | Now | **Target** | Add |
| --- | ---: | ---: | ---: |
| train | 8 | **12** | +4 in 2018–2021 |
| val | 12 | **16** | +4 in 2022 |
| test | 12 | **16** | +4 across 2023–2024 |
| **Total** | 32 | **44** | +12 |

**Balance (hard):** after thicken, each of DJF/MAM/JJA/SON has **≥4** ICs in val and **≥4** in test; train keeps ≥2 per season. ≤1 IC per calendar month per split-year. Global ERA5 **ARCO** ICs only (Nepal CDS crop ≠ IC). Continue ids **ic33+**.

### Concrete adds

**Train +4 (ic33–ic36)**

| id | time | season | region_class | label |
| --- | --- | --- | --- | --- |
| ic33 | 2018-04-15 | MAM | non_asia | N_Atlantic |
| ic34 | 2019-10-15 | SON | asia_ex_nepal | East_Asia |
| ic35 | 2020-03-15 | MAM | monsoon_adjacent | Bay_of_Bengal_premonsoon |
| ic36 | 2021-09-15 | SON | non_asia | N_Pacific |

**Val +4 (ic37–ic40)**

| id | time | season | region_class | label |
| --- | --- | --- | --- | --- |
| ic37 | 2022-01-05 | DJF | asia_ex_nepal | East_Asia |
| ic38 | 2022-04-05 | MAM | monsoon_adjacent | Bay_of_Bengal_premonsoon |
| ic39 | 2022-07-05 | JJA | non_asia | tropics_Atlantic_Africa |
| ic40 | 2022-10-05 | SON | asia_ex_nepal | West_Pacific |

**Test +4 (ic41–ic44)**

| id | time | season | region_class | label |
| --- | --- | --- | --- | --- |
| ic41 | 2023-02-15 | DJF | monsoon_adjacent | NE_India_Myanmar_fringe |
| ic42 | 2023-06-15 | JJA | non_asia | SH_midlat |
| ic43 | 2024-03-15 | MAM | non_asia | N_Atlantic |
| ic44 | 2024-08-15 | JJA | asia_ex_nepal | West_Pacific_typhoon |

±1 day OK if ARCO hole; document `time_adjusted_from`.

### Re-score after A (required before any new model)
1. Stage ARCO globals + pairs for new ICs only.
2. Tier-0: re-fit train-only on **ic01–08+ic33–36**; score full val16 + test16 → new beat-this (val primary).
3. v1.2b **zero-shot** on thick-2 set (no retrain) → confirm/revoke INTERIM PASS.
4. Leonard freezes new bars + call. **Only then** new architecture / diffusion may unlock.

---

## While ERA5 Nepal CDS archive runs (PID 611595)

| Do | Don’t |
| --- | --- |
| Finish **B** (labels/docs) | New Tier-A train / diffusion |
| Stage **A** ARCO ICs + pairs when ready (independent of Nepal CDS crop) | Block A on CDS reaching present |
| Watch/log CDS progress; don’t kill PID 611595 | Use Nepal crop files as FCN3 ICs |
| GitHub sync / docs | Overwrite frozen v0…v1_2b / G0 verifying |

Nepal CDS backfill is **regional archive** for later IMDAA/G1 — **not** a blocker for ARCO global IC thicken.

## Non-goals until A+B complete
- v1-diff / CorrDiff
- New residual capacity / Tier-B
- Claiming G1
