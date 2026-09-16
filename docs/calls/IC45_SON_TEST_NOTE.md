# ic45 optional test SON — note

**Date:** 2026-09-15 PT · **Mode:** stage + pairs + **eval_only** (zero-shot) · **NO retrain**

## Choice

| Field | Value |
| --- | --- |
| id | **ic45** |
| time | **2023-10-15T00:00:00** |
| season | **SON** |
| region_class | **asia_ex_nepal** |
| region_label | East_Asia |
| split | test |

**Why:** Thick-2 soft miss was test SON=3 (wanted ≥4). `2024-10-15` already used as **ic32** (monsoon_adjacent Bay_of_Bengal_retreat). Oct 2023 free. `asia_ex_nepal` balances region_class on test.

## Pipeline

1. ARCO global npy staged → `data/tier0_ics/ic_20231015T0000_global.npy`
2. Targets (ARCO crop, CDS untouched) + FCN3 forecast crop (GPU, 2 members) under `runs/phase0/tier0_holdout_expand_v2/pairs/`
3. Living residual **eval_only** → sidecar `runs/phase0/tier_a/v1_2b_thick2_train/ic45_zs_eval/` (parent weights/results **untouched**)

## Numbers (t2m RMSE tier-A)

| | Living (test16) | +ic45 ZS (test17) |
| --- | ---: | ---: |
| Val pooled | 1.759883 | 1.759883 |
| Test pooled | 1.779660 | 1.778875 |
| Val +120h | 1.962487 | 1.962487 |
| Test +120h | 1.838313 | 1.837104 |

Δ test pooled vs living: **-0.000785** K (noise-scale; still well under thick-2 test bar 1.918383).

## Non-claims

- Does **not** replace living residual bars (canonical remain test16 living JSON).
- `interim_era5` / `g1_claimable=false`.
- CDS PID **611595** left alone.

Sidecar JSON: `runs/phase0/tier_a/v1_2b_thick2_train/ic45_zs_eval/ic45_son_note.json`
