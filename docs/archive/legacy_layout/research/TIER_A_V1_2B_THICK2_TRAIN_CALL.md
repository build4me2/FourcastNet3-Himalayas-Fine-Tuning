# Tier-A v1.2b thick-2 retrain call (Leonard) — 2026-09-14

**Artifacts (Spark)**
- `~/fourcastnet/runs/phase0/tier_a/v1_2b_thick2_train/tier_a_v1_2b_thick2_train_results.json`
- `best_residual.pt` same dir
- Recipe: `TIER_A_V1_2B_THICK2_TRAIN_RECIPE.md` · bars: `HOLDOUT_THICK2_CALL.md`

## Call

| Gate | Result |
| --- | --- |
| Val pooled < 1.988588 | **PASS** — **1.760** |
| Test pooled < 1.918383 | **PASS** — **1.780** |
| Val +120 h ≤ 2.178842 | **PASS** — **1.962** (Δ −0.216) |
| Eligible ckpt (≥1) | **PASS** — **52** saves · reload=`composite_eligible` (ep 395) |
| **INTERIM PASS** | **YES — CONFIRM** |
| Diffusion / G1 / FCN3 FT | **NO** |

## Living residual (LOCKED)

| Role | Path |
| --- | --- |
| **NEW living** | `runs/phase0/tier_a/v1_2b_thick2_train/` |
| Prior expand-train v1.2b | `v1_2b/` → historical (still frozen) |
| Thick-2 ZS | `v1_2b_thick2/` → **historical** (ZS demoted) |
| v1.2 FAIL | `v1_2/` → keep |

Do **not** overwrite any of the above.

## vs zero-shot thick-2

| | ZS | **Retrain** |
| --- | ---: | ---: |
| Val | 1.829 | **1.760** |
| Test | 1.897 | **1.780** |
| Val +120 h | 2.015 | **1.962** |

Retrain clearly dominates ZS under the same bars — expected when early-stop uses thick-2 val.

Honesty: test +120 h still ≈ raw (1.838 vs 1.832, tiny regress) — **not** a gate.

## Next

1. Howard: mirror this call to Spark `docs/research/`; echo living path in status docs.
2. **No diffusion** unless Manisha assigns Phase 3.
3. Optional later: seed sweep / winds table / test SON+1 IC — not required for this PASS.

## Non-claims
- Not G1 / IMDAA / CorrDiff / published skill  
- interim_era5 · years hard-locked  
