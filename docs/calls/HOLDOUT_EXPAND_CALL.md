# Holdout expand call (Leonard) — 2026-09-12

**Artifacts**
- Tier-0: `~/fourcastnet/runs/phase0/tier0_holdout_expand/tier0_holdout_expand_metrics.json`
- v1.1 zero-shot: `~/fourcastnet/runs/phase0/tier_a/v1_1_expand/tier_a_v1_1_expand_results.json`
- Recipe: `docs/calls/HOLDOUT_IC_EXPAND_RECIPE.md`
- Thin set remains frozen under `runs/phase0/tier0_holdout/` and `tier_a/v1_1/`

**Set:** train 8 / val **12** / test **12** · `holdout_expand=v1` · provisional years unchanged · no retrain of v1.1

## Call

| Layer | Result |
| --- | --- |
| Expand protocol (8/12/12, seasons) | **PASS** |
| **New beat-this** | **FROZEN** (below) — supersedes thin-set bars for all future Tier-A |
| **v1.1 INTERIM PASS on thick set** | **CONFIRM YES** (zero-shot vs new bars + expand-val +120 h raw) |
| Diffusion / new Tier-A train | **Still NO** unless Manisha orders |

## New beat-this (freeze now)

Script `recommended_beat_this_source=test` is **rejected** — primary stays **val** (designated holdout year; do not cherry-pick easier test).

| Role | Split | Exact lin RMSE bar | Headline |
| --- | --- | ---: | ---: |
| **Primary** | val 2022 (12 ICs) | strictly **< 1.987014** | < **1.987 K** |
| **Secondary** | test 2023–24 (12 ICs) | strictly **< 1.897298** | < **1.897 K** |
| **+120 h hard (v1.x)** | val +120 h ≤ **expand-val raw +120 h** | ≤ **2.131516** | use expand raw, not thin 2.220 |

Thin-set bars (val 1.948661 / test 1.850338 / raw+120h 2.220) remain **historical only** (v0/v0.1/v1/v1.1 thin calls).

## v1.1 zero-shot vs new bars

| Gate | Number | Bar | Result |
| --- | ---: | ---: | --- |
| Val pooled | **1.808** | < 1.987014 | **PASS** |
| Test pooled | **1.826** | < 1.897298 | **PASS** |
| Val +120 h | **1.998** | ≤ 2.131516 expand raw | **PASS** (−0.133 vs expand raw) |

**CONFIRM:** v1.1 remains **INTERIM PASS** on the thick holdout (zero-shot; ckpt unchanged).
claim_level stays `interim_era5`; `g1_claimable=false`; `provisional_years=true`.

### Honesty deltas (thin → thick)

| | Thin v1.1 | Thick zero-shot |
| --- | ---: | ---: |
| Val | 1.716 | 1.808 |
| Test | 1.812 | 1.826 |
| Val +120 h | 2.015 | 1.998 |

Slight pooled degradation on thicker N (expected); still clears **new** (slightly softer) Tier-0 val bar. Test +120 h still regresses vs raw on expand (1.819 vs 1.733) — not a gate; document only.

Note: Howard expand results JSON still echoes **old** thin bars in `beat_this` / `gates` text — mechanical `tier_a_interim_pass=true` there is against thin refs. **This call** is authoritative for expand bars.

## Next eng (Howard)

1. Freeze expand metrics + this call as the living beat-this reference.
2. Patch future result JSON templates to echo **expand** bars (1.987014 / 1.897298 / raw+120h 2.131516).
3. **No** new Tier-A train and **no** diffusion until Manisha asks — v1.1 thick PASS stands.
4. Optional later: more ICs or year hard-lock; then re-score again.

## Non-claims
- Not G1 / IMDAA / CorrDiff / FCN3 FT
- Not year hard-lock
- Not license to start v1-diff
