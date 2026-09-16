# Holdout thick-2 call (Leonard) — 2026-09-13

**Unlock A complete** (after B year hard-lock).  
**Artifacts (Spark)**
- Tier-0: `~/fourcastnet/runs/phase0/tier0_holdout_expand_v2/tier0_holdout_expand_v2_metrics.json`
- v1.2b ZS: `~/fourcastnet/runs/phase0/tier_a/v1_2b_thick2/tier_a_v1_2b_thick2_results.json`
- Prior expand (historical): `tier0_holdout_expand/` · living residual weights: `tier_a/v1_2b/best_residual.pt`

**Set:** train **12** / val **16** / test **16** · `holdout_expand=v2` · `provisional_years=false` · `year_split_frozen=true` HARD

## Call

| Layer | Result |
| --- | --- |
| Thick-2 protocol (12/16/16) | **PASS** (all ic33–ic44 staged; no ARCO holes) |
| **New beat-this** | **FROZEN** (below) — supersedes expand-v1 bars |
| **v1.2b INTERIM PASS on thick-2** | **CONFIRM YES** (zero-shot vs new bars + thick-2 val +120 h raw) |
| New architecture / diffusion | Still **NO** until Manisha assigns (A+B unlocks *eligibility*, not auto-start) |

## New beat-this (freeze now)

Script `recommended_beat_this_source=test` **rejected** — primary stays **val**.

| Role | Split | Exact lin RMSE | Headline |
| --- | --- | ---: | ---: |
| **Primary** | val 2022 (16 ICs) | strictly **< 1.988588** | < **1.989 K** |
| **Secondary** | test 2023–24 (16 ICs) | strictly **< 1.918383** | < **1.918 K** |
| **+120 h hard** | val +120 h ≤ thick-2 raw | ≤ **2.178842** | use thick-2 raw |

Expand-v1 bars (val 1.987014 / test 1.897298 / +120h 2.131516) → **historical only**.

## v1.2b zero-shot vs new bars

| Gate | Number | Bar | Result |
| --- | ---: | ---: | --- |
| Val pooled | **1.829** | < 1.988588 | **PASS** |
| Test pooled | **1.897** | < 1.918383 | **PASS** |
| Val +120 h | **2.015** | ≤ 2.178842 | **PASS** (−0.164) |
| Eligible-ckpt rule | ZS / `eval_only_frozen` | exempt | OK |

**CONFIRM:** v1.2b remains **INTERIM PASS** on thick-2 (weights unchanged).  
`claim_level=interim_era5` · `g1_claimable=false` · years hard-locked.

### Honesty
- Test pooled **1.8965** is tight vs old expand test bar 1.8973 (would still pass old); comfortable vs new 1.918.
- Test +120 h still regresses vs raw (1.955 vs 1.832) — not a gate.
- **Recipe soft miss:** test **SON=3** (wanted ≥4). Val SON=4 OK. Not a protocol FAIL; optional later add one test SON IC (ic45) without reopening architecture.

## Next eng (Howard)

1. Mirror this call to `~/fourcastnet/docs/research/HOLDOUT_THICK2_CALL.md`.
2. Echo **thick-2** bars in future result JSON templates.
3. Keep CDS 611595 alone; no new Tier-A train / diffusion until Manisha assigns.
4. Optional (low priority): add one test SON IC to hit ≥4; re-score test only if she wants recipe-perfect seasons.

## Non-claims
- Not G1 / CorrDiff / FCN3 FT  
- Not auto-license for v1-diff — Manisha must assign  
