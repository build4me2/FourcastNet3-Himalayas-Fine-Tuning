# Tier-A v1-diff call (Leonard) — 2026-09-14

**Artifacts (Spark)**
- `~/fourcastnet/runs/phase0/tier_a/v1_diff/tier_a_v1_diff_results.json`
- `best_diffusion.pt` same dir
- Recipe: `TIER_A_V1_DIFF_RECIPE.md`

**Arch:** custom 2D EDM `ResidualEDM2D` · ~111k params · target ERA5−(FCN3+frozen residual mean) · FCN3 + living residual **FROZEN**

## Call

| Layer | Result |
| --- | --- |
| Thick-2 bars (val/test/+120h) | **PASS** on one-step 0-noise decode |
| vs living residual (material lift) | **FAIL** — Δ val ~**−1.7e-5** (noise); test slightly worse |
| ens-mean K=4 | **Worse** (val ~1.94 / test ~1.97) — do not use for promote |
| `diff_improves_residual` | **false** |
| **Product / INTERIM PASS as upgrade** | **NULL — NO MATERIAL LIFT** |
| Promote diffusion over living residual? | **NO** |

## Living residual (unchanged)

**Canonical remains:** `runs/phase0/tier_a/v1_2b_thick2_train/`  
v1-diff dir kept as **null experiment artifact** (do not delete; do not overwrite living).

## Headline (one-step 0-noise)

| | Living | v1-diff |
| --- | ---: | ---: |
| Val | 1.7599 | 1.7598 |
| Test | 1.7798 | 1.7799 |
| Val +120 h | ~1.962 | 1.962 |

Diffusion collapsed to near-identity leftover (expected when target is residual-of-residual after a strong mean).

## Next eng (Howard)

1. Mirror already on Spark `docs/research/TIER_A_V1_DIFF_CALL.md` (this file). Copy to manii when reachable.
2. **Stop scaling this v1-diff setup** (recipe kill: no lift → don’t grow capacity on same target).
3. Idle on GPU science unless Manisha assigns a **different** Phase-3 fork, e.g.:
   - diffusion target = ERA5−FCN3 **without** residual mean in the target (parallel mean, not leftover), or
   - winds/CRPS-first probabilistic gate, or
   - pause Phase 3 and write up residual-only story.
4. CDS 611595 leave alone.

## Non-claims
- Not CorrDiff NVIDIA parity  
- Not a license to Tier-B  
- Null is a valid publishable outcome (residual mean sufficient under this protocol)  
