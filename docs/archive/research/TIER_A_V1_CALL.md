# Tier-A v1-reg call (Leonard) — 2026-09-11

**Artifacts**
- `~/fourcastnet/runs/phase0/tier_a/v1/tier_a_v1_results.json`
- `best_residual.pt` + metrics CSV same dir
- Note: `docs/research/TIER_A_V1_NOTE.md`
- Sketch freeze: `docs/research/TIER_A_V1_SKETCH_CALL.md`

**Method:** ElevCondResidualUNet (~665k), w_120=3.0, composite ckpt (reject +120 h regress), FCN3 frozen, no diffusion.

## Call

| Gate | Result |
| --- | --- |
| Val pooled &lt; 1.948661 | **PASS** — 1.796 |
| Test pooled &lt; 1.850338 | **FAIL** — **1.950** (worse than raw 1.893) |
| Val +120 h ≤ raw 2.220 | **PASS** — 2.018 (−0.202) |
| **Tier-A v1-reg INTERIM PASS** | **NO — FAIL** |
| Start v1-diff? | **NO** |

Diagnosis: +120 h hard gate is **operational** (good), but the larger head **overfit** 4 val ICs (best eligible ~ep 100). Val-only + long-lead win with test regression = classic small-N FAIL.

## Headline vs prior

| Split | raw | v0 | v0.1 | v1-reg |
| --- | ---: | ---: | ---: | ---: |
| val | 1.949 | 1.892 | 1.889 | **1.796** |
| test | 1.893 | **1.733** | 1.786 | **1.950** ✗ |
| val +120 h | 2.220 | 2.382 | 2.354 | **2.018** ✓ |

v0 remains best test pooled on the shelf. v1-reg wins val/+120 h but loses the holdout that matters for PASS.

## Next — v1.1 iterate (reg/capacity), still no diffusion

New dir: `runs/phase0/tier_a/v1_1/` (keep `v1/` frozen as FAIL reference).

| Knob | v1 | **v1.1 default** |
| --- | --- | --- |
| `base_channels` | 32 | **16 or 24** |
| dropout | 0.10 | **0.20** |
| `w_120` | 3.0 | **1.5** (still ≥1; keep composite +120 h reject) |
| weight_decay | (as v1) | **≥ 1e-3** if not already |
| epochs / patience | 400 / 80 | keep composite; consider patience **40–60** |
| data | 8/4/4 | unchanged until Manisha year hard-lock / more ICs |

### Still required for v1.1 PASS
Same three gates as sketch (val bar, test bar, val +120 h ≤ raw).

### Explicit non-moves
- **No v1-diff / CorrDiff** until a v1.x cut passes test (pathway: debug bars before diffusion).
- No FCN3 weight FT.
- Do not overwrite `v0/`, `v0_1/`, or `v1/`.

### Optional (only if v1.1 still test-fails)
- Milder model + leave-one-IC-out on train for early-stop proxy (still never tune on test).
- Ask Manisha for more val/test ICs or year hard-lock — N=4 is the binding constraint.

## Next eng (Howard)

1. Freeze `v1/` as FAIL artifact.
2. Implement **v1.1** with smaller/regularized residual + `w_120=1.5`; ping `tier_a_v1_1_results.json`.
3. Do **not** start diffusion.

## Non-claims
- Not plateau (missed test bar — underfit/overfit issue, not skill ceiling for diffusion)
- Not G1 / CorrDiff / FCN3 FT  
